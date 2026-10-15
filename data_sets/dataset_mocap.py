import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
import numpy as np
from cprint import cprint
from .dataset_base import DatasetBase
from data_sets.common.quaternion import *
from data_sets.motion_representation import rotate_repr_smpl, get_repr_smplx, transform_repr_smpl
from data_sets.motion_representation import rotate_repr_smpl_2, face_joint_indx
from utils.other_utils import SMPLH_KINTREE
from utils.representation_registry import RepresentationRegistry
from typing import Tuple, Dict, Optional, Any, List
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Literal
import tqdm

from .motion_aware_sampling import (
    compute_motion_stats,
    compute_sample_weights,
    MotionStats,
)

class DatasetMocap(DatasetBase):
    """
    A base dataloader for pre-processed motion capture datasets (AIST, TotalCapture, 3DPW).
    Handles common data processing tasks like normalization and batch formatting.
    """
    def __init__(self, opt: dict,
                input_noise=True,
                noise_std_joint=0.0,
                 noise_std_smplx_global_rot=0.0,
                 noise_std_smplx_body_rot=0.0,
                 noise_std_smplx_trans=0.0,
                 noise_std_smplx_betas=0.0,
                 clip_len=33,
                 cano_jts_pos: bool = False,
                 slice_strategy: Literal['random_uniform_len', 'full_sequence',
                                        'high_motion_weighted', 'high_motion_focused']='full_sequence',
                 align_to_yaxis: bool = False,
                 norm_params_path='',
                 # Motion-aware sampling parameters
                 motion_weight_strategy: Literal['jerk', 'accel', 'combined'] = 'combined',
                 motion_weight_temperature: float = 1.0,
                 motion_weight_min: float = 0.1,
                 motion_fps: float = 30.0,
                 repr_registry: RepresentationRegistry = None,
                 mask_scheme_registry=None,
                 allow_eval_norm_params_compute: bool = False,
                 save_eval_norm_params: bool = False,
                **kwargs):
        super().__init__(repr_registry=repr_registry, mask_scheme_registry=mask_scheme_registry, **kwargs)
        self.root = opt['root']
        self.split = opt['split']

        self.num_joints = 22 # Standardize to 22 body joints.
        self.input_noise = input_noise
        self.clip_len = clip_len
        self.align_to_yaxis = align_to_yaxis
        
        # Motion-aware sampling config
        self.motion_weight_strategy = motion_weight_strategy
        self.motion_weight_temperature = motion_weight_temperature
        self.motion_weight_min = motion_weight_min
        self.motion_fps = motion_fps
        self.motion_stats_list: List[MotionStats] = []
        self.sample_weights: Optional[np.ndarray] = None
        self.allow_eval_norm_params_compute = allow_eval_norm_params_compute
        self.save_eval_norm_params = save_eval_norm_params
        
        # This data attribute will be populated by the child classes (AIST, etc.)
        self.data = None
        # NOTE: train_norm_params.pkl is the default because norm parameters are produced during training and therefore named `*_train_norm_params.pkl`.
        default_norm_filename = f"{self.__class__.__name__}_train_norm_params.pkl"
        self.norm_params_path = self._resolve_norm_params_path(norm_params_path, default_norm_filename)
        self._norm_params_ready = False
        # assert 'R_world_joints' not in self._repr_list, "Global joint rotations not supported in MocapDataset"
        # assert 'local_vel' not in self._repr_list, "Relative velocity representation required (repr_abs_only=True)"
        self._slice_strategy = slice_strategy
        self.cano_jts_pos = cano_jts_pos
        self._cumsum_clips: np.ndarray = np.array([0])  # Initialized properly in _precompute_clip_cumsum

        assert hasattr(self, "floor_height_offset"), "Child class must define self.floor_height_offset"

        self.noise_std_params_dict = {'global_orient': noise_std_smplx_global_rot,
                                      'transl': noise_std_smplx_trans,
                                      'body_pose': noise_std_smplx_body_rot,
                                      'betas': noise_std_smplx_betas,}

    def _precompute_clip_cumsum(self):
        """
        Precompute cumulative sum of sequence lengths for O(log n) clip-to-sequence mapping.
        Must be called by child classes after loading self.data.
        """
        if self.data is None or 'pose' not in self.data:
            return
        
        seq_lengths = np.array([len(seq) for seq in self.data['pose']])
        # cumsum[i] = total frames from sequences [0, i), cumsum[0] = 0
        self._cumsum_clips = np.concatenate([[0], np.cumsum(seq_lengths)])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        if self.data is None or 'name' not in self.data:
            return 0
        if self._slice_strategy == 'full_sequence':
            return len(self.data['pose'])
        elif self._slice_strategy in ('random_uniform_len', 'high_motion_weighted', 'high_motion_focused'):
            # Use precomputed cumsum for total frame count
            total_frames = int(self._cumsum_clips[-1])
            return total_frames // self.clip_len
        else:
            raise ValueError(f"Invalid slice strategy: {self._slice_strategy}")

    def _get_item_from_data(self, index: int) -> tuple[str, torch.Tensor]:
        """
        Abstract method to be implemented by child classes.
        It should fetch the sequence name and 3D joint data for a given index.
        """
        raise NotImplementedError("Child dataloader must implement _get_item_from_data.")

    def __getitem__(self, index: int) -> dict:
        return self._build_item(index)

    def _build_item(self, index: int, compute_noise: bool = True) -> dict:
        """
        Retrieves, processes, and formats a data item at a given index.
        """
        # 1. Fetch the raw sequence data using the child class's implementation.
        if self._slice_strategy == 'full_sequence':
            name = self.data['name'][index]
        elif self._slice_strategy == 'random_uniform_len':
            name = index
        else:
            raise ValueError(f"Invalid slice strategy: {self._slice_strategy}")
        name, smplx_params_dict = self._get_item_from_data(index)
        T, *_ = smplx_params_dict['transl'].shape

        # First align to the inertial frame with homogenous_transform.
        J = self.smplx_neutral(**smplx_params_dict).joints[:, 0:22, :]  # Shape: (T, 22, 3)
        repr_dict_clean = {
            "smplx_trans": smplx_params_dict['transl'],
            "smplx_rot_6d": rotmat_to_rot6d(aa_to_rotmat(smplx_params_dict['global_orient'].reshape(-1, 3))).reshape(smplx_params_dict['global_orient'].shape[0], 1, 6),
            "smplx_body_pose_6d": rotmat_to_rot6d(aa_to_rotmat(smplx_params_dict['body_pose'].reshape(-1, 3))).reshape(smplx_params_dict['body_pose'].shape[0], 21, 6),
            "smplx_betas": smplx_params_dict['betas'],
            "local_positions": J.reshape(J.shape[0], -1),
    }
        rot_repr_dict = rotate_repr_smpl_2(repr_dict_clean, rotation_matrix=torch.from_numpy(self.homogenous_transform_align_[:3, :3]).float().to(smplx_params_dict['transl'].device))

        PER_FRAME_FLOOR_HEIGHT_OFFSET = True
        first_frame_root_xy = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[0:1, 0:1, :2] # Shape: (1, 1, 2)
        if not PER_FRAME_FLOOR_HEIGHT_OFFSET:
            # Compute floor_height_offset and the x/y frame offset from the rotated repr.
            floor_height_offset = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[..., 2].min() + self.floor_height_offset

            # Option1: use constant floor height offset.
            transform = torch.tensor([[1, 0, 0, -first_frame_root_xy[0,0,0]],       
                                    [0, 1, 0, -first_frame_root_xy[0,0,1]],       
                                    [0, 0, 1, -floor_height_offset],       
                                    [0, 0, 0, 1]], dtype=torch.float32).to(smplx_params_dict['transl'].device)

        else:
            # Option2: use per-timestamp floor height offset
            floor_height_offset = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[..., 2].amin(dim=1, keepdim=False)
            transform = torch.eye(4).unsqueeze(0).to(smplx_params_dict['transl'].device).repeat(T, 1, 1)
            transform[:, 0, 3] = -first_frame_root_xy[0,0,0]
            transform[:, 1, 3] = -first_frame_root_xy[0,0,1]
            transform[:, 2, 3] = -floor_height_offset

        cano_repr_dict = transform_repr_smpl(rot_repr_dict, transform=transform)

        cano_smplx_params_dict = {
            "transl": cano_repr_dict["smplx_trans"].numpy(force=True),
            "body_pose": rotmat_to_aa(rot6d_to_rotmat(cano_repr_dict['smplx_body_pose_6d'])).numpy(force=True),
            "global_orient": rotmat_to_aa(rot6d_to_rotmat(cano_repr_dict['smplx_rot_6d'])).numpy(force=True),
            "betas": cano_repr_dict['smplx_betas'].numpy(force=True),
        }

        
        cano_smplx_params_tensor_dict = {
            "transl": cano_repr_dict['smplx_trans'],
            "body_pose": rotmat_to_aa(rot6d_to_rotmat(cano_repr_dict['smplx_body_pose_6d'])).reshape(-1, 63),
            "global_orient": rotmat_to_aa(rot6d_to_rotmat(cano_repr_dict['smplx_rot_6d'])),
            "betas": cano_repr_dict['smplx_betas'],
            "jaw_pose": torch.zeros((T, 3)).to(self.device),
            "leye_pose": torch.zeros((T, 3)).to(self.device),
            "reye_pose": torch.zeros((T, 3)).to(self.device),
            "left_hand_pose": torch.zeros((T, 45)).to(self.device),
            "right_hand_pose": torch.zeros((T, 45)).to(self.device),
            "expression": torch.zeros((T, 10)).to(self.device),
        }

        include_noise = self.input_noise and compute_noise
        if include_noise:
            cano_smplx_params_noisy_dict = {}
            for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
                if param_name == 'transl' or param_name == 'betas':
                    noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smplx_params_dict[param_name].shape)
                    cano_smplx_params_noisy_dict[param_name] = cano_smplx_params_dict[param_name] + noise_1
                elif param_name == 'global_orient':
                    global_orient_mat = R.from_rotvec(cano_smplx_params_dict['global_orient'])  # [145, 3, 3]
                    global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
                    noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                    global_orient_angle_noisy = global_orient_angle + noise_global_rot
                    global_orient_rotvec_noisy = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec().reshape(-1, 3)
                    cano_smplx_params_noisy_dict[param_name] = global_orient_rotvec_noisy
                elif param_name == 'body_pose':
                    body_pose_mat = R.from_rotvec(cano_smplx_params_dict['body_pose'].reshape(-1, 3))
                    body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
                    noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                    body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                    body_pose_rotvec_noisy = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
                    cano_smplx_params_noisy_dict[param_name] = body_pose_rotvec_noisy

            for param_name in ['jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
                cano_smplx_params_noisy_dict[param_name] = cano_smplx_params_tensor_dict[param_name].numpy(force=True)
            
            cano_smplx_params_tensor_noisy_dict = {k: torch.from_numpy(v).to(self.device, dtype=self.dtype) for k,v in cano_smplx_params_noisy_dict.items()}
            cano_positions_noisy = self.smplx_neutral(**cano_smplx_params_tensor_noisy_dict).joints[:, 0:22].detach().cpu().numpy()  # [clip_len, 22, 3]


        J = self.smplx_neutral(**cano_smplx_params_tensor_dict).joints[:, 0:22, :]  # (T, 22, 3)

        ######################################## create motion representation
        repr_dict_clean = get_repr_smplx(positions=cano_repr_dict['local_positions'].reshape(-1, self.num_joints, 3).numpy(force=True),
                                        cano_jts_pos=False,
                                       smplx_params_dict=cano_smplx_params_dict,
                                       feet_vel_thre=5e-5,
                                       smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                       device=self.device,
                                       modality_names=self._repr_list)  # a dict of reprs
        if include_noise:
            repr_dict_noisy = get_repr_smplx(positions=cano_positions_noisy,
                                            cano_jts_pos=False,
                                            smplx_params_dict=cano_smplx_params_noisy_dict,
                                            feet_vel_thre=5e-5,
                                            smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                            device=self.device,
                                            modality_names=self._repr_list)  # a dict of reprs

                
        
        align_to_yaxis = self.align_to_yaxis
        if align_to_yaxis:
            positions_clean = repr_dict_clean['local_positions'].reshape(-1, self.num_joints, 3)
            l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
            across1 = positions_clean[:, r_hip] - positions_clean[:, l_hip]
            across2 = positions_clean[:, sdr_r] - positions_clean[:, sdr_l]
            across = across1 + across2
            # Robust normalization to avoid NaN when across is near-zero
            across_norm = np.maximum(np.sqrt((across ** 2).sum(axis=-1)), 1e-8)
            across = across / across_norm[:, np.newaxis]
            forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
            # Robust normalization: when across is parallel to z-axis, forward is near-zero
            forward_norm = np.maximum(np.sqrt((forward ** 2).sum(axis=-1)), 1e-8)
            forward = forward / forward_norm[..., np.newaxis]

            target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
            root_rot_quat = qbetween_np(forward, target)
            # Handle NaN quaternions (from degenerate poses) by copying from adjacent frame
            if np.isnan(root_rot_quat).any():
                nan_frames = np.where(np.isnan(root_rot_quat).any(axis=-1))[0]
                for idx in nan_frames:
                    root_rot_quat[idx] = root_rot_quat[idx - 1] if idx > 0 else np.array([1.0, 0.0, 0.0, 0.0])

            rot_mat = torch.from_numpy(R.from_quat(root_rot_quat[0], scalar_first=True).as_matrix()).float()

            tensor_repr_dict_clean = {k: torch.from_numpy(v).float() for k, v in repr_dict_clean.items() if isinstance(v, np.ndarray)}
            if include_noise:
                tensor_repr_dict_noisy = {k: torch.from_numpy(v).float() for k, v in repr_dict_noisy.items() if isinstance(v, np.ndarray)}

            repr_dict_clean = rotate_repr_smpl_2(repr_dict=tensor_repr_dict_clean, rotation_matrix=rot_mat)
            if include_noise:
                repr_dict_noisy = rotate_repr_smpl_2(repr_dict=tensor_repr_dict_noisy, rotation_matrix=rot_mat)
                positions_noisy = repr_dict_noisy['local_positions'].reshape(-1, 22, 3).numpy()
            
            repr_dict_clean = {k: v.numpy(force=True) for k, v in repr_dict_clean.items()}
            if include_noise:
                repr_dict_noisy = {k: v.numpy(force=True) for k, v in repr_dict_noisy.items()}

        
        
        item_dict = {}
        batch_dtype = np.float32
        item_dict['motion_repr_clean'] = np.concatenate([repr_dict_clean[key] for key in self._repr_list], axis=-1).astype(batch_dtype) # [clip_len-1, 263]

        if include_noise:
            positions_noisy = repr_dict_noisy['local_positions'].reshape(-1, 22, 3)
            item_dict['noisy_joints'] = positions_noisy.astype(batch_dtype)  # [clip_len, 22, 3]
            item_dict['motion_repr_noisy'] = np.concatenate([repr_dict_noisy[key] for key in self._repr_list], axis=-1).astype(batch_dtype) # [clip_len-1, 263]

        # Meta-information for debugging and analysis.
        # Store transformation info for inverse recovery during visualization
        item_dict['meta'] = {
            'name': name,
            'split': self.split,
            'translation_xy_offset': first_frame_root_xy[0,0,:2].numpy(force=True),
            'floor_height_offset': floor_height_offset.numpy(force=True) if isinstance(floor_height_offset, torch.Tensor) else floor_height_offset,
            'homogenous_transform_align': self.homogenous_transform_align_,
            'yaxis_align_rot': rot_mat.numpy(force=True) if align_to_yaxis else np.eye(3),
            'original_trans_z': smplx_params_dict['transl'][:, 2].numpy(force=True),  # For precise inverse transform
        }

        dbg = False
        if dbg:
            from utils.repr_adapter import decode_smplx_from_full_repr
            from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
            motion_repr = torch.from_numpy(item_dict['motion_repr_clean']).unsqueeze(0).float()
            output_rot_flag = 'R_world_joints' in self._repr_list
            _, joints, verts = decode_smplx_from_full_repr(
                full_repr=motion_repr,
                repr_list=self._repr_list,
                repr_indices=self._repr_indices,
                smplx_model=self.smplx_neutral,
                output_global_rotations=output_rot_flag,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=True,
            )
            cfg = SmplxVisConfig(window_title="Mocap Debug", num_repeats=1, fixed_cam=False, camera_follow=True)
            visualize_smplx_sequence(
                joints=joints,
                vertices=verts,
                faces=self.smplx_neutral.faces,
                parent_indices=SMPLH_KINTREE,
                batch_idx=0,
                cfg=cfg,
            )

        return item_dict

    def _iter_sequences_for_norm(self):
        """
        Generator that yields repr_dict for each unique source sequence.
        
        This iterates over source files (not clips), processing entire sequences
        at once to avoid IO thrashing and redundant computation.
        
        Yields:
            Dict[str, np.ndarray]: Motion representation dict for each sequence
        """
        if self.data is None or 'name' not in self.data:
            return
        
        num_sequences = len(self.data['name'])
        
        # Temporarily switch to full_sequence mode for efficient processing
        original_strategy = self._slice_strategy
        self._slice_strategy = 'full_sequence'
        
        try:
            for seq_idx in range(num_sequences):
                try:
                    # _build_item with full_sequence strategy processes entire sequence
                    sample = self._build_item(seq_idx, compute_noise=False)
                    motion_repr = sample['motion_repr_clean']
                    
                    # Extract individual repr arrays
                    repr_dict = {}
                    for repr_name, (start, end) in self._repr_indices.items():
                        repr_dict[repr_name] = motion_repr[:, start:end]
                    
                    yield repr_dict
                    
                except Exception as e:
                    if self.logging_enabled:
                        seq_name = self.data['name'][seq_idx]
                        cprint.warn(f"[WARN] Skipping sequence {seq_name}: {e}")
                    continue
        finally:
            # Restore original strategy
            self._slice_strategy = original_strategy

    def _collect_repr_lists_for_norm(self) -> dict:
        """
        Efficiently collect normalization statistics using file-centric iteration.
        
        Instead of iterating over clips (which causes IO thrashing for datasets
        with many overlapping clips from few source files), this iterates over
        unique source sequences once, using streaming statistics.
        
        Returns:
            Pre-computed norm_params dict (not raw repr_lists)
        """
        if self.data is None or 'name' not in self.data:
            raise RuntimeError("Cannot compute normalization parameters: no data loaded.")
        
        num_sequences = len(self.data['name'])
        if num_sequences == 0:
            raise RuntimeError("Cannot compute normalization parameters: dataset is empty.")

        if self.logging_enabled:
            cprint.info(f"[INFO] Computing normalization params efficiently for "
                       f"{self.__class__.__name__} ({self.split}) using {num_sequences} sequences.")

        # Use streaming computation from base class
        return self._compute_norm_params_streaming(
            self._iter_sequences_for_norm(),
            num_items=num_sequences,
            desc=f"Processing {self.__class__.__name__} sequences"
        )

    def _compute_motion_stats_for_dataset(self):
        """Compute motion statistics for all sequences in the dataset."""
        if self.data is None:
            return
        
        self.motion_stats_list = []
        for idx in range(len(self.data['pose'])):
            # Get joint positions for this sequence
            name, smplx_params_dict = self._get_item_from_data(idx)
            J = self.smplx_neutral(**smplx_params_dict).joints[:, 0:22, :].detach().cpu().numpy()
            stats = compute_motion_stats(J, fps=self.motion_fps)
            self.motion_stats_list.append(stats)
        
        # Compute sample weights
        if self._slice_strategy in ('high_motion_weighted', 'high_motion_focused'):
            self.sample_weights = compute_sample_weights(
                self.motion_stats_list,
                weight_strategy=self.motion_weight_strategy,
                temperature=self.motion_weight_temperature,
                min_weight=self.motion_weight_min,
            )
            if self.logging_enabled:
                jerks = np.array([s.mean_jerk for s in self.motion_stats_list])
                accels = np.array([s.mean_accel for s in self.motion_stats_list])
                cprint.info(f'[INFO] Motion stats - Jerk: mean={jerks.mean():.2f}, std={jerks.std():.2f}; '
                           f'Accel: mean={accels.mean():.2f}, std={accels.std():.2f}')

    def get_weighted_sampler(self) -> 'torch.utils.data.WeightedRandomSampler':
        """
        Get a WeightedRandomSampler for use with DataLoader.
        """
        from torch.utils.data import WeightedRandomSampler
        
        n = len(self)
        if self._slice_strategy in ('high_motion_weighted', 'high_motion_focused') and self.sample_weights is not None:
            weights = self.sample_weights
        else:
            weights = np.ones(n) / n
        
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=n,
            replacement=True,
        )

    def _initialize_norm_params(self):
        """
        Compute or load normalization params once data has been loaded.
        """
        if self._norm_params_ready:
            return

        if self.split == 'train':
            # _collect_repr_lists_for_norm now returns norm_params directly
            self.norm_params = self._collect_repr_lists_for_norm()
            if (not self.use_ddp) or (self.dist_rank == 0):
                self._save_norm_params_to_path(self.norm_params_path, protocol=2)
            if self.use_ddp:
                dist.barrier()
        else:
            if self.use_ddp:
                dist.barrier()
            try:
                self.norm_params = self._load_norm_params_from_path(self.norm_params_path)
            except FileNotFoundError as exc:
                if not self.allow_eval_norm_params_compute:
                    raise FileNotFoundError(
                        f"Norm params not found at {self.norm_params_path}. "
                        "Refusing to compute normalization from a non-train split because "
                        "that leaks evaluation-set statistics and can overwrite training "
                        "artifacts. Provide the training norm params path, or set "
                        "allow_eval_norm_params_compute=True for an explicit debug-only run."
                    ) from exc
                if self.logging_enabled:
                    cprint.warn(
                        f"[WARN] Norm params not found at {self.norm_params_path}; "
                        f"computing from {self.split} split for debug-only evaluation."
                    )
                self.norm_params = self._collect_repr_lists_for_norm()
                if self.save_eval_norm_params and ((not self.use_ddp) or (self.dist_rank == 0)):
                    self._save_norm_params_to_path(self.norm_params_path, protocol=2)
                if self.use_ddp:
                    dist.barrier()

        self._norm_params_ready = True

# --- Specific Dataset Implementations ---

class DatasetAIST(DatasetMocap):
    """Dataset for the preprocessed AIST++ dataset."""
    def __init__(self, opt: dict, **kwargs):
        self.floor_height_offset = 0.10
        self.homogenous_transform_align_ = np.array([[1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]])

        # First, call the parent constructor to set up common attributes
        super().__init__(opt, **kwargs)
        
        # Load the specific AIST data file
        data_path = os.path.join(self.root, self.split + '.pt')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"AIST data file not found at: {data_path}")
        self.data = torch.load(data_path)
        # Filter out clips with length smaller than args.clip_len
        # clip_len_seqs = np.array([len(_) for _ in self.data['pose']])
        # keep_idx = clip_len_seqs >= self.clip_len
        # for k, v in self.data.items():
        #     self.data[k] = self.data[k][slice(keep_idx)]
        self._precompute_clip_cumsum()
        self._initialize_norm_params()
        cprint.ok(f'✅ AIST Dataset ({self.split}) initialized, found {len(self)} sequences.')

    def _get_item_from_data(self, index: int) -> tuple[str, torch.Tensor]:
        if self._slice_strategy == 'random_uniform_len':
            # Map global index to sequence using binary search on cumsum
            global_frame_idx = index * self.clip_len
            seq_ind = np.searchsorted(self._cumsum_clips, global_frame_idx, side='right') - 1
            seq_ind = min(seq_ind, len(self.data['name']) - 1)  # Clamp to valid range
            
            name = self.data['name'][seq_ind]
            T = self.data['pose'][seq_ind].shape[0]
            
            # Deterministic start position within the sequence based on global_frame_idx
            in_seq_frame_idx = global_frame_idx - int(self._cumsum_clips[seq_ind])
            # Clamp to ensure we get a valid clip window
            start_t = min(in_seq_frame_idx, max(0, T - self.clip_len))
            end_t = min(start_t + self.clip_len, T)
        elif self._slice_strategy == 'full_sequence':
            name = self.data['name'][index]
            T = self.data['pose'][index].shape[0]
            start_t = 0
            end_t = T
            seq_ind = index

        downsample_rate = 1
        smpl_body_pose_aa = self.data['pose'][seq_ind][start_t: end_t]      # (N, 24, 6)
        smpl_trans = self.data['tran'][seq_ind][start_t: end_t]  # (N, 3)
        T, _ = smpl_body_pose_aa.shape
        smplx_rot_aa = smpl_body_pose_aa[:, :3]  # (N, 3)
        smplx_body_pose_aa = smpl_body_pose_aa[:, 3:66]  # (N, 63)

        smplx_betas = torch.zeros((T, 10), dtype=torch.float32)  # (N, 10)

        device = self.smplx_neutral.pose_mean.device

        smplx_params_dict = {
            "transl": smpl_trans.to(device),
            "body_pose": smplx_body_pose_aa.to(device),
            "global_orient": smplx_rot_aa.to(device),
            "betas": smplx_betas.to(device),
            "jaw_pose": torch.zeros((T, 3)).to(device),
            "leye_pose": torch.zeros((T, 3)).to(device),
            "reye_pose": torch.zeros((T, 3)).to(device),
            "left_hand_pose": torch.zeros((T, 45)).to(device),
            "right_hand_pose": torch.zeros((T, 45)).to(device),
            "expression": torch.zeros((T, 10)).to(device),
        }
        return name, smplx_params_dict

class DatasetTotalCapture(DatasetMocap):
    """Dataset for the preprocessed TotalCapture dataset."""
    def __init__(self, opt: dict, **kwargs):
        self.floor_height_offset = 0.00
        self.homogenous_transform_align_ = np.array([[1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]])
        # The preprocessing script only creates a 'test' split for TotalCapture
        opt['split'] = 'test' 
        super().__init__(opt, **kwargs)

        # Load the specific TotalCapture data file
        data_path = os.path.join(self.root, 'test.pt')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"TotalCapture data file not found at: {data_path}")
        self.data = torch.load(data_path)

        self._precompute_clip_cumsum()
        self._initialize_norm_params()
        cprint.ok(f'✅ TotalCapture Dataset ({self.split}) initialized, found {len(self)} sequences.')

    def _get_item_from_data(self, index: int) -> tuple[str, torch.Tensor]:
        if self._slice_strategy == 'random_uniform_len':
            # Map global index to sequence using binary search on cumsum
            global_frame_idx = index * self.clip_len
            seq_ind = np.searchsorted(self._cumsum_clips, global_frame_idx, side='right') - 1
            seq_ind = min(seq_ind, len(self.data['name']) - 1)  # Clamp to valid range
            
            name = self.data['name'][seq_ind]
            T = self.data['pose'][seq_ind].shape[0]
            
            # Deterministic start position within the sequence based on global_frame_idx
            in_seq_frame_idx = global_frame_idx - int(self._cumsum_clips[seq_ind])
            # Clamp to ensure we get a valid clip window
            start_t = min(in_seq_frame_idx, max(0, T - self.clip_len))
            end_t = min(start_t + self.clip_len, T)
        elif self._slice_strategy == 'full_sequence':
            name = self.data['name'][index]
            T = self.data['pose'][index].shape[0]
            seq_ind = index
            start_t = 0
            end_t = T

        downsample_rate = 1
        smpl_body_pose_aa = self.data['pose'][seq_ind][start_t:end_t][::downsample_rate].reshape(-1, 72)      # (N, 72)
        smpl_trans = self.data['tran'][seq_ind][start_t:end_t][::downsample_rate]  # (N, 3)
        T, _ = smpl_body_pose_aa.shape
        smplx_rot_aa = smpl_body_pose_aa[:, :3]  # (N, 3)
        smplx_body_pose_aa = smpl_body_pose_aa[:, 3:66]  # (N, 63)

        smplx_betas = torch.zeros((T, 10), dtype=torch.float32)  # (N, 10)

        device = self.smplx_neutral.pose_mean.device

        smplx_params_dict = {
            "transl": smpl_trans.to(device),
            "body_pose": smplx_body_pose_aa.to(device),
            "global_orient": smplx_rot_aa.to(device),
            "betas": smplx_betas.to(device),
            "jaw_pose": torch.zeros((T, 3)).to(device),
            "leye_pose": torch.zeros((T, 3)).to(device),
            "reye_pose": torch.zeros((T, 3)).to(device),
            "left_hand_pose": torch.zeros((T, 45)).to(device),
            "right_hand_pose": torch.zeros((T, 45)).to(device),
            "expression": torch.zeros((T, 10)).to(device),
        }
        return name, smplx_params_dict

class Dataset3DPW(DatasetMocap):
    """Dataset for the preprocessed 3DPW dataset."""
    def __init__(self, opt: dict, **kwargs):
        opt = opt.copy()
        self.floor_height_offset = 0.00
        self.homogenous_transform_align_ = np.array([[1, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 0, 1]])
        opt.setdefault('split', 'test')
        kwargs.setdefault('motion_fps', 60.0)
        super().__init__(opt, **kwargs)

        # Load the specific 3DPW data file
        data_path = os.path.join(self.root, f'{self.split}.pt')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"3DPW data file not found at: {data_path}")
        self.data = torch.load(data_path)
        if 'pose' not in self.data and 'posec' in self.data:
            self.data['pose'] = [rotmat_to_aa(posec).reshape(-1, 72) for posec in self.data['posec']]
        
        self._precompute_clip_cumsum()
        self._initialize_norm_params()
        cprint.ok(f'✅ 3DPW Dataset ({self.split}) initialized, found {len(self)} sequences.')

    def _get_item_from_data(self, index: int) -> tuple[str, torch.Tensor]:
        if self._slice_strategy == 'random_uniform_len':
            # Map global index to sequence using binary search on cumsum
            global_frame_idx = index * self.clip_len
            seq_ind = np.searchsorted(self._cumsum_clips, global_frame_idx, side='right') - 1
            seq_ind = min(seq_ind, len(self.data['name']) - 1)  # Clamp to valid range
            
            name = self.data['name'][seq_ind]
            T = self.data['pose'][seq_ind].shape[0]
            
            # Deterministic start position within the sequence based on global_frame_idx
            in_seq_frame_idx = global_frame_idx - int(self._cumsum_clips[seq_ind])
            # Clamp to ensure we get a valid clip window
            start_t = min(in_seq_frame_idx, max(0, T - self.clip_len))
            end_t = min(start_t + self.clip_len, T)
        elif self._slice_strategy == 'full_sequence':
            seq_ind = index
            name = self.data['name'][seq_ind]
            T = self.data['pose'][seq_ind].shape[0]
            start_t = 0
            end_t = T
        else:
            raise ValueError(f"Unsupported slice strategy: {self._slice_strategy}")

        downsample_rate = 1
        smpl_body_pose_aa = self.data['pose'][seq_ind][start_t:end_t][::downsample_rate].reshape(-1, 72)      # (N, 72)
        smpl_trans = self.data['tranc'][seq_ind][start_t:end_t][::downsample_rate]  # (N, 3)
        T, _ = smpl_body_pose_aa.shape
        smplx_rot_aa = smpl_body_pose_aa[:, :3]  # (N, 3)
        smplx_body_pose_aa = smpl_body_pose_aa[:, 3:66]  # (N, 63)

        device = self.smplx_neutral.pose_mean.device

        if 'shape' in self.data and len(self.data['shape']) > seq_ind and self.data['shape'][seq_ind] is not None:
            betas = self.data['shape'][seq_ind].to(device)
            smplx_betas = betas.view(1, -1).expand(T, -1)
        else:
            smplx_betas = torch.zeros((T, 10), dtype=torch.float32, device=device)  # (N, 10)

        smplx_params_dict = {
            "transl": smpl_trans.to(device),
            "body_pose": smplx_body_pose_aa.to(device),
            "global_orient": smplx_rot_aa.to(device),
            "betas": smplx_betas,
            "jaw_pose": torch.zeros((T, 3)).to(device),
            "leye_pose": torch.zeros((T, 3)).to(device),
            "reye_pose": torch.zeros((T, 3)).to(device),
            "left_hand_pose": torch.zeros((T, 45)).to(device),
            "right_hand_pose": torch.zeros((T, 45)).to(device),
            "expression": torch.zeros((T, 10)).to(device),
        }
        return name, smplx_params_dict


if __name__ == '__main__':
    # This is a demonstration of how to use the dataloaders.

    
    # --- Configuration ---
    # Replace './data' with the actual parent directory of your preprocessed datasets.
    # The structure should be:
    # ./data/
    #   ├── aist/
    #   │   ├── train.pt
    #   │   └── test.pt
    #   ├── totalcapture/
    #   │   └── test.pt
    #   └── 3dpw/
    #       └── test.pt

    from utils import dist_util
    from ipdb_safety_net import ipdb_safety_net
    from pathlib import Path
    from omegaconf import OmegaConf
    from utils.registry_builders import build_mask_scheme_registry, build_representation_registry

    ipdb_safety_net()

    repo_root = Path(__file__).resolve().parents[1]
    repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/default.yaml")
    mask_cfg = OmegaConf.load(repo_root / "cfg_files/conf/mask_registry/default.yaml")
    repr_registry = build_representation_registry(OmegaConf.to_container(repr_cfg, resolve=True))
    mask_scheme_registry = build_mask_scheme_registry(OmegaConf.to_container(mask_cfg, resolve=True))
    
    base_data_dir = 'datasets/RobustCap/dataset_work' 
    
    datasets_to_test = {
        # "AIST_train": {
        #     'class': DatasetAIST,
        #     'opt': {'root': os.path.join(base_data_dir, 'AIST'), 'split': 'train'}
        # },
        # "TotalCapture_test": {
        #     'class': DatasetTotalCapture,
        #     'opt': {'root': os.path.join(base_data_dir, 'TotalCapture'), 'split': 'test'}
        # },
        "3DPW_test": {
            'class': Dataset3DPW,
            'opt': {'root': os.path.join(base_data_dir, '3DPW'), 'split': 'test'}
        }
    }

    for name, config in datasets_to_test.items():
        print("-" * 50)
        print(f"🚀 Testing Dataset: {name}")
        
        # Initialize the dataset
        dataset = config['class'](opt=config['opt'],
                                logdir="runs/your_run",
                                device=dist_util.dev(),
                                noise_std_joint=0.0,
                                noise_std_smplx_global_rot=2.0,
                                noise_std_smplx_body_rot=2.0,
                                noise_std_smplx_trans=0.03,
                                noise_std_smplx_betas=0.0,
                                dtype=torch.float32,
                                body_model_path="data/body_models/smplx_model",
                                joints_num=22,
                                use_ddp=False,
                                clip_len=33,
                                disable_cache=False,
                                slice_strategy='full_sequence',
                                mean_std=True,
                                norm_strategy='meanstd',
                                input_noise=True,
                                repr_registry=repr_registry,
                                mask_scheme_registry=mask_scheme_registry,)
        
        if len(dataset) == 0:
            cprint.warn(f"⚠️  Dataset {name} is empty. Skipping.")
            continue

        # NOTE: batch_size=1 since sequences may have variable lengths
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Fetch and inspect one batch
        for bach_idx, batch in enumerate(dataloader):
            # print(f"    Successfully loaded a batch of size: {len(batch['meta']['name'])}")
            
            # --- Inspecting the batch content ---
            motion_repr = batch['motion_repr_clean']
            # cond_mask = batch['cond_mask']
            # meta = batch['meta']
            
            print(f"    Batch 'motion_repr_clean' shape: {motion_repr.shape}")
            # print(f"    Batch 'cond_mask' shape: {cond_mask.shape}")
            # print(f"    Sequence names in batch: {meta['name']}")
            
            # Check normalization parameters for the first item in the batch
            # first_item_norm = meta['normalization_params']
            # xy_offset = first_item_norm['translation_xy_offset'][0]
            # z_offset = first_item_norm['floor_height_offset'][0]
            # print(f"    Normalization for first item: XY offset={xy_offset}, Floor offset={z_offset:.4f}")

            
            print("-" * 50 + "\n")
