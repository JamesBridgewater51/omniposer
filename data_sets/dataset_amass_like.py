from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_sets.motion_representation import *
import pickle as pkl
from utils.representation_registry import RepresentationRegistry
import torch.distributed as dist
from typing import Literal, cast, Optional, List, Generator, Dict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from cprint import cprint

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover - plotting is optional
    plt = None
    sns = None

from .dataset_base import DatasetBase
from .running_stats import RunningStatsDict
from .motion_aware_sampling import (
    compute_motion_stats, 
    compute_sample_weights,
    MotionStats,
    MotionAwareSliceSampler,
)

class DatasetAMASSLike(DatasetBase):
    def __init__(self,
                 preprocessed_dataset_root='',
                 datasets=None,
                 split='train',
                 slice_strategy: Literal['random_uniform_len', 'full_sequence',
                                        'high_motion_weighted']='random_uniform_len',
                 spacing=1,
                 clip_len=145,
                 input_noise=False,
                 sep_noise: bool=False,
                 noise_std_joint=0.0,
                 noise_std_smplx_global_rot=0.0,
                 noise_std_smplx_body_rot=0.0,
                 noise_std_smplx_trans=0.0,
                 noise_std_smplx_betas=0.0,
                 load_noise=False,
                 loaded_smplx_noise_dict=None,
                 align_to_yaxis=False,
                 norm_params_path='',
                 # Motion-aware sampling parameters
                 motion_weight_strategy: Literal['jerk', 'accel', 'combined'] = 'combined',
                 motion_weight_temperature: float = 1.0,
                 motion_weight_min: float = 0.1,
                 motion_fps: float = 30.0,
                 body_repr_batch_size: int = 256,
                 repr_registry: RepresentationRegistry = None,
                 mask_scheme_registry=None,
                 **kwargs
                 ):

        super(DatasetAMASSLike,self).__init__(repr_registry=repr_registry, mask_scheme_registry=mask_scheme_registry, **kwargs)

        self.preprocessed_dataset_root = preprocessed_dataset_root
        self.split = split
        self.clip_len = clip_len
        self._slice_strategy = slice_strategy
        self.spacing = spacing

        # Motion-aware sampling config
        self.motion_weight_strategy = motion_weight_strategy
        self.motion_weight_temperature = motion_weight_temperature
        self.motion_weight_min = motion_weight_min
        self.motion_fps = motion_fps
        self.body_repr_batch_size = int(body_repr_batch_size)
        if self.body_repr_batch_size <= 0:
            raise ValueError(f"body_repr_batch_size must be positive, got {body_repr_batch_size}.")
        self.motion_stats_list: List[MotionStats] = []
        self.sample_weights: Optional[np.ndarray] = None

        ########################################## configs about how to add noise
        self.input_noise = input_noise
        self.sep_noise = sep_noise  # add different noise to joint-based repr, and smplx-based repr, separately
        self.noise_std_joint = noise_std_joint  # set if sep_noise=True
        self.noise_std_params_dict = {'global_orient': noise_std_smplx_global_rot,
                                      'transl': noise_std_smplx_trans,
                                      'body_pose': noise_std_smplx_body_rot,
                                      'betas': noise_std_smplx_betas,}
        self.load_noise = load_noise  # load preset noise (for reproducibility)
        self.loaded_smplx_noise_dict = loaded_smplx_noise_dict  # loaded preset noise
        self.align_to_yaxis = align_to_yaxis

        # NOTE: train_norm_params.pkl is the default because norm parameters are produced during training and therefore named `*_train_norm_params.pkl`.
        default_norm_filename = f"{self.__class__.__name__}_train_norm_params.pkl"
        self.norm_params_path = self._resolve_norm_params_path(norm_params_path, default_norm_filename)

        self.repr_list_dict = {}
        for repr_name in self._repr_list:
            self.repr_list_dict[repr_name] = []
        self.smplx_params_list_dict = {}  # each item: params of all samples
        for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
            self.smplx_params_list_dict[param_name] = []

        # these `self.*_list` store the cleaned and noisy joint positions, populated in `create_body_repr`, used in `__getitem__`
        self.joints_clean_list = []
        if self._slice_strategy == 'full_sequence':
            self.seq_name_list = []
        if self.input_noise and (not self.sep_noise):
            self.joints_noisy_list = []
            self.repr_list_dict_noisy = {}
            for repr_name in self._repr_list:
                self.repr_list_dict_noisy[repr_name] = []

        # these `*_clip_list` store preprocessed data, populated in  `read_data`, used in `create_body_repr` to create motion_repr. for `self.*_list`
        self.joints_clip_list = []
        self.smplx_clip_list = []
        if self._slice_strategy == 'full_sequence':
            self.seq_name_clip_list = []

        # Optionally: reprs we explicitly DON'T want to normalize (keeps previous behavior)
        self._normalize_mask = {r: True for r in self._repr_list}
        self._normalize_mask['foot_contact'] = False   # keep as-is (0/1 labels)

        if not self.disable_cache:
            # Add cache-related attributes
            self.cache_dir = Path(preprocessed_dataset_root) / '.cache'
            self.cache_dir.mkdir(exist_ok=True)
            
            # Create a unique cache key based on all relevant parameters
            cache_params = {
                'slice_strategy': slice_strategy,
                'spacing': spacing,
                'split': split,
                'clip_len': self.clip_len,
                'input_noise': input_noise,
                'sep_noise': sep_noise,
                'noise_std_joint': noise_std_joint,
                'noise_std_params': self.noise_std_params_dict,
                'load_noise': load_noise,
                'cano_jts_pos': self.cano_jts_pos,
                'amass_datasets': sorted(datasets) if datasets is not None else None,
                'joints_num': self.joints_num,
                'modalities': self._repr_list,
                'mean_std': self.mean_std,
                'norm_strategy': self.norm_strategy,
                # Motion-aware sampling params
                'motion_weight_strategy': motion_weight_strategy,
                'motion_weight_temperature': motion_weight_temperature,
                'motion_weight_min': motion_weight_min,
                'body_repr_batch_size': self.body_repr_batch_size,
            }
            
            # Create deterministic cache key
            self._cache_key = self._get_cache_key(cache_params)
            # Memmap-based cache: use a directory per cache key instead of a single .pkl file
            self.cache_subdir = self.cache_dir / self._cache_key
            self.cache_file = self.cache_subdir / 'meta.pkl'  # Small metadata file
            
            # Try to load from cache first
            if self._try_load_cache():
                if self.logging_enabled:
                    cprint.info(f"[INFO] Loaded dataset from cache: {self.cache_subdir}")
                # NOTE: if the cache is valid and split == 'train', AMASS_mean.pkl and AMASS_std.pkl must be written to disk, otherwise the 'test' split would fail to find them under logdir.
                # This is a workaround to ensure that the mean/std stats are always saved when the dataset is initialized. But it would eat up disk space.
                if self.split == 'train':
                    with open(self.norm_params_path, 'wb') as result_file:
                        pkl.dump(self.norm_params, result_file, protocol=2)
                return
            
        # If cache miss or invalid or <self.disable_cache>, proceed with normal initialization
        
        ######################################## read data and compute the motion reprentations
        self.read_data(datasets)
        self.create_body_repr()

        if not self.disable_cache:
            # Save to cache after initialization (only rank 0 saves in DDP mode)
            self._save_to_cache()
            # Synchronize all processes after cache save in DDP mode
            if self.use_ddp:
                dist.barrier()

    def _get_cache_key(self, params):
        """Generate a deterministic hash key from parameters."""
        # Convert params to a canonical string representation
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _try_load_cache(self):
        """Attempt to load dataset state from cache using memmap for large arrays.
        
        Uses numpy memmap in read-only mode for zero-copy access to large arrays,
        avoiding memory explosion when loading datasets with hundreds of thousands of clips.
        """
        if not self.cache_file.exists():
            return False
            
        try:
            with open(self.cache_file, 'rb') as f:
                meta = pkl.load(f)
                
            # Verify cache version (increment when format changes)
            if meta.get('version') != 4:  # Version 4 = memmap-based cache
                cprint.warn("[Warning] Cache version mismatch.")
                return False
            
            # Load small metadata attributes from pickle
            small_attrs = ['n_samples', '_repr_indices', 'norm_params']
            if self._slice_strategy == 'high_motion_weighted':
                small_attrs.extend(['motion_stats_list', 'sample_weights'])
            
            for attr in small_attrs:
                if attr not in meta:
                    cprint.warn(f"[Warning] Cache meta missing attribute: {attr}")
                    return False
                setattr(self, attr, meta[attr])
            
            # Load large arrays via memmap (read-only for zero-copy access)
            cache_subdir = self.cache_subdir
            
            # Required memmap files
            required_files = ['joints_clean.npy', 'smplx_params.npz']
            required_files.extend([f'repr_{name}.npy' for name in self._repr_list])
            if self.input_noise and not self.sep_noise:
                required_files.append('joints_noisy.npy')
                required_files.extend([f'repr_noisy_{name}.npy' for name in self._repr_list])
            
            for fname in required_files:
                fpath = cache_subdir / fname
                if not fpath.exists():
                    cprint.warn(f"[Warning] Cache missing file: {fpath}")
                    return False
            
            # Load joints_clean as memmap
            self.joints_clean_list = np.load(cache_subdir / 'joints_clean.npy', mmap_mode='r')
            
            # Load repr_list_dict as memmaps
            self.repr_list_dict = {}
            for repr_name in self._repr_list:
                self.repr_list_dict[repr_name] = np.load(
                    cache_subdir / f'repr_{repr_name}.npy', mmap_mode='r'
                )
            
            # Load smplx_params_list_dict from npz (smaller, so use regular load)
            smplx_data = np.load(cache_subdir / 'smplx_params.npz')
            self.smplx_params_list_dict = {
                'global_orient': smplx_data['global_orient'],
                'transl': smplx_data['transl'],
                'body_pose': smplx_data['body_pose'],
                'betas': smplx_data['betas'],
            }
            
            # Load noisy data if applicable
            if self.input_noise and not self.sep_noise:
                self.joints_noisy_list = np.load(
                    cache_subdir / 'joints_noisy.npy', mmap_mode='r'
                )
                self.repr_list_dict_noisy = {}
                for repr_name in self._repr_list:
                    self.repr_list_dict_noisy[repr_name] = np.load(
                        cache_subdir / f'repr_noisy_{repr_name}.npy', mmap_mode='r'
                    )
            
            # These are empty after cache load (not needed post-processing)
            self.joints_clip_list = []
            self.smplx_clip_list = []
            
            return True
            
        except (EOFError, pkl.UnpicklingError, KeyError, ValueError) as e:
            # Handle corrupt cache files - clean up the entire cache subdir
            import shutil
            if self.cache_subdir.exists():
                shutil.rmtree(self.cache_subdir, ignore_errors=True)
            return False

    def _save_to_cache(self):
        """Save current dataset state to cache using memmap for large arrays.
        
        Uses numpy memmap to write large arrays incrementally to disk,
        avoiding memory explosion during serialization of large datasets.
        
        In DDP mode, only rank 0 saves the cache to avoid race conditions.
        Other ranks will wait and load from the cache after rank 0 finishes.
        """
        # In DDP mode, only rank 0 should save the cache
        if self.use_ddp and self.dist_rank != 0:
            return
        
        # Create cache subdirectory atomically
        temp_subdir = self.cache_dir / f'{self._cache_key}.tmp.{os.getpid()}'
        temp_subdir.mkdir(parents=True, exist_ok=True)
        
        n_clips = len(self.joints_clean_list)
        
        # Determine shapes from first sample
        first_joints = np.asarray(self.joints_clean_list[0])
        clip_len, n_joints, _ = first_joints.shape  # (T, J, 3)
        
        # Save joints_clean as memmap
        joints_clean_path = temp_subdir / 'joints_clean.npy'
        joints_memmap = np.lib.format.open_memmap(
            str(joints_clean_path),
            mode='w+',
            dtype=np.float32,
            shape=(n_clips, clip_len, n_joints, 3),
        )
        for i, jc in enumerate(self.joints_clean_list):
            joints_memmap[i] = np.asarray(jc, dtype=np.float32)
        joints_memmap.flush()
        
        # Save repr_list_dict as memmaps
        for repr_name in self._repr_list:
            repr_data = self.repr_list_dict[repr_name]
            first_repr = np.asarray(repr_data[0] if isinstance(repr_data, list) else repr_data[0])
            repr_shape = first_repr.shape  # (T-1, D) or similar
            
            repr_path = temp_subdir / f'repr_{repr_name}.npy'
            repr_memmap = np.lib.format.open_memmap(
                str(repr_path),
                mode='w+',
                dtype=np.float32,
                shape=(n_clips,) + repr_shape,
            )
            for i, r in enumerate(repr_data if isinstance(repr_data, list) else repr_data):
                repr_memmap[i] = np.asarray(r, dtype=np.float32)
            repr_memmap.flush()
        
        # Save smplx_params_list_dict as compressed npz (smaller than joints/repr)
        smplx_arrays = {}
        for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
            smplx_arrays[param_name] = np.stack(
                [np.asarray(p, dtype=np.float32) for p in self.smplx_params_list_dict[param_name]],
                axis=0
            )
        np.savez(temp_subdir / 'smplx_params.npz', **smplx_arrays)
        
        # Save noisy data if applicable
        if self.input_noise and not self.sep_noise:
            # joints_noisy_list
            joints_noisy_path = temp_subdir / 'joints_noisy.npy'
            first_noisy = np.asarray(self.joints_noisy_list[0])
            noisy_shape = first_noisy.shape
            joints_noisy_memmap = np.lib.format.open_memmap(
                str(joints_noisy_path),
                mode='w+',
                dtype=np.float32,
                shape=(n_clips,) + noisy_shape,
            )
            for i, jn in enumerate(self.joints_noisy_list):
                joints_noisy_memmap[i] = np.asarray(jn, dtype=np.float32)
            joints_noisy_memmap.flush()
            
            # repr_list_dict_noisy
            for repr_name in self._repr_list:
                repr_data_noisy = self.repr_list_dict_noisy[repr_name]
                first_repr_noisy = np.asarray(repr_data_noisy[0] if isinstance(repr_data_noisy, list) else repr_data_noisy[0])
                repr_noisy_shape = first_repr_noisy.shape
                
                repr_noisy_path = temp_subdir / f'repr_noisy_{repr_name}.npy'
                repr_noisy_memmap = np.lib.format.open_memmap(
                    str(repr_noisy_path),
                    mode='w+',
                    dtype=np.float32,
                    shape=(n_clips,) + repr_noisy_shape,
                )
                for i, r in enumerate(repr_data_noisy if isinstance(repr_data_noisy, list) else repr_data_noisy):
                    repr_noisy_memmap[i] = np.asarray(r, dtype=np.float32)
                repr_noisy_memmap.flush()
        
        # Save small metadata to pickle
        meta = {
            'version': 4,  # Version 4 = memmap-based cache
            'n_samples': self.n_samples,
            '_repr_indices': self._repr_indices,
            'norm_params': self.norm_params,
        }
        
        if self._slice_strategy == 'high_motion_weighted':
            meta['motion_stats_list'] = self.motion_stats_list
            meta['sample_weights'] = self.sample_weights
        
        temp_meta_file = temp_subdir / 'meta.pkl'
        with open(temp_meta_file, 'wb') as f:
            pkl.dump(meta, f, protocol=4)
        
        # Atomic rename of cache subdirectory
        if self.cache_subdir.exists():
            import shutil
            shutil.rmtree(self.cache_subdir, ignore_errors=True)
        temp_subdir.rename(self.cache_subdir)

    def clear_cache(self):
        """Clear all cache files (including memmap-based cache directories)."""
        import shutil
        if self.cache_dir.exists():
            # Clear old pickle-based caches
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            # Clear new memmap-based cache directories
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    shutil.rmtree(subdir, ignore_errors=True)

    def get_weighted_sampler(self) -> 'torch.utils.data.WeightedRandomSampler':
        """
        Get a WeightedRandomSampler for use with DataLoader.
        
        For high_motion_weighted strategy, returns a sampler
        that oversamples high jerk/acceleration clips.
        For other strategies, returns a uniform sampler.
        
        Returns:
            WeightedRandomSampler instance
        """
        from torch.utils.data import WeightedRandomSampler
        
        n = len(self)
        if self._slice_strategy == 'high_motion_weighted' and self.sample_weights is not None:
            weights = self.sample_weights
        else:
            weights = np.ones(n) / n
        
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=n,
            replacement=True,
        )

    def get_motion_stats_summary(self) -> dict:
        """
        Get summary statistics of motion characteristics across all clips.
        
        Returns:
            Dictionary with motion statistics summary
        """
        if not self.motion_stats_list:
            return {}
        
        jerks = np.array([s.mean_jerk for s in self.motion_stats_list])
        accels = np.array([s.mean_accel for s in self.motion_stats_list])
        vels = np.array([s.mean_velocity for s in self.motion_stats_list])
        
        return {
            'jerk': {'mean': jerks.mean(), 'std': jerks.std(), 'min': jerks.min(), 'max': jerks.max(),
                    'p25': np.percentile(jerks, 25), 'p75': np.percentile(jerks, 75)},
            'accel': {'mean': accels.mean(), 'std': accels.std(), 'min': accels.min(), 'max': accels.max(),
                     'p25': np.percentile(accels, 25), 'p75': np.percentile(accels, 75)},
            'velocity': {'mean': vels.mean(), 'std': vels.std(), 'min': vels.min(), 'max': vels.max(),
                        'p25': np.percentile(vels, 25), 'p75': np.percentile(vels, 75)},
            'sample_weights': {
                'min': self.sample_weights.min() if self.sample_weights is not None else None,
                'max': self.sample_weights.max() if self.sample_weights is not None else None,
                'ratio': (self.sample_weights.max() / self.sample_weights.min()) if self.sample_weights is not None else None,
            }
        }

    def divide_clip(self, dataset_name=Optional[str]):
        preprocessed_dataset_joints_dir = os.path.join(self.preprocessed_dataset_root, 'pose_data_fps_30')
        preprocessed_dataset_smpl_dir = os.path.join(self.preprocessed_dataset_root, 'smpl_data_fps_30')
        if dataset_name:
            seq_joints_paths = glob.glob(os.path.join(preprocessed_dataset_joints_dir, dataset_name, '**/*.npy'), recursive=True)  # name list of all npz sequence files in current dataset
            seq_smpl_paths = glob.glob(os.path.join(preprocessed_dataset_smpl_dir, dataset_name, '**/*.npy'), recursive=True)  # name list of all npz sequence files in current dataset
        else:
            seq_joints_paths = glob.glob(os.path.join(preprocessed_dataset_joints_dir, '**/*.npy'), recursive=True)  # name list of all npz sequence files in current dataset
            seq_smpl_paths = glob.glob(os.path.join(preprocessed_dataset_smpl_dir, '**/*.npy'), recursive=True)  # name list of all npz sequence files in current dataset
        seq_joints_paths = sorted(seq_joints_paths)
        seq_smpl_paths = sorted(seq_smpl_paths)
        # print('reading sequences in %s...' % (dataset_name))
        for seq_joint_path, seq_smpl_path in zip(seq_joints_paths, seq_smpl_paths):
            seq_joints = np.load(seq_joint_path)  # [seq_len, 25, 3]
            seq_smplx = np.load(seq_smpl_path) # [seq_len, 178]
            if self.split == 'test':
                seq_joints = seq_joints[1:-1]
                seq_smplx = seq_smplx[1:-1]
            N = len(seq_joints)  # total frame number of the current sequence
            
            # Handle different slice strategies
            if self._slice_strategy == 'full_sequence':
                self.joints_clip_list.append(seq_joints)
                self.smplx_clip_list.append(seq_smplx)
                self.seq_name_clip_list.append(seq_joint_path)
            elif N >= self.clip_len:
                # For random_uniform_len, high_motion_weighted
                num_valid_clip = int(N / self.clip_len)
                for i in range(num_valid_clip):
                    joints_clip = seq_joints[(self.clip_len * i):self.clip_len * (i + 1)]  # [clip_len, 25, 3]
                    smplx_clip = seq_smplx[(self.clip_len * i):self.clip_len * (i + 1)]  # [clip_len, ]
                    self.joints_clip_list.append(joints_clip)
                    self.smplx_clip_list.append(smplx_clip)

    def read_data(self, datasets):
        if datasets is not None:
            for dataset_name in tqdm(datasets, disable=not self.logging_enabled):
                self.divide_clip(dataset_name)
        else:
            self.divide_clip(None)
        self.n_samples = len(self.joints_clip_list)
        if self.logging_enabled:
            cprint.info('[INFO] {} set: get {} sub clips in total.'.format(self.split, self.n_samples))

    def _canonicalize_seq(self, seq_joints, seq_smplx):
        
        smplx_params_dict = {'global_orient': seq_smplx[:, 0:3],
                                'transl': seq_smplx[:, 3:6],
                                'betas': seq_smplx[:, 6:16],
                                'body_pose': seq_smplx[:, 16:(16+63)],
                                }  # for the clip

        ######################################## canonicalize for GT sequence
        cano_positions, cano_smplx_params_dict = cano_seq_smplx(positions=seq_joints,
                                                                align_to_yaxis=False,
                                                                smplx_params_dict=smplx_params_dict,
                                                                smpl_model=self.smplx_neutral, device=self.device)

        ######################################## add noise to smplx params
        smplx_noise_dict = self.smplx_noise_dict
        if self.input_noise and (not self.sep_noise):
            cano_smplx_params_dict_noisy = {}
            for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
                if param_name == 'transl' or param_name == 'betas':
                    if self.load_noise:
                        noise_1 = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                    else:
                        noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smplx_params_dict[param_name].shape)
                    cano_smplx_params_dict_noisy[param_name] = cano_smplx_params_dict[param_name] + noise_1
                    if param_name not in smplx_noise_dict.keys():
                        smplx_noise_dict[param_name] = []
                    smplx_noise_dict[param_name].append(noise_1)
                elif param_name == 'global_orient':
                    global_orient_mat = R.from_rotvec(cano_smplx_params_dict['global_orient'])  # [145, 3, 3]
                    global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
                    if self.load_noise:
                        noise_global_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                    else:
                        noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                    global_orient_angle_noisy = global_orient_angle + noise_global_rot
                    cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
                    if param_name not in smplx_noise_dict.keys():
                        smplx_noise_dict[param_name] = []
                    smplx_noise_dict[param_name].append(noise_global_rot)  #  [145, 3] in euler angle
                elif param_name == 'body_pose':
                    body_pose_mat = R.from_rotvec(cano_smplx_params_dict['body_pose'].reshape(-1, 3))
                    body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
                    if self.load_noise:
                        noise_body_pose_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
                    else:
                        noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                    body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                    cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
                    if param_name not in smplx_noise_dict.keys():
                        smplx_noise_dict[param_name] = []
                    smplx_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, 21, 3))  # [145, 21, 3]  in euler angle

            ### using FK to obtain noisy joint positions from noisy smplx params
            smplx_params_dict_noisy_torch = {}
            for key in cano_smplx_params_dict_noisy.keys():
                smplx_params_dict_noisy_torch[key] = torch.FloatTensor(cano_smplx_params_dict_noisy[key]).to(self.device)
            bs = smplx_params_dict_noisy_torch['transl'].shape[0]
            # we do not consider face/hand details in RoHM
            smplx_params_dict_noisy_torch['jaw_pose'] = torch.zeros(bs, 3).to(self.device)
            smplx_params_dict_noisy_torch['leye_pose'] = torch.zeros(bs, 3).to(self.device)
            smplx_params_dict_noisy_torch['reye_pose'] = torch.zeros(bs, 3).to(self.device)
            smplx_params_dict_noisy_torch['left_hand_pose'] = torch.zeros(bs, 45).to(self.device)
            smplx_params_dict_noisy_torch['right_hand_pose'] = torch.zeros(bs, 45).to(self.device)
            smplx_params_dict_noisy_torch['expression'] = torch.zeros(bs, 10).to(self.device)
            cano_positions_noisy = self.smplx_neutral(**smplx_params_dict_noisy_torch).joints[:, 0:22].detach().cpu().numpy()  # [clip_len, 22, 3]

        ######################################## create motion representation
        repr_dict = get_repr_smplx(positions=cano_positions,
                                    cano_jts_pos=self.cano_jts_pos,
                                    smplx_params_dict=cano_smplx_params_dict,
                                    feet_vel_thre=5e-5,
                                    smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                    device=self.device,
                                    modality_names=self._repr_list)  # a dict of reprs
        if self.input_noise and (not self.sep_noise):
            repr_dict_noisy = get_repr_smplx(positions=cano_positions_noisy,
                                            cano_jts_pos=self.cano_jts_pos,
                                            smplx_params_dict=cano_smplx_params_dict_noisy,
                                            feet_vel_thre=5e-5,
                                            smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                            device=self.device,
                                            modality_names=self._repr_list)  # a dict of reprs
        
        output = {
            "cano_positions": cano_positions,
            "cano_smplx_params_dict": cano_smplx_params_dict,
            "repr_dict": repr_dict,
        }
        if self.input_noise and (not self.sep_noise):
            output["cano_positions_noisy"] = cano_positions_noisy
            output["cano_smplx_params_dict_noisy"] = cano_smplx_params_dict_noisy
            output["repr_dict_noisy"] = repr_dict_noisy
            output["smplx_noise_dict"] = smplx_noise_dict
        return output

    def create_body_repr(self):
        """Create body representation. Dispatches to batched or sequential processing."""
        # Use batched processing for fixed-length clips (much faster)
        fixed_len_strategies = ('random_uniform_len', 'high_motion_weighted')
        use_batched = (
            self._slice_strategy in fixed_len_strategies
            and not self.sep_noise  # Separate noise not supported in batched version
            and self.device.type == 'cuda'  # Batched is most beneficial on GPU
            and len(self.joints_clip_list) > 0
        )
        
        if use_batched:
            self._create_body_repr_batched(batch_size=self.body_repr_batch_size)
        else:
            self._create_body_repr_sequential()
    
    def _create_body_repr_batched(self, batch_size: int = 256):
        """GPU-batched body representation creation for 20-50x speedup."""
        from data_sets.batched_repr_utils import (
            cano_seq_smplx_batched,
            get_repr_smplx_batched,
            add_noise_to_smplx_params_batched,
            smplx_forward_batched,
        )
        
        self.smplx_noise_dict = {}
        
        # Initialize streaming statistics for norm params computation
        repr_dims = {m.name: m.dim for m in self.repr_registry.modalities}
        running_stats = RunningStatsDict(repr_dims, device=torch.device('cpu'))
        compute_streaming_stats = (self.split == 'train')
        
        # Get indices with spacing applied
        indices = list(range(0, self.n_samples, self.spacing))
        n_clips = len(indices)
        
        # Check if noise should be applied
        apply_noise = self.input_noise and (not self.sep_noise)
        
        if self.logging_enabled:
            noise_str = " with noise augmentation" if apply_noise else ""
            cprint.info(f'[INFO] Using GPU-batched processing{noise_str} for {n_clips} clips (batch_size={batch_size})')
        
        # Process in batches
        for batch_start in tqdm(range(0, n_clips, batch_size), 
                                 desc="Batched processing", 
                                 disable=not self.logging_enabled):
            batch_end = min(batch_start + batch_size, n_clips)
            batch_indices = indices[batch_start:batch_end]
            curr_batch_size = len(batch_indices)
            
            # Stack clips into batched tensors
            batch_joints = torch.stack([
                torch.from_numpy(self.joints_clip_list[i][:, :self.joints_num, :]).float()
                for i in batch_indices
            ], dim=0).to(self.device)  # (B, T, J, 3)
            
            batch_smplx = torch.stack([
                torch.from_numpy(self.smplx_clip_list[i]).float()
                for i in batch_indices
            ], dim=0).to(self.device)  # (B, T, D)
            
            # Parse SMPLX params
            smplx_params_dict = {
                'global_orient': batch_smplx[:, :, 0:3],
                'transl': batch_smplx[:, :, 3:6],
                'betas': batch_smplx[:, :, 6:16],
                'body_pose': batch_smplx[:, :, 16:79],
            }
            
            # Canonicalize (batched)
            cano_positions, cano_params = cano_seq_smplx_batched(
                batch_joints, smplx_params_dict, device=self.device
            )
            
            # Compute representations (batched) for clean data
            repr_dict = get_repr_smplx_batched(
                cano_positions,
                cano_params,
                feet_vel_thre=5e-5,
                cano_jts_pos=self.cano_jts_pos,
                smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                device=self.device,
                modality_names=self._repr_list
            )
            
            # Noise augmentation (batched)
            if apply_noise:
                # Add noise to SMPLX params
                cano_params_noisy, batch_noise_dict = add_noise_to_smplx_params_batched(
                    cano_params,
                    self.noise_std_params_dict,
                    device=self.device
                )
                
                # Compute noisy joint positions via SMPLX forward kinematics
                cano_positions_noisy = smplx_forward_batched(
                    self.smplx_neutral,
                    cano_params_noisy,
                    device=self.device
                )
                
                # Compute representations for noisy data
                repr_dict_noisy = get_repr_smplx_batched(
                    cano_positions_noisy,
                    cano_params_noisy,
                    feet_vel_thre=5e-5,
                    cano_jts_pos=self.cano_jts_pos,
                    smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                    device=self.device,
                    modality_names=self._repr_list
                )
                
                # Move noisy data to CPU
                cano_positions_noisy_cpu = cano_positions_noisy.cpu().numpy()
                repr_dict_noisy_cpu = {k: v.cpu().numpy() for k, v in repr_dict_noisy.items()}
                cano_params_noisy_cpu = {k: v.cpu().numpy() for k, v in cano_params_noisy.items()}
                batch_noise_cpu = {k: v.cpu().numpy() for k, v in batch_noise_dict.items()}
            
            # Move to CPU and append to lists
            cano_positions_cpu = cano_positions.cpu().numpy()
            repr_dict_cpu = {k: v.cpu().numpy() for k, v in repr_dict.items()}
            cano_params_cpu = {k: v.cpu().numpy() for k, v in cano_params.items()}
            
            for i in range(curr_batch_size):
                self.joints_clean_list.append(cano_positions_cpu[i])
                
                for repr_name in self._repr_list:
                    if repr_name in repr_dict_cpu:
                        self.repr_list_dict[repr_name].append(repr_dict_cpu[repr_name][i])
                
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    self.smplx_params_list_dict[param_name].append(cano_params_cpu[param_name][i])
                
                # Append noisy data if applicable
                if apply_noise:
                    self.joints_noisy_list.append(cano_positions_noisy_cpu[i])
                    
                    for repr_name in self._repr_list:
                        if repr_name in repr_dict_noisy_cpu:
                            self.repr_list_dict_noisy[repr_name].append(repr_dict_noisy_cpu[repr_name][i])
                
                # Compute motion stats for high_motion strategies
                if self._slice_strategy == 'high_motion_weighted':
                    stats = compute_motion_stats(cano_positions_cpu[i], fps=self.motion_fps)
                    self.motion_stats_list.append(stats)
                
                # Update streaming statistics for norm params
                if compute_streaming_stats:
                    tensor_repr = {
                        k: torch.from_numpy(repr_dict_cpu[k][i])
                        for k in repr_dict_cpu
                        if k in repr_dims
                    }
                    running_stats.update(tensor_repr)
        
        # Compute sample weights for high_motion strategies
        if self._slice_strategy == 'high_motion_weighted' and self.motion_stats_list:
            self.sample_weights = compute_sample_weights(
                self.motion_stats_list,
                weight_strategy=self.motion_weight_strategy,
                temperature=self.motion_weight_temperature,
                min_weight=self.motion_weight_min,
            )
            if self.logging_enabled:
                jerks = np.array([s.mean_jerk for s in self.motion_stats_list])
                accels = np.array([s.mean_accel for s in self.motion_stats_list])
                cprint.info(f'[INFO] Motion stats - Jerk: mean={jerks.mean():.2f}, std={jerks.std():.2f}, '
                           f'max={jerks.max():.2f}; Accel: mean={accels.mean():.2f}, std={accels.std():.2f}')
                cprint.info(f'[INFO] Sample weights - min={self.sample_weights.min():.4f}, '
                           f'max={self.sample_weights.max():.4f}, ratio={self.sample_weights.max()/self.sample_weights.min():.2f}x')
        
        # Finalize representation arrays and norm params
        self._finalize_body_repr(running_stats, compute_streaming_stats)

    def _create_body_repr_sequential(self):
        """Original sequential body representation creation (for variable-length or with noise)."""
        self.smplx_noise_dict = {}
        
        # Initialize streaming statistics for norm params computation
        repr_dims = {m.name: m.dim for m in self.repr_registry.modalities}
        running_stats = RunningStatsDict(repr_dims, device=torch.device('cpu'))
        compute_streaming_stats = (self.split == 'train')
        
        # ``spacing`` subsamples fixed-length clips. In full-sequence mode each
        # entry is already one complete sequence, so skipping by spacing would
        # silently drop evaluation sequences.
        sample_indices = range(self.n_samples) if self._slice_strategy == 'full_sequence' else range(0, self.n_samples, self.spacing)
        for i in tqdm(sample_indices, disable=not self.logging_enabled):
            source_data_joints = self.joints_clip_list[i][:, 0:self.joints_num, :]  # [T, 22, 3]
            source_data_smplx = self.smplx_clip_list[i]  # [T, 178]
            if self._slice_strategy == 'full_sequence':
                source_data_seq_name = self.seq_name_clip_list[i]
            output = self._canonicalize_seq(source_data_joints, source_data_smplx)

            ############### clean data repr gt
            self.joints_clean_list.append(output["cano_positions"])
            if self._slice_strategy == 'full_sequence':
                self.seq_name_list.append(source_data_seq_name)
            for repr_name in self._repr_list:
                self.repr_list_dict[repr_name].append(output["repr_dict"][repr_name])
            for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                self.smplx_params_list_dict[param_name].append(output["cano_smplx_params_dict"][param_name])

            if self.input_noise and (not self.sep_noise):
                self.joints_noisy_list.append(output["cano_positions_noisy"])
                for repr_name in self._repr_list:
                    self.repr_list_dict_noisy[repr_name].append(output["repr_dict_noisy"][repr_name])

            # Compute motion stats for high_motion strategies
            if self._slice_strategy == 'high_motion_weighted':
                stats = compute_motion_stats(output["cano_positions"], fps=self.motion_fps)
                self.motion_stats_list.append(stats)
            
            # Update streaming statistics for norm params
            if compute_streaming_stats:
                tensor_repr = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in output["repr_dict"].items()
                    if k in repr_dims
                }
                running_stats.update(tensor_repr)

        # Compute sample weights for high_motion strategies
        if self._slice_strategy == 'high_motion_weighted' and self.motion_stats_list:
            self.sample_weights = compute_sample_weights(
                self.motion_stats_list,
                weight_strategy=self.motion_weight_strategy,
                temperature=self.motion_weight_temperature,
                min_weight=self.motion_weight_min,
            )
            if self.logging_enabled:
                jerks = np.array([s.mean_jerk for s in self.motion_stats_list])
                accels = np.array([s.mean_accel for s in self.motion_stats_list])
                cprint.info(f'[INFO] Motion stats - Jerk: mean={jerks.mean():.2f}, std={jerks.std():.2f}, '
                           f'max={jerks.max():.2f}; Accel: mean={accels.mean():.2f}, std={accels.std():.2f}')
                cprint.info(f'[INFO] Sample weights - min={self.sample_weights.min():.4f}, '
                           f'max={self.sample_weights.max():.4f}, ratio={self.sample_weights.max()/self.sample_weights.min():.2f}x')

        # Finalize representation arrays and norm params
        self._finalize_body_repr(running_stats, compute_streaming_stats)
    
    def _finalize_body_repr(self, running_stats: RunningStatsDict, compute_streaming_stats: bool):
        """Finalize representation arrays and compute norm params."""
        # Strategies that produce fixed-length clips
        fixed_len_strategies = ('random_uniform_len', 'high_motion_weighted')

        def _as_object_array(items):
            """Store variable-length sequences without triggering numpy ragged stacking."""
            arr = np.empty(len(items), dtype=object)
            arr[:] = items
            return arr
        
        for repr_name in self._repr_list:
            if self._slice_strategy in fixed_len_strategies:
                self.repr_list_dict[repr_name] = np.asarray(self.repr_list_dict[repr_name])  # each item: [N, T-1, d]
            elif self._slice_strategy == 'full_sequence':
                self.repr_list_dict[repr_name] = _as_object_array(self.repr_list_dict[repr_name])  # each item: [T-1, d]
                if self.input_noise and (not self.sep_noise) and hasattr(self, "repr_list_dict_noisy"):
                    self.repr_list_dict_noisy[repr_name] = _as_object_array(self.repr_list_dict_noisy[repr_name])

        if self.split == 'train':
            # Use streaming statistics computed during the loop
            stats_dict = running_stats.compute()
            self.norm_params = {}
            for repr_name in self._repr_list:
                if repr_name in stats_dict:
                    self.norm_params[repr_name] = stats_dict[repr_name]
                else:
                    dim = self.repr_registry.get_modality(repr_name).dim
                    self.norm_params[repr_name] = {
                        'mean': np.zeros(dim, dtype=np.float32),
                        'std': np.ones(dim, dtype=np.float32),
                    }
            # Apply special handling for certain modalities
            if self.mean_std and self.norm_strategy == 'meanstd':
                ignore_norm_keys = ['smplx_betas', 'foot_contact', 'smplx_rot_6d', 'smplx_body_pose_6d']
                for repr_name in ignore_norm_keys:
                    if repr_name in self.norm_params:
                        self.norm_params[repr_name]['mean'] = np.zeros_like(self.norm_params[repr_name]['mean'])
                        self.norm_params[repr_name]['std'] = np.ones_like(self.norm_params[repr_name]['std'])
            
            if (not self.use_ddp) or (self.dist_rank == 0):
                self._save_norm_params_to_path(self.norm_params_path, protocol=2)
            if self.use_ddp:
                dist.barrier()
        else:
            if self.use_ddp:
                dist.barrier() # avoid non-master process loading mean/std stats before master process is able to save them.
            self.norm_params = self._load_norm_params_from_path(self.norm_params_path)

    def __len__(self):
        return len(self.joints_clean_list)

    def __getitem__(self, index):
        positions_clean = self.joints_clean_list[index]
        repr_dict_clean = {}
        for repr_name in self._repr_list:
            repr_dict_clean[repr_name] = self.repr_list_dict[repr_name][index]  # [clip_len, d]

        ####################################### add noise
        if self.input_noise:
            if self.sep_noise:
                ## add different noise to smplx params and joint positions separately
                smplx_params_dict_clean = {}
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    smplx_params_dict_clean[param_name] = self.smplx_params_list_dict[param_name][index]  # [clip_len, d]
                #### add noise to smpl params
                smplx_params_dict_noisy = {}
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    noise = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name],
                                             size=smplx_params_dict_clean[param_name].shape)
                    smplx_params_dict_noisy[param_name] = smplx_params_dict_clean[param_name] + noise
                ### add noise to joint positions
                noise_0 = np.random.normal(loc=0.0, scale=self.noise_std_joint, size=positions_clean.shape)
                positions_noisy = positions_clean + noise_0  # [clip_len, 22, 3]
                positions_noisy = positions_noisy.astype(np.float32)
                ######## creat noisy repr
                repr_dict_noisy = get_repr_smplx(positions=positions_noisy,
                             cano_jts_pos=self.cano_jts_pos,
                             smplx_params_dict=smplx_params_dict_noisy,
                             smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                             device=torch.device('cpu'),
                             modality_names=self._repr_list)
                # NOTE: positions_noisy must be updated from repr_dict_noisy['local_positions'], because get_repr_smplx reduces clip_len by one.
                positions_noisy = repr_dict_noisy['local_positions'].reshape(-1, 22, 3).numpy() # [clip_len-1, 22, 3]
            else:
                ## add noise to smplx params, and noisy joints obtained by FK (our setup)
                positions_noisy = self.joints_noisy_list[index]
                repr_dict_noisy = {}
                for repr_name in self._repr_list:
                    repr_dict_noisy[repr_name] = self.repr_list_dict_noisy[repr_name][index]  # [clip_len, d]

        ####################################### get data items
        item_dict = {}
        if self._slice_strategy == 'full_sequence':
            clip_name = self.seq_name_list[index]
        else:
            # self.seq_name_list only exists for full_sequence; use index as a stable identifier.
            clip_name = str(index)
        item_dict['meta'] = {
            'name': clip_name,
            'split': self.split,
            'clip_start': 0,
            'clip_end': int(positions_clean.shape[0]),
        }
        
        ####################################### add some data arg. here.
        do_data_aug = False
        if do_data_aug:
            from data_sets.motion_representation import rotate_repr_smpl

            angle = np.random.uniform(low=-180.0, high=180.0)
            rot_mat = torch.tensor([[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                    [0, 0, 1]], dtype=torch.float32)

            tensor_repr_dict_clean = {k: torch.from_numpy(v.copy()).float() for k, v in repr_dict_clean.items() if isinstance(v, np.ndarray)}
            if self.input_noise:
                tensor_repr_dict_noisy = {k: torch.from_numpy(v.copy()).float() for k, v in repr_dict_noisy.items() if isinstance(v, np.ndarray)}

            repr_dict_clean = rotate_repr_smpl(repr_dict=tensor_repr_dict_clean, rotation_matrix=rot_mat)
            if self.input_noise:
                repr_dict_noisy = rotate_repr_smpl(repr_dict=tensor_repr_dict_noisy, rotation_matrix=rot_mat)
                positions_noisy = (rot_mat @ torch.from_numpy(positions_noisy).permute(0, 2, 1)).permute(0, 2, 1).numpy()
            
            repr_dict_clean = {k: v.numpy(force=True) for k, v in repr_dict_clean.items()}
            if self.input_noise:
                repr_dict_noisy = {k: v.numpy(force=True) for k, v in repr_dict_noisy.items()}
        
        align_to_yaxis = self.align_to_yaxis
        if align_to_yaxis:
            from data_sets.motion_representation import rotate_repr_smpl_2, face_joint_indx
            l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
            across1 = positions_clean[:, r_hip] - positions_clean[:, l_hip]
            across2 = positions_clean[:, sdr_r] - positions_clean[:, sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]
            forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
            forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

            target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
            root_rot_quat = qbetween_np(forward, target)
            ### several frames have nan values
            root_rot_quat_t = torch.from_numpy(root_rot_quat)
            if torch.isnan(root_rot_quat_t).any():
                if torch.isnan(root_rot_quat_t[0]).any():
                    root_rot_quat_t[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_rot_quat_t.dtype)
                
                valid_mask = ~torch.isnan(root_rot_quat_t).any(dim=-1)
                indices = torch.arange(len(root_rot_quat_t), device=root_rot_quat_t.device)
                last_valid_idx = (indices * valid_mask.long()).cummax(dim=0)[0]
                root_rot_quat = root_rot_quat_t[last_valid_idx].numpy()

            assert not np.isnan(root_rot_quat).any()

            rot_mat = torch.from_numpy(R.from_quat(root_rot_quat[0], scalar_first=True).as_matrix()).float()

            tensor_repr_dict_clean = {k: torch.from_numpy(v.copy()).float() for k, v in repr_dict_clean.items() if isinstance(v, np.ndarray)}
            if self.input_noise:
                tensor_repr_dict_noisy = {k: torch.from_numpy(v.copy()).float() for k, v in repr_dict_noisy.items() if isinstance(v, np.ndarray)}

            repr_dict_clean = rotate_repr_smpl_2(repr_dict=tensor_repr_dict_clean, rotation_matrix=rot_mat)
            if self.input_noise:
                repr_dict_noisy = rotate_repr_smpl_2(repr_dict=tensor_repr_dict_noisy, rotation_matrix=rot_mat)
                positions_noisy = repr_dict_noisy['local_positions'].reshape(-1, 22, 3).numpy()
            
            repr_dict_clean = {k: v.numpy(force=True) for k, v in repr_dict_clean.items()}
            if self.input_noise:
                repr_dict_noisy = {k: v.numpy(force=True) for k, v in repr_dict_noisy.items()}

        
        batch_dtype = np.float32
        item_dict['motion_repr_clean'] = np.concatenate([repr_dict_clean[key] for key in self._repr_list], axis=-1).astype(batch_dtype) # [clip_len-1, 263]
        if self.input_noise:
            item_dict['noisy_joints'] = positions_noisy.astype(batch_dtype)  # [clip_len, 22, 3]
            item_dict['motion_repr_noisy'] = np.concatenate([repr_dict_noisy[key] for key in self._repr_list], axis=-1).astype(batch_dtype) # [clip_len-1, 263]

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
            cfg = SmplxVisConfig(window_title="AMASS Debug", num_repeats=1, fixed_cam=False, camera_follow=True)
            visualize_smplx_sequence(
                joints=joints,
                vertices=verts,
                faces=self.smplx_neutral.faces,
                parent_indices=SMPLH_KINTREE,
                batch_idx=0,
                cfg=cfg,
            )

        
        return item_dict

    # ====================== Visualization & Dataset Statistics ====================== #

    def _get_flat_repr_array(self, repr_name: str, use_noisy: bool = False):
        """
        Stack all clips and time-steps for a given repr into a 2D numpy array: [N_total_frames, dim].

        Parameters
        ----------
        repr_name : str
            Name of the representation in self._repr_list.
        use_noisy : bool
            If True, use noisy reprs (requires self.input_noise and self.sep_noise == False or True accordingly).
        """
        if repr_name not in self._repr_list:
            raise ValueError(f"Unknown repr_name '{repr_name}'. Available: {self._repr_list}")

        if use_noisy:
            if not self.input_noise:
                raise RuntimeError("Noisy representations requested but input_noise=False for this dataset.")
            if not hasattr(self, "repr_list_dict_noisy"):
                raise RuntimeError("repr_list_dict_noisy not initialized. Check input_noise/sep_noise flags.")
            repr_list = self.repr_list_dict_noisy[repr_name]
        else:
            repr_list = self.repr_list_dict[repr_name]

        # repr_list may already be a numpy array (after create_body_repr)
        if isinstance(repr_list, np.ndarray):
            arr = repr_list
        else:
            arr = np.asarray(repr_list)

        # random_uniform_len: shape (N_clips, T-1, D)
        # full_sequence: dtype=object, each entry [T-1, D]; stack over time
        if arr.dtype == object:
            frames = [clip.reshape(-1, clip.shape[-1]) for clip in arr]
            arr_flat = np.concatenate(frames, axis=0)
        else:
            arr_flat = arr.reshape(-1, arr.shape[-1])
        return arr_flat

    def _get_flat_joints_array(self, use_noisy: bool = False):
        """
        Stack all clips and time-steps for joints into 2D array [N_total_frames, J*3].
        """
        if use_noisy:
            if not self.input_noise:
                raise RuntimeError("Noisy joints requested but input_noise=False for this dataset.")
            if not hasattr(self, "joints_noisy_list"):
                raise RuntimeError("joints_noisy_list not initialized. Check input_noise/sep_noise flags.")
            src_list = self.joints_noisy_list
        else:
            src_list = self.joints_clean_list

        clips = [np.asarray(c).reshape(-1, c.shape[-2] * c.shape[-1]) for c in src_list]
        return np.concatenate(clips, axis=0)

    def _normalize_flat_repr_array(self, flat_arr: np.ndarray, repr_name: str):
        """
        Apply the dataset's normalization (self.norm_params, self.norm_strategy) to a flat repr array.

        We reuse DatasetBase._normalize_full_tensor by embedding this repr into a full feature
        tensor with zeros for all other reprs. This guarantees consistency with training-time
        normalization without re-implementing every strategy here.
        """
        if not self.norm_params:
            raise RuntimeError("norm_params is empty. Make sure dataset was initialized with split='train' "
                               "or norm_params loaded for 'test' / 'full_sequence'.")

        if repr_name not in self._repr_indices:
            raise ValueError(f"repr_name '{repr_name}' not found in _repr_indices.")

        n_samples, dim = flat_arr.shape
        total_dim = self.body_feat_dim
        device = self.device
        dtype = torch.float32

        full = torch.zeros((n_samples, total_dim), device=device, dtype=dtype)
        s, e = self._repr_indices[repr_name]
        full[:, s:e] = torch.from_numpy(flat_arr).to(device=device, dtype=dtype)

        full_norm_obj = self._normalize_full_tensor(full, device=device, dtype=dtype, store_sample_norms=False)
        full_norm = cast(torch.Tensor, full_norm_obj)
        # Use narrow to avoid type-checker complaints about multi-dimensional slicing
        width = int(e - s)
        norm_slice = full_norm.narrow(1, int(s), width).detach().cpu().numpy()
        return norm_slice

    def _plot_histogram(self, values: np.ndarray, title: str, xlabel: str, save_path: str,
                        bins: int = 100, kde: bool = True):
        """
        Helper for 1D histogram plotting. Expects values to be 1D array.
        """
        if plt is None:
            if self.logging_enabled:
                cprint.warn("[WARN] matplotlib/seaborn not available; skipping plot:", save_path)
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        flat = values.astype(np.float32).ravel()
        # Avoid degenerate plots if all values are equal
        if flat.size == 0:
            return

        plt.figure(figsize=(6, 4))
        if sns is not None and kde:
            sns.histplot(flat, bins=bins, kde=True, stat="density")
        else:
            plt.hist(flat, bins=bins, density=True, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_xyz_histograms(self, values: np.ndarray, title_prefix: str, xlabel_prefix: str,
                             save_dir: str, suffix: str = "", bins: int = 100, kde: bool = True):
        """
        Plot separate histograms for x, y, z components of 3D data.
        
        Parameters
        ----------
        values : np.ndarray
            Array of shape (N, 3) or (N, J*3) where J is number of joints.
            For (N, J*3), assumes flattened joint positions that need reshaping.
        title_prefix : str
            Prefix for plot titles (e.g., "smplx_trans (raw, clean)")
        xlabel_prefix : str
            Prefix for x-axis labels (e.g., "smplx_trans")
        save_dir : str
            Directory to save plots
        suffix : str
            Suffix for filenames (e.g., "_raw_clean")
        bins : int
            Number of bins for histograms
        kde : bool
            Whether to overlay KDE
        """
        if plt is None:
            if self.logging_enabled:
                cprint.warn("[WARN] matplotlib/seaborn not available; skipping xyz plots in:", save_dir)
            return

        os.makedirs(save_dir, exist_ok=True)

        # Handle shape: could be (N, 3) or (N, J*3) for joints
        if values.shape[1] == 3:
            # Direct 3D data (e.g., smplx_trans)
            x_vals = values[:, 0].astype(np.float32)
            y_vals = values[:, 1].astype(np.float32)
            z_vals = values[:, 2].astype(np.float32)
        elif values.shape[1] % 3 == 0:
            # Flattened joint positions (e.g., local_positions: N, 66 -> N, 22, 3)
            num_joints = values.shape[1] // 3
            values_reshaped = values.reshape(-1, num_joints, 3)
            # Extract all x, y, z across all joints
            x_vals = values_reshaped[:, :, 0].ravel().astype(np.float32)
            y_vals = values_reshaped[:, :, 1].ravel().astype(np.float32)
            z_vals = values_reshaped[:, :, 2].ravel().astype(np.float32)
        else:
            if self.logging_enabled:
                cprint.warn(f"[WARN] Cannot split xyz for array shape {values.shape}, skipping")
            return

        # Plot X component
        if x_vals.size > 0:
            self._plot_histogram(
                x_vals,
                title=f"{title_prefix} - X component",
                xlabel=f"{xlabel_prefix} X value",
                save_path=os.path.join(save_dir, f"{xlabel_prefix}_x{suffix}_hist.png"),
                bins=bins,
                kde=kde
            )

        # Plot Y component
        if y_vals.size > 0:
            self._plot_histogram(
                y_vals,
                title=f"{title_prefix} - Y component",
                xlabel=f"{xlabel_prefix} Y value",
                save_path=os.path.join(save_dir, f"{xlabel_prefix}_y{suffix}_hist.png"),
                bins=bins,
                kde=kde
            )

        # Plot Z component
        if z_vals.size > 0:
            self._plot_histogram(
                z_vals,
                title=f"{title_prefix} - Z component",
                xlabel=f"{xlabel_prefix} Z value",
                save_path=os.path.join(save_dir, f"{xlabel_prefix}_z{suffix}_hist.png"),
                bins=bins,
                kde=kde
            )

    def _compute_basic_stats(self, values: np.ndarray):
        """
        Compute a set of basic descriptive statistics for a 2D array [N, D].
        """
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        stats = {
            "num_samples": int(flat.shape[0]),
            "dim": int(flat.shape[1]),
            "mean_per_dim": flat.mean(axis=0).tolist(),
            "std_per_dim": flat.std(axis=0).tolist(),
            "min_per_dim": flat.min(axis=0).tolist(),
            "max_per_dim": flat.max(axis=0).tolist(),
            "p01_per_dim": np.percentile(flat, 1, axis=0).tolist(),
            "p25_per_dim": np.percentile(flat, 25, axis=0).tolist(),
            "p50_per_dim": np.percentile(flat, 50, axis=0).tolist(),
            "p75_per_dim": np.percentile(flat, 75, axis=0).tolist(),
            "p99_per_dim": np.percentile(flat, 99, axis=0).tolist(),
        }
        return stats

    def visualize_dataset_statistics(self,
                                     output_root=None,
                                     include_noisy: bool = True,
                                     max_frames_per_repr: int = 1_000_000,
                                     save_json: bool = True):
        """
        Visualize and summarize statistics for all representations and joints.

        This covers:
        - Raw (unnormalized) repr distributions for each repr_name.
        - Noisy repr distributions (if input_noise enabled and include_noisy=True).
        - Normalized repr distributions using the same normalization as in DatasetBase/_batch_util.
        - Per-repr mean/std (and other stats) from norm_params.
        - Clean/noisy joint position distributions.

        Parameters
        ----------
        output_root : str
            Directory where plots and JSON stats will be written. Defaults to <logdir>/dataset_stats.
        include_noisy : bool
            Whether to include noisy repr/joints visualizations (if available).
        max_frames_per_repr : int
            Upper bound on number of frames to sample per repr for visualization to keep memory manageable.
        save_json : bool
            If True, saves detailed statistics as JSON alongside plots.
        """
        if output_root is None:
            output_root = os.path.join(self.logdir, "dataset_stats")
        os.makedirs(output_root, exist_ok=True)

        if self.logging_enabled:
            cprint.info(f"[INFO] Visualizing dataset statistics into: {output_root}")

        all_stats = {"repr_stats": {}, "norm_stats": {}, "joints_stats": {}}

        # ---- 1) Per-repr raw / noisy / normalized distributions ---- #
        for repr_name in self._repr_list:
            if self.logging_enabled:
                cprint.info(f"[INFO] Processing repr: {repr_name}")

            repr_dir = os.path.join(output_root, "repr", repr_name)
            os.makedirs(repr_dir, exist_ok=True)

            # Raw / clean
            clean_arr = self._get_flat_repr_array(repr_name, use_noisy=False)
            if clean_arr.shape[0] > max_frames_per_repr:
                idx = np.random.choice(clean_arr.shape[0], size=max_frames_per_repr, replace=False)
                clean_arr = clean_arr[idx]

            repr_stats = {"raw": self._compute_basic_stats(clean_arr)}

            # Special handling for xyz-splittable reprs: smplx_trans and local_positions
            if repr_name in ('smplx_trans', 'local_positions'):
                # Plot xyz components separately
                self._plot_xyz_histograms(
                    clean_arr,
                    title_prefix=f"{repr_name} (raw, clean)",
                    xlabel_prefix=repr_name,
                    save_dir=repr_dir,
                    suffix="_raw_clean"
                )
                # Also plot overall histogram for completeness
                self._plot_histogram(
                    clean_arr,
                    title=f"{repr_name} (raw, clean, all dims)",
                    xlabel=f"{repr_name} value",
                    save_path=os.path.join(repr_dir, f"{repr_name}_raw_clean_hist.png"),
                )
            else:
                # Histogram on scalar distribution (flatten over dims)
                self._plot_histogram(
                    clean_arr,
                    title=f"{repr_name} (raw, clean)",
                    xlabel=f"{repr_name} value",
                    save_path=os.path.join(repr_dir, f"{repr_name}_raw_clean_hist.png"),
                )

            # Noisy (if requested and available)
            if include_noisy and self.input_noise and hasattr(self, "repr_list_dict_noisy"):
                noisy_arr = self._get_flat_repr_array(repr_name, use_noisy=True)
                if noisy_arr.shape[0] > max_frames_per_repr:
                    idx = np.random.choice(noisy_arr.shape[0], size=max_frames_per_repr, replace=False)
                    noisy_arr = noisy_arr[idx]
                repr_stats["raw_noisy"] = self._compute_basic_stats(noisy_arr)
                
                if repr_name in ('smplx_trans', 'local_positions'):
                    # Plot xyz components separately
                    self._plot_xyz_histograms(
                        noisy_arr,
                        title_prefix=f"{repr_name} (raw, noisy)",
                        xlabel_prefix=repr_name,
                        save_dir=repr_dir,
                        suffix="_raw_noisy"
                    )
                    # Also plot overall histogram
                    self._plot_histogram(
                        noisy_arr,
                        title=f"{repr_name} (raw, noisy, all dims)",
                        xlabel=f"{repr_name} value",
                        save_path=os.path.join(repr_dir, f"{repr_name}_raw_noisy_hist.png"),
                    )
                else:
                    self._plot_histogram(
                        noisy_arr,
                        title=f"{repr_name} (raw, noisy)",
                        xlabel=f"{repr_name} value",
                        save_path=os.path.join(repr_dir, f"{repr_name}_raw_noisy_hist.png"),
                    )

            # Normalized (clean) using same normalization as training
            if self.norm_params:
                norm_clean_arr = self._normalize_flat_repr_array(clean_arr, repr_name)
                repr_stats["normalized"] = self._compute_basic_stats(norm_clean_arr)
                
                if repr_name in ('smplx_trans', 'local_positions'):
                    # Plot xyz components separately
                    self._plot_xyz_histograms(
                        norm_clean_arr,
                        title_prefix=f"{repr_name} (normalized, clean)",
                        xlabel_prefix=repr_name,
                        save_dir=repr_dir,
                        suffix="_normalized_clean"
                    )
                    # Also plot overall histogram
                    self._plot_histogram(
                        norm_clean_arr,
                        title=f"{repr_name} (normalized, clean, all dims)",
                        xlabel=f"{repr_name} normalized value",
                        save_path=os.path.join(repr_dir, f"{repr_name}_normalized_clean_hist.png"),
                    )
                else:
                    self._plot_histogram(
                        norm_clean_arr,
                        title=f"{repr_name} (normalized, clean)",
                        xlabel=f"{repr_name} normalized value",
                        save_path=os.path.join(repr_dir, f"{repr_name}_normalized_clean_hist.png"),
                    )

            all_stats["repr_stats"][repr_name] = repr_stats

            if save_json:
                with open(os.path.join(repr_dir, f"{repr_name}_stats.json"), "w") as f:
                    json.dump(repr_stats, f)

        # ---- 2) Norm-parameter statistics (mean/std/etc per repr) ---- #
        norm_dir = os.path.join(output_root, "norm_params")
        os.makedirs(norm_dir, exist_ok=True)
        if self.norm_params:
            for repr_name, params in self.norm_params.items():
                if self.logging_enabled:
                    cprint.info(f"[INFO] Processing norm_params for repr: {repr_name}")

                repr_norm_dir = os.path.join(norm_dir, repr_name)
                os.makedirs(repr_norm_dir, exist_ok=True)

                norm_stats = {}
                for key, val in params.items():
                    arr = np.asarray(val, dtype=np.float64).reshape(-1)
                    norm_stats[key] = {
                        "num_entries": int(arr.size),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "p01": float(np.percentile(arr, 1)),
                        "p25": float(np.percentile(arr, 25)),
                        "p50": float(np.percentile(arr, 50)),
                        "p75": float(np.percentile(arr, 75)),
                        "p99": float(np.percentile(arr, 99)),
                    }

                    # Plot histogram of norm parameter values (e.g. std distribution)
                    self._plot_histogram(
                        arr,
                        title=f"{repr_name} norm param: {key}",
                        xlabel=f"{repr_name} {key}",
                        save_path=os.path.join(repr_norm_dir, f"{repr_name}_norm_{key}_hist.png"),
                    )

                all_stats["norm_stats"][repr_name] = norm_stats

                if save_json:
                    with open(os.path.join(repr_norm_dir, f"{repr_name}_norm_stats.json"), "w") as f:
                        json.dump(norm_stats, f)

        # ---- 3) Joints distributions (clean / noisy) ---- #
        joints_dir = os.path.join(output_root, "joints")
        os.makedirs(joints_dir, exist_ok=True)

        # Clean joints
        joints_clean = self._get_flat_joints_array(use_noisy=False)
        joints_stats_clean = self._compute_basic_stats(joints_clean)
        all_stats["joints_stats"]["clean"] = joints_stats_clean
        
        # Plot xyz components separately for joints
        self._plot_xyz_histograms(
            joints_clean,
            title_prefix="Joints (clean)",
            xlabel_prefix="joints",
            save_dir=joints_dir,
            suffix="_clean"
        )
        # Also plot overall histogram
        self._plot_histogram(
            joints_clean,
            title="Joints (clean, all coords)",
            xlabel="Joint coordinate value",
            save_path=os.path.join(joints_dir, "joints_clean_hist.png"),
        )

        # Noisy joints
        if include_noisy and self.input_noise and hasattr(self, "joints_noisy_list"):
            try:
                joints_noisy = self._get_flat_joints_array(use_noisy=True)
                joints_stats_noisy = self._compute_basic_stats(joints_noisy)
                all_stats["joints_stats"]["noisy"] = joints_stats_noisy
                
                # Plot xyz components separately for joints
                self._plot_xyz_histograms(
                    joints_noisy,
                    title_prefix="Joints (noisy)",
                    xlabel_prefix="joints",
                    save_dir=joints_dir,
                    suffix="_noisy"
                )
                # Also plot overall histogram
                self._plot_histogram(
                    joints_noisy,
                    title="Joints (noisy, all coords)",
                    xlabel="Joint coordinate value",
                    save_path=os.path.join(joints_dir, "joints_noisy_hist.png"),
                )
            except RuntimeError:
                pass

        # Save global summary JSON
        if save_json:
            with open(os.path.join(output_root, "dataset_stats_summary.json"), "w") as f:
                json.dump(all_stats, f)

        if self.logging_enabled:
            cprint.info(f"[INFO] Finished dataset statistics visualization. Output at: {output_root}")
