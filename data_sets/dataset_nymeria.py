"""
Dataset for the Nymeria dataset using Zarr for high-performance I/O.

This module provides DatasetNymeria class that loads pre-processed SMPLX data
from a Zarr store with chunked, compressed storage for maximum efficiency.

Zarr store structure:
    <zarr_root>/<seq-name>/
        - global_orient: (T, 3) axis-angle global orientation
        - transl: (T, 3) translation
        - body_pose: (T, 63) body pose in axis-angle (21 joints * 3)
        - betas: (1, 10) shape parameters
"""

import os
import json
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import zarr
import numcodecs
from cprint import cprint
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.other_utils import SMPLH_KINTREE
from torch.utils.data import DataLoader

from .dataset_mocap import DatasetMocap
from data_sets.motion_representation import (
    get_repr_smplx, rotate_repr_smpl_2, face_joint_indx, transform_repr_smpl
)
from data_sets.common.quaternion import (
    qbetween_np, rotmat_to_rot6d, aa_to_rotmat, rot6d_to_rotmat, rotmat_to_aa
)

# Disable internal threading to avoid contention with DataLoader workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
numcodecs.blosc.set_nthreads(1)

# Zarr v2/v3 compatibility
ZARR_V3 = int(zarr.__version__.split('.')[0]) >= 3


def _open_zarr_group(path: str, mode: str = 'r') -> zarr.Group:
    """Open Zarr group with v2/v3 compatibility."""
    if ZARR_V3:
        return zarr.open_group(path, mode=mode)
    else:
        store = zarr.DirectoryStore(path)
        return zarr.open_group(store, mode=mode)


def _get_group_keys(group: zarr.Group):
    """Get group keys with v2/v3 compatibility."""
    if ZARR_V3:
        return list(group.members())
    else:
        return list(group.group_keys())


class DatasetNymeria(DatasetMocap):
    """
    High-performance Zarr-based dataloader for the Nymeria dataset.
    
    Key optimizations:
    - Zarr chunked storage with LZ4 compression for fast random access
    - Lazy loading: only reads requested frame ranges (no full sequence load)
    - Process-local Zarr handles: each DataLoader worker opens its own connection
    - Cached metadata: instant initialization via JSON sidecar file
    - Motion representation caching: precompute once, reuse in __getitem__
    - Betas caching: avoid repeated reads of constant shape parameters
    - CPU tensor output: device transfer deferred to training loop
    
    Recommended DataLoader settings for CPU-intensive workloads:
        DataLoader(
            dataset,
            num_workers=4,  # Reduce from default; too many workers cause contention
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,  # Queue multiple batches per worker
            pin_memory=True,  # Faster GPU transfer
        )
    """
    SPLIT_RATIO = 0.85
    SPLIT_SEED = 0
    SPLIT_META_FILENAME = "nymeria_split.json"
    TRIM_FRAMES = 20

    def __init__(
        self,
        opt: dict,
        input_noise: bool = False,
        noise_std_joint: float = 0.0,
        noise_std_smplx_global_rot: float = 0.0,
        noise_std_smplx_body_rot: float = 0.0,
        noise_std_smplx_trans: float = 0.0,
        noise_std_smplx_betas: float = 0.0,
        **kwargs,
    ):
        self.homogenous_transform_align_ = np.eye(4, dtype=np.float32)
        self.floor_height_offset = 0.0
        norm_params_path = kwargs.get("norm_params_path", "")
        self.num_compute_workers = kwargs.get("num_compute_workers")
        self.norm_spacing = int(kwargs.get("norm_spacing"))

        super().__init__(
            opt,
            input_noise=input_noise,
            noise_std_joint=noise_std_joint,
            noise_std_smplx_global_rot=noise_std_smplx_global_rot,
            noise_std_smplx_body_rot=noise_std_smplx_body_rot,
            noise_std_smplx_trans=noise_std_smplx_trans,
            noise_std_smplx_betas=noise_std_smplx_betas,
            **kwargs,
        )

        if not norm_params_path:
            train_norm_filename = f"{self.__class__.__name__}_train_norm_params.pkl"
            self.norm_params_path = self._resolve_norm_params_path("", train_norm_filename)

        self.zarr_path = self.root
        assert self.zarr_path.endswith('.zarr') and os.path.isdir(self.zarr_path), \
            f"Zarr store not found at {self.zarr_path}"
        self._zarr_store: Optional[zarr.Group] = None
        self._betas_cache: Dict[str, np.ndarray] = {}  # Cache betas per sequence (name -> betas array)
        
        self.data = {'pose': [], 'name': [], 'path': []}
        self._load_metadata()
        self._apply_split()
        
        # Motion representation cache infrastructure
        self._motion_repr_cache: np.ndarray = np.empty((0, 0, 0), dtype=np.float32)  # Indexed by clip index
        self._clip_index_list: list[Tuple[str, int, int]] = []  # (seq_name, start, end) for each clip
        self._cache_ready = False
        cache_key = self._get_cache_key()
        self._cache_dir = Path(self.zarr_path).parent / f".nymeria_cache_{cache_key[:12]}"
        
        self._initialize_norm_params()

    def _get_zarr_store(self) -> zarr.Group:
        """Get process-local Zarr store handle (lazy initialization)."""
        if self._zarr_store is None:
            self._zarr_store = _open_zarr_group(self.zarr_path, mode='r')
        return self._zarr_store

    def _get_cache_key(self) -> str:
        """Generate a deterministic hash key from parameters affecting motion representation."""
        cache_params = {
            'clip_len': self.clip_len,
            'repr_list': self._repr_list,
            'split': self.split,
            'align_to_yaxis': self.align_to_yaxis,
            'trim_frames': self.TRIM_FRAMES,
            'cano_jts_pos': self.cano_jts_pos,
        }
        payload = json.dumps(cache_params, sort_keys=True, separators=(',', ':'), ensure_ascii=True, default=str)
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()

    def _try_load_cache(self) -> bool:
        """Attempt to load motion representation cache from disk. Returns True if successful."""
        cache_file = self._cache_dir / "motion_repr_cache.npy"
        clip_list_file = self._cache_dir / "clip_index_list.npz"
        
        if not cache_file.exists() or not clip_list_file.exists():
            return False

        clip_data = np.load(clip_list_file)
        seq_names = clip_data["seq_names"]
        starts = clip_data["starts"]
        ends = clip_data["ends"]

        if not (len(seq_names) == len(starts) == len(ends)):
            if self.logging_enabled:
                cprint.warn("[WARN] Cache clip index list has inconsistent lengths.")
            return False

        self._clip_index_list = list(zip(seq_names.tolist(), starts.tolist(), ends.tolist()))
        self._motion_repr_cache = np.load(cache_file, mmap_mode='r')

        if len(self._motion_repr_cache) != len(self._clip_index_list):
            if self.logging_enabled:
                cprint.warn("[WARN] Cache motion repr and clip index lengths mismatch.")
            return False

        if self.logging_enabled:
            cprint.info(f"[INFO] Loaded motion repr cache: {len(self._motion_repr_cache)} clips from {self._cache_dir}")
        self._cache_ready = True
        return True

    def _build_and_save_cache(self, repr_generator: Iterator[Dict[str, np.ndarray]], num_clips: int) -> None:
        """Build motion representation cache from generator and save to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._motion_repr_cache = np.empty((0, 0, 0), dtype=np.float32)
        self._clip_index_list = self._build_global_clip_list()
        
        if self.logging_enabled:
            cprint.info(f"[INFO] Building motion repr cache for {num_clips} clips...")
        
        memmap = None
        cached_count = 0
        clip_len = None
        repr_dim = None
        cache_file = self._cache_dir / "motion_repr_cache.npy"
        clip_list_file = self._cache_dir / "clip_index_list.npz"
        for repr_dict in tqdm(repr_generator, total=num_clips, desc="Building cache", disable=not self.logging_enabled):
            # Concatenate repr_dict to motion_repr array (same as _build_item output)
            motion_repr = np.concatenate([repr_dict[key] for key in self._repr_list], axis=-1).astype(np.float32)
            if memmap is None:
                clip_len, repr_dim = motion_repr.shape
                memmap = np.lib.format.open_memmap(
                    cache_file,
                    mode='w+',
                    dtype=np.float32,
                    shape=(num_clips, clip_len, repr_dim),
                )
            memmap[cached_count] = motion_repr
            cached_count += 1

        if memmap is None or cached_count == 0:
            if self.logging_enabled:
                cprint.warn("[WARN] No motion representations were cached.")
            return

        memmap.flush()

        assert cached_count == num_clips

        self._clip_index_list = self._clip_index_list[:cached_count]
        max_name_len = max(len(name) for name, _, _ in self._clip_index_list)
        seq_names = np.array([name for name, _, _ in self._clip_index_list], dtype=f'<U{max_name_len}')
        starts = np.array([start for _, start, _ in self._clip_index_list], dtype=np.int64)
        ends = np.array([end for _, _, end in self._clip_index_list], dtype=np.int64)
        np.savez(clip_list_file, seq_names=seq_names, starts=starts, ends=ends)

        self._motion_repr_cache = np.load(cache_file, mmap_mode='r')
        
        if self.logging_enabled:
            cprint.info(f"[INFO] Saved motion repr cache to {self._cache_dir}")
        self._cache_ready = True

    def _get_effective_length(self, total_len: int) -> int:
        return max(0, total_len - self.TRIM_FRAMES)

    def _get_cached_betas(self, seq_name: str) -> np.ndarray:
        """Get betas for a sequence, using cache to avoid repeated Zarr reads."""
        if seq_name not in self._betas_cache:
            store = self._get_zarr_store()
            self._betas_cache[seq_name] = np.asarray(store[seq_name]['betas'][:])
        return self._betas_cache[seq_name]

    def _load_metadata(self):
        """Load sequence metadata from JSON sidecar or scan Zarr store."""
        meta_path = os.path.join(self.zarr_path, "nymeria_meta.json")
        skipped = []
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            for seq_name, frame_count in meta.items():
                effective_len = self._get_effective_length(frame_count)
                if effective_len <= 0:
                    skipped.append(seq_name)
                    continue
                self.data['name'].append(seq_name)
                self.data['pose'].append(range(effective_len))
                self.data['path'].append(seq_name)
            if self.logging_enabled:
                cprint.info(f"[INFO] Loaded {len(meta)} sequences from cached metadata")
        else:
            if self.logging_enabled:
                cprint.info(f"[INFO] Scanning Zarr store for metadata (one-time)...")
            store = self._get_zarr_store()
            meta = {}
            for item in _get_group_keys(store):
                seq_name = item[0] if isinstance(item, tuple) else item
                frame_count = np.asarray(store[seq_name]['transl']).shape[0]
                meta[seq_name] = frame_count
                effective_len = self._get_effective_length(frame_count)
                if effective_len <= 0:
                    skipped.append(seq_name)
                    continue
                self.data['name'].append(seq_name)
                self.data['pose'].append(range(effective_len))
                self.data['path'].append(seq_name)
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            if self.logging_enabled:
                cprint.info(f"[INFO] Cached metadata for {len(meta)} sequences")
        if self.logging_enabled and skipped:
            cprint.warn(f"[WARN] Skipped {len(skipped)} sequences shorter than {self.TRIM_FRAMES} frames after trimming.")

    def _load_or_create_split_names(self, all_names: list[str]) -> Tuple[list[str], list[str]]:
        split_path = os.path.join(self.zarr_path, self.SPLIT_META_FILENAME)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split_meta = json.load(f)
            train_names = split_meta.get("train", [])
            test_names = split_meta.get("test", [])
            if self.logging_enabled:
                cprint.info(f"[INFO] Loaded Nymeria split from {split_path}")
            return train_names, test_names

        names = sorted(all_names)
        rng = np.random.RandomState(self.SPLIT_SEED)
        rng.shuffle(names)
        num_total = len(names)
        num_train = int(np.floor(num_total * self.SPLIT_RATIO))
        if num_total > 1:
            num_train = max(1, min(num_train, num_total - 1))
        train_names = names[:num_train]
        test_names = names[num_train:]

        split_meta = {
            "train": train_names,
            "test": test_names,
            "seed": self.SPLIT_SEED,
            "ratio": self.SPLIT_RATIO,
        }
        try:
            with open(split_path, 'w') as f:
                json.dump(split_meta, f, indent=2)
            if self.logging_enabled:
                cprint.info(f"[INFO] Cached Nymeria split to {split_path}")
        except OSError as exc:
            if self.logging_enabled:
                cprint.warn(f"[WARN] Could not write split file {split_path}: {exc}")

        return train_names, test_names

    def _apply_split(self) -> None:
        if self.split == "val":
            if self.logging_enabled:
                cprint.warn("[WARN] Nymeria does not define a val split; using test split instead.")
            split = "test"
        else:
            split = self.split

        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split '{self.split}' for Nymeria")

        all_data = {k: list(v) for k, v in self.data.items()}
        all_names = all_data.get("name", [])
        train_names, test_names = self._load_or_create_split_names(all_names)
        selected_names = train_names if split == "train" else test_names

        name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        self.data = {k: [] for k in all_data.keys()}
        missing = []
        for name in selected_names:
            idx = name_to_idx.get(name)
            if idx is None:
                missing.append(name)
                continue
            for key in self.data.keys():
                self.data[key].append(all_data[key][idx])

        if self.logging_enabled:
            if missing:
                cprint.warn(f"[WARN] Split list contains {len(missing)} missing sequences.")
            unused = set(all_names) - set(selected_names)
            cprint.info(f"[INFO] Nymeria split '{split}': {len(self.data['name'])} sequences "
                        f"(unused={len(unused)}, ratio={self.SPLIT_RATIO:.2f})")

    def _get_item_from_data(self, index: int) -> Tuple[str, dict]:
        """
        Load SMPLX parameters from Zarr store.
        
        For full_sequence: loads the entire sequence (no trimming)
        For random_uniform_len: raises NotImplementedError (use cache instead)
        
        Returns:
            (seq_name, smplx_params_dict) tuple
        """
        store = self._get_zarr_store()
        
        if self._slice_strategy == 'full_sequence':
            seq_name = self.data['name'][index]
            start, end = 0, None  # Load all frames, no trim
        else:
            raise NotImplementedError("Use cache for random_uniform_len strategy")
        
        seq_group = store[seq_name]
        
        # Read data from Zarr
        global_orient = torch.from_numpy(np.asarray(seq_group['global_orient'][start:end]))
        transl = torch.from_numpy(np.asarray(seq_group['transl'][start:end]))
        body_pose = torch.from_numpy(np.asarray(seq_group['body_pose'][start:end]))
        betas_raw = self._get_cached_betas(seq_name)
        betas = torch.from_numpy(betas_raw)
        
        T = transl.shape[0]
        device = self.device
        dtype = self.dtype
        
        # Expand betas to T frames
        if betas.shape[0] == 1:
            betas = betas.expand(T, -1)
        
        smplx_params_dict = {
            "transl": transl.to(device, dtype=dtype),
            "body_pose": body_pose.to(device, dtype=dtype),
            "global_orient": global_orient.to(device, dtype=dtype),
            "betas": betas.to(device, dtype=dtype),
            "jaw_pose": torch.zeros((T, 3), device=device, dtype=dtype),
            "leye_pose": torch.zeros((T, 3), device=device, dtype=dtype),
            "reye_pose": torch.zeros((T, 3), device=device, dtype=dtype),
            "left_hand_pose": torch.zeros((T, 45), device=device, dtype=dtype),
            "right_hand_pose": torch.zeros((T, 45), device=device, dtype=dtype),
            "expression": torch.zeros((T, 10), device=device, dtype=dtype),
        }
        
        return seq_name, smplx_params_dict

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        if self.data is None or 'name' not in self.data:
            return 0
        if self._slice_strategy == 'full_sequence':
            return len(self.data['pose'])
        elif self._slice_strategy in ('random_uniform_len', 'high_motion_weighted', 'high_motion_focused'):
            return len(self._clip_index_list)
        else:
            raise ValueError(f"Invalid slice strategy: {self._slice_strategy}")

    def __getitem__(self, index: int) -> dict:
        """Get a data item, using cache if available to skip expensive computation."""
        # The cache is guaranteed to hit: these were already computed when norm_params was derived.
        assert self._cache_ready
        if self._slice_strategy == 'random_uniform_len':
            # Use cached motion representation directly
            if index < len(self._motion_repr_cache):
                motion_repr = self._motion_repr_cache[index]
                seq_name, start, end = self._clip_index_list[index]
                
                item_dict = {
                    'motion_repr_clean': motion_repr,
                    'meta': {
                        'name': seq_name,
                        'split': self.split,
                        'clip_start': start,
                        'clip_end': end,
                    }
                }
                if self.input_noise:
                    item_dict['motion_repr_noisy'] = np.array(motion_repr, copy=True)
                    if 'local_positions' in self._repr_indices:
                        pos_start, pos_end = self._repr_indices['local_positions']
                        local_positions = motion_repr[:, pos_start:pos_end]
                        item_dict['noisy_joints'] = local_positions.reshape(-1, self.num_joints, 3)
                    else:
                        item_dict['noisy_joints'] = np.zeros(
                            (motion_repr.shape[0], self.num_joints, 3),
                            dtype=np.float32,
                        )
                return item_dict
        elif self._slice_strategy == 'full_sequence':
            return super().__getitem__(index)
        else:
            raise ValueError(f"Unsupported slice_strategy: {self._slice_strategy}")
        
    # ====================== Streaming Normalization ====================== #

    def _iter_clip_ranges_for_norm(self, total_len: int) -> Iterator[Tuple[int, int]]:
        """Yield non-overlapping clip ranges matching random_uniform_len sampling."""
        effective_len = self._get_effective_length(total_len)
        if effective_len < self.clip_len:
            return
        end_limit = self.TRIM_FRAMES + effective_len
        for start in range(self.TRIM_FRAMES, end_limit - self.clip_len + 1, self.clip_len):
            yield start, start + self.clip_len

    def _compute_repr_for_zarr_chunk(self, seq_group: zarr.Group, start: int, end: int, seq_name: str) -> Dict[str, np.ndarray]:
        """Compute motion representation for a Zarr chunk."""
        T = end - start

        global_orient = torch.from_numpy(np.asarray(seq_group['global_orient'][start:end])).to(self.device, dtype=self.dtype)
        transl = torch.from_numpy(np.asarray(seq_group['transl'][start:end])).to(self.device, dtype=self.dtype)
        body_pose = torch.from_numpy(np.asarray(seq_group['body_pose'][start:end])).to(self.device, dtype=self.dtype)
        betas_raw = self._get_cached_betas(seq_name)
        betas = torch.from_numpy(betas_raw).to(self.device, dtype=self.dtype)
        betas = betas.expand(T, -1) if betas.shape[0] == 1 else betas[start:end]

        smplx_params = {
            "transl": transl, "body_pose": body_pose, "global_orient": global_orient, "betas": betas,
            "jaw_pose": torch.zeros((T, 3), device=self.device, dtype=self.dtype),
            "leye_pose": torch.zeros((T, 3), device=self.device, dtype=self.dtype),
            "reye_pose": torch.zeros((T, 3), device=self.device, dtype=self.dtype),
            "left_hand_pose": torch.zeros((T, 45), device=self.device, dtype=self.dtype),
            "right_hand_pose": torch.zeros((T, 45), device=self.device, dtype=self.dtype),
            "expression": torch.zeros((T, 10), device=self.device, dtype=self.dtype),
        }

        with torch.no_grad():
            J = self.smplx_neutral(**smplx_params).joints[:, :22, :]

        repr_dict = {
            "smplx_trans": transl,
            "smplx_rot_6d": rotmat_to_rot6d(aa_to_rotmat(global_orient.reshape(-1, 3))).reshape(T, 1, 6),
            "smplx_body_pose_6d": rotmat_to_rot6d(aa_to_rotmat(body_pose.reshape(-1, 3))).reshape(T, 21, 6),
            "smplx_betas": betas,
            "local_positions": J.reshape(T, -1),
        }

        rot_mat = torch.from_numpy(self.homogenous_transform_align_[:3, :3]).float().to(self.device)
        repr_dict = rotate_repr_smpl_2(repr_dict, rotation_matrix=rot_mat)

        positions = repr_dict['local_positions'].reshape(T, self.num_joints, 3)
        
        # Per-chunk canonicalization: use this chunk's first frame for xy offset
        # This matches __getitem__ / _build_item behavior for consistent norm params
        first_frame_root_xy = positions[0, 0, :2]
        floor_height_offset = positions[..., 2].amin().item()

        transform = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(T, 1, 1)
        transform[:, 0, 3] = -first_frame_root_xy[0]
        transform[:, 1, 3] = -first_frame_root_xy[1]
        transform[:, 2, 3] = -floor_height_offset

        repr_dict = transform_repr_smpl(repr_dict, transform=transform)

        cano_positions = repr_dict['local_positions'].reshape(T, self.num_joints, 3).cpu().numpy()
        cano_smplx_params = {
            "transl": repr_dict["smplx_trans"].cpu().numpy(),
            "body_pose": rotmat_to_aa(rot6d_to_rotmat(repr_dict['smplx_body_pose_6d'])).cpu().numpy(),
            "global_orient": rotmat_to_aa(rot6d_to_rotmat(repr_dict['smplx_rot_6d'])).cpu().numpy(),
            "betas": repr_dict['smplx_betas'].cpu().numpy(),
        }

        repr_dict_clean = get_repr_smplx(
            positions=cano_positions, cano_jts_pos=self.cano_jts_pos, smplx_params_dict=cano_smplx_params,
            feet_vel_thre=5e-5,
            smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
            device=self.device, modality_names=self._repr_list
        )

        if self.align_to_yaxis:
            positions_clean = repr_dict_clean['local_positions'].reshape(-1, self.num_joints, 3)
            l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
            across = (positions_clean[:, r_hip] - positions_clean[:, l_hip]) + (positions_clean[:, sdr_r] - positions_clean[:, sdr_l])
            across = across / (np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis] + 1e-8)
            forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
            forward = forward / (np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis] + 1e-8)

            target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
            root_rot_quat = qbetween_np(forward, target)
            if np.isnan(root_rot_quat).any():
                nan_idx = np.where(np.isnan(root_rot_quat).any(axis=1))[0]
                for idx in nan_idx:
                    root_rot_quat[idx] = root_rot_quat[idx - 1] if idx > 0 else np.array([1, 0, 0, 0])

            rot_mat = torch.from_numpy(R.from_quat(root_rot_quat[0], scalar_first=True).as_matrix()).float()
            tensor_repr = {k: torch.from_numpy(v).float() for k, v in repr_dict_clean.items() if isinstance(v, np.ndarray)}
            repr_dict_clean = rotate_repr_smpl_2(repr_dict=tensor_repr, rotation_matrix=rot_mat)
            repr_dict_clean = {k: v.numpy() for k,v in repr_dict_clean.items()}

        return repr_dict_clean

    def _build_global_clip_list(self) -> list[Tuple[str, int, int]]:
        """Build a list of all (seq_name, start, end) clip tuples across the dataset."""
        store = self._get_zarr_store()
        all_clips = []
        for name in self.data['name']:
            total_len = np.asarray(store[name]['transl']).shape[0]
            for start, end in self._iter_clip_ranges_for_norm(total_len):
                all_clips.append((name, start, end))
        return all_clips

    def _initialize_norm_params(self):
        """Compute or load normalization parameters and motion representation cache."""
        if self._norm_params_ready:
            return

        # Step 1: Try to load cached motion representations
        cache_loaded = self._try_load_cache()
        
        # Step 2: Build cache if it doesn't exist (for train split or when computing norm params)
        if not cache_loaded:
            # Build cache with ALL clips (not subsampled like norm params)
            all_clips = self._build_global_clip_list()
            num_all_clips = len(all_clips)
            
            if num_all_clips == 0:
                raise RuntimeError("No data files found.")
            
            # Determine parallelism level
            num_workers = self.num_compute_workers or 1
            
            if num_workers <= 1:
                # Single-threaded: use simple generator
                def all_clips_generator():
                    store = self._get_zarr_store()
                    for name, start, end in all_clips:
                        seq_group = store[name]
                        yield self._compute_repr_for_zarr_chunk(seq_group, start, end, name)
                
                self._build_and_save_cache(all_clips_generator(), num_all_clips)
            else:
                # Multi-threaded: use ThreadPoolExecutor for parallel computation
                # ThreadPoolExecutor works well here because:
                # 1. Zarr I/O releases GIL
                # 2. PyTorch GPU operations release GIL
                # 3. Avoids CUDA context issues with multiprocessing
                if self.logging_enabled:
                    cprint.info(f"[INFO] Using {num_workers} threads for parallel cache building...")
                
                def parallel_clips_generator():
                    store = self._get_zarr_store()
                    
                    def compute_clip(clip_info):
                        name, start, end = clip_info
                        seq_group = store[name]
                        return self._compute_repr_for_zarr_chunk(seq_group, start, end, name)
                    
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # Submit all tasks in batch and yield results in order
                        futures = [executor.submit(compute_clip, clip) for clip in all_clips]
                        for future in futures:
                            yield future.result()

                self._build_and_save_cache(parallel_clips_generator(), num_all_clips)
        
        # Step 3: Compute or load norm params
        if self.split == 'train':
            # Compute norm params from cached data using subsampling (norm_spacing)
            if self._cache_ready and len(self._motion_repr_cache) > 0:
                # Use cached data for norm params computation with subsampling
                total_clips = len(self._motion_repr_cache)
                spacing = max(1, self.norm_spacing)
                selected_indices = list(range(0, total_clips, spacing))
                
                if self.logging_enabled:
                    cprint.info(f"[INFO] Computing norm params from cached data: {len(selected_indices)}/{total_clips} clips (spacing={spacing})")
                
                def cached_repr_generator():
                    for idx in selected_indices:
                        # Convert motion_repr back to repr_dict format for norm computation
                        motion_repr = self._motion_repr_cache[idx]
                        repr_dict = {}
                        for repr_name, (start, end) in self._repr_indices.items():
                            repr_dict[repr_name] = motion_repr[:, start:end]
                        yield repr_dict
                
                self.norm_params = self._compute_norm_params_streaming(
                    cached_repr_generator(),
                    num_items=len(selected_indices),
                    desc="Computing norm params from cache",
                )
            
            if (not self.use_ddp) or (self.dist_rank == 0):
                self._save_norm_params_to_path(self.norm_params_path, protocol=2)
            if self.use_ddp:
                dist.barrier()
        else:
            # Non-train split: load existing norm params
            if self.use_ddp:
                dist.barrier()
            self.norm_params = self._load_norm_params_from_path(self.norm_params_path)

        self._norm_params_ready = True

    def get_sequence_info(self, index: int) -> Dict:
        """Get metadata for a specific sequence."""
        if index >= len(self.data['name']):
            raise IndexError(f"Index {index} out of range")
        name = self.data['name'][index]
        length = len(self.data['pose'][index])
        store = self._get_zarr_store()
        betas = np.asarray(store[name]['betas'][:])
        return {'name': name, 'num_frames': length, 'duration_seconds': length / 30.0, 'betas_mean': betas.mean(axis=0).tolist()}

    # ====================== Visualization Utilities ====================== #

    def visualize_sequence(
        self,
        index: int,
        num_keyframes: int = 10,
        show_skeleton: bool = True,
        show_mesh: bool = True,
        camera_follow: bool = True,
        fixed_cam: bool = False,
        window_title: Optional[str] = None,
        num_repeats: int = 1,
        orbit_period: int = 180,
        orbit_radius: float = 2.5,
        orbit_height: float = 1.2,
    ) -> None:
        """
        Visualize a single sequence from the dataset.

        This method is only available when slice_strategy is 'full_sequence'.

        Args:
            index: Sequence index in the dataset
            num_keyframes: Number of keyframes to display (evenly spaced)
            show_skeleton: Whether to show skeleton joints and bones
            show_mesh: Whether to show SMPL-X mesh
            camera_follow: Whether the camera follows the subject with an orbital path
            fixed_cam: Use a fixed camera pose (disables camera_follow)
            window_title: Window title override (defaults to sequence name)
            num_repeats: Number of times to repeat the animation
            orbit_period: Frames per camera orbit
            orbit_radius: Orbit radius around the subject
            orbit_height: Orbit height offset
        """
        # if self._slice_strategy != 'full_sequence':
        #     raise ValueError("visualize_sequence() only works with slice_strategy='full_sequence'")

        if index >= len(self):
            raise IndexError(f"Index {index} out of range (dataset has {len(self)} sequences)")

        # Get the processed item
        item = self.__getitem__(index)
        motion_repr = item['motion_repr_clean']
        meta = item['meta']
        name = meta['name']

        T = motion_repr.shape[0]

        if self.logging_enabled:
            cprint.info(f"[VIS] Visualizing sequence '{name}' with {T} frames")

        from utils.repr_adapter import decode_smplx_from_full_repr
        from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence

        motion_repr_t = torch.from_numpy(motion_repr).unsqueeze(0).float()
        output_rot_flag = 'R_world_joints' in self._repr_list
        _, joints_t, verts_t = decode_smplx_from_full_repr(
            full_repr=motion_repr_t,
            repr_list=self._repr_list,
            repr_indices=self._repr_indices,
            smplx_model=self.smplx_neutral,
            output_global_rotations=output_rot_flag,
            cano_jts_pos=self.cano_jts_pos,
            return_verts=show_mesh,
        )

        if num_keyframes is not None and num_keyframes > 0:
            total_frames = joints_t.shape[1]
            if total_frames > num_keyframes:
                keyframes = torch.linspace(
                    0, total_frames - 1, steps=num_keyframes, device=joints_t.device
                ).long()
                joints_t = joints_t.index_select(1, keyframes)
                if verts_t is not None:
                    verts_t = verts_t.index_select(1, keyframes)

        cfg = SmplxVisConfig(
            window_title=window_title or f"Nymeria: {name}",
            num_repeats=num_repeats,
            fixed_cam=fixed_cam,
            camera_follow=camera_follow,
            orbit_period=orbit_period,
            orbit_radius=orbit_radius,
            orbit_height=orbit_height,
            show_mesh=show_mesh,
            show_skeleton=show_skeleton,
        )
        visualize_smplx_sequence(
            joints=joints_t if show_skeleton else None,
            vertices=verts_t if show_mesh else None,
            faces=self.smplx_neutral.faces if show_mesh else None,
            parent_indices=SMPLH_KINTREE,
            batch_idx=0,
            cfg=cfg,
        )

    def visualize_dataset_samples(
        self,
        num_samples: int = 5,
        num_keyframes: int = 5,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Visualize multiple random samples from the dataset.

        Args:
            num_samples: Number of sequences to visualize
            num_keyframes: Number of keyframes per sequence
            save_dir: If provided, save visualizations to this directory
        """
        if self._slice_strategy != 'full_sequence':
            raise ValueError("visualize_dataset_samples() only works with slice_strategy='full_sequence'")

        num_samples = min(num_samples, len(self))
        indices = np.random.choice(len(self), size=num_samples, replace=False)

        if save_dir and self.logging_enabled:
            cprint.warn("[VIS] save_dir is ignored; visualization is interactive only.")

        for i, idx in enumerate(indices):
            cprint.info(f"[VIS] Sample {i + 1}/{num_samples}")
            self.visualize_sequence(
                index=idx,
                num_keyframes=num_keyframes,
            )

    def print_dataset_summary(self) -> None:
        """Print dataset summary."""
        num_seqs = len(self.data['name'])
        total_frames = sum(len(p) for p in self.data['pose'])
        print(f"\n{'='*60}\nNymeria Dataset Summary ({self.split})\n{'='*60}")
        print(f"  Zarr store:     {self.zarr_path}")
        print(f"  Sequences:      {num_seqs}")
        print(f"  Total frames:   {total_frames}")
        print(f"  Avg frames/seq: {total_frames/num_seqs:.1f}" if num_seqs else "  N/A")
        print(f"  Slice strategy: {self._slice_strategy}")
        print(f"  Clip length:    {self.clip_len}\n{'='*60}\n")


# ====================== Zarr Conversion Utility ====================== #

def convert_pt_to_zarr(
    input_root: str = "data/nymeria/processed",
    output_zarr_root: str = "data/nymeria/",
    time_chunk_size: int = 150,
):
    """Convert .pt files to one Zarr store per config group with LZ4 compression."""
    import glob
    import copy
    import hashlib
    
    def _find_config_path(pt_path: str) -> Optional[str]:
        candidates = [
            pt_path.replace(".pt", "_config.yaml"),
            pt_path.replace(".pt", "_config.yml"),
            pt_path.replace(".pt", "_config.json"),
        ]
        return next((c for c in candidates if os.path.exists(c)), None)

    def _load_config(config_path: str) -> dict:
        if config_path.endswith((".yaml", ".yml")):
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)
        with open(config_path, "r") as f:
            return json.load(f)

    def _drop_path(cfg: dict, keys: tuple[str, ...]) -> None:
        cursor = cfg
        for key in keys[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return
            cursor = cursor[key]
        if isinstance(cursor, dict):
            cursor.pop(keys[-1], None)

    def _normalize_config(cfg: dict) -> dict:
        if not isinstance(cfg, dict):
            return cfg
        normalized = copy.deepcopy(cfg)
        for key in ("meta", "results", "hydra"):
            normalized.pop(key, None)
        for key in ("timestamp", "mvnx_path", "output_params", "output_config", "output_dir"):
            normalized.pop(key, None)
        for path in (
            ("paths", "mvnx_path"),
            ("paths", "out_dir"),
            ("paths", "output_params"),
            ("paths", "output_config"),
            ("paths", "output_dir"),
        ):
            _drop_path(normalized, path)
        return normalized

    def _hash_config(cfg: dict) -> str:
        payload = json.dumps(
            cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _get_by_path(cfg: dict, keys: tuple[str, ...]):
        cursor = cfg
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor

    def _get_mvnx_name(cfg: dict) -> Optional[str]:
        for path in (("paths", "mvnx_path"), ("meta", "mvnx_path")):
            mvnx_path = _get_by_path(cfg, path)
            if isinstance(mvnx_path, str) and mvnx_path:
                return os.path.splitext(os.path.basename(mvnx_path))[0]
        mvnx_name = _get_by_path(cfg, ("meta", "mvnx_name"))
        if isinstance(mvnx_name, str) and mvnx_name:
            return mvnx_name
        return None

    def _build_hp_str(cfg: dict) -> Optional[str]:
        if not isinstance(cfg, dict):
            return None
        collision = cfg.get("collision", {}) if isinstance(cfg.get("collision"), dict) else {}
        collision_method = collision.get("method")
        num_iters = _get_by_path(cfg, ("conversion", "num_iters"))
        if num_iters is None:
            num_iters = _get_by_path(cfg, ("optimization", "num_iters"))
        collision_weight = _get_by_path(cfg, ("conversion", "loss_weights", "collision"))
        if collision_weight is None:
            collision_weight = _get_by_path(cfg, ("loss_weights", "collision"))
        if not collision_method or num_iters is None or collision_weight is None:
            return None
        try:
            num_iters = int(num_iters)
        except (TypeError, ValueError):
            pass
        try:
            collision_weight = float(collision_weight)
            cw_str = f"{collision_weight:.0e}".replace("+", "")
        except (TypeError, ValueError):
            cw_str = str(collision_weight)
        hp_parts = [
            f"col_{collision_method}",
            f"iter{num_iters}",
            f"cw{cw_str}",
        ]
        if collision.get("use_joint_constraint"):
            hp_parts.append("jc")
        if collision.get("use_staged_repair"):
            hp_parts.append("sr")
        smoother_method = _get_by_path(cfg, ("smoother", "method"))
        if smoother_method is None:
            smoother_method = _get_by_path(cfg, ("temporal_smoother", "method"))
        if smoother_method and smoother_method != "none":
            hp_parts.append(f"sm_{smoother_method}")
        return "_".join(hp_parts)

    def _group_name(pt_paths: list[str], cfg: dict, cfg_hash: str) -> str:
        stems = {os.path.splitext(os.path.basename(p))[0] for p in pt_paths}
        if len(stems) == 1:
            return next(iter(stems))
        hp_str = _build_hp_str(cfg)
        mvnx_name = _get_mvnx_name(cfg)
        if hp_str:
            return f"{mvnx_name}_{hp_str}" if mvnx_name else hp_str
        return f"config_{cfg_hash[:8]}"

    def _write_group_config(output_dir: str, cfg_norm: dict, cfg_path: str) -> None:
        ext = os.path.splitext(cfg_path)[1].lower()
        if ext in (".yaml", ".yml"):
            from omegaconf import OmegaConf
            cfg_yaml = OmegaConf.create(cfg_norm)
            out_name = "config.yaml" if ext == ".yaml" else "config.yml"
            OmegaConf.save(cfg_yaml, os.path.join(output_dir, out_name))
        else:
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(cfg_norm, f, indent=2, ensure_ascii=True)

    files = sorted(glob.glob(os.path.join(input_root, '**', '*.pt'), recursive=True))
    if not files:
        print(f"No .pt files found in {input_root}")
        return

    grouped = {}
    skipped = 0
    for fpath in files:
        config_path = _find_config_path(fpath)
        if not config_path:
            print(f"[WARN] Missing config for {fpath}; skipping.")
            skipped += 1
            continue
        cfg = _load_config(config_path)
        cfg_norm = _normalize_config(cfg)
        cfg_hash = _hash_config(cfg_norm)
        group = grouped.setdefault(cfg_hash, {
            "files": [],
            "config_path": config_path,
            "config_norm": cfg_norm,
            "config_raw": cfg,
        })
        group["files"].append(fpath)

    if not grouped:
        print("No .pt files with matching config files found.")
        return

    os.makedirs(output_zarr_root, exist_ok=True)
    print(f"Found {len(grouped)} config group(s) from {len(files)} .pt files (skipped={skipped}).")

    for idx, (cfg_hash, group) in enumerate(sorted(grouped.items()), start=1):
        pt_paths = group["files"]
        group_name = _group_name(pt_paths, group["config_raw"], cfg_hash)
        if group_name.endswith(".zarr"):
            output_zarr = os.path.join(output_zarr_root, group_name)
        else:
            output_zarr = os.path.join(output_zarr_root, f"{group_name}.zarr")
        print(f"[{idx}/{len(grouped)}] Converting {len(pt_paths)} sequences to {output_zarr}...")
        os.makedirs(output_zarr, exist_ok=True)
        root = _open_zarr_group(output_zarr, mode='w')

        # Zarr v3 uses different codec API
        if ZARR_V3:
            from zarr.codecs import BloscCodec
            compressors = [BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')]
        else:
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

        meta = {}
        for fpath in tqdm(pt_paths, desc=f"Converting {group_name}"):
            seq_name = os.path.basename(os.path.dirname(fpath))
            data = torch.load(fpath, map_location='cpu', weights_only=True)
            seq_group = root.require_group(seq_name)
            for key, tensor in data.items():
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                if not isinstance(arr, np.ndarray):
                    continue
                chunks = (min(arr.shape[0], time_chunk_size),) + arr.shape[1:] if arr.ndim > 0 else arr.shape
                if ZARR_V3:
                    seq_group.create_array(key, data=arr, chunks=chunks, compressors=compressors, overwrite=True)
                else:
                    seq_group.create_dataset(key, data=arr, chunks=chunks, compressor=compressor, overwrite=True)
            if "transl" in data:
                meta[seq_name] = int(data["transl"].shape[0])
            else:
                meta[seq_name] = 0

        with open(os.path.join(output_zarr, "nymeria_meta.json"), 'w') as f:
            json.dump(meta, f, ensure_ascii=True)
        _write_group_config(output_zarr, group["config_norm"], group["config_path"])
        print(f"Done. Converted {len(meta)} sequences for {group_name}.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        convert_pt_to_zarr()
    else:
        from pathlib import Path
        from omegaconf import OmegaConf
        from utils import dist_util
        from utils.registry_builders import build_mask_scheme_registry, build_representation_registry
        
        repo_root = Path(__file__).resolve().parents[1]
        # repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/xyz_only.yaml")
        repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/xyz_rot.yaml")
        mask_cfg = OmegaConf.load(repo_root / "cfg_files/conf/mask_registry/custom_body_parts_xyz.yaml")
        repr_registry = build_representation_registry(OmegaConf.to_container(repr_cfg, resolve=True))
        mask_scheme_registry = build_mask_scheme_registry(OmegaConf.to_container(mask_cfg, resolve=True))

        dataset = DatasetNymeria(
            opt={'root': str(repo_root / "data/nymeria_dataset.zarr"), 'split': 'train'},
            logdir=str(repo_root / "runs" / "nymeria_test"),
            device=dist_util.dev(),
            # device=torch.device('cpu'),
            dtype=torch.float32,
            body_model_path=str(repo_root / "data/body_models/smplx_model"),
            joints_num=22,
            use_ddp=False,
            clip_len=33,
            disable_cache=True,
            # slice_strategy='full_sequence',
            slice_strategy='random_uniform_len',
            num_compute_workers=0,
            norm_spacing=100,
            mean_std=True,
            norm_strategy='meanstd',
            input_noise=False,
            align_to_yaxis=True,
            repr_registry=repr_registry,
            mask_scheme_registry=mask_scheme_registry,
            logging_enabled=True,
        )
        dataset.print_dataset_summary()
        
        # train_dl = DataLoader(dataset, batch_size=384, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        train_dl = DataLoader(dataset, batch_size=384, num_workers=0, drop_last=True)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"motion_repr_clean shape: {sample['motion_repr_clean'].shape}")
        
        for index, batch in tqdm(enumerate(train_dl)):
            pass

        # for sample_idx in range(len(dataset)):
        #     dataset.visualize_sequence(index=sample_idx, num_keyframes=dataset[sample_idx]['motion_repr_clean'].shape[0])
            
            
