from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_sets.motion_representation import *
import pickle as pkl
import torch.distributed as dist
from typing import Literal, Iterator, Dict, Generator
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from utils.representation_registry import RepresentationRegistry
from .running_stats import RunningStatsDict

class DatasetBase(data.Dataset):
    def __init__(self,
                logdir=None,
                device='cpu',
                dtype=torch.float32,
                body_model_path=None,
                disable_cache=False,
                use_ddp=False,
                dist_rank=0,
                joints_num=22,
                head_joint_idx=[15],
                torso_joint_idx=[12, 9, 6, 3],
                logging_enabled=False,
                mean_std=True,
                norm_strategy='meanstd',
                clip_percentiles=(1.0, 99.0),
                repr_registry: RepresentationRegistry = None,
                cano_jts_pos: bool = False,
                *args, **kwargs):
        self.logdir = logdir
        self.device = device
        self.dtype = dtype
        self.smplx_neutral = smplx.create(model_path=body_model_path, model_type="smplx",
                        gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)
        self.joints_num = joints_num
        self.head_joint_idx = head_joint_idx
        self.torso_joint_idx = torso_joint_idx
        self.logging_enabled = logging_enabled
        self.mean_std = mean_std
        self.norm_strategy = norm_strategy
        self.clip_percentiles = clip_percentiles
        self.disable_cache = disable_cache
        self.use_ddp = use_ddp
        self.dist_rank = dist_rank
        self.cano_jts_pos = cano_jts_pos

        ########################################## configs for motion representation
        if repr_registry is None:
            raise ValueError("repr_registry must be provided explicitly; defaults are not allowed.")
        self.repr_registry = repr_registry
        # Required modality groupings
        if not self.repr_registry.traj_modalities:
            raise ValueError("repr_registry.traj_modalities must be specified.")
        if 'traj' not in self.repr_registry.cond_modalities or 'pose' not in self.repr_registry.cond_modalities:
            raise ValueError("repr_registry.cond_modalities must include keys 'traj' and 'pose'.")
        if 'traj' not in self.repr_registry.predicted_modalities or 'pose' not in self.repr_registry.predicted_modalities:
            raise ValueError("repr_registry.predicted_modalities must include keys 'traj' and 'pose'.")

        self.traj_repr_name_list = list(self.repr_registry.traj_modalities)
        # Cond reprs for traj and pose tasks (can be different)
        self.cond_repr_name_list_traj = list(self.repr_registry.cond_modalities.get('traj', []))
        self.cond_repr_name_list_pose = list(self.repr_registry.cond_modalities.get('pose', []))
        # Local/pose reprs are all modalities not in traj list (order preserved)
        self.local_repr_name_list = [m.name for m in self.repr_registry.modalities if m.name not in self.traj_repr_name_list]

        self._repr_list = [m.name for m in self.repr_registry.modalities]

        ## get dimensions for features
        self.body_feat_dim = 0
        self.traj_feat_dim = 0
        self.pose_feat_dim = 0
        self.cond_feat_dim_traj = 0
        self.cond_feat_dim_pose = 0

        # Store start and end indices for quick lookup in __getitem__
        self._repr_indices = {}
        current_dim = 0
        for m in self.repr_registry.modalities:
            repr_name = m.name
            dim = m.dim
            self._repr_indices[repr_name] = (current_dim, current_dim + dim)
            current_dim += dim
            if repr_name in self.traj_repr_name_list:
                self.traj_feat_dim += dim
            if repr_name in self.local_repr_name_list:
                self.pose_feat_dim += dim
            if repr_name in self.cond_repr_name_list_traj:
                self.cond_feat_dim_traj += dim
            if repr_name in self.cond_repr_name_list_pose:
                self.cond_feat_dim_pose += dim
            self.body_feat_dim += dim

        # Optionally: reprs we explicitly DON'T want to normalize (keeps previous behavior)
        self._normalize_mask = {r: True for r in self._repr_list}
        # Keep foot_contact untouched unless caller changes normalize mask explicitly
        if 'foot_contact' in self._normalize_mask:
            self._normalize_mask['foot_contact'] = False

        self.norm_params = {}

        # Mask scheme registry must be provided by caller/config; fail if absent
        self.mask_scheme_registry = kwargs.get('mask_scheme_registry', None)
        if self.mask_scheme_registry is None:
            raise ValueError("mask_scheme_registry must be provided explicitly.")

    def get_traj_features(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Extract trajectory features from a full feature tensor.

        Args:
            tensor: Full feature tensor containing all modalities
            dim: Dimension along which features are organized (default: -1)

        Returns:
            Tensor containing only trajectory features
        """
        from utils.feature_accessor import select_modalities
        return select_modalities(tensor, self.traj_repr_name_list, self._repr_indices, dim=dim)

    def get_pose_features(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Extract pose features from a full feature tensor.

        Args:
            tensor: Full feature tensor containing all modalities
            dim: Dimension along which features are organized (default: -1)

        Returns:
            Tensor containing only pose features (local_repr_name_list)
        """
        from utils.feature_accessor import select_modalities
        return select_modalities(tensor, self.local_repr_name_list, self._repr_indices, dim=dim)

    def get_cond_features(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Extract conditioning features from a full feature tensor.
        Uses traj conditioning modalities by default (for backward compatibility).

        Args:
            tensor: Full feature tensor containing all modalities
            dim: Dimension along which features are organized (default: -1)

        Returns:
            Tensor containing only conditioning features
        """
        from utils.feature_accessor import select_modalities
        return select_modalities(tensor, self.cond_repr_name_list, self._repr_indices, dim=dim)

    def get_cond_features_for_task(self, tensor: torch.Tensor, task: str, dim: int = -1) -> torch.Tensor:
        """
        Extract task-specific conditioning features from a full feature tensor.

        Args:
            tensor: Full feature tensor containing all modalities
            task: Task type ('traj' or 'pose')
            dim: Dimension along which features are organized (default: -1)

        Returns:
            Tensor containing only task-specific conditioning features

        Raises:
            ValueError: If task is not 'traj' or 'pose'
        """
        from utils.feature_accessor import select_modalities
        if task == 'traj':
            return select_modalities(tensor, self.cond_repr_name_list_traj, self._repr_indices, dim=dim)
        elif task == 'pose':
            return select_modalities(tensor, self.cond_repr_name_list_pose, self._repr_indices, dim=dim)
        else:
            raise ValueError(f"task must be 'traj' or 'pose', got '{task}'")

    def get_cond_feat_dim_for_task(self, task: str) -> int:
        """
        Get conditioning feature dimension for a specific task.

        Args:
            task: Task type ('traj' or 'pose')

        Returns:
            Conditioning feature dimension for the specified task

        Raises:
            ValueError: If task is not 'traj' or 'pose'
        """
        if task == 'traj':
            return self.cond_feat_dim_traj
        elif task == 'pose':
            return self.cond_feat_dim_pose
        else:
            raise ValueError(f"task must be 'traj' or 'pose', got '{task}'")

    def get_modality_norm_stats(self, modality_names: list) -> dict:
        """
        Get concatenated normalization statistics for specific modalities.

        Args:
            modality_names: List of modality names to get stats for

        Returns:
            Dictionary with 'mean' and 'std' keys (or other stats depending on norm_strategy),
            where values are concatenated numpy arrays for the specified modalities
        """
        if not self.norm_params:
            raise RuntimeError("Normalization parameters not computed yet. Call compute/load norm params first.")

        # Determine which stats keys are available based on norm_strategy
        if self.norm_strategy == 'meanstd':
            stat_keys = ['mean', 'std']
        elif self.norm_strategy == 'minmax':
            stat_keys = ['min', 'max']
        elif self.norm_strategy == 'maxabs':
            stat_keys = ['maxabs']
        elif self.norm_strategy == 'meannorm':
            stat_keys = ['mean', 'min', 'max']
        elif self.norm_strategy == 'robust':
            stat_keys = ['median', 'iqr']
        else:
            # For other strategies, return all available keys
            first_modality = list(self.norm_params.keys())[0]
            stat_keys = list(self.norm_params[first_modality].keys())

        result = {}
        for key in stat_keys:
            parts = []
            for modality_name in modality_names:
                if modality_name not in self.norm_params:
                    raise KeyError(f"Modality '{modality_name}' not found in norm_params")
                if key not in self.norm_params[modality_name]:
                    continue  # Skip if this stat doesn't exist for this modality
                parts.append(self.norm_params[modality_name][key])
            if parts:
                result[key] = np.concatenate(parts, axis=0)

        return result

    def _normalize_full_tensor(self, full_tensor: torch.Tensor, device, dtype=torch.float32,
                            store_sample_norms=False):
        """
        full_tensor: (..., total_dim) torch tensor (float)
        returns normalized tensor (same shape).
        If store_sample_norms=True and strategy == 'l2': returns (normalized, norms) so you can invert later.
        """
        # Clone to avoid in-place operations on tensors that might be part of the computation graph
        out = full_tensor.to(device=device, dtype=dtype).clone()
        sample_norms = None
        _EPS = 1e-8

        for repr_name in self._repr_list:
            s = self._repr_indices[repr_name]
            s = slice(s[0], s[1])
            if not self._normalize_mask[repr_name]:
                continue
            P = self.norm_params[repr_name]

            if self.norm_strategy == 'maxabs':
                maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] / (maxabs + _EPS)

            elif self.norm_strategy == 'meannorm':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                denom = (max_t - min_t)
                denom[denom.abs() < _EPS] = 1.0
                out[..., s] = ((out[..., s] - mean_t) * 2.0) / denom

            elif self.norm_strategy == 'robust':
                median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype)
                iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype)
                iqr_t[iqr_t.abs() < _EPS] = 1.0
                out[..., s] = (out[..., s] - median_t) / iqr_t

            elif self.norm_strategy == 'minmax':
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                out[..., s] = ((out[..., s] * 2.0) - (min_t + max_t)) / (max_t - min_t + _EPS)

            elif self.norm_strategy == 'meanstd':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype)
                std_t[std_t.abs() < _EPS] = 1.0
                out[..., s] = (out[..., s] - mean_t) / std_t

            elif self.norm_strategy in ('quantile_uniform', 'quantile_normal'):
                # runtime quantile transform: we interpolate using numpy for this batch (works but may be slower).
                # If you have sklearn available you can use QuantileTransformer.transform on numpy arrays (faster).
                q_levels = P['q_levels'] / 100.0  # [0..1]
                q_vals = P['q_values']  # (nq, dim)
                q_eps = 1e-6
                # convert this repr to numpy for interpolation (small batch cost)
                orig = out[..., s].cpu().numpy()  # shape (..., dim)
                shape = orig.shape
                flat = orig.reshape(-1, shape[-1])  # (B, dim)
                B = flat.shape[0]
                mapped = np.empty_like(flat, dtype=np.float32)

                for j in range(flat.shape[1]):
                    # q_vals[:, j] monotonic increasing
                    kq = q_vals[:, j]
                    # use np.interp: map x->p in [0,1]
                    # clamp outside range using q_eps / (1-q_eps)
                    p = np.interp(flat[:, j], kq, q_levels, left=0.0 + q_eps, right=1.0 - q_eps)
                    if self.norm_strategy == 'quantile_uniform':
                        mapped[:, j] = p.astype(np.float32)
                    else:  # quantile_normal
                        # map p->normal quantile. uses scipy 
                        from scipy.stats import norm as _norm
                        mapped[:, j] = _norm.ppf(p).astype(np.float32)

                out_np = mapped.reshape(shape)
                out[..., s] = torch.from_numpy(out_np).to(device=device, dtype=dtype)

            elif self.norm_strategy == 'boxcox':
                # apply per-feature boxcox forward (numpy-based)
                lam = P['lambda']
                shift = P['shift']
                orig = out[..., s].cpu().numpy().astype(np.float32)
                # forward elementwise
                res = np.empty_like(orig)
                for j in range(orig.shape[-1]):
                    x = orig[..., j] + shift[j]
                    lj = float(lam[j])
                    if abs(lj) < 1e-12:
                        res[..., j] = np.log(x)
                    else:
                        res[..., j] = (np.power(x, lj) - 1.0) / lj
                out[..., s] = torch.from_numpy(res).to(device=device, dtype=dtype)

            elif self.norm_strategy == 'yeojohnson':
                lam = P['lambda']
                orig = out[..., s].cpu().numpy().astype(np.float32)
                res = np.empty_like(orig)
                for j in range(orig.shape[-1]):
                    x = orig[..., j]
                    lj = float(lam[j])
                    # forward yj:
                    # for x >= 0:
                    #   if lambda == 0: y = log(x+1)
                    #   else: y = ((x+1)**lambda -1)/lambda
                    # for x < 0:
                    #   if lambda == 2: y = -log(-x+1)
                    #   else: y = - ( (-x+1) ** (2-lambda) -1 ) / (2-lambda)
                    xv = x
                    if np.isscalar(xv):
                        xv = np.array([xv])
                    # vectorized:
                    pos = xv >= 0
                    neg = ~pos
                    y = np.zeros_like(xv)
                    if abs(lj) < 1e-12:
                        # lambda == 0
                        y[pos] = np.log(xv[pos] + 1.0)
                    else:
                        y[pos] = (np.power(xv[pos] + 1.0, lj) - 1.0) / lj
                    if abs(lj - 2.0) < 1e-12:
                        y[neg] = -np.log(-xv[neg] + 1.0)
                    else:
                        y[neg] = - (np.power(-xv[neg] + 1.0, 2.0 - lj) - 1.0) / (2.0 - lj)
                    # NOTE: yeojohnson can produce NaNs for a lot of reasons (large y, large |lambda|, etc), don't use it currently.
                    if np.isnan(y).any():
                        breakpoint()
                    res[..., j] = y
                out[..., s] = torch.from_numpy(res).to(device=device, dtype=dtype)

            elif self.norm_strategy in ('pca_whiten', 'zca_whiten'):
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                W = torch.from_numpy(P['whiten_W']).to(device=device, dtype=dtype)  # (dim, dim)
                # subtract mean and multiply by W^T (so that x_centered @ W.T)
                xc = out[..., s] - mean_t
                # flatten last dim for matmul:
                orig_shape = xc.shape
                xc_flat = xc.reshape(-1, orig_shape[-1])
                whitened = torch.matmul(xc_flat, W.T)
                out[..., s] = whitened.reshape(orig_shape)

            elif self.norm_strategy == 'l2':
                # normalize per-sample vectors to unit L2
                vec = out[..., s]
                norm = vec.norm(dim=-1, keepdim=True).clamp_min(_EPS)
                normalized = vec / norm
                if store_sample_norms:
                    # we return norms too
                    if sample_norms is None:
                        sample_norms = {}
                    sample_norms[repr_name] = norm  # keep for inverse
                out[..., s] = normalized

            elif self.norm_strategy in ('clip_percentile', 'winsorize'):
                lo = torch.from_numpy(P['clip_low']).to(device=device, dtype=dtype)
                hi = torch.from_numpy(P['clip_high']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s].clamp(lo, hi)

            else:
                # fallback: no-op
                pass

        if store_sample_norms:
            return out, sample_norms
        return out


    def _unnormalize_full_tensor(self, 
                full_tensor: torch.Tensor, 
                device, 
                dtype=torch.float32, 
                sample_norms=None):
        """
        Inverse of _normalize_full_tensor for invertible transforms.
        For non-invertible transforms (clip, winsorize) we raise.
        For l2, sample_norms must be provided to recover original magnitudes.
        """
        # Clone to avoid in-place operations on tensors that might be part of the computation graph
        out = full_tensor.to(device=device, dtype=dtype).clone()
        _EPS=1e-8

        for repr_name in self._repr_list:
            s = self._repr_indices[repr_name]
            s = slice(s[0], s[1])
            if not self._normalize_mask.get(repr_name, True):
                continue
            P = self.norm_params[repr_name]

            if self.norm_strategy == 'maxabs':
                maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * maxabs

            elif self.norm_strategy == 'meannorm':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * ((max_t - min_t) / 2.0) + mean_t

            elif self.norm_strategy == 'robust':
                median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype)
                iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype)
                iqr_t[iqr_t.abs() < _EPS] = 1.0
                out[..., s] = out[..., s] * iqr_t + median_t

            elif self.norm_strategy == 'minmax':
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                out[..., s] = ((max_t - min_t) * out[..., s] + (min_t + max_t)) / 2.0

            elif self.norm_strategy == 'meanstd':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * std_t + mean_t

            elif self.norm_strategy in ('quantile_uniform', 'quantile_normal'):
                q_levels = P['q_levels'] / 100.0
                q_vals = P['q_values']  # (nq, dim)
                shape = out[..., s].shape
                flat = out[..., s].cpu().numpy().reshape(-1, shape[-1])
                inv = np.empty_like(flat, dtype=np.float32)
                for j in range(flat.shape[1]):
                    if self.norm_strategy == 'quantile_uniform':
                        p = flat[:, j]
                    else:
                        # quantile_normal: map normal back to p using cdf
                        from scipy.stats import norm as _norm
                        p = _norm.cdf(flat[:, j])

                    # now p in (0,1) -> interpolate p -> original value using stored q_vals
                    qx = q_vals[:, j]
                    inv[:, j] = np.interp(p, q_levels, qx).astype(np.float32)
                out_np = inv.reshape(shape)
                out[..., s] = torch.from_numpy(out_np).to(device=device, dtype=dtype)

            elif self.norm_strategy == 'boxcox':
                lam = P['lambda']
                shift = P['shift']
                orig = out[..., s].cpu().numpy().astype(np.float32)
                res = np.empty_like(orig)
                for j in range(orig.shape[-1]):
                    y = orig[..., j]
                    lj = float(lam[j])
                    if abs(lj) < 1e-12:
                        res[..., j] = np.exp(y) - shift[j]
                    else:
                        res[..., j] = np.power(lj * y + 1.0, 1.0 / lj) - shift[j]
                out[..., s] = torch.from_numpy(res).to(device=device, dtype=dtype)

            elif self.norm_strategy == 'yeojohnson':
                lam = P['lambda']
                orig = out[..., s].cpu().numpy().astype(np.float32)
                res = np.empty_like(orig)
                for j in range(orig.shape[-1]):
                    y = orig[..., j]
                    lj = float(lam[j])
                    # inverse Yeo-Johnson:
                    # if y >= 0:
                    #   if lambda == 0: x = exp(y)-1
                    #   else: x = (lambda*y + 1) ** (1/lambda) - 1
                    # if y < 0:
                    #   if lambda == 2: x = - (exp(-y) - 1)
                    #   else: x = - ( (-(2-lambda)*y + 1) ** (1/(2-lambda)) - 1 )
                    yv = y
                    pos = yv >= 0
                    neg = ~pos
                    x = np.zeros_like(yv)
                    if abs(lj) < 1e-12:
                        x[pos] = np.exp(yv[pos]) - 1.0
                    else:
                        x[pos] = np.power(lj * yv[pos] + 1.0, 1.0 / lj) - 1.0
                    if abs(lj - 2.0) < 1e-12:
                        x[neg] = - (np.exp(-yv[neg]) - 1.0)
                    else:
                        x[neg] = - (np.power((-(2.0 - lj) * yv[neg] + 1.0), 1.0 / (2.0 - lj)) - 1.0)
                    # NOTE: yeojohnson can produce NaNs for a lot of reasons (large y, large |lambda|, etc), don't use it currently
                    if np.isnan(x).any():
                        breakpoint()
                    res[..., j] = x
                out[..., s] = torch.from_numpy(res).to(device=device, dtype=dtype)

            elif self.norm_strategy in ('pca_whiten', 'zca_whiten'):
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                W_inv = torch.from_numpy(P['whiten_inv']).to(device=device, dtype=dtype)
                xc = out[..., s]
                orig_shape = xc.shape
                xc_flat = xc.reshape(-1, orig_shape[-1])
                dewhitened = torch.matmul(xc_flat, W_inv.T)
                out[..., s] = (dewhitened.reshape(orig_shape) + mean_t)

            elif self.norm_strategy == 'l2':
                # to invert l2 normalization you must provide original norms per sample
                if sample_norms is None or repr_name not in sample_norms:
                    raise RuntimeError("Cannot unnormalize 'l2' without sample_norms. Supply norms from normalization stage.")
                norms = sample_norms[repr_name]  # tensor same leading dims with shape (..., 1)
                out[..., s] = out[..., s] * norms

            elif self.norm_strategy in ('clip_percentile', 'winsorize'):
                raise RuntimeError("clip_percentile/winsorize are not invertible")

            else:
                # no-op
                pass

        return out

    def _get_cond_layout(self, task: str):
        """
        Get layout information for the cond tensor.
        
        Args:
            task: 'traj' or 'pose'
            
        Returns:
            Tuple of (joint_modalities, global_modalities, num_joints, per_joint_total, joint_dim)
        """
        registry = self.repr_registry
        cond_names = registry.cond_modalities.get(task, [])
        
        joint_modalities = []
        global_modalities = []
        for name in cond_names:
            mod = registry.get_modality(name)
            if mod.is_joint_based:
                joint_modalities.append(mod)
            else:
                global_modalities.append(mod)
        
        num_joints = 0
        if joint_modalities:
            num_joints = registry.joint_count_for_topology(joint_modalities[0].topology) or 0
        
        per_joint_total = sum(m.per_joint_dim or 0 for m in joint_modalities)
        joint_dim = num_joints * per_joint_total
        
        return joint_modalities, global_modalities, num_joints, per_joint_total, joint_dim

    def _normalize_cond_tensor(
        self,
        cond_tensor: torch.Tensor,
        task: str,
        device,
        dtype=torch.float32,
        store_sample_norms: bool = False,
    ):
        """
        Normalize conditioning tensor using same strategy as motion_repr.
        
        Cond tensor layout: [joint-interleaved (num_joints * per_joint_total)] | [global modalities]
        
        Args:
            cond_tensor: Conditioning tensor [bs, T, cond_dim]
            task: 'traj' or 'pose'
            device: Target device
            dtype: Target dtype
            store_sample_norms: If True and strategy is 'l2', returns (normalized, sample_norms)
            
        Returns:
            Normalized tensor, or (normalized, sample_norms) if store_sample_norms=True and l2
        """
        out = cond_tensor.to(device=device, dtype=dtype).clone()
        sample_norms = {} if store_sample_norms else None
        _EPS = 1e-8
        
        joint_mods, global_mods, num_joints, per_joint_total, joint_dim = self._get_cond_layout(task)
        
        # === Joint-interleaved section ===
        if joint_dim > 0 and per_joint_total > 0:
            # Shape: [bs, T, num_joints, per_joint_total]
            joint_section = out[..., :joint_dim].reshape(*out.shape[:-1], num_joints, per_joint_total)
            
            cur = 0
            for mod in joint_mods:
                pj_dim = mod.per_joint_dim or 0
                if pj_dim == 0 or mod.name not in self.norm_params:
                    cur += pj_dim
                    continue
                    
                P = self.norm_params[mod.name]
                
                if self.norm_strategy == 'meanstd':
                    mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                    std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype)
                    std_t = std_t.clamp(min=_EPS)
                    # Reshape to [num_joints, pj_dim] for broadcasting
                    mean_2d = mean_t.reshape(num_joints, pj_dim)
                    std_2d = std_t.reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = (
                        joint_section[..., cur:cur+pj_dim] - mean_2d
                    ) / std_2d
                    
                elif self.norm_strategy == 'l2':
                    # L2 normalize per sample, flattening all joints for this modality
                    orig_shape = joint_section[..., cur:cur+pj_dim].shape
                    vec = joint_section[..., cur:cur+pj_dim].reshape(*out.shape[:-1], -1)
                    norm = vec.norm(dim=-1, keepdim=True).clamp(min=_EPS)
                    normalized = vec / norm
                    joint_section[..., cur:cur+pj_dim] = normalized.reshape(orig_shape)
                    if store_sample_norms:
                        sample_norms[f'cond_joint_{mod.name}'] = norm
                        
                elif self.norm_strategy == 'maxabs':
                    maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype).clamp(min=_EPS)
                    maxabs_2d = maxabs.reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = joint_section[..., cur:cur+pj_dim] / maxabs_2d
                    
                elif self.norm_strategy == 'minmax':
                    min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    denom = (max_t - min_t).clamp(min=_EPS)
                    joint_section[..., cur:cur+pj_dim] = (
                        (joint_section[..., cur:cur+pj_dim] * 2.0) - (min_t + max_t)
                    ) / denom
                    
                elif self.norm_strategy == 'robust':
                    median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    iqr_t = iqr_t.clamp(min=_EPS)
                    joint_section[..., cur:cur+pj_dim] = (
                        joint_section[..., cur:cur+pj_dim] - median_t
                    ) / iqr_t
                    
                # Other strategies: no-op for now (can extend as needed)
                
                cur += pj_dim
            
            # NOTE: use `.clone()` since otherwise this error would throw: `RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.  `
            out[..., :joint_dim] = joint_section.reshape(*out.shape[:-1], joint_dim).clone()
        
        # === Global section ===
        global_offset = joint_dim
        for mod in global_mods:
            if mod.name not in self.norm_params:
                global_offset += mod.dim
                continue
                
            P = self.norm_params[mod.name]
            s = slice(global_offset, global_offset + mod.dim)
            
            if self.norm_strategy == 'meanstd':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype).clamp(min=_EPS)
                out[..., s] = (out[..., s] - mean_t) / std_t
                
            elif self.norm_strategy == 'l2':
                vec = out[..., s]
                norm = vec.norm(dim=-1, keepdim=True).clamp(min=_EPS)
                out[..., s] = vec / norm
                if store_sample_norms:
                    sample_norms[f'cond_global_{mod.name}'] = norm
                    
            elif self.norm_strategy == 'maxabs':
                maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype).clamp(min=_EPS)
                out[..., s] = out[..., s] / maxabs
                
            elif self.norm_strategy == 'minmax':
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                denom = (max_t - min_t).clamp(min=_EPS)
                out[..., s] = ((out[..., s] * 2.0) - (min_t + max_t)) / denom
                
            elif self.norm_strategy == 'robust':
                median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype)
                iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype).clamp(min=_EPS)
                out[..., s] = (out[..., s] - median_t) / iqr_t
            
            global_offset += mod.dim
        
        if store_sample_norms and self.norm_strategy == 'l2':
            return out, sample_norms
        return out

    def _unnormalize_cond_tensor(
        self,
        cond_tensor: torch.Tensor,
        task: str,
        device,
        dtype=torch.float32,
        sample_norms=None,
    ):
        """
        Inverse of _normalize_cond_tensor.
        
        Args:
            cond_tensor: Normalized conditioning tensor [bs, T, cond_dim]
            task: 'traj' or 'pose'
            device: Target device
            dtype: Target dtype
            sample_norms: Required for 'l2' strategy to recover original magnitudes
            
        Returns:
            Unnormalized tensor
        """
        out = cond_tensor.to(device=device, dtype=dtype).clone()
        _EPS = 1e-8
        
        joint_mods, global_mods, num_joints, per_joint_total, joint_dim = self._get_cond_layout(task)
        
        # === Joint-interleaved section ===
        if joint_dim > 0 and per_joint_total > 0:
            joint_section = out[..., :joint_dim].reshape(*out.shape[:-1], num_joints, per_joint_total)
            
            cur = 0
            for mod in joint_mods:
                pj_dim = mod.per_joint_dim or 0
                if pj_dim == 0 or mod.name not in self.norm_params:
                    cur += pj_dim
                    continue
                    
                P = self.norm_params[mod.name]
                
                if self.norm_strategy == 'meanstd':
                    mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                    std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype)
                    mean_2d = mean_t.reshape(num_joints, pj_dim)
                    std_2d = std_t.reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = (
                        joint_section[..., cur:cur+pj_dim] * std_2d + mean_2d
                    )
                    
                elif self.norm_strategy == 'l2':
                    norm_key = f'cond_joint_{mod.name}'
                    if sample_norms is None or norm_key not in sample_norms:
                        raise RuntimeError(f"Cannot unnormalize l2-normalized cond without sample_norms['{norm_key}']")
                    orig_shape = joint_section[..., cur:cur+pj_dim].shape
                    vec = joint_section[..., cur:cur+pj_dim].reshape(*out.shape[:-1], -1)
                    joint_section[..., cur:cur+pj_dim] = (vec * sample_norms[norm_key]).reshape(orig_shape)
                    
                elif self.norm_strategy == 'maxabs':
                    maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype)
                    maxabs_2d = maxabs.reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = joint_section[..., cur:cur+pj_dim] * maxabs_2d
                    
                elif self.norm_strategy == 'minmax':
                    min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = (
                        (max_t - min_t) * joint_section[..., cur:cur+pj_dim] + (min_t + max_t)
                    ) / 2.0
                    
                elif self.norm_strategy == 'robust':
                    median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype).reshape(num_joints, pj_dim)
                    joint_section[..., cur:cur+pj_dim] = (
                        joint_section[..., cur:cur+pj_dim] * iqr_t + median_t
                    )
                
                cur += pj_dim
            
            out[..., :joint_dim] = joint_section.reshape(*out.shape[:-1], joint_dim)
        
        # === Global section ===
        global_offset = joint_dim
        for mod in global_mods:
            if mod.name not in self.norm_params:
                global_offset += mod.dim
                continue
                
            P = self.norm_params[mod.name]
            s = slice(global_offset, global_offset + mod.dim)
            
            if self.norm_strategy == 'meanstd':
                mean_t = torch.from_numpy(P['mean']).to(device=device, dtype=dtype)
                std_t = torch.from_numpy(P['std']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * std_t + mean_t
                
            elif self.norm_strategy == 'l2':
                norm_key = f'cond_global_{mod.name}'
                if sample_norms is None or norm_key not in sample_norms:
                    raise RuntimeError(f"Cannot unnormalize l2-normalized cond without sample_norms['{norm_key}']")
                out[..., s] = out[..., s] * sample_norms[norm_key]
                
            elif self.norm_strategy == 'maxabs':
                maxabs = torch.from_numpy(P['maxabs']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * maxabs
                
            elif self.norm_strategy == 'minmax':
                min_t = torch.from_numpy(P['min']).to(device=device, dtype=dtype)
                max_t = torch.from_numpy(P['max']).to(device=device, dtype=dtype)
                out[..., s] = ((max_t - min_t) * out[..., s] + (min_t + max_t)) / 2.0
                
            elif self.norm_strategy == 'robust':
                median_t = torch.from_numpy(P['median']).to(device=device, dtype=dtype)
                iqr_t = torch.from_numpy(P['iqr']).to(device=device, dtype=dtype)
                out[..., s] = out[..., s] * iqr_t + median_t
            
            global_offset += mod.dim
        
        return out

    def _resolve_norm_params_path(self, provided_path: str, default_filename: str) -> str:
        """
        Decide where to load/save normalization params.
        """
        if provided_path:
            return provided_path
        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            return os.path.join(self.logdir, default_filename)
        return default_filename

    def _save_norm_params_to_path(self, path: str, protocol: int = 2) -> None:
        """
        Persist current normalization parameters to disk.
        """
        if not path:
            return
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'wb') as result_file:
            pkl.dump(self.norm_params, result_file, protocol=protocol)

    def _load_norm_params_from_path(self, path: str):
        """
        Load normalization parameters from disk.
        """
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Normalization params file not found: {path}")
        with open(path, 'rb') as f:
            return pkl.load(f)

    def _flatten_repr_array_for_norm(self, repr_name: str, arr):
        """
        Convert stored repr array/list/object array into a 2D numpy array [N, dim].
        """
        dim = self.repr_registry.get_modality(repr_name).dim
        if isinstance(arr, list):
            # Concatenate arr if it's a list of arrays to avoid object dtype (important for mean/std/etc)
            assert len(arr) > 0, "arr is empty"
            arr = np.concatenate(arr, axis=0)
        if isinstance(arr, np.ndarray):
            if arr.size == 0:
                return np.zeros((0, dim), dtype=np.float32)
            if arr.dtype == object:
                frames = []
                for clip in arr:
                    if clip is None:
                        continue
                    frames.append(clip.reshape(-1, clip.shape[-1]))
                if not frames:
                    return np.zeros((0, dim), dtype=np.float32)
                return np.concatenate(frames, axis=0)
            return arr.reshape(-1, arr.shape[-1])
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            return np.zeros((0, dim), dtype=np.float32)
        return arr_np.reshape(-1, arr_np.shape[-1])

    def _compute_norm_params_from_repr_lists(self, repr_list_dict: dict) -> dict:
        """
        Compute normalization statistics for each representation using stored clips.
        """
        if not repr_list_dict:
            raise ValueError("repr_list_dict is empty; cannot compute normalization parameters.")

        norm_params = {}
        _eps = 1e-8
        for repr_name in self._repr_list:
            if repr_name not in repr_list_dict:
                raise KeyError(f"Missing representation '{repr_name}' in repr_list_dict.")
            arr_flat = self._flatten_repr_array_for_norm(repr_name, repr_list_dict[repr_name])
            if arr_flat.size == 0:
                raise ValueError(f"No data available to compute norm params for '{repr_name}'.")

            P = {}
            if self.norm_strategy == 'meanstd':
                P['mean'] = arr_flat.mean(axis=0).astype(np.float32)
                P['std'] = arr_flat.std(axis=0).astype(np.float32)
            elif self.norm_strategy == 'minmax':
                P['max'] = arr_flat.max(axis=0).astype(np.float32)
                P['min'] = arr_flat.min(axis=0).astype(np.float32)
            elif self.norm_strategy == 'maxabs':
                P['maxabs'] = np.max(np.abs(arr_flat), axis=0).astype(np.float32)
            elif self.norm_strategy == 'meannorm':
                meanv = arr_flat.mean(axis=0).astype(np.float32)
                minv = arr_flat.min(axis=0).astype(np.float32)
                maxv = arr_flat.max(axis=0).astype(np.float32)
                rng = (maxv - minv).astype(np.float32)
                rng[rng < _eps] = 1.0
                P['mean'] = meanv
                P['min'] = minv
                P['max'] = maxv
            elif self.norm_strategy == 'robust':
                med = np.median(arr_flat, axis=0).astype(np.float32)
                q25 = np.percentile(arr_flat, 25, axis=0).astype(np.float32)
                q75 = np.percentile(arr_flat, 75, axis=0).astype(np.float32)
                iqr = (q75 - q25).astype(np.float32)
                iqr[iqr < _eps] = 1.0
                P['median'] = med
                P['iqr'] = iqr
            elif self.norm_strategy in ('quantile_uniform', 'quantile_normal'):
                quantile_n = 100
                nq = min(int(quantile_n), max(16, arr_flat.shape[0]))
                q_levels = np.linspace(0.0, 100.0, nq).astype(np.float32)
                q_vals = np.percentile(arr_flat, q_levels, axis=0).astype(np.float32)
                P['q_levels'] = q_levels
                P['q_values'] = q_vals
            elif self.norm_strategy == 'boxcox':
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='box-cox', standardize=False)
                pt.fit(arr_flat)
                P['lambda'] = pt.lambdas_.astype(np.float32)
                minvals = arr_flat.min(axis=0)
                shifts = np.zeros_like(minvals)
                shifts[minvals <= 0] = (1.0 - minvals[minvals <= 0])
                P['shift'] = shifts.astype(np.float32)
            elif self.norm_strategy == 'yeojohnson':
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                pt.fit(arr_flat)
                lambdas = pt.lambdas_.astype(np.float32)
                lambdas[np.abs(lambdas) >= 2.0] = 1.0
                P['lambda'] = lambdas
            elif self.norm_strategy in ('pca_whiten', 'zca_whiten'):
                whiten_reg = 1e-6
                X = arr_flat - arr_flat.mean(axis=0, keepdims=True)
                cov = np.cov(X, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + whiten_reg))
                W = (eigvecs @ D_inv_sqrt) @ eigvecs.T
                P['mean'] = arr_flat.mean(axis=0).astype(np.float32)
                P['whiten_W'] = W.astype(np.float32)
                P['whiten_inv'] = np.linalg.pinv(W).astype(np.float32)
            elif self.norm_strategy == 'l2':
                # L2 does not require dataset-wide statistics.
                P['mean'] = np.zeros(arr_flat.shape[-1], dtype=np.float32)
            elif self.norm_strategy in ('clip_percentile', 'winsorize'):
                lower, upper = self.clip_percentiles
                lo = np.percentile(arr_flat, lower, axis=0)
                hi = np.percentile(arr_flat, upper, axis=0)
                P['clip_low'] = lo.astype(np.float32)
                P['clip_high'] = hi.astype(np.float32)
            else:
                # Default fallback (no-op stats)
                P['mean'] = arr_flat.mean(axis=0).astype(np.float32)
                P['std'] = arr_flat.std(axis=0).astype(np.float32)

            if self.mean_std and self.norm_strategy == 'meanstd':
                ignore_norm_keys = ['smplx_betas', 'foot_contact', 'smplx_rot_6d', 'smplx_body_pose_6d']
                if repr_name in ignore_norm_keys:
                    P['mean'] = np.zeros_like(P['mean'])
                    P['std'] = np.ones_like(P['std'])

            norm_params[repr_name] = {k: v.copy() for k, v in P.items()}

        return norm_params

    def _compute_norm_params_streaming(
        self, 
        repr_iterator: Generator[Dict[str, np.ndarray], None, None],
        num_items: int = 0,
        desc: str = "Computing norm params"
    ) -> dict:
        """
        Compute normalization parameters using streaming/online statistics.
        
        This method uses Welford's algorithm to compute mean/std/min/max
        without storing all data in memory. It accepts a generator that
        yields repr_dict for each sequence/clip.
        
        Args:
            repr_iterator: Generator yielding Dict[repr_name, np.ndarray] 
                          where each array has shape (T, dim)
            num_items: Total number of items for progress bar (0 to disable)
            desc: Description for progress bar
            
        Returns:
            norm_params dict compatible with _normalize_full_tensor
        """
        # Initialize running statistics
        repr_dims = {m.name: m.dim for m in self.repr_registry.modalities}
        running_stats = RunningStatsDict(repr_dims, device=torch.device('cpu'))
        
        # Process each item from the iterator
        iterator = tqdm(repr_iterator, total=num_items if num_items > 0 else None, 
                       desc=desc, disable=not self.logging_enabled)
        
        for repr_dict in iterator:
            # Convert numpy arrays to tensors and update stats
            tensor_repr = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in repr_dict.items()
                if k in repr_dims
            }
            running_stats.update(tensor_repr)
        
        # Finalize statistics
        stats_dict = running_stats.compute()
        
        # Convert to norm_params format and apply special handling
        norm_params = {}
        for repr_name in self._repr_list:
            if repr_name in stats_dict:
                norm_params[repr_name] = stats_dict[repr_name]
            else:
                # Fallback for missing reprs
                dim = self.repr_registry.get_modality(repr_name).dim
                norm_params[repr_name] = {
                    'mean': np.zeros(dim, dtype=np.float32),
                    'std': np.ones(dim, dtype=np.float32),
                }
        
        # Apply special handling for certain modalities (same as _compute_norm_params_from_repr_lists)
        if self.mean_std and self.norm_strategy == 'meanstd':
            ignore_norm_keys = ['smplx_betas', 'foot_contact', 'smplx_rot_6d', 'smplx_body_pose_6d']
            for repr_name in ignore_norm_keys:
                if repr_name in norm_params:
                    norm_params[repr_name]['mean'] = np.zeros_like(norm_params[repr_name]['mean'])
                    norm_params[repr_name]['std'] = np.ones_like(norm_params[repr_name]['std'])
        
        return norm_params
