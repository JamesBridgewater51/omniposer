"""
Online/streaming statistics accumulator using Welford's algorithm.

This module provides efficient computation of mean, variance, min, max
over large datasets without storing all data in memory.
"""

import torch
import numpy as np
from typing import Dict, Optional, List


class RunningStats:
    """
    Accumulates statistics (mean, std, min, max) incrementally using Welford's algorithm.
    
    This allows computing exact statistics over arbitrarily large datasets
    while only keeping O(dim) memory footprint.
    """
    
    def __init__(self, dim: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float64):
        """
        Initialize running statistics accumulators.
        
        Args:
            dim: Feature dimension
            device: Torch device for computation
            dtype: Use float64 for numerical stability in accumulation
        """
        self.dim = dim
        self.device = device
        self.dtype = dtype
        
        # Welford's algorithm accumulators
        self.n = 0  # Total sample count
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.M2 = torch.zeros(dim, device=device, dtype=dtype)  # Sum of squared deviations
        
        # Min/Max tracking
        self.min_val = torch.full((dim,), float('inf'), device=device, dtype=dtype)
        self.max_val = torch.full((dim,), float('-inf'), device=device, dtype=dtype)
    
    def update(self, batch: torch.Tensor) -> None:
        """
        Update statistics with a new batch of data.
        
        Uses Welford's online algorithm for numerically stable mean/variance computation.
        
        Args:
            batch: Tensor of shape (N, dim) or (T, dim) containing new samples
        """
        if batch.numel() == 0:
            return
            
        # Flatten to (N, dim)
        batch = batch.reshape(-1, self.dim).to(device=self.device, dtype=self.dtype)
        batch_n = batch.shape[0]
        
        # Update min/max
        batch_min = batch.min(dim=0).values
        batch_max = batch.max(dim=0).values
        self.min_val = torch.minimum(self.min_val, batch_min)
        self.max_val = torch.maximum(self.max_val, batch_max)
        
        # Welford's algorithm for batch update
        # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)  # Population variance
        batch_M2 = batch_var * batch_n
        
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            # Parallel/batch Welford update
            n_total = self.n + batch_n
            delta = batch_mean - self.mean
            
            # Update mean
            self.mean = self.mean + delta * (batch_n / n_total)
            
            # Update M2 (sum of squared deviations)
            self.M2 = self.M2 + batch_M2 + delta.pow(2) * (self.n * batch_n / n_total)
            
            self.n = n_total
    
    def update_from_stats(self, mean: np.ndarray, std: np.ndarray, n: int,
                          min_val: Optional[np.ndarray] = None, 
                          max_val: Optional[np.ndarray] = None) -> None:
        """
        Update statistics by merging pre-computed statistics from another source.
        
        Uses Welford's parallel algorithm to combine two sets of statistics.
        
        Args:
            mean: Mean array of shape (dim,)
            std: Standard deviation array of shape (dim,)
            n: Number of samples these statistics were computed from
            min_val: Optional min array of shape (dim,)
            max_val: Optional max array of shape (dim,)
        """
        if n == 0:
            return
        
        batch_mean = torch.from_numpy(mean).to(device=self.device, dtype=self.dtype)
        batch_var = torch.from_numpy(std ** 2).to(device=self.device, dtype=self.dtype)
        batch_M2 = batch_var * n
        
        if min_val is not None:
            batch_min = torch.from_numpy(min_val).to(device=self.device, dtype=self.dtype)
            self.min_val = torch.minimum(self.min_val, batch_min)
        if max_val is not None:
            batch_max = torch.from_numpy(max_val).to(device=self.device, dtype=self.dtype)
            self.max_val = torch.maximum(self.max_val, batch_max)
        
        if self.n == 0:
            self.n = n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            n_total = self.n + n
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (n / n_total)
            self.M2 = self.M2 + batch_M2 + delta.pow(2) * (self.n * n / n_total)
            self.n = n_total
    
    def compute(self) -> Dict[str, np.ndarray]:
        """
        Finalize and return computed statistics.
        
        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' as numpy arrays (float32)
        """
        if self.n == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        
        variance = self.M2 / self.n
        std = torch.sqrt(variance.clamp(min=1e-8))
        
        return {
            'mean': self.mean.cpu().numpy().astype(np.float32),
            'std': std.cpu().numpy().astype(np.float32),
            'min': self.min_val.cpu().numpy().astype(np.float32),
            'max': self.max_val.cpu().numpy().astype(np.float32),
        }


class RunningStatsDict:
    """
    Manages multiple RunningStats instances, one per representation modality.
    """
    
    def __init__(self, repr_dims: Dict[str, int], device: torch.device = torch.device('cpu')):
        """
        Args:
            repr_dims: Dictionary mapping repr_name -> dimension
            device: Torch device for computation
        """
        self.stats = {
            name: RunningStats(dim, device=device)
            for name, dim in repr_dims.items()
        }
    
    def update(self, repr_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update all statistics with a batch of representation data.
        
        Args:
            repr_dict: Dictionary mapping repr_name -> tensor of shape (T, dim)
        """
        for name, tensor in repr_dict.items():
            if name in self.stats and tensor is not None:
                self.stats[name].update(tensor)
    
    def compute(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Finalize and return all computed statistics.
        
        Returns:
            Nested dictionary: repr_name -> {'mean', 'std', 'min', 'max'}
        """
        return {name: stats.compute() for name, stats in self.stats.items()}


def merge_norm_params(
    datasets: List,
    mean_std: bool = True,
    norm_strategy: str = 'meanstd'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Merge normalization parameters from multiple datasets using Welford's parallel algorithm.
    
    Each dataset must have:
    - norm_params: Dict[repr_name, {'mean', 'std', ...}]
    - A way to get frame count (via _get_total_frames() or len())
    
    Args:
        datasets: List of dataset objects with norm_params attribute
        mean_std: Whether to apply special handling for certain modalities
        norm_strategy: Normalization strategy (currently only 'meanstd' supported for merging)
        
    Returns:
        Merged norm_params dict
    """
    if not datasets:
        raise ValueError("No datasets provided for merging norm_params")
    
    # Get repr list from first dataset
    repr_list = datasets[0]._repr_list
    repr_dims = {m.name: m.dim for m in datasets[0].repr_registry.modalities}
    
    # Initialize running stats for each repr
    merged_stats = {name: RunningStats(dim) for name, dim in repr_dims.items()}
    
    for dataset in datasets:
        if not hasattr(dataset, 'norm_params') or dataset.norm_params is None:
            raise ValueError(f"Dataset {dataset} has no norm_params")
        
        # Get frame count for this dataset
        n_frames = _get_dataset_frame_count(dataset)
        if n_frames == 0:
            raise ValueError(f"Dataset {dataset} has no frames")
        
        # Merge each repr's statistics
        for repr_name in repr_list:
            if repr_name not in dataset.norm_params:
                raise ValueError(f"Dataset {dataset} has no norm_params for repr {repr_name}")
            
            params = dataset.norm_params[repr_name]
            mean = params.get('mean')
            std = params.get('std')
            
            if mean is None or std is None:
                raise ValueError(f"Dataset {dataset} has no mean or std for repr {repr_name}")
            
            min_val = params.get('min')
            max_val = params.get('max')
            
            merged_stats[repr_name].update_from_stats(
                mean=mean, std=std, n=n_frames,
                min_val=min_val, max_val=max_val
            )
    
    # Compute final merged statistics
    merged_norm_params = {}
    for repr_name in repr_list:
        if merged_stats[repr_name].n > 0:
            merged_norm_params[repr_name] = merged_stats[repr_name].compute()
        else:
            # Fallback for missing reprs
            dim = repr_dims.get(repr_name, 1)
            merged_norm_params[repr_name] = {
                'mean': np.zeros(dim, dtype=np.float32),
                'std': np.ones(dim, dtype=np.float32),
            }
    
    # Apply special handling for certain modalities
    if mean_std and norm_strategy == 'meanstd':
        ignore_norm_keys = ['smplx_betas', 'foot_contact', 'smplx_rot_6d', 'smplx_body_pose_6d']
        for repr_name in ignore_norm_keys:
            if repr_name in merged_norm_params:
                merged_norm_params[repr_name]['mean'] = np.zeros_like(merged_norm_params[repr_name]['mean'])
                merged_norm_params[repr_name]['std'] = np.ones_like(merged_norm_params[repr_name]['std'])
    
    return merged_norm_params


def _get_dataset_frame_count(dataset) -> int:
    """
    Get total frame count from a dataset for weighted statistics merging.
    """
    # Try different methods to get frame count
    if hasattr(dataset, '_get_total_frames'):
        return dataset._get_total_frames()
    
    # For DatasetAMASSLike: sum of sequence lengths
    if hasattr(dataset, 'joints_clean_list'):
        return sum(len(clip) for clip in dataset.joints_clean_list)
    
    # For DatasetMocap: sum from data['pose']
    if hasattr(dataset, 'data') and dataset.data is not None and 'pose' in dataset.data:
        return sum(len(seq) for seq in dataset.data['pose'])
    
    # Fallback: use dataset length * estimated clip length
    if hasattr(dataset, 'clip_len'):
        return len(dataset) * dataset.clip_len
    
    raise ValueError(f"Dataset {dataset} has no frame count statistics provided.")
