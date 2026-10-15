"""
Sliding window utilities for processing long sequences.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configuration for sliding window processing."""
    window_size: int
    overlap_size: int
    
    @property
    def stride(self) -> int:
        return self.window_size - self.overlap_size


class SlidingWindowProcessor:
    """Handles sliding window decomposition and fusion for long sequences."""
    
    def __init__(self, config: WindowConfig, device: torch.device):
        self.config = config
        self.device = device
        self._overlap_weights_cache: Dict[int, torch.Tensor] = {}
    
    def get_overlap_weights(self, seq_len: int) -> torch.Tensor:
        """Get canonical overlap weights for blending windows."""
        if seq_len not in self._overlap_weights_cache:
            weights = np.minimum(
                self.config.overlap_size,
                np.minimum(
                    np.arange(1, max(seq_len, self.config.window_size) + 1),
                    np.arange(1, max(seq_len, self.config.window_size) + 1)[::-1],
                ),
            ) / self.config.overlap_size
            self._overlap_weights_cache[seq_len] = torch.from_numpy(weights).to(self.device).float()
        return self._overlap_weights_cache[seq_len]
    
    def compute_num_windows(self, seq_len: int) -> int:
        """Compute number of windows needed for a sequence."""
        if seq_len > self.config.window_size:
            return (seq_len - self.config.window_size) // self.config.stride + 2
        return 1
    
    def compute_padded_length(self, seq_len: int) -> int:
        """Compute padded sequence length to fit all windows."""
        num_windows = self.compute_num_windows(seq_len)
        return (num_windows - 1) * self.config.stride + self.config.window_size
    
    def split_into_windows(
        self,
        batch: Dict[str, torch.Tensor],
        keys_to_slice: List[str],
        repr_indices: Dict[str, Tuple[int, int]],
        cond_type: str = 'traj',
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split a batch into overlapping windows.
        
        Args:
            batch: Input batch dictionary
            keys_to_slice: Keys to slice along time dimension
            repr_indices: Representation indices for feature extraction
            cond_type: Type of conditioning - 'traj' (66 features: 22*3 joint positions)
                       or 'pose' (75 features: 9 smplx params + 66 joint positions)
            
        Returns:
            List of window dictionaries with root_init_xy info
        """
        seq_len = batch['motion_repr_clean'].shape[1]
        num_windows = self.compute_num_windows(seq_len)
        windows = []
        
        for win_idx in range(num_windows):
            start_t = win_idx * self.config.stride
            end_t = start_t + self.config.window_size
            pad_len = max(0, end_t - seq_len)
            
            win_data = {}
            for key in keys_to_slice:
                if key not in batch:
                    continue
                val = batch[key]
                if isinstance(val, torch.Tensor) and val.dim() > 1 and val.shape[1] >= end_t - pad_len:
                    win_val = val[:, start_t:end_t - pad_len, ...].clone()
                    if pad_len > 0:
                        # Pad by repeating last frame
                        pad_shape = [1] * (win_val.dim() - 2)
                        win_val = torch.cat([win_val, win_val[:, [-1], ...].repeat(1, pad_len, *pad_shape)], dim=1)
                    win_data[key] = win_val
                else:
                    win_data[key] = val
            
            win_data['_window_meta'] = {
                'start_t': start_t,
                'end_t': end_t,
                'pad_len': pad_len,
                'win_idx': win_idx,
            }
            
            windows.append(win_data)
        
        return windows
    
    def fuse_windows(
        self,
        window_outputs: List[torch.Tensor],
        seq_len: int,
        feat_dim: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Fuse overlapping window outputs using weighted blending.
        
        Args:
            window_outputs: List of window output tensors [bs, window_size, feat_dim]
            seq_len: Original sequence length
            feat_dim: Feature dimension
            batch_size: Batch size
            
        Returns:
            Fused output tensor [bs, seq_len, feat_dim]
        """
        padded_len = self.compute_padded_length(seq_len)
        output = torch.zeros([batch_size, padded_len, feat_dim], device=self.device)
        weights = torch.zeros([batch_size, padded_len], device=self.device)
        
        canonical_weights = self.get_overlap_weights(seq_len)[:self.config.window_size, None]
        
        for win_idx, win_output in enumerate(window_outputs):
            start_t = win_idx * self.config.stride
            end_t = start_t + self.config.window_size
            
            output[:, start_t:end_t, :] += win_output * canonical_weights
            weights[:, start_t:end_t] += canonical_weights[..., 0]
        
        # Normalize by weights
        output = output / weights[..., None].clamp(min=1e-8)
        
        # Remove padding
        if padded_len > seq_len:
            output = output[:, :seq_len, :]
        
        return output
