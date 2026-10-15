"""
Normalization utilities for test inference.

Handles the conversion between different feature layouts used in motion_repr vs cond tensors.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass

# Per-joint representation dimensions
JOINT_REPR_DIM_DICT = {
    'local_positions': 3,
    'R_world_joints': 6,
    'R_root_joints': 6,
    'local_vel': 3,
    'R_world_joints_vel': 6,
    'R_root_joints_vel': 6,
}


@dataclass
class NormalizationStats:
    """Container for normalization statistics in different layouts."""
    # Full motion_repr stats (concatenated layout)
    motion_repr_mean: torch.Tensor
    motion_repr_std: torch.Tensor
    
    # Interleaved per-joint stats for traj cond
    traj_cond_mean: torch.Tensor
    traj_cond_std: torch.Tensor
    
    # Interleaved per-joint stats for pose cond (includes traj features prefix)
    pose_cond_mean: torch.Tensor
    pose_cond_std: torch.Tensor


def get_interleaved_stats_for_features(
    global_mean_vec: torch.Tensor,
    global_std_vec: torch.Tensor,
    dataset: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts and interleaves statistics for 'local_positions' and 
    optionally 'R_world_joints' from the global mean/std vectors.
    
    The cond tensor uses an interleaved per-joint layout:
        For each joint: [local_pos(3), R_world(6), R_root(6), ...]
    
    This function reorders the statistics to match that layout.
    
    Args:
        global_mean_vec: Mean vector in motion_repr concatenated layout
        global_std_vec: Std vector in motion_repr concatenated layout
        dataset: Dataset instance with _repr_indices and joints_num
        
    Returns:
        Tuple of (interleaved_mean, interleaved_std)
    """
    joints_num = dataset.joints_num
    
    # Derive flags from dataset's _repr_list (Purpose 2: conditioning tensor structure)
    # This determines what features are present in the conditioning tensor layout
    include_global_joint_rotations = 'R_world_joints' in dataset._repr_list
    repr_abs_only = 'local_vel' not in dataset._repr_list
    
    # Get stats for local_positions
    lp_start, lp_end = dataset._repr_indices['local_positions']
    mean_lp_flat = global_mean_vec[lp_start:lp_end]
    std_lp_flat = global_std_vec[lp_start:lp_end]

    if not repr_abs_only:
        s_lv, e_lv = dataset._repr_indices['local_vel']
        mean_lv_flat = global_mean_vec[s_lv:e_lv]
        std_lv_flat = global_std_vec[s_lv:e_lv]

    if not include_global_joint_rotations:
        if not repr_abs_only:
            return torch.cat([mean_lp_flat, mean_lv_flat], dim=-1), torch.cat([std_lp_flat, std_lv_flat], dim=-1)
        else:
            return mean_lp_flat, std_lp_flat

    # Get stats for R_world_joints
    gjrot_start, gjrot_end = dataset._repr_indices['R_world_joints']
    mean_gjrot_flat = global_mean_vec[gjrot_start:gjrot_end]
    std_gjrot_flat = global_std_vec[gjrot_start:gjrot_end]

    # Get stats for R_root_joints
    ljrot_start, ljrot_end = dataset._repr_indices['R_root_joints']
    mean_ljrot_flat = global_mean_vec[ljrot_start:ljrot_end]
    std_ljrot_flat = global_std_vec[ljrot_start:ljrot_end]

    # Reshape to (joints_num, dim_per_feature)
    mean_lp_reshaped = mean_lp_flat.reshape(joints_num, 3)
    std_lp_reshaped = std_lp_flat.reshape(joints_num, 3)
    mean_gjrot_reshaped = mean_gjrot_flat.reshape(joints_num, 6)
    std_gjrot_reshaped = std_gjrot_flat.reshape(joints_num, 6)
    mean_ljrot_reshaped = mean_ljrot_flat.reshape(joints_num, 6)
    std_ljrot_reshaped = std_ljrot_flat.reshape(joints_num, 6)

    if not repr_abs_only:
        # Include velocity features
        s_lv, e_lv = dataset._repr_indices['local_vel']
        mean_lv_flat = global_mean_vec[s_lv:e_lv]
        std_lv_flat = global_std_vec[s_lv:e_lv]

        s_gjrot_vel, e_gjrot_vel = dataset._repr_indices['R_world_joints_vel']
        mean_gjrot_vel_flat = global_mean_vec[s_gjrot_vel:e_gjrot_vel]
        std_gjrot_vel_flat = global_std_vec[s_gjrot_vel:e_gjrot_vel]

        s_ljrot_vel, e_ljrot_vel = dataset._repr_indices['R_root_joints_vel']
        mean_ljrot_vel_flat = global_mean_vec[s_ljrot_vel:e_ljrot_vel]
        std_ljrot_vel_flat = global_std_vec[s_ljrot_vel:e_ljrot_vel]

        # Reshape velocity tensors
        mean_lv_reshaped = mean_lv_flat.reshape(joints_num, 3)
        std_lv_reshaped = std_lv_flat.reshape(joints_num, 3)
        mean_gjrot_vel_reshaped = mean_gjrot_vel_flat.reshape(joints_num, 6)
        std_gjrot_vel_reshaped = std_gjrot_vel_flat.reshape(joints_num, 6)
        mean_ljrot_vel_reshaped = mean_ljrot_vel_flat.reshape(joints_num, 6)
        std_ljrot_vel_reshaped = std_ljrot_vel_flat.reshape(joints_num, 6)

        # Interleaved layout: [lp(3), gjrot(6), ljrot(6), lv(3), gjrot_vel(6), ljrot_vel(6)] per joint
        interleaved_dim_per_joint = 3 + 6 + 6 + 3 + 6 + 6  # 30
        total_dim = joints_num * interleaved_dim_per_joint
        
        interleaved_mean = torch.empty(total_dim, device=global_mean_vec.device, dtype=global_mean_vec.dtype)
        interleaved_std = torch.empty(total_dim, device=global_std_vec.device, dtype=global_std_vec.dtype)

        for j_idx in range(joints_num):
            offset = j_idx * interleaved_dim_per_joint
            interleaved_mean[offset : offset + 3] = mean_lp_reshaped[j_idx, :]
            interleaved_mean[offset + 3 : offset + 9] = mean_gjrot_reshaped[j_idx, :]
            interleaved_mean[offset + 9 : offset + 15] = mean_ljrot_reshaped[j_idx, :]
            interleaved_mean[offset + 15 : offset + 18] = mean_lv_reshaped[j_idx, :]
            interleaved_mean[offset + 18 : offset + 24] = mean_gjrot_vel_reshaped[j_idx, :]
            interleaved_mean[offset + 24 : offset + 30] = mean_ljrot_vel_reshaped[j_idx, :]

            interleaved_std[offset : offset + 3] = std_lp_reshaped[j_idx, :]
            interleaved_std[offset + 3 : offset + 9] = std_gjrot_reshaped[j_idx, :]
            interleaved_std[offset + 9 : offset + 15] = std_ljrot_reshaped[j_idx, :]
            interleaved_std[offset + 15 : offset + 18] = std_lv_reshaped[j_idx, :]
            interleaved_std[offset + 18 : offset + 24] = std_gjrot_vel_reshaped[j_idx, :]
            interleaved_std[offset + 24 : offset + 30] = std_ljrot_vel_reshaped[j_idx, :]

        return interleaved_mean, interleaved_std
    else:
        # Interleaved layout: [lp(3), gjrot(6), ljrot(6)] per joint
        interleaved_dim_per_joint = 3 + 6 + 6  # 15
        total_dim = joints_num * interleaved_dim_per_joint
        
        interleaved_mean = torch.empty(total_dim, device=global_mean_vec.device, dtype=global_mean_vec.dtype)
        interleaved_std = torch.empty(total_dim, device=global_std_vec.device, dtype=global_std_vec.dtype)

        for j_idx in range(joints_num):
            offset = j_idx * interleaved_dim_per_joint
            interleaved_mean[offset : offset + 3] = mean_lp_reshaped[j_idx, :]
            interleaved_mean[offset + 3 : offset + 9] = mean_gjrot_reshaped[j_idx, :]
            interleaved_mean[offset + 9 : offset + 15] = mean_ljrot_reshaped[j_idx, :]

            interleaved_std[offset : offset + 3] = std_lp_reshaped[j_idx, :]
            interleaved_std[offset + 3 : offset + 9] = std_gjrot_reshaped[j_idx, :]
            interleaved_std[offset + 9 : offset + 15] = std_ljrot_reshaped[j_idx, :]

        return interleaved_mean, interleaved_std


def create_normalization_stats(
    dataset: Any,
    device: torch.device,
) -> NormalizationStats:
    """
    Create all normalization statistics needed for inference.
    
    Args:
        dataset: Dataset instance with norm_params and _repr_indices
        device: Target device for tensors
        
    Returns:
        NormalizationStats containing all required statistics
    """
    # Build full motion_repr stats (concatenated layout)
    motion_repr_mean = torch.from_numpy(
        np.concatenate([dataset.norm_params[rn]['mean'] for rn in dataset._repr_list])
    ).to(device)
    motion_repr_std = torch.from_numpy(
        np.concatenate([dataset.norm_params[rn]['std'] for rn in dataset._repr_list])
    ).to(device)
    
    # Build interleaved stats for traj cond
    traj_cond_mean, traj_cond_std = get_interleaved_stats_for_features(
        global_mean_vec=motion_repr_mean,
        global_std_vec=motion_repr_std,
        dataset=dataset,
    )
    
    # Build stats for pose cond
    # New pose cond layout: [joint-based interleaved] | [global modalities including traj]
    # Global modalities for pose task include smplx_trans and smplx_rot_6d (traj modalities)
    
    # Get traj modality stats
    traj_modality_means = []
    traj_modality_stds = []
    for name in dataset.traj_repr_name_list:
        s, e = dataset._repr_indices[name]
        traj_modality_means.append(motion_repr_mean[s:e])
        traj_modality_stds.append(motion_repr_std[s:e])
    traj_stats_mean = torch.cat(traj_modality_means) if traj_modality_means else torch.tensor([], device=device)
    traj_stats_std = torch.cat(traj_modality_stds) if traj_modality_stds else torch.tensor([], device=device)
    
    # Pose cond = [interleaved joint features] | [traj features]
    pose_cond_mean = torch.cat([traj_cond_mean, traj_stats_mean])
    pose_cond_std = torch.cat([traj_cond_std, traj_stats_std])
    
    return NormalizationStats(
        motion_repr_mean=motion_repr_mean,
        motion_repr_std=motion_repr_std,
        traj_cond_mean=traj_cond_mean,
        traj_cond_std=traj_cond_std,
        pose_cond_mean=pose_cond_mean,
        pose_cond_std=pose_cond_std,
    )
