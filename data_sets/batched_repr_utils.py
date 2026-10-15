"""
Batched GPU utilities for motion representation processing.

This module provides GPU-accelerated, batched versions of motion processing functions
to enable 20-50x speedup over sequential numpy/scipy operations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from smplx.lbs import batch_rodrigues
import smplx
from einops import rearrange

# Import existing utilities that are already GPU-compatible
from data_sets.common.quaternion import qrot, qbetween, qnormalize
from utils.other_utils import estimate_angular_velocity
from utils.smplx_utils import local_to_global_pose_rotations
from data_sets.motion_representation import face_joint_indx, fid_r, fid_l, foot_detect_torch


def batched_rotvec_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert batched axis-angle (rotvec) to rotation matrices using batch_rodrigues.
    
    Args:
        rotvec: Rotation vectors of shape (*, 3) where * can be any number of dimensions
        
    Returns:
        Rotation matrices of shape (*, 3, 3)
    """
    original_shape = rotvec.shape[:-1]
    rotvec_flat = rotvec.reshape(-1, 3)
    rotmat_flat = batch_rodrigues(rotvec_flat)  # (N, 3, 3)
    return rotmat_flat.reshape(*original_shape, 3, 3)


def batched_rotmat_to_6d(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 6D rotation representation.
    
    Args:
        rotmat: Rotation matrices of shape (*, 3, 3)
        
    Returns:
        6D rotation representation of shape (*, 6)
    """
    # Take first two columns and flatten
    return rotmat[..., :2].reshape(*rotmat.shape[:-2], 6)


def batched_rotmat_to_rotvec(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle (rotation vector) representation.
    Uses the smplx rotation_matrix_to_angle_axis internally.
    
    Args:
        rotmat: Rotation matrices of shape (*, 3, 3)
        
    Returns:
        Rotation vectors of shape (*, 3)
    """
    from utils.konia_transform import rotation_matrix_to_angle_axis
    original_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.contiguous().reshape(-1, 3, 3)
    rotvec_flat = rotation_matrix_to_angle_axis(rotmat_flat)  # (N, 3)
    return rotvec_flat.reshape(*original_shape, 3)


def batched_rotmat_to_euler_zxy(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to ZXY intrinsic euler angles (in degrees).
    
    Pure PyTorch implementation for GPU batched processing.
    Matches scipy's Rotation.as_euler('zxy', degrees=True) convention.
    
    Args:
        rotmat: Rotation matrices of shape (*, 3, 3)
        
    Returns:
        Euler angles in degrees of shape (*, 3) as (z, x, y) rotations
    """
    # ZXY intrinsic = YXZ extrinsic convention
    # For ZXY intrinsic: R = Rz @ Rx @ Ry
    #
    # R = [cy*cz - sx*sy*sz,  -cx*sz,  cz*sy + cy*sx*sz]
    #     [cy*sz + cz*sx*sy,   cx*cz,  sy*sz - cy*cz*sx]
    #     [-cx*sy,             sx,     cx*cy           ]
    #
    # Where cx=cos(x), sx=sin(x), etc.
    
    original_shape = rotmat.shape[:-2]
    r = rotmat.reshape(-1, 3, 3)
    
    # x = arcsin(r[2,1])
    sx = r[:, 2, 1].clamp(-1.0, 1.0)  # sin(x)
    x = torch.asin(sx)
    
    # Check for gimbal lock (x close to +/- 90 degrees)
    cos_x = torch.cos(x)
    gimbal_lock = cos_x.abs() < 1e-6
    
    # Normal case: cos(x) != 0
    # z = atan2(r[1,0], r[1,1])  -- from derived formula
    # y = atan2(-r[2,0], r[2,2])
    z = torch.atan2(r[:, 1, 0], r[:, 1, 1])
    y = torch.atan2(-r[:, 2, 0], r[:, 2, 2])
    
    # Gimbal lock case: set z=0, compute y from r[0,0] and r[0,2]
    # When cos(x) ≈ 0, x ≈ ±90°
    # If x = 90° (sx=1):  y - z = atan2(r[0,2], r[0,0])
    # If x = -90° (sx=-1): y + z = atan2(-r[0,2], r[0,0])
    z_gimbal = torch.zeros_like(z)
    y_gimbal = torch.where(
        sx > 0,
        torch.atan2(r[:, 0, 2], r[:, 0, 0]),
        torch.atan2(-r[:, 0, 2], r[:, 0, 0])
    )
    
    z = torch.where(gimbal_lock, z_gimbal, z)
    y = torch.where(gimbal_lock, y_gimbal, y)
    
    # Stack and convert to degrees
    euler_rad = torch.stack([z, x, y], dim=-1)  # (N, 3)
    euler_deg = torch.rad2deg(euler_rad)
    
    return euler_deg.reshape(*original_shape, 3)


def batched_euler_zxy_to_rotmat(euler_deg: torch.Tensor) -> torch.Tensor:
    """
    Convert ZXY intrinsic euler angles (in degrees) to rotation matrices.
    
    Pure PyTorch implementation for GPU batched processing.
    Matches scipy's Rotation.from_euler('zxy', degrees=True) convention.
    
    Args:
        euler_deg: Euler angles in degrees of shape (*, 3) as (z, x, y) rotations
        
    Returns:
        Rotation matrices of shape (*, 3, 3)
    """
    original_shape = euler_deg.shape[:-1]
    euler = torch.deg2rad(euler_deg.reshape(-1, 3))
    
    z, x, y = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cz, sz = torch.cos(z), torch.sin(z)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    
    # R = Rz @ Rx @ Ry
    # Build rotation matrices column by column
    r00 = cy * cz - sx * sy * sz
    r01 = -cx * sz
    r02 = cz * sy + cy * sx * sz
    
    r10 = cy * sz + cz * sx * sy
    r11 = cx * cz
    r12 = sy * sz - cy * cz * sx
    
    r20 = -cx * sy
    r21 = sx
    r22 = cx * cy
    
    rotmat = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (N, 3, 3)
    
    return rotmat.reshape(*original_shape, 3, 3)


def add_noise_to_smplx_params_batched(
    cano_smplx_params_dict: Dict[str, torch.Tensor],
    noise_std_params_dict: Dict[str, float],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Add noise to SMPLX parameters in a batched manner.
    
    Noise is added:
    - Directly to transl and betas (additive Gaussian)
    - In euler angle space for rotations (global_orient, body_pose)
    
    Uses scipy for euler conversion to match sequential behavior exactly.
    
    Args:
        cano_smplx_params_dict: Canonical SMPLX params, each shape (B, T, dim)
        noise_std_params_dict: Dict of noise std for each param type
        device: Computation device
        
    Returns:
        cano_smplx_params_dict_noisy: Noisy SMPLX params
        smplx_noise_dict: Dict of noise tensors for each param
    """
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    B, T, _ = cano_smplx_params_dict['transl'].shape
    
    cano_smplx_params_dict_noisy = {}
    smplx_noise_dict = {}
    
    for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
        param = cano_smplx_params_dict[param_name]  # (B, T, dim)
        noise_std = noise_std_params_dict.get(param_name, 0.0)
        
        if param_name in ['transl', 'betas']:
            # Direct additive Gaussian noise
            noise = torch.randn_like(param) * noise_std
            cano_smplx_params_dict_noisy[param_name] = param + noise
            smplx_noise_dict[param_name] = noise
            
        elif param_name == 'global_orient':
            # Add noise in euler angle space using scipy for correctness
            # global_orient: (B, T, 3) axis-angle
            param_np = param.cpu().numpy()
            original_shape = param_np.shape  # (B, T, 3)
            
            # Flatten to (B*T, 3) for scipy operations
            rotvec_flat = param_np.reshape(-1, 3)
            
            # Convert to euler using scipy
            euler_deg = R.from_rotvec(rotvec_flat).as_euler('zxy', degrees=True)
            
            # Generate noise
            noise_np = np.random.normal(0, noise_std, euler_deg.shape).astype(np.float32)
            euler_deg_noisy = euler_deg + noise_np
            
            # Convert back to rotvec
            rotvec_noisy = R.from_euler('zxy', euler_deg_noisy, degrees=True).as_rotvec().astype(np.float32)
            rotvec_noisy = rotvec_noisy.reshape(original_shape)
            
            cano_smplx_params_dict_noisy[param_name] = torch.from_numpy(rotvec_noisy).to(device)
            smplx_noise_dict[param_name] = torch.from_numpy(noise_np.reshape(B, T, 3)).to(device)
            
        elif param_name == 'body_pose':
            # body_pose: (B, T, 63) -> (B, T, 21, 3) axis-angle
            param_np = param.cpu().numpy()
            B, T, D = param_np.shape
            
            # Reshape to (B*T*21, 3)
            rotvec_flat = param_np.reshape(-1, 3)
            
            # Convert to euler using scipy
            euler_deg = R.from_rotvec(rotvec_flat).as_euler('zxy', degrees=True)
            
            # Generate noise
            noise_np = np.random.normal(0, noise_std, euler_deg.shape).astype(np.float32)
            euler_deg_noisy = euler_deg + noise_np
            
            # Convert back to rotvec
            rotvec_noisy = R.from_euler('zxy', euler_deg_noisy, degrees=True).as_rotvec().astype(np.float32)
            rotvec_noisy = rotvec_noisy.reshape(B, T, D)
            
            cano_smplx_params_dict_noisy[param_name] = torch.from_numpy(rotvec_noisy).to(device)
            smplx_noise_dict[param_name] = torch.from_numpy(noise_np.reshape(B, T, 21, 3)).to(device)
    
    return cano_smplx_params_dict_noisy, smplx_noise_dict


def smplx_forward_batched(
    smplx_model,
    smplx_params_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Batched SMPLX forward pass to compute joint positions from parameters.
    
    Args:
        smplx_model: SMPLX model instance
        smplx_params_dict: Dict with 'global_orient', 'transl', 'betas', 'body_pose'
                          Each shape (B, T, dim) or will be flattened to (B*T, dim)
        device: Computation device
        
    Returns:
        Joint positions of shape (B, T, 22, 3)
    """
    # Get batch and time dimensions
    transl = smplx_params_dict['transl']
    B, T, _ = transl.shape
    
    # Flatten batch and time dimensions for SMPLX forward
    params_flat = {}
    for key in ['global_orient', 'transl', 'betas', 'body_pose']:
        param = smplx_params_dict[key]
        if key == 'body_pose':
            # body_pose might be (B, T, 63) or (B, T, 21, 3)
            params_flat[key] = param.reshape(B * T, -1)
        else:
            params_flat[key] = param.reshape(B * T, -1)
    
    # Add zero poses for face/hands (we don't model them in RoHM)
    BT = B * T
    params_flat['jaw_pose'] = torch.zeros(BT, 3, device=device)
    params_flat['leye_pose'] = torch.zeros(BT, 3, device=device)
    params_flat['reye_pose'] = torch.zeros(BT, 3, device=device)
    params_flat['left_hand_pose'] = torch.zeros(BT, 45, device=device)
    params_flat['right_hand_pose'] = torch.zeros(BT, 45, device=device)
    params_flat['expression'] = torch.zeros(BT, 10, device=device)
    
    # SMPLX forward pass
    with torch.no_grad():
        smplx_output = smplx_model(**params_flat)
        joints = smplx_output.joints[:, 0:22]  # (B*T, 22, 3)
    
    # Reshape back to (B, T, 22, 3)
    return joints.reshape(B, T, 22, 3)


def cano_seq_smplx_batched(
    positions: torch.Tensor,
    smplx_params_dict: Dict[str, torch.Tensor],
    device: torch.device = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Batched canonicalization for motion sequences.
    
    Canonicalize such that:
    - Z+ axis is up
    - X/Y coordinate of frame 0's root is at origin
    - Foot on floor
    
    NOTE: align_to_yaxis is NOT supported (set to False by default in original code)
    
    Args:
        positions: Joint positions of shape (B, T, J, 3), Z-up
        smplx_params_dict: Dict with keys 'global_orient', 'transl', 'betas', 'body_pose'
                          Each value has shape (B, T, dim)
        device: Target device
        
    Returns:
        cano_positions: Canonicalized positions (B, T, J, 3)
        cano_smplx_params_dict: Canonicalized SMPLX parameters
    """
    if device is None:
        device = positions.device
    
    B, T, J, _ = positions.shape
    cano_positions = positions.clone()
    
    # Put on floor: find minimum Z across all joints and frames for each batch
    floor_height = cano_positions.amin(dim=(1, 2))[:, 2]  # (B,)
    cano_positions[:, :, :, 2] -= floor_height.view(B, 1, 1)
    
    # Translate so root (joint 0) XY at frame 0 is at origin
    root_pos_init = cano_positions[:, 0, 0, :]  # (B, 3) - root of first frame
    root_pose_init_xy = root_pos_init * torch.tensor([1, 1, 0], device=device, dtype=positions.dtype)  # (B, 3)
    cano_positions = cano_positions - root_pose_init_xy.view(B, 1, 1, 3)
    
    # Build transformation matrix for SMPLX params
    # transf_matrix_1: translation
    transf_matrix_1 = torch.eye(4, device=device, dtype=positions.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    transf_matrix_1[:, 0, 3] = -root_pose_init_xy[:, 0]
    transf_matrix_1[:, 1, 3] = -root_pose_init_xy[:, 1]
    transf_matrix_1[:, 2, 3] = -floor_height
    
    # transf_matrix_2: identity (no rotation since align_to_yaxis=False)
    transf_matrix_2 = torch.eye(4, device=device, dtype=positions.dtype).unsqueeze(0).expand(B, -1, -1)
    
    transf_matrix = torch.bmm(transf_matrix_2, transf_matrix_1)  # (B, 4, 4)
    
    # Compute delta_T = positions[:, :, 0] - transl  (pelvis offset from transl)
    delta_T = positions[:, :, 0, :] - smplx_params_dict['transl']  # (B, T, 3)
    
    # Update SMPLX global rotation and translation
    cano_smplx_params_dict = update_globalRT_batched(
        smplx_params_dict, transf_matrix, delta_T
    )
    
    return cano_positions, cano_smplx_params_dict


def update_globalRT_batched(
    body_param_dict: Dict[str, torch.Tensor],
    trans_to_target_origin: torch.Tensor,
    delta_T: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Batched update of global rotation and translation for SMPLX params.
    
    Args:
        body_param_dict: Dict with 'global_orient', 'transl', etc. Each shape (B, T, dim)
        trans_to_target_origin: Transformation matrix (B, 4, 4)
        delta_T: Pelvis offset (B, T, 3)
        
    Returns:
        Updated body_param_dict with new global_orient and transl
    """
    B, T, _ = body_param_dict['transl'].shape
    device = body_param_dict['transl'].device
    dtype = body_param_dict['transl'].dtype
    
    # Get rotation from axis-angle
    body_R_angle = body_param_dict['global_orient']  # (B, T, 3)
    body_R_mat = batched_rotvec_to_rotmat(body_R_angle)  # (B, T, 3, 3)
    
    body_T = body_param_dict['transl']  # (B, T, 3)
    
    # Build body transformation matrices
    body_mat = torch.zeros(B, T, 4, 4, device=device, dtype=dtype)
    body_mat[:, :, :3, :3] = body_R_mat
    body_mat[:, :, :3, 3] = body_T + delta_T
    body_mat[:, :, 3, 3] = 1
    
    # Apply transformation: new = trans @ body
    # trans_to_target_origin: (B, 4, 4), body_mat: (B, T, 4, 4)
    trans_expanded = trans_to_target_origin.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, 4, 4)
    body_mat_new = torch.matmul(trans_expanded, body_mat)  # (B, T, 4, 4)
    
    # Extract new rotation (as axis-angle) and translation
    body_R_new_mat = body_mat_new[:, :, :3, :3]  # (B, T, 3, 3)
    body_R_new = batched_rotmat_to_rotvec(body_R_new_mat)  # (B, T, 3)
    body_T_new = body_mat_new[:, :, :3, 3]  # (B, T, 3)
    
    # Create output dict (copy other params)
    result = {k: v.clone() for k, v in body_param_dict.items()}
    result['global_orient'] = body_R_new
    result['transl'] = body_T_new - delta_T
    
    return result


def get_repr_smplx_batched(
    positions: torch.Tensor,
    smplx_params_dict: Dict[str, torch.Tensor],
    feet_vel_thre: float = 5e-5,
    cano_jts_pos: bool = True,
    smplx_model: Optional[smplx.SMPLX] = None,
    device: torch.device = None,
    modality_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Batched computation of motion representation.
    
    Args:
        positions: Joint positions (B, T, J, 3), canonicalized
        smplx_params_dict: Canonicalized SMPLX params, each (B, T, dim)
        feet_vel_thre: Velocity threshold for foot contact
        cano_jts_pos: Whether positions are already canonicalized
        smplx_model: SMPLX model for global rotations (optional)
        device: Computation device
        modality_names: List of modalities to compute
        
    Returns:
        data_dict: Motion representation dict, each value (B, T-1, dim)
    """
    if device is None:
        device = positions.device
    
    if modality_names is None:
        modality_names = ['smplx_rot_6d', 'smplx_trans', 'local_positions', 
                          'smplx_body_pose_6d', 'smplx_betas', 'foot_contact']
    
    modality_set = set(modality_names)
    
    need_velocity = any(m in modality_set for m in [
        'smplx_rot_vel', 'smplx_trans_vel', 'local_vel', 
        'smplx_body_pose_vel', 'R_world_joints_vel', 'R_root_joints_vel'
    ])
    need_global_rotations = any(m in modality_set for m in [
        'R_world_joints', 'R_root_joints', 'R_world_joints_vel', 'R_root_joints_vel'
    ])
    
    B, T, J, _ = positions.shape
    global_positions = positions.clone()
    
    # Foot contact detection
    feet_l, feet_r = foot_detect_torch(positions, feet_vel_thre, up_axis='z')
    # feet_l, feet_r: (B, T-1, 2)
    
    # Get forward direction
    l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
    across1 = positions[:, :, r_hip] - positions[:, :, l_hip]  # (B, T, 3)
    across2 = positions[:, :, sdr_r] - positions[:, :, sdr_l]  # (B, T, 3)
    across = across1 + across2
    across = across / (torch.sqrt((across ** 2).sum(dim=-1, keepdim=True)) + 1e-8)
    
    # Forward = cross([0,0,1], across)
    z_up = torch.tensor([[0, 0, 1]], device=device, dtype=positions.dtype).expand(B, T, 3)
    forward = torch.cross(z_up, across, dim=-1)
    forward = forward / (torch.sqrt((forward ** 2).sum(dim=-1, keepdim=True)) + 1e-8)
    
    # Get root rotation quaternion (rotate forward to [0,1,0])
    target = torch.tensor([[0, 1, 0]], device=device, dtype=positions.dtype).expand(B, T, 3)
    root_rot_quat = qbetween(forward, target)  # (B, T, 4)
    
    # Handle NaN values by copying from previous frame
    nan_mask = torch.isnan(root_rot_quat).any(dim=-1)  # (B, T)
    if nan_mask.any():
        # Replace NaN frames with previous frame's quaternion
        for b in range(B):
            for t in range(T):
                if nan_mask[b, t]:
                    root_rot_quat[b, t] = root_rot_quat[b, t-1] if t > 0 else torch.tensor([1, 0, 0, 0], device=device, dtype=positions.dtype)
    
    if cano_jts_pos:
        # First frame should be identity (aligned to y+)
        root_rot_quat[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=positions.dtype)
    
    # Local joint positions
    local_positions = positions.clone()  # (B, T, J, 3)
    if cano_jts_pos:
        # Subtract root XY - extract root X and Y first before subtracting
        root_x = local_positions[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        root_y = local_positions[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        local_positions = local_positions.clone()  # Avoid in-place issues
        local_positions[:, :, :, 0] = local_positions[:, :, :, 0] - root_x.squeeze(-1)
        local_positions[:, :, :, 1] = local_positions[:, :, :, 1] - root_y.squeeze(-1)
        
        # Rotate each frame to face y+
        # Expand quaternion for all joints: (B, T, J, 4)
        quat_expanded = root_rot_quat.unsqueeze(2).expand(-1, -1, J, -1)
        local_positions = qrot(quat_expanded, local_positions)
        
        if need_velocity:
            # Local velocity: rotate the global position differences
            global_vel = global_positions[:, 1:] - global_positions[:, :-1]  # (B, T-1, J, 3)
            quat_vel = root_rot_quat[:, :-1].unsqueeze(2).expand(-1, -1, J, -1)  # (B, T-1, J, 4)
            local_vel = qrot(quat_vel, global_vel)  # (B, T-1, J, 3)
    else:
        if need_velocity:
            local_vel = global_positions[:, 1:] - global_positions[:, :-1]
    
    # SMPLX-based representation
    smplx_rot_aa = smplx_params_dict['global_orient']  # (B, T, 3)
    smplx_rot_mat = batched_rotvec_to_rotmat(smplx_rot_aa)  # (B, T, 3, 3)
    smplx_rot_6d = batched_rotmat_to_6d(smplx_rot_mat)  # (B, T, 6)
    
    if need_velocity:
        dRdt = smplx_rot_mat[:, 1:] - smplx_rot_mat[:, :-1]  # (B, T-1, 3, 3)
        smplx_rot_vel = estimate_angular_velocity(smplx_rot_mat[:, :-1], dRdt)  # (B, T-1, 3)
    
    smplx_trans = smplx_params_dict['transl']  # (B, T, 3)
    if need_velocity:
        smplx_trans_vel = smplx_trans[:, 1:] - smplx_trans[:, :-1]  # (B, T-1, 3)
    
    # Body pose
    smplx_body_pose_aa = smplx_params_dict['body_pose']  # (B, T, 63)
    smplx_body_pose_mat = batched_rotvec_to_rotmat(
        smplx_body_pose_aa.reshape(B, T, 21, 3)
    )  # (B, T, 21, 3, 3)
    smplx_body_pose_6d = batched_rotmat_to_6d(smplx_body_pose_mat)  # (B, T, 21, 6)
    
    if need_velocity:
        dRdt_body = smplx_body_pose_mat[:, 1:] - smplx_body_pose_mat[:, :-1]  # (B, T-1, 21, 3, 3)
        smplx_body_pose_vel = estimate_angular_velocity(
            smplx_body_pose_mat[:, :-1], dRdt_body
        )  # (B, T-1, 21, 3)
    
    smplx_betas = smplx_params_dict['betas']  # (B, T, 10)
    
    T_out = T - 1  # Output frames
    
    # Global rotations if needed
    if need_global_rotations and smplx_model is not None:
        with torch.no_grad():
            smplx_model.to(device)
            # Flatten batch and time dimensions
            global_orient_flat = smplx_rot_aa.reshape(B * T, 3)
            body_pose_flat = smplx_body_pose_aa.reshape(B * T, 63)
            
            J_transformed, R_world_joints, R_root_joints = local_to_global_pose_rotations(
                smplx_model=smplx_model,
                global_orient_aa=global_orient_flat,
                body_pose_aa=body_pose_flat,
                return_joint_positions=True
            )  # (B*T, 55, 3, 3)
            
            R_world_joints = R_world_joints.reshape(B, T, -1, 3, 3)[:, :, :22]  # (B, T, 22, 3, 3)
            R_root_joints = R_root_joints.reshape(B, T, -1, 3, 3)[:, :, :22]  # (B, T, 22, 3, 3)
            
            from utils.konia_transform import matrix_to_rotation_6d
            R_world_joints_6d = matrix_to_rotation_6d(R_world_joints)  # (B, T, 22, 6)
            R_root_joints_6d = matrix_to_rotation_6d(R_root_joints)  # (B, T, 22, 6)
            
            R_world_joints_6d_flat = R_world_joints_6d[:, :T_out].reshape(B, T_out, -1)
            R_root_joints_6d_flat = R_root_joints_6d[:, :T_out].reshape(B, T_out, -1)
            
            if need_velocity:
                dRdt_world = R_world_joints[:, 1:] - R_world_joints[:, :-1]
                dRdt_root = R_root_joints[:, 1:] - R_root_joints[:, :-1]
                R_world_joints_vel = estimate_angular_velocity(R_world_joints[:, :-1], dRdt_world)
                R_root_joints_vel = estimate_angular_velocity(R_root_joints[:, :-1], dRdt_root)
                R_world_joints_vel_flat = R_world_joints_vel.reshape(B, T_out, -1)
                R_root_joints_vel_flat = R_root_joints_vel.reshape(B, T_out, -1)
    
    # Build output dictionary
    data_dict = {
        'smplx_rot_6d': smplx_rot_6d[:, :T_out],  # (B, T-1, 6)
        'smplx_trans': smplx_trans[:, :T_out],  # (B, T-1, 3)
        'local_positions': local_positions[:, :T_out].reshape(B, T_out, -1),  # (B, T-1, J*3)
        'smplx_body_pose_6d': smplx_body_pose_6d[:, :T_out].reshape(B, T_out, -1),  # (B, T-1, 126)
        'smplx_betas': smplx_betas[:, :T_out],  # (B, T-1, 10)
        'foot_contact': torch.cat([feet_l, feet_r], dim=-1),  # (B, T-1, 4)
    }
    
    if need_velocity:
        data_dict['smplx_rot_vel'] = smplx_rot_vel  # (B, T-1, 3)
        data_dict['smplx_trans_vel'] = smplx_trans_vel  # (B, T-1, 3)
        data_dict['local_vel'] = local_vel.reshape(B, T_out, -1)  # (B, T-1, J*3)
        if not need_global_rotations:
            data_dict['smplx_body_pose_vel'] = smplx_body_pose_vel.reshape(B, T_out, -1)
    
    if need_global_rotations and smplx_model is not None:
        data_dict['R_world_joints'] = R_world_joints_6d_flat
        data_dict['R_root_joints'] = R_root_joints_6d_flat
        if need_velocity:
            data_dict['R_world_joints_vel'] = R_world_joints_vel_flat
            data_dict['R_root_joints_vel'] = R_root_joints_vel_flat
    
    return data_dict


def process_clips_batched(
    joints_list: List[torch.Tensor],
    smplx_list: List[torch.Tensor],
    joints_num: int,
    smplx_model: Optional[smplx.SMPLX],
    modality_names: List[str],
    device: torch.device,
    batch_size: int = 512
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Process all clips in batches.
    
    Args:
        joints_list: List of joint position arrays, each (T, J, 3)
        smplx_list: List of SMPLX param arrays, each (T, D)
        joints_num: Number of joints to use (typically 22)
        smplx_model: SMPLX model instance
        modality_names: List of modality names to compute
        device: GPU device
        batch_size: Number of clips per batch
        
    Returns:
        cano_positions_list: List of canonicalized positions
        repr_list: List of representation dicts
        cano_params_list: List of canonicalized SMPLX params
    """
    from tqdm import tqdm
    
    n_clips = len(joints_list)
    
    # Check if all clips have the same length
    clip_lens = [j.shape[0] for j in joints_list]
    fixed_len = len(set(clip_lens)) == 1
    
    cano_positions_list = []
    repr_list = []
    cano_params_list = []
    
    for batch_start in tqdm(range(0, n_clips, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_clips)
        batch_indices = range(batch_start, batch_end)
        
        # Stack clips into batch tensors
        batch_joints = torch.stack([
            torch.from_numpy(joints_list[i][:, :joints_num, :]).float() 
            for i in batch_indices
        ], dim=0).to(device)  # (B, T, J, 3)
        
        batch_smplx = torch.stack([
            torch.from_numpy(smplx_list[i]).float() 
            for i in batch_indices
        ], dim=0).to(device)  # (B, T, D)
        
        # Parse SMPLX params
        smplx_params_dict = {
            'global_orient': batch_smplx[:, :, 0:3],
            'transl': batch_smplx[:, :, 3:6],
            'betas': batch_smplx[:, :, 6:16],
            'body_pose': batch_smplx[:, :, 16:79],
        }
        
        # Canonicalize
        cano_positions, cano_params = cano_seq_smplx_batched(
            batch_joints, smplx_params_dict, device=device
        )
        
        # Compute representations
        repr_dict = get_repr_smplx_batched(
            cano_positions,
            cano_params,
            feet_vel_thre=5e-5,
            cano_jts_pos=True,
            smplx_model=smplx_model if 'R_world_joints' in modality_names else None,
            device=device,
            modality_names=modality_names
        )
        
        # Move to CPU and convert to individual clips
        cano_positions_cpu = cano_positions.cpu()
        repr_dict_cpu = {k: v.cpu() for k, v in repr_dict.items()}
        cano_params_cpu = {k: v.cpu() for k, v in cano_params.items()}
        
        for i in range(len(batch_indices)):
            cano_positions_list.append(cano_positions_cpu[i].numpy())
            repr_list.append({k: v[i].numpy() for k, v in repr_dict_cpu.items()})
            cano_params_list.append({k: v[i].numpy() for k, v in cano_params_cpu.items()})
    
    return cano_positions_list, repr_list, cano_params_list


def benchmark_batched_processing(n_clips: int = 100, batch_size: int = 64, device: str = 'cuda'):
    """
    Benchmark batched vs sequential processing.
    """
    import time
    import numpy as np
    from data_sets.motion_representation import get_repr_smplx, cano_seq_smplx
    
    print(f"Benchmarking with {n_clips} clips, batch_size={batch_size}")
    
    # Generate random test data
    T, J = 33, 22
    np.random.seed(42)
    
    joints_list = [np.random.randn(T, J, 3).astype(np.float32) for _ in range(n_clips)]
    smplx_list = [np.random.randn(T, 79).astype(np.float32) for _ in range(n_clips)]
    
    # Make the data look more realistic
    for i in range(n_clips):
        joints_list[i][:, :, 2] = np.abs(joints_list[i][:, :, 2])  # Z positive
        smplx_list[i][:, 3:6] = joints_list[i][:, 0, :]  # transl = root joint
    
    modality_names = ['smplx_rot_6d', 'smplx_trans', 'local_positions', 
                      'smplx_body_pose_6d', 'smplx_betas', 'foot_contact']
    
    # Benchmark batched
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.time()
    cano_list, repr_list, params_list = process_clips_batched(
        joints_list, smplx_list, J, None, modality_names, device, batch_size
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_batched = time.time() - t0
    
    print(f"Batched processing: {t_batched:.3f}s ({n_clips/t_batched:.1f} clips/s)")
    
    # Benchmark sequential (only if small enough)
    if n_clips <= 100:
        t0 = time.time()
        for i in range(n_clips):
            smplx_params = {
                'global_orient': smplx_list[i][:, 0:3],
                'transl': smplx_list[i][:, 3:6],
                'betas': smplx_list[i][:, 6:16],
                'body_pose': smplx_list[i][:, 16:79],
            }
            cano_pos, cano_params = cano_seq_smplx(
                joints_list[i][:, :J], smplx_params, align_to_yaxis=False
            )
            repr_dict = get_repr_smplx(
                cano_pos, cano_params, modality_names=modality_names
            )
        t_sequential = time.time() - t0
        
        print(f"Sequential processing: {t_sequential:.3f}s ({n_clips/t_sequential:.1f} clips/s)")
        print(f"Speedup: {t_sequential/t_batched:.1f}x")
    
    return t_batched
