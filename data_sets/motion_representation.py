# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/EricGuo5513/HumanML3D

from threading import local
from data_sets.common.quaternion import *
from utils.konia_transform import *
from typing import Optional, Dict, Tuple

import math
from utils.other_utils import *
from utils.smplx_utils import local_to_global_pose_rotations, global_to_local_pose_rotations
from utils import dist_util
from einops import einsum
import smplx

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
# joints_num = 22
head_joint_indx = 15



def foot_detect(positions, thres, up_axis='y'):
    if up_axis == 'y':
        up_axis_dim = 1
    elif up_axis == 'z':
        up_axis_dim = 2
    # velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.18, 0.15])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1,fid_l,up_axis_dim]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
    # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1,fid_r,up_axis_dim]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
    # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r

def foot_detect_torch(positions: torch.Tensor, thres: float, up_axis: str = 'y') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of foot contact detection
    
    Args:
        positions: Joint positions tensor [..., T, J, 3]
        thres: Velocity threshold for foot contact
        up_axis: Vertical axis, either 'y' or 'z'
    
    Returns:
        tuple of (feet_l, feet_r) tensors with foot contact flags
    """
    up_axis_dim = 1 if up_axis == 'y' else 2
    velfactor = torch.tensor([thres, thres], device=positions.device)
    heightfactor = torch.tensor([0.18, 0.15], device=positions.device)

    # Compute squared velocity differences for left foot
    feet_l_x = (positions[..., 1:, fid_l, 0] - positions[..., :-1, fid_l, 0]).pow(2)
    feet_l_y = (positions[..., 1:, fid_l, 1] - positions[..., :-1, fid_l, 1]).pow(2)
    feet_l_z = (positions[..., 1:, fid_l, 2] - positions[..., :-1, fid_l, 2]).pow(2)
    feet_l_h = positions[..., :-1, fid_l, up_axis_dim]
    
    # Detect left foot contacts
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    feet_l = feet_l.to(dtype=torch.float)

    # Compute squared velocity differences for right foot 
    feet_r_x = (positions[..., 1:, fid_r, 0] - positions[..., :-1, fid_r, 0]).pow(2)
    feet_r_y = (positions[..., 1:, fid_r, 1] - positions[..., :-1, fid_r, 1]).pow(2)
    feet_r_z = (positions[..., 1:, fid_r, 2] - positions[..., :-1, fid_r, 2]).pow(2)
    feet_r_h = positions[..., :-1, fid_r, up_axis_dim]
    
    # Detect right foot contacts
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    feet_r = feet_r.to(dtype=torch.float)

    return feet_l, feet_r

############################## aria glasses
def cano_seq_camera(traj_clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs canonicalization on an Aria camera trajectory clip.
    The trajectory is transformed so that the first frame is at the origin (0,0,0)
    with an identity rotation.

    Args:
        traj_clip (np.ndarray): Trajectory clip of shape [T, 7] (tx, ty, tz, qw, qx, qy, qz).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - cano_traj_clip: Canonicalized trajectory clip [T, 7].
            - inv_transf_matrix: The inverse transformation matrix [4, 4] that was applied.
    """
    # Extract the pose of the first frame
    p0 = traj_clip[0, :3]
    q0 = traj_clip[0, 3:]  # qw, qx, qy, qz

    # Create the transformation matrix for the first frame's pose
    r0 = R.from_quat(q0[[1, 2, 3, 0]])  # Scipy expects [x, y, z, w]
    T0 = np.eye(4)
    T0[:3, :3] = r0.as_matrix()
    T0[:3, 3] = p0

    # Compute the inverse transformation to bring the first frame to the origin
    inv_T0 = np.linalg.inv(T0)

    # Apply the inverse transformation to all frames in the clip
    cano_traj_clip = np.zeros_like(traj_clip)
    for i in range(traj_clip.shape[0]):
        pi = traj_clip[i, :3]
        qi = traj_clip[i, 3:]
        ri = R.from_quat(qi[[1, 2, 3, 0]])
        Ti = np.eye(4)
        Ti[:3, :3] = ri.as_matrix()
        Ti[:3, 3] = pi

        # Transform the pose
        Ti_cano = inv_T0 @ Ti

        # Extract the new position and quaternion
        pi_cano = Ti_cano[:3, 3]
        ri_cano = R.from_matrix(Ti_cano[:3, :3])
        qi_cano = ri_cano.as_quat()  # [x, y, z, w]

        cano_traj_clip[i, :3] = pi_cano
        cano_traj_clip[i, 3:] = qi_cano[[3, 0, 1, 2]]  # Convert back to [w, x, y, z]

    return cano_traj_clip, inv_T0


def get_repr_aria_head(cano_clip: np.ndarray, full_repr_list: list) -> dict:
    """
    Calculates a full-body motion representation from a canonicalized Aria head trajectory.
    All body parts other than the root/head are zero-filled.

    Args:
        cano_clip (np.ndarray): Canonicalized trajectory clip of shape [T, 7].
        full_repr_list (list): The list of all expected representation keys.

    Returns:
        dict: A dictionary of motion representation features, compatible with AMASS.
    """
    num_frames = cano_clip.shape[0]
    num_output_frames = num_frames - 1

    # --- 1. Calculate Head/Root Trajectory Features ---
    head_pos = cano_clip[:, :3]
    head_quat = cano_clip[:, 3:]  # (w, x, y, z)

    # Linear velocity (in the local frame of the previous step)
    head_vel = head_pos[1:] - head_pos[:-1]
    head_vel = qrot_np(qinv_np(head_quat[:-1]), head_vel)

    # 6D Rotation
    head_rot_mat = R.from_quat(head_quat[:, [1, 2, 3, 0]]).as_matrix()
    head_rot_6d = rotmat_to_rot6d(head_rot_mat)

    # Angular velocity
    delta_quat = qmul_np(head_quat[1:], qinv_np(head_quat[:-1]))
    delta_rot_mat = R.from_quat(delta_quat[:, [1, 2, 3, 0]]).as_matrix()
    head_rot_vel = R.from_matrix(delta_rot_mat).as_rotvec()

    # --- 2. Create the Representation Dictionary ---
    data_dict = {
        # Treat head as the root
        'root_rot_angle': np.arctan2(head_quat[:-1, 3:4], head_quat[:-1, 0:1]),
        'root_rot_angle_vel': np.arctan2(delta_quat[:, 3:4], delta_quat[:, 0:1]),
        'root_l_pos': head_pos[:-1, :2], # Z is up, so XY is the ground plane
        'root_l_vel': head_vel[:, :2],
        'root_height': head_pos[:-1, 2:3],
        # Treat head as the smplx root
        'smplx_rot_6d': head_rot_6d[:-1],
        'smplx_rot_vel': head_rot_vel,
        'smplx_trans': head_pos[:-1],
        'smplx_trans_vel': head_vel,
    }

    # --- 3. Zero-fill all other body-related features ---
    for key in full_repr_list:
        if key not in data_dict:
            dim = REPR_DIM_DICT[key]
            data_dict[key] = np.zeros((num_output_frames, dim), dtype=np.float32)

    return data_dict

#################################### ADT dataset
def cano_seq_adt(positions, align_to_yaxis=True):
    """
    Performs canonicalization on an ADT motion sequence (3D joint positions).
    This function mirrors the logic of cano_seq_smplx but operates only on joint data.
    The input is assumed to be Z-up, which is consistent with ADT's scene frame.

    - The sequence is aligned so frame 0 faces the Y+ axis.
    - The root's XY coordinate at frame 0 is at the origin.
    - The lowest point in the sequence is placed on the Z=0 floor.

    Args:
        positions (np.ndarray): Original joint positions of shape [T, J, 3].
        align_to_yaxis (bool): Whether to align the initial orientation.

    Returns:
        np.ndarray: Canonicalized joint positions.
    """
    cano_positions = positions.copy()
    
    # Joint indices for SMPL 22-joint structure
    # Based on smpl_joint_names_22 in DatasetADT
    r_hip, l_hip, sdr_r, sdr_l = 2, 1, 17, 16

    # 1. Put on Floor (Z-axis is up)
    floor_height = np.nanmin(cano_positions[..., 2])
    cano_positions[:, :, 2] -= floor_height

    # 2. Translate so the root (pelvis, joint 0) XY at frame 0 is at the origin
    root_pos_init_xy = cano_positions[0, 0, :] * np.array([1, 1, 0])
    cano_positions = cano_positions - root_pos_init_xy

    # 3. Rotate so that at frame 0, the character faces the Y+ axis
    if align_to_yaxis:
        joints_frame0 = cano_positions[0]
        across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
        across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
        
        x_axis = (across1 + across2)
        x_axis[2] = 0
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        transf_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)
        cano_positions = np.matmul(cano_positions, transf_rotmat)
        
    return cano_positions


def get_repr_adt(positions, feet_vel_thre=5e-5, cano_jts_pos=True):
    """
    Calculates the motion representation for an ADT sequence.
    Mirrors `get_repr_smplx` but only computes features derivable from joint positions.
    SMPL-X specific features are returned as arrays of zeros to maintain compatibility.
    """
    num_frames, num_joints, _ = positions.shape
    
    # Indices based on the 22-joint structure
    face_joint_indx = [2, 1, 17, 16] # r_hip, l_hip, r_shoulder, l_shoulder
    l_hip, r_hip, sdr_l, sdr_r = 1, 2, 16, 17 # Corrected indices
    fid_r, fid_l = [8, 11], [7, 10] # r_ankle, r_foot, l_ankle, l_foot

    # --- Joint-based trajectory representation ---
    feet_l, feet_r = foot_detect(positions, feet_vel_thre, up_axis='z')

    across = (positions[:, r_hip] - positions[:, l_hip]) + (positions[:, sdr_r] - positions[:, sdr_l])
    across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-8)
    forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)
    
    target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
    root_rot_quat = qbetween_np(forward, target)
    if np.isnan(root_rot_quat).any():
        # A simple fix for potential NaNs
        for i in range(len(root_rot_quat)):
            if np.isnan(root_rot_quat[i]).any():
                root_rot_quat[i] = root_rot_quat[i-1] if i > 0 else np.array([1.0, 0.0, 0.0, 0.0])

    if cano_jts_pos:
        root_rot_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])

    root_rot_quat_vel = qmul_np(root_rot_quat[1:], qinv_np(root_rot_quat[:-1]))

    root_l_pos = positions[:, 0, :]
    root_height = positions[:, 0, 2:3]
    root_l_vel = positions[1:, 0, :] - positions[:-1, 0, :]
    if cano_jts_pos:
        root_l_vel = qrot_np(root_rot_quat[1:], root_l_vel)

    root_rot_angle = np.arctan2(root_rot_quat[:, 3:4], root_rot_quat[:, 0:1])
    root_rot_angle_vel = np.arctan2(root_rot_quat_vel[:, 3:4], root_rot_quat_vel[:, 0:1])

    # --- Joint-based local pose representation ---
    local_positions = positions.copy()
    if cano_jts_pos:
        local_positions[..., :2] -= local_positions[:, 0:1, :2]
        local_positions = qrot_np(np.repeat(root_rot_quat[:, None], num_joints, axis=1), local_positions)

    local_vel = positions[1:] - positions[:-1]
    if cano_jts_pos:
        local_vel = qrot_np(np.repeat(root_rot_quat[:-1, None], num_joints, axis=1), local_vel)

    num_output_frames = num_frames - 1

    # --- Create Zero-filled representations for SMPL-X features ---
    zero_smplx_rot_6d = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_rot_6d']))
    zero_smplx_rot_vel = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_rot_vel']))
    zero_smplx_trans = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_trans']))
    zero_smplx_trans_vel = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_trans_vel']))
    zero_smplx_body_pose_6d = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_body_pose_6d']))
    zero_smplx_betas = np.zeros((num_output_frames, REPR_DIM_DICT['smplx_betas']))

    data_dict = {
        'root_rot_angle': root_rot_angle[:-1], 'root_rot_angle_vel': root_rot_angle_vel,
        'root_l_pos': root_l_pos[:-1, :2], 'root_l_vel': root_l_vel[:, :2],
        'root_height': root_height[:-1],
        'local_positions': local_positions[:-1].reshape(num_output_frames, -1),
        'local_vel': local_vel.reshape(num_output_frames, -1),
        'foot_contact': np.concatenate([feet_l, feet_r], axis=-1),
        'smplx_rot_6d': zero_smplx_rot_6d, 'smplx_rot_vel': zero_smplx_rot_vel,
        'smplx_trans': zero_smplx_trans, 'smplx_trans_vel': zero_smplx_trans_vel,
        'smplx_body_pose_6d': zero_smplx_body_pose_6d, 'smplx_betas': zero_smplx_betas,
    }

    return data_dict

def cano_seq_smplx(positions, smplx_params_dict, preset_floor_height=None, align_to_yaxis=True, return_transf_mat=False, smpl_model=None, device='cpu'):
    '''
    Perform canonicalization to the original motion sequence, such that:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    Use for AMASS and PROX (coordinate system z axis up)
    input:
        - positions: original joint positions (z axis up)
        - smplx_params_dict: original smplx params
        - preset_floor_height: if not None, the preset floor height
        - return_transf_mat: if True, also return the transf matrix for canonicalization
    Output:
        - cano_positions: canonicalized joint positions
        - cano_smplx_params_dict: canonicalized smplx params
        - transf_matrix (if return_transf_mat): the transf matrix for canonicalization
    '''
    assert not align_to_yaxis, "currently align_to_yaxis is not correct, it does not conform to smplx_model's forward equation"
    assert not align_to_yaxis, "currently align_to_yaxis is not correct, it does not conform to smplx_model's forward equation"
    ##### positions: z axis up
    cano_positions = positions.copy()
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

    ######################## Put on Floor
    if preset_floor_height:
        floor_height = preset_floor_height
    else:
        floor_height = cano_positions.min(axis=0).min(axis=0)[2]
    cano_positions[:, :, 2] -= floor_height  # z: up-axis, foot on ground
    # print(floor_height)

    ######################## transl such that XY for frame 0 is at origin
    root_pos_init = cano_positions[0]  # [22, 3]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    cano_positions = cano_positions - root_pose_init_xy

    if align_to_yaxis:
        ######################## transfrom such that frame 0 faces y+ axis
        joints_frame0 = cano_positions[0] # [N, 3] joints of first frame
        across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
        across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
        x_axis = across1 + across2
        x_axis[-1] = 0
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        transf_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)  # [3, 3]
        cano_positions = np.matmul(cano_positions, transf_rotmat)  # [T(/bs), 22, 3]

    ######################## canonicalization transf matrix for smpl params
    transf_matrix_1 = np.array([[1, 0, 0, -root_pose_init_xy[0]],
                                [0, 1, 0, -root_pose_init_xy[1]],
                                [0, 0, 1, -floor_height],
                                [0, 0, 0, 1]])
    transf_matrix_2 = np.zeros([4, 4])
    transf_matrix_2[0:3, 0:3] = transf_rotmat.transpose() if align_to_yaxis else np.identity(3)
    transf_matrix_2[-1, -1] = 1
    transf_matrix = np.matmul(transf_matrix_2, transf_matrix_1)
    cano_smplx_params_dict = update_globalRT_for_smplx(smplx_params_dict, transf_matrix,
                                                    delta_T=positions[:, 0]-smplx_params_dict['transl'])

    if not return_transf_mat:
        return cano_positions, cano_smplx_params_dict
    else:
        return cano_positions, cano_smplx_params_dict, transf_matrix


def cano_seq_smplx_egobody(positions, smplx_params_dict, preset_floor_height=None, return_transf_mat=False, smpl_model=None, device='cpu'):
    '''
    Perform canonicalization to the input motion sequence, similar as cano_seq_smplx,
    but the original input here is y-axis up, use for EgoBody (EgoBody scene coordinate system is y axis up)
    Output:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    '''
    ##### positions: y axis up
    cano_positions = positions.copy()
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

    ######################## Put on Floor
    if preset_floor_height:
        floor_height = preset_floor_height
    else:
        floor_height = cano_positions.min(axis=0).min(axis=0)[1]
    cano_positions[:, :, 1] -= floor_height
    # print(floor_height)

    ######################## transl to XZ at origin
    root_pos_init = cano_positions[0]  # [22, 3]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    cano_positions = cano_positions - root_pose_init_xz  # such that the x/z of first frame is 0

    ######################## transfrom to frame 0 face z+ axis
    joints_frame0 = cano_positions[0] # [N, 3] joints of first frame
    across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
    across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
    x_axis = across1 + across2
    # x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
    x_axis[1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    transf_rotmat = np.stack([x_axis, z_axis, y_axis], axis=1)  # [3, 3]
    transf_rotmat = -transf_rotmat # to make transf_rotmat a valid rotation matrix
    cano_positions = np.matmul(cano_positions, transf_rotmat)  # after this y axis down

    ########################## rotate such to make z axis up
    # rotate around x axis for -90 degrees
    trans_rot_x = np.array([[1, 0, 0],
                            [0, math.cos(-math.pi/2), -math.sin(-math.pi/2)],
                            [0, math.sin(-math.pi/2), math.cos(-math.pi/2)]])
    # rotate around z axis for 180 degrees
    trans_rot_z = np.array([[math.cos(math.pi), -math.sin(math.pi), 0],
                            [math.sin(math.pi), math.cos(math.pi), 0],
                            [0, 0, 1]])
    add_trans = np.matmul(trans_rot_z, trans_rot_x)
    cano_positions = np.matmul(cano_positions, add_trans.T) # after this z axis up

    ######################## canonicalization for smplx params
    transf_matrix_1 = np.array([[1, 0, 0, -root_pose_init_xz[0]],
                                [0, 1, 0, -floor_height],
                                [0, 0, 1, -root_pose_init_xz[2]],
                                [0, 0, 0, 1]])
    transf_matrix_2 = np.identity(4)
    transf_matrix_2[0:3, 0:3] = transf_rotmat.T
    transf_matrix_3 = np.identity(4)
    transf_matrix_3[0:3, 0:3] = add_trans
    transf_matrix = np.matmul(transf_matrix_3, np.matmul(transf_matrix_2, transf_matrix_1))
    cano_smplx_params_dict = update_globalRT_for_smplx(smplx_params_dict, transf_matrix,
                                                      # smplx_model=smpl_model, device=device,
                                                       delta_T=positions[:, 0]-smplx_params_dict['transl'],)

    if not return_transf_mat:
        return cano_positions, cano_smplx_params_dict
    else:
        return cano_positions, cano_smplx_params_dict, transf_matrix

def populate_repr_dict(repr_dict: Dict[str, torch.Tensor], smplx_model: smplx.SMPLX, recover_global:bool) -> None:
    """
    Populate the repr_dict.
    When recover_global is True, recover R_world_joints && R_root_joints from smplx_body_pose_6d. Otherwise, recover smplx_body_pose_rot_6d from R_world_joints.
    """
            
    if recover_global:

        assert 'smplx_body_pose_6d' in repr_dict, "When recover_global is True, 'smplx_body_pose_6d' must be present in repr_dict."
        assert 'smplx_rot_6d' in repr_dict, "When recover_global is True, 'smplx_rot_6d' must be present in repr_dict."

        bs, T, _ = repr_dict['smplx_body_pose_6d'].shape # Shape: bs, T, 21*6

        smplx_body_pose_6d_flat = repr_dict['smplx_body_pose_6d'] # bs, T, 21*6
        smplx_body_pose_rotmat = rot6d_to_rotmat(smplx_body_pose_6d_flat.reshape(bs * T, 21, 6)) # (bs*T
        smplx_body_pose_aa = rotation_matrix_to_angle_axis(smplx_body_pose_rotmat.reshape(bs * T, 21, 3, 3)).reshape(bs * T, -1) # (bs*T, 21*3)

        global_orient_rotmat = rot6d_to_rotmat(repr_dict['smplx_rot_6d']) # (bs, T, 3)
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_rotmat.reshape(bs * T, 3, 3)).reshape(bs * T, 3) # (bs*T, 3)

        J_transformed, R_world_joints, R_root_joints = local_to_global_pose_rotations(smplx_model=smplx_model, global_orient_aa=global_orient_aa.reshape(bs * T, 3), body_pose_aa=smplx_body_pose_aa, return_joint_positions=True) # (seq_len, 55, 3, 3)
        R_world_joints_22 = R_world_joints.reshape(bs, T, -1, 3, 3)[:, :, :22, :, :] # (bs, seq_len, 22, 3, 3)
        R_root_joints_22 = R_root_joints.reshape(bs, T, -1, 3, 3)[:, :, :22, :, :] # (bs, seq_len, 22, 3, 3)
        R_world_joints_rot6d_22 = rotmat_to_rot6d(R_world_joints_22)
        R_root_joints_rot6d_22 = rotmat_to_rot6d(R_root_joints_22)
        R_world_joints_rot6d_22_flat = R_world_joints_rot6d_22.reshape(bs, T, -1)
        R_root_joints_rot6d_22_flat = R_root_joints_rot6d_22.reshape(bs, T, -1)

        repr_dict['R_world_joints'] = R_world_joints_rot6d_22_flat # (bs, T, 22*6)
        repr_dict['R_root_joints'] = R_root_joints_rot6d_22_flat # (bs, T, 22*6)

    else:
        assert 'R_world_joints' in repr_dict, "When recover_global is False, 'R_world_joints' must be present in repr_dict."
        bs, T, _ = repr_dict['R_world_joints'].shape # Shape: bs, T, 22*6
        
        # Get predicted global rotations for all 22 joints from unnormalized representation
        grot_22j_rot6d_flat = repr_dict['R_world_joints'] # bs, T, 22*6
        grot_22j_rotmat = rot6d_to_rotmat(grot_22j_rot6d_flat.reshape(bs * T, 22, 6)) # (bs*T, 22, 3, 3)
        
        # Use the updated global_to_local_pose_rotations function
        # It expects (batch_size, num_joints, 3, 3)
        grot_aa, smplx_body_pose_aa = global_to_local_pose_rotations(
            smplx_model, # Pass the SMPLX model instance
            grot_22j_rotmat # Pass the (bs*T, 22, 3, 3) tensor
        )
        
        # derived_global_orient_aa is (bs*T, 3)
        # derived_smplx_body_pose_aa is (bs*T, 21*3)

        smplx_body_pose_rotmat = angle_axis_to_rotation_matrix(
            smplx_body_pose_aa.reshape(-1, 3) # (bs*T*21, 3)
        ).reshape(bs * T, smplx_model.NUM_BODY_JOINTS, 3, 3) # (bs*T, 21, 3, 3)
        smplx_body_pose_6d_flat = rotmat_to_rot6d(smplx_body_pose_rotmat).reshape(bs, T, -1) # (bs, T, 21*6)

        repr_dict['smplx_body_pose_6d'] = smplx_body_pose_6d_flat

    # Reconstruct local_positions if missing or contains non-finite placeholders (e.g., inf from pose sampling)
    local_pos = repr_dict.get('local_positions')
    needs_local_rebuild = (
        local_pos is None
        or (torch.is_tensor(local_pos) and not torch.isfinite(local_pos).all())
        or (isinstance(local_pos, np.ndarray) and not np.isfinite(local_pos).all())
    )
    if needs_local_rebuild:
        if smplx_model is None:
            raise ValueError("Cannot reconstruct local_positions without a smplx_model.")

        smpl_keys = ('smplx_rot_6d', 'smplx_body_pose_6d', 'smplx_trans', 'smplx_betas')
        missing_smpl = [k for k in smpl_keys if k not in repr_dict]
        if missing_smpl:
            raise ValueError(f"Cannot reconstruct local_positions; missing SMPL-X keys: {missing_smpl}")

        ref_tensor = next(
            (repr_dict[k] for k in smpl_keys if torch.is_tensor(repr_dict[k])),
            None,
        )
        if ref_tensor is not None:
            device = ref_tensor.device
            dtype = ref_tensor.dtype
        else:
            device = next(smplx_model.parameters()).device
            dtype = torch.float32

        smplx_input = {
            k: repr_dict[k].to(device=device, dtype=dtype) if torch.is_tensor(repr_dict[k])
            else torch.as_tensor(repr_dict[k], device=device, dtype=dtype)
            for k in smpl_keys
        }

        with torch.no_grad():
            joints = recover_from_repr_smpl(
                smplx_input,
                recover_mode='smplx_params',
                smplx_model=smplx_model,
                return_verts=False,
                return_full_joints=False,
            )
        if isinstance(joints, tuple):
            joints = joints[0]

        repr_dict['local_positions'] = joints.reshape(*joints.shape[:-2], -1).to(device=device, dtype=dtype)

    return repr_dict
        


def get_repr_smplx(positions, smplx_params_dict, feet_vel_thre=5e-5, cano_jts_pos=True, smplx_model=None, device=dist_util.dev(), modality_names=None):
    '''
    calculate the motion representation for input sequence
    input:
        - positions: input joint positions
        - smplx_params_dict: input smplx params
        - feet_vel_thre: velocity threshold for foot contact label
        - cano_jts_pos: whether to canonicalize joint positions
        - smplx_model: SMPL-X model (required for R_world_joints/R_root_joints)
        - device: computation device
        - modality_names: list of modality names to compute. If None, computes basic modalities only.
            Supported modalities:
            - Basic (always computed): smplx_rot_6d, smplx_trans, local_positions, smplx_body_pose_6d, smplx_betas, foot_contact
            - Velocity: smplx_rot_vel, smplx_trans_vel, local_vel, smplx_body_pose_vel
            - Global rotations (requires smplx_model): R_world_joints, R_root_joints, R_world_joints_vel, R_root_joints_vel
    Output:
        - data_dict: our motion representation (both traj and local pose)
    '''
    # Default modalities if not specified
    if modality_names is None:
        modality_names = ['smplx_rot_6d', 'smplx_trans', 'local_positions', 'smplx_body_pose_6d', 'smplx_betas', 'foot_contact']

    modality_set = set(modality_names)

    # Determine what to compute based on requested modalities
    need_velocity = any(m in modality_set for m in ['smplx_rot_vel', 'smplx_trans_vel', 'local_vel', 'smplx_body_pose_vel', 'R_world_joints_vel', 'R_root_joints_vel'])
    need_global_rotations = any(m in modality_set for m in ['R_world_joints', 'R_root_joints', 'R_world_joints_vel', 'R_root_joints_vel'])
    global_positions = positions.copy()

    """ Get Foot Contacts """
    feet_l, feet_r = foot_detect(positions, feet_vel_thre, up_axis='z')  # feet_l/feet_r: [seq_len, 2]

    ##################### joint-based repr #####################
    '''Get Forward Direction'''
    l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
    across1 = positions[:, r_hip] - positions[:, l_hip]
    across2 = positions[:, sdr_r] - positions[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]
    forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    '''Get Root Rotation and rotation velocity'''
    target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
    root_rot_quat = qbetween_np(forward, target)
    ### several frames have nan values
    if np.isnan(root_rot_quat).sum() > 0:
        frame_idx = np.where(np.isnan(root_rot_quat) == True)[0][0]
        root_rot_quat[frame_idx] = root_rot_quat[frame_idx - 1]
    if cano_jts_pos:
        '''Assume the `cano_seq_smplx` func has been called, so the first frame is aligned to y+ axis'''
        root_rot_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])

    '''local joint positions'''
    local_positions = positions.copy()  # [seq_len, 22, 3]
    if cano_jts_pos:
        local_positions[..., 0] -= local_positions[:, 0:1, 0]
        local_positions[..., 1] -= local_positions[:, 0:1, 1]
        '''for each frame, local pose face y+'''
        local_positions = qrot_np(np.repeat(root_rot_quat[:, None], local_positions.shape[1], axis=1), local_positions)

        if need_velocity:
            '''local joint velocity'''
            local_vel = qrot_np(np.repeat(root_rot_quat[:-1, None], global_positions.shape[1], axis=1),
                                global_positions[1:] - global_positions[:-1])
    else:
        if need_velocity:
            local_vel = global_positions[1:] - global_positions[:-1]

    ##################### smplx-based repr #####################
    '''smpl global_orient matrix'''
    smplx_rot_aa = smplx_params_dict['global_orient']   # [seq_len, 3]
    smplx_rot_mat = R.from_rotvec(smplx_rot_aa).as_matrix()  # [seq_len, 3, 3]
    smplx_rot_6d = smplx_rot_mat[..., :-1].reshape(-1, 6)  # [seq_len, 6]

    if need_velocity:
        '''smpl global_orient velocity'''
        dRdt = smplx_rot_mat[1:] - smplx_rot_mat[0:-1]  # [seq_len-1, 3, 3]
        smplx_rot_vel = estimate_angular_velocity_np(smplx_rot_mat[0:-1], dRdt)

    '''smpl transl and velocity'''
    smplx_trans = smplx_params_dict['transl']  # [seq_len, 3]
    if need_velocity:
        smplx_trans_vel = (smplx_trans[1:] - smplx_trans[:-1]).copy()

    '''smpl body pose'''
    smplx_body_pose_aa = smplx_params_dict['body_pose']  # [seq_len, 63]
    smplx_body_pose_mat = R.from_rotvec(smplx_body_pose_aa.reshape(-1, 3)).as_matrix().reshape(len(smplx_body_pose_aa), -1, 3, 3)  # [seq_len*21, 3, 3]
    smplx_body_pose_6d = smplx_body_pose_mat[..., :-1].reshape([smplx_body_pose_mat.shape[0], -1, 6])
    if need_velocity:
        '''smpl body pose orient velocity'''
        dRdt_body = smplx_body_pose_mat[1:] - smplx_body_pose_mat[0:-1]  # [seq_len-1, 21, 3, 3]
        smplx_body_pose_vel = estimate_angular_velocity_np(smplx_body_pose_mat[0:-1], dRdt_body)

    '''smpl shape'''
    smplx_betas = smplx_params_dict['betas']  # [seq_len, 10]

    T = smplx_trans.shape[0]  - 1 # total output frames is 1 frame less than absolute definitions of trajectory.

    if need_global_rotations and smplx_model is not None:
        with torch.no_grad():
            smplx_model.to(device)
            global_orient_t = torch.from_numpy(smplx_rot_aa).float().to(device)
            body_pose_t = torch.from_numpy(smplx_body_pose_aa).float().to(device)
            J_transformed, R_world_joints, R_root_joints = local_to_global_pose_rotations(smplx_model=smplx_model, global_orient_aa=global_orient_t, body_pose_aa=body_pose_t, return_joint_positions=True) # (seq_len, 55, 3, 3)
            R_world_joints_22 = R_world_joints[:, :22, :, :] # (seq_len, 22, 3, 3)
            R_root_joints_22 = R_root_joints[:, :22, :, :] # (seq_len, 22, 3, 3)
            R_world_joints_rot6d_22 = rotmat_to_rot6d(R_world_joints_22).cpu().numpy()
            R_root_joints_rot6d_22 = rotmat_to_rot6d(R_root_joints_22).cpu().numpy()
            R_world_joints_rot6d_22_flat = R_world_joints_rot6d_22[:T].reshape(T, -1)
            R_root_joints_rot6d_22_flat = R_root_joints_rot6d_22[:T].reshape(T, -1)
            if need_velocity:
                dRdt_R_world_joints = R_world_joints_22[1:] - R_world_joints_22[0:-1]
                dRdt_R_root_joints = R_root_joints_22[1:] - R_root_joints_22[0:-1]
                R_world_joints_vel_22 = estimate_angular_velocity(R_world_joints_22[0:-1], dRdt_R_world_joints)
                R_world_joints_vel_22_flat = R_world_joints_vel_22.reshape(T, -1).cpu().numpy()
                R_root_joints_vel_22 = estimate_angular_velocity(R_root_joints_22[0:-1], dRdt_R_root_joints)
                R_root_joints_vel_22_flat = R_root_joints_vel_22.reshape(T, -1).cpu().numpy()

    ################### final full body repr #####################
    data_dict = { 'smplx_rot_6d': smplx_rot_6d[0:-1],
                 'smplx_trans': smplx_trans[0:-1],
                 'local_positions': local_positions[0:-1].reshape(T, -1),
                 'smplx_body_pose_6d': smplx_body_pose_6d[0:-1].reshape(T, -1),
                 'smplx_betas': smplx_betas[0:-1],
                 'foot_contact': np.concatenate([feet_l, feet_r], axis=-1),
                 }
    if need_velocity:
        data_dict['smplx_rot_vel'] = smplx_rot_vel
        data_dict['smplx_trans_vel'] = smplx_trans_vel
        data_dict['local_vel'] = local_vel.reshape(T, -1)
        if not need_global_rotations:
            data_dict['smplx_body_pose_vel'] = smplx_body_pose_vel.reshape(T, -1)

    if need_global_rotations and smplx_model is not None:
        data_dict['R_world_joints'] = R_world_joints_rot6d_22_flat
        data_dict['R_root_joints'] = R_root_joints_rot6d_22_flat
        if need_velocity:
            data_dict['R_world_joints_vel'] = R_world_joints_vel_22_flat
            data_dict['R_root_joints_vel'] = R_root_joints_vel_22_flat

    debug = False
    if debug:
        smplx_params_dict_torch = {k: torch.from_numpy(v).to(device, dtype=torch.float32) for k, v in smplx_params_dict.items()}
        seq_len = smplx_params_dict_torch['global_orient'].shape[0]
        # NOTE: it has been verified that smplx_params_dict passed through smplx_model fk layer can recover positions. Jul15.
        output = smplx_model(betas=smplx_params_dict_torch['betas'], global_orient=smplx_params_dict_torch['global_orient'], body_pose=smplx_params_dict_torch['body_pose'], transl=smplx_params_dict_torch['transl'], jaw_pose=torch.zeros(seq_len, 3).to(device, dtype=torch.float32), left_hand_pose=torch.zeros(seq_len, 45).to(device, dtype=torch.float32), right_hand_pose=torch.zeros(seq_len, 45).to(device, dtype=torch.float32), leye_pose=torch.zeros(seq_len, 3).to(device, dtype=torch.float32), reye_pose=torch.zeros(seq_len, 3).to(device, dtype=torch.float32), expression=torch.zeros(seq_len, 10).to(device, dtype=torch.float32), return_verts=True, return_full_pose=True)
        from utils.vis_util import visualize_temporal_sequence
        from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
        visualize_temporal_sequence(output.vertices[:, ::3, :].numpy(force=True))
        _R_world_joints_22 = einsum(R_world_joints_22[:, 0:1, :, :], R_root_joints_22, 'seq_len idx_root i j, seq_len num_jts j k -> seq_len num_jts i k')
        # NOTE: this is confirmed that the R_world_joints_22 && _R_world_joints_22 are correct and are same. Jul 15
        cfg = SmplxVisConfig(window_title="SMPL-X Debug", num_repeats=1, fixed_cam=False, camera_follow=True)
        visualize_smplx_sequence(
            joints=J_transformed[:, :22, :].unsqueeze(0),
            vertices=output.vertices.unsqueeze(0),
            faces=smplx_model.faces,
            parent_indices=SMPLH_KINTREE,
            batch_idx=0,
            cfg=cfg,
        )

    return data_dict


def recover_root_rot_pos(data, root_traj_repr='abs', up_axis='y', cano_jts_pos=True):
    '''
    Recover joint-based trajectory (root linear position and rotation) from full motion representation
    input:
        - data: motion representation
        - root_traj_repr: 'abs'/'rel', absolute or relative joint-based trajectory representation
        - up_axis: y/z
    output:
        - r_rot_quat: joint-based root rotation
        - r_pos: joint-based root linear position
    '''
    if up_axis == 'y':
        up_axis_dim, face_axis_dim, quat_sin_dim = 1, 2, 2
    elif up_axis == 'z':
        up_axis_dim, face_axis_dim, quat_sin_dim = 2, 1, 3
    else:
        print('[ERROR] up_axis not setup correctly.')
        exit()

    if root_traj_repr == 'abs':
        r_rot_ang = data[..., 0]
        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., quat_sin_dim] = torch.sin(r_rot_ang)
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., [0, face_axis_dim]] = data[..., 1:3]
        r_pos[..., up_axis_dim] = data[..., 3]
    elif root_traj_repr == 'rel':
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get up-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., quat_sin_dim] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, face_axis_dim]] = data[..., :-1, 1:3]
        if cano_jts_pos:
            '''Add Y-axis rotation to root position'''
            r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., up_axis_dim] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_repr_smpl(data_dict, recover_mode='joint_abs_traj', cano_jts_pos=True, smplx_model=None, return_verts=False, return_full_joints=False):
    '''
    Recover joint/vertices positions from the motion representation
    input:
        - data_dict: motion representation
        - recover_mode:
            joint_abs_traj: recover from joint-based repr using absolute joint-based traj repr
            joint_rel_traj: recover from joint-based repr using relative joint-based traj repr
            smplx_params: recover from smplx-based repr
        - return_verts: return body vertices
        - return_full_joints: return all 127 smplx joints
    output:
        - smplx_joints: smplx joints, only include 22 main body joints if not return_full_joints
        - smplx_verts: smplx vertices
    '''
    if recover_mode not in ['joint_abs_traj', 'joint_rel_traj', 'smplx_params']:
        print('[ERROR] recover_mode incorrect! in func recover_from_repr_smpl()')
    if recover_mode[0:5] == 'joint':
        root_height = data_dict['root_height']
        if recover_mode == 'joint_abs_traj':
            root_rot_angle = data_dict['root_rot_angle']
            root_l_pos = data_dict['root_l_pos']
            root_traj_repr = torch.cat([root_rot_angle, root_l_pos, root_height], axis=-1)  # [..., 4]
            r_rot_quat, r_pos = recover_root_rot_pos(data=root_traj_repr, root_traj_repr='abs', up_axis='z', cano_jts_pos=cano_jts_pos)
        elif recover_mode == 'joint_rel_traj':
            root_rot_angle_vel = data_dict['root_rot_angle_vel']
            root_l_vel = data_dict['root_l_vel']
            root_traj_repr = torch.cat([root_rot_angle_vel, root_l_vel, root_height], axis=-1)  # [..., 4]
            r_rot_quat, r_pos = recover_root_rot_pos(data=root_traj_repr, root_traj_repr='rel', up_axis='z', cano_jts_pos=cano_jts_pos)

        if cano_jts_pos:
            positions = data_dict['local_positions'][..., 3:]  # [..., 22*3]
            positions = positions.view(positions.shape[:-1] + (-1, 3))  # [..., 22, 3]
            '''Add up-axis rotation to local joints'''
            positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
            '''Add root transl (projected on ground plane) to joints'''
            positions[..., 0] += r_pos[..., 0:1]
            positions[..., 1] += r_pos[..., 1:2]
            '''Concate root and joints'''
            positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        else:
            positions = data_dict['local_positions']  # [..., 22*3]
            positions = positions.view(positions.shape[:-1] + (-1, 3))  # [..., 22, 3]

        return positions

    elif recover_mode == 'smplx_params':
        bs = len(data_dict['smplx_rot_6d'])
        global_orient_mat = rot6d_to_rotmat(data_dict['smplx_rot_6d'].reshape(-1, 6))  # [bs*T, 3, 3]
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat) # [bs*T, 3]
        body_pose_mat = rot6d_to_rotmat(data_dict['smplx_body_pose_6d'].reshape(-1, 6))  # [bs*T*21, 3, 3]
        body_pose_aa = rotation_matrix_to_angle_axis(body_pose_mat).reshape(-1, 21, 3)  # [bs*T, 21, 3]
        smplx_params_dict = {'transl': data_dict['smplx_trans'].reshape(-1, 3),
                             'global_orient': global_orient_aa,
                             'body_pose': body_pose_aa.reshape(-1, 63),
                             'betas': data_dict['smplx_betas'].reshape(-1, 10),
                             'jaw_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'leye_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'reye_pose': torch.zeros(len(global_orient_aa), 3).to(global_orient_aa.device),
                             'left_hand_pose': torch.zeros(len(global_orient_aa), 45).to(global_orient_aa.device),
                             'right_hand_pose': torch.zeros(len(global_orient_aa), 45).to(global_orient_aa.device),
                             'expression': torch.zeros(len(global_orient_aa), 10).to(global_orient_aa.device)}
        smplx_output = smplx_model(**smplx_params_dict)
        joints = smplx_output.joints
        if return_full_joints:
            smplx_joints = joints.reshape(bs, -1, joints.shape[1], 3)
        else:
            smplx_joints = joints[:, 0:22].reshape(bs, -1, min(22, joints.shape[1]), 3)
        if return_verts and hasattr(smplx_output, 'vertices') and smplx_output.vertices is not None:
            verts = smplx_output.vertices
            smplx_verts = verts.reshape(bs, -1, verts.shape[1], 3)
            return smplx_joints, smplx_verts
        else:
            return smplx_joints


def rotate_repr_smpl(repr_dict, rotation_matrix):
    """
    Apply rotation to motion representation dictionary.
    
    Args:
        repr_dict: Dictionary containing motion representation data
        rotation_matrix: 3x3 rotation matrix as torch.Tensor
    """
    # Get the shape prefix (batch dimensions before the last dimension)
    *batch_dims, _ = repr_dict['smplx_trans'].shape
    
    J = repr_dict['local_positions'].view(*batch_dims, 22, 3) # [..., 22, 3]
    # root offset before rotation.
    J_root_ = J[..., 0:1, :] - repr_dict['smplx_trans'].unsqueeze(-2)  # [..., 1, 3]
    RJ_root_ = torch.matmul(J_root_, rotation_matrix.transpose(-1, -2))  # [..., 1, 3]
    RJ = torch.matmul(J, rotation_matrix.transpose(-1, -2))  # [..., 22, 3]
    new_J = RJ + J_root_ - RJ_root_ # [..., 22, 3]
    repr_dict['local_positions'] = new_J.view(*batch_dims, 22 * 3)

    rotated_smplx_rot_6d = rotate_6d_repr(repr_dict['smplx_rot_6d'], rotation_matrix).reshape(*batch_dims, 6)

    # Apply rotation to smplx_trans [..., 3]
    repr_dict['smplx_trans'] = torch.matmul(repr_dict['smplx_trans'], rotation_matrix.transpose(-1, -2))
    # # Apply rotation to smplx_rot_6d
    repr_dict['smplx_rot_6d'] = rotated_smplx_rot_6d

    return repr_dict

def rotate_repr_smpl_2(repr_dict, rotation_matrix):
    """
    Apply rotation to motion representation dictionary.
    
    Args:
        repr_dict: Dictionary containing motion representation data
        rotation_matrix: 3x3 rotation matrix as torch.Tensor
    """
    # Get the shape prefix (batch dimensions before the last dimension)
    *batch_dims, _ = repr_dict['smplx_trans'].shape
    
    J = repr_dict['local_positions'].view(*batch_dims, 22, 3) # [..., 22, 3]
    # root offset before rotation.
    J_root_ = J[..., 0:1, :] - repr_dict['smplx_trans'].unsqueeze(-2)  # [..., 1, 3]
    RJ_root_ = torch.matmul(J_root_, rotation_matrix.transpose(-1, -2))  # [..., 1, 3]
    RJ = torch.matmul(J, rotation_matrix.transpose(-1, -2))  # [..., 22, 3]

    new_J = torch.matmul(J, rotation_matrix.transpose(-1, -2))
    repr_dict['local_positions'] = new_J.view(*batch_dims, 22 * 3)

    rotated_smplx_rot_6d = rotate_6d_repr(repr_dict['smplx_rot_6d'], rotation_matrix).reshape(*batch_dims, 6)

    # Apply rotation to smplx_trans [..., 3]
    repr_dict['smplx_trans'] = RJ_root_.squeeze(-2) - J_root_.squeeze(-2) + torch.matmul(repr_dict['smplx_trans'], rotation_matrix.transpose(-1, -2))
    # Apply rotation to smplx_rot_6d
    repr_dict['smplx_rot_6d'] = rotated_smplx_rot_6d

    return repr_dict


def transform_repr_smpl(repr_dict, transform: torch.Tensor):
    """
    Apply a 4x4 homogeneous transform to the motion representation.

    Args:
        repr_dict: Dictionary containing motion representation data.
        transform: 4x4 homogeneous transform (torch.Tensor) or batched [..., 4, 4].
                   Transform is applied as p' = R p + t for positions and to rotations via R.
    """
    # Determine batch prefix from smplx_trans: [..., 3]
    *batch_dims, _ = repr_dict['smplx_trans'].shape
    device = repr_dict['smplx_trans'].device

    # Ensure transform is a torch tensor on the correct device
    transform = transform.to(device)

    # Broadcast transform to batch dims if necessary
    if transform.dim() == 2:
        transform = transform.view(*(1,) * len(batch_dims), 4, 4).expand(*batch_dims, 4, 4)
    else:
        # If provided transform has fewer leading dims, broadcast to batch_dims
        if transform.shape[: -2] != tuple(batch_dims):
            transform = transform.view(*(1,) * len(batch_dims), 4, 4).expand(*batch_dims, 4, 4)

    # Extract rotation R and translation t
    R = transform[..., :3, :3]          # [..., 3, 3]
    # Note: t shape is [..., 3]
    t = transform[..., :3, 3]

    # --- Transform joint positions using homogeneous coordinates ---
    J = repr_dict['local_positions'].view(*batch_dims, 22, 3)  # [..., 22, 3]
    # root offset before rotation.
    J_root_ = J[..., 0:1, :] - repr_dict['smplx_trans'].unsqueeze(-2)  # [..., 1, 3]
    RJ_root_ = torch.einsum('...ij,...jk->...ik', J_root_, R.transpose(-2, -1))  # [..., 1, 3]
    RJ = torch.einsum('...ij,...jk->...ik', J, R.transpose(-2, -1))  # [..., 22, 3]
    new_J = RJ + J_root_ - RJ_root_ + t.unsqueeze(-2) # [..., 22, 3]
    repr_dict['local_positions'] = new_J.reshape(*batch_dims, 22 * 3)

    # --- Transform root translation (smplx_trans) using homogeneous coordinates ---
    trans = repr_dict['smplx_trans']  # [..., 3]
    ones_t = torch.ones(*batch_dims, 1, device=device, dtype=trans.dtype)  # [..., 1]
    trans_h = torch.cat([trans, ones_t], dim=-1)  # [..., 4]
    new_trans_h = torch.einsum('...i,...ij->...j', trans_h, transform.transpose(-2, -1))  # [..., 4]
    repr_dict['smplx_trans'] = new_trans_h[..., :3]

    # --- Rotate 6D root rotation representation using R (upper-left 3x3) ---
    # rotate_6d_repr expects the rotation matrix (3x3 or batched), so pass R
    rotated_smplx_rot_6d = rotate_6d_repr(repr_dict['smplx_rot_6d'], R)
    repr_dict['smplx_rot_6d'] = rotated_smplx_rot_6d.reshape(*batch_dims, 6)

    return repr_dict
