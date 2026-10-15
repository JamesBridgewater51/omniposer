import numpy as np
import cv2
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import datetime
import os, json, sys
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from enum import IntEnum
from pathlib import Path
import subprocess

TRANS_ROT_MASK_RATIO = 0.65

MASK_SCHEME_CHOICES = ['lower', 'upper', 'full', 'random', 'head_only', 'head_with_two_hands', 'head_with_two_hands_and_two_feets']
TEST_TIME_MASK_SCHEME_CHOICES = ['random', 'head_only', 'head_with_two_hands', 'head_with_two_hands_and_two_feets']

####### configs for motion representation: joint-based + smplx_based
REPR_LIST = ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel', 'root_height', # joint-based traj
             'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel',  # smplx-based traj
             'local_positions', 'local_vel',  # joint-based local pose
             'smplx_body_pose_6d',  # smplx-based local pose
             'smplx_betas',  # smplx body shape
             'foot_contact', ]  # [foot contact labels]
# dimension for each categody of the motion representation
REPR_DIM_DICT = {'root_rot_angle': 1,
                 'root_rot_angle_vel': 1,
                 'root_l_pos': 2,
                 'root_l_vel': 2,
                 'root_height': 1,
                 'smplx_rot_6d': 6,
                 'smplx_rot_vel': 3,
                 'smplx_trans': 3,
                 'smplx_trans_vel': 3, # end of trajNet input
                 'local_positions': 22 * 3,
                 'local_vel': 22 * 3,
                 'R_world_joints': 22 * 6,
                 'R_root_joints': 22 * 6,
                 'R_world_joints_vel': 22 * 3,
                 'R_root_joints_vel': 22 * 3,
                 'smplx_body_pose_6d': 21 * 6,
                 'smplx_body_pose_vel': 21 * 3,
                 'smplx_betas': 10,
                 'foot_contact': 4, } # end of PostNet input.

# dimension of each joint-specific modality of the motion repr.
JOINT_REPR_DIM_DICT = {
    'local_positions' : 3,
    'local_vel': 3,
    'R_world_joints': 6,
    'R_root_joints': 6,
    'R_world_joints_vel': 3,
    'R_root_joints_vel': 3,
    'smplx_body_pose_6d': 6,
    'smplx_body_pose_vel': 3
}

# estimated floor height from scene mesh
# z axis up
prox_floor_height = {'N0Sofa': -0.9843093165454873,
                     'MPH1Library': -0.34579620031341207,
                     'N3Library': -0.6736229583361132,
                     'N3Office': -0.7772727989022952,
                     'BasementSittingBooth': -0.767080139846674,
                     'MPH8': -0.41432886722717904,
                     'MPH11': -0.7169139211234009,
                     'MPH16': -0.8408992040141058,
                     'MPH112': -0.6419028605753081,
                     'N0SittingBooth': -0.6677103008966809,
                     'N3OpenArea': -1.0754909672969915,
                     'Werkraum': -0.6777057869851316}

# y axis up
egobody_floor_height = {'seminar_g110': -1.660,
                        'seminar_d78': -0.810,
                        'seminar_j716': -0.8960,
                        'seminar_g110_0315': -0.73,
                        'seminar_d78_0318': -1.03,
                        'seminar_g110_0415': -0.77}

LIMBS_BODY_SMPL = [(15, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         # (20, 22),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         # (21, 23),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11),]

SMPLH_KINTREE = [
                -1,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                9,
                9,
                12,
                13,
                14,
                16,
                17,
                18,
                19,
            ]  # Parent indices for each of 22 SMPLH joints

class SMPLJointID(IntEnum):
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    MAX_SMPL_JOINT = 22

class SMPLXJointID(IntEnum):
    # --- Body (0-21) Same as SMPL ---
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21

    # --- Face (22-24) Derived from FLAME ---
    JAW = 22
    LEFT_EYE = 23
    RIGHT_EYE = 24

    # --- Left Hand (25-39) Derived from MANO ---
    # Note: the MANO finger order is Index, Middle, Pinky, Ring, Thumb.
    # This is counter-intuitive; take care with the ordering.
    
    # Index Finger
    LEFT_INDEX1 = 25
    LEFT_INDEX2 = 26
    LEFT_INDEX3 = 27
    
    # Middle Finger
    LEFT_MIDDLE1 = 28
    LEFT_MIDDLE2 = 29
    LEFT_MIDDLE3 = 30
    
    # Pinky Finger
    LEFT_PINKY1 = 31
    LEFT_PINKY2 = 32
    LEFT_PINKY3 = 33
    
    # Ring Finger
    LEFT_RING1 = 34
    LEFT_RING2 = 35
    LEFT_RING3 = 36
    
    # Thumb
    LEFT_THUMB1 = 37
    LEFT_THUMB2 = 38
    LEFT_THUMB3 = 39

    # --- Right Hand (40-54) Derived from MANO ---
    # Same order as the left hand.
    
    # Index Finger
    RIGHT_INDEX1 = 40
    RIGHT_INDEX2 = 41
    RIGHT_INDEX3 = 42
    
    # Middle Finger
    RIGHT_MIDDLE1 = 43
    RIGHT_MIDDLE2 = 44
    RIGHT_MIDDLE3 = 45
    
    # Pinky Finger
    RIGHT_PINKY1 = 46
    RIGHT_PINKY2 = 47
    RIGHT_PINKY3 = 48
    
    # Ring Finger
    RIGHT_RING1 = 49
    RIGHT_RING2 = 50
    RIGHT_RING3 = 51
    
    # Thumb
    RIGHT_THUMB1 = 52
    RIGHT_THUMB2 = 53
    RIGHT_THUMB3 = 54
    
    # End of joints
    # SMPL-X joints typically come as (Batch, 55, 3) or (Batch, 165).
    MAX_SMPLX_JOINT = 55

SMPLJointVisProb_EgoexoLike = np.zeros((SMPLJointID.MAX_SMPL_JOINT,), dtype=np.float32)

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

COCO_JOINT_NAMES = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
]


LIMBS_BODY_COCO = [
    # Face connections
    (0, 1),   # nose -> left-eye
    (0, 2),   # nose -> right-eye
    (1, 3),   # left-eye -> left-ear
    (2, 4),   # right-eye -> right-ear
    
    # Upper body (shoulders, arms)
    (5, 6),   # left-shoulder -> right-shoulder (torso)
    (5, 7),   # left-shoulder -> left-elbow
    (7, 9),   # left-elbow -> left-wrist
    (6, 8),   # right-shoulder -> right-elbow
    (8, 10),  # right-elbow -> right-wrist
    
    # Lower body (hips, legs)
    (11, 12), # left-hip -> right-hip (pelvis)
    (5, 11),  # left-shoulder -> left-hip (side torso)
    (6, 12),  # right-shoulder -> right-hip (side torso)
    (11, 13), # left-hip -> left-knee
    (13, 15), # left-knee -> left-ankle
    (12, 14), # right-hip -> right-knee
    (14, 16), # right-knee -> right-ankle
]

COCO_KINTREE =[
                -1,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                12,
                11,
                11,
                12,
                13,
                14,
            ]

# SMPL joint sets for masking schemes
HEAD_ID_SET = {15}  # head
HANDS_IDS_SET = {20, 21}  # SMPL wrist joints (L_Wrist, R_Wrist) - often referred to as hands
FEET_IDS_SET = {10, 11}  # SMPL foot joints (L_Foot, R_Foot)

NECK_ID_SET = {12}
SPINE_IDS_SET = {3, 6, 9}  # Spine1, Spine2, Spine3
PELVIS_ID_SET = {0} # Often considered the root
COLLAR_IDS_SET = {13, 14}  # L_Collar, R_Collar
SHOULDER_IDS_SET = {16, 17}  # L_Shoulder, R_Shoulder
ELBOW_IDS_SET = {18, 19}  # L_Elbow, R_Elbow
HIP_IDS_SET = {1, 2}  # L_Hip, R_Hip
KNEE_IDS_SET = {4, 5}  # L_Knee, R_Knee
ANKLE_IDS_SET = {7, 8}  # L_Ankle, R_Ankle

# Individual joint IDs based on SMPL_JOINT_NAMES order
# Pelvis (root) is 0
LEFT_HIP_ID = 1
RIGHT_HIP_ID = 2
LEFT_KNEE_ID = 4
RIGHT_KNEE_ID = 5
LEFT_ANKLE_ID = 7
RIGHT_ANKLE_ID = 8
LEFT_FOOT_ID = 10
RIGHT_FOOT_ID = 11
# Neck is 12
# Head is 15
LEFT_SHOULDER_ID = 16
RIGHT_SHOULDER_ID = 17
LEFT_ELBOW_ID = 18
RIGHT_ELBOW_ID = 19
LEFT_WRIST_ID = 20 # Referred to as left hand
RIGHT_WRIST_ID = 21 # Referred to as right hand


# Individual joint ID sets
LEFT_HIP_ID_SET = {LEFT_HIP_ID}
RIGHT_HIP_ID_SET = {RIGHT_HIP_ID}
LEFT_FOOT_ID_SET = {LEFT_FOOT_ID}
RIGHT_FOOT_ID_SET = {RIGHT_FOOT_ID}
LEFT_SHOULDER_ID_SET = {LEFT_SHOULDER_ID}
RIGHT_SHOULDER_ID_SET = {RIGHT_SHOULDER_ID}
LEFT_HAND_ID_SET = {LEFT_WRIST_ID} # Using wrist as hand
RIGHT_HAND_ID_SET = {RIGHT_WRIST_ID} # Using wrist as hand
LEFT_KNEE_ID_SET = {LEFT_KNEE_ID}
RIGHT_KNEE_ID_SET = {RIGHT_KNEE_ID}
LEFT_ANKLE_ID_SET = {LEFT_ANKLE_ID}
RIGHT_ANKLE_ID_SET = {RIGHT_ANKLE_ID}
LEFT_ELBOW_ID_SET = {LEFT_ELBOW_ID}
RIGHT_ELBOW_ID_SET = {RIGHT_ELBOW_ID}


# Assuming 22 SMPL joints (0-21) based on SMPL_JOINT_NAMES
ALL_SMPL_JOINT_IDS_SET = set(range(len(SMPL_JOINT_NAMES)))


def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def save_omegaconf_config(logdir, config):
    """
    Save OmegaConf configuration to logdir.
    
    Args:
        logdir: Path to save directory.
        config: OmegaConf DictConfig object.
    """
    from omegaconf import OmegaConf
    
    # Save as YAML
    yaml_path = os.path.join(logdir, "config.yaml")
    print(f"[*] CONFIG path: {yaml_path}")
    with open(yaml_path, 'w') as fp:
        OmegaConf.save(config, fp)
    
    # Also save as JSON for backward compatibility
    json_path = os.path.join(logdir, "params.json")
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(json_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)


def load_config_from_logdir(logdir):
    """
    Load configuration from a previous training run's logdir.
    
    Args:
        logdir: Path to the logdir containing params.json
        
    Returns:
        dict: Dictionary containing the loaded configuration, or None if file doesn't exist
    """
    param_path = os.path.join(logdir, "params.json")
    if not os.path.exists(param_path):
        print(f"[WARNING] Config file not found at {param_path}")
        return None
    
    print(f"[*] Loading config from: {param_path}")
    with open(param_path, 'r') as fp:
        config_dict = json.load(fp)
    return config_dict


def load_omegaconf_from_logdir(logdir: str):
    """
    Load OmegaConf configuration from a previous training run's logdir.
    
    Prioritizes config.yaml (OmegaConf native format) over params.json.
    
    Args:
        logdir: Path to the logdir containing config.yaml or params.json
        
    Returns:
        DictConfig: OmegaConf DictConfig object
    """
    from omegaconf import OmegaConf, DictConfig
    
    # Try loading config.yaml first (Hydra/OmegaConf native format)
    yaml_path = os.path.join(logdir, "config.yaml")
    if os.path.exists(yaml_path):
        print(f"[*] Loading OmegaConf config from: {yaml_path}")
        config = OmegaConf.load(yaml_path)
        return config
    
    # Fallback to params.json and convert to OmegaConf
    json_path = os.path.join(logdir, "params.json")
    if os.path.exists(json_path):
        print(f"[*] Loading config from JSON (legacy): {json_path}")
        with open(json_path, 'r') as fp:
            config_dict = json.load(fp)
        return OmegaConf.create(config_dict)
    
    raise FileNotFoundError(f"No config file found in {logdir}. Expected config.yaml or params.json.")

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                             np.asarray(cam['k']))[0].squeeze()

def perspective_projection(points,
                           # translation,
                           focal_length,
                           camera_center=None,
                           rotation=None):
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    # points = points + translation.unsqueeze(1)
    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]



def update_globalRT_for_smplx(body_param_dict, trans_to_target_origin, smplx_model=None, device=None, delta_T=None):
    '''
    input:
        body_param_dict:
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
        delta_T: pelvis location?
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    bs = len(body_param_dict['transl'])

    if delta_T is None:
        body_param_dict_torch = {}
        for key in body_param_dict.keys():
            body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(device)
        body_param_dict_torch['transl'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)

        body_param_dict_torch['jaw_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['leye_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['reye_pose'] = torch.zeros(bs, 3).to(device)
        body_param_dict_torch['left_hand_pose'] = torch.zeros(bs, 45).to(device)
        body_param_dict_torch['right_hand_pose'] = torch.zeros(bs, 45).to(device)
        body_param_dict_torch['expression'] = torch.zeros(bs, 10).to(device)

        smpl_out = smplx_model(**body_param_dict_torch)
        delta_T = smpl_out.joints[:,0,:] # (bs, 3,)
        delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict['global_orient']
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [bs, 3,3] rotation mat
    body_T = body_param_dict['transl']
    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1,:-1] = body_R_mat
    body_mat[:, :-1, -1] = body_T + delta_T
    body_mat[:, -1, -1] = 1

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    trans_to_target_origin = np.expand_dims(trans_to_target_origin, axis=0)  # [1, 4]
    trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)  # [bs, 4]

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    body_R_new = R.from_matrix(body_mat_new[:, :-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:, :-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(-1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(-1,3)
    return body_params_dict_new


def estimate_angular_velocity(rot_seq, dRdt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    # dRdt = self.estimate_linear_velocity(rot_seq, h)

    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], dim=-1)  # [B, T, ..., 3]
    return w


def estimate_angular_velocity_np(rot_seq, dRdt):
    # rot_seq: [B, T, ..., 3, 3]
    # dRdt: [B, T, ..., 3, 3]
    R = rot_seq
    RT = np.swapaxes(R, -2, -1)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)  # [B, T, ..., 3]
    return w


def rotation_geodesic_distance_np(R1, R2):
    """Compute the angular distance between two rotation matrices.
    Supports inputs with arbitrary dimensions ending in (3, 3), as long as R1 and R2 have the same shape."""
    R_rel = np.matmul(np.swapaxes(R1, -2, -1), R2)
    trace = np.clip((np.trace(R_rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)  # Clipping for numerical stability
    angle = np.arccos(trace)
    return angle

def rotation_geodesic_distance_torch(R1, R2):
    """Compute the angular distance between two rotation matrices.
    Supports inputs with arbitrary dimensions ending in (3, 3), as long as R1 and R2 have the same shape."""
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)
    trace = torch.clamp((torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)  # Clamping for numerical stability
    angle = torch.acos(trace)
    return angle

def ipdb_safety_net():
    import signal
    import traceback as tb
    """Attaches a "safety net" for unexpected errors in a Python script.

    When called, PDB will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.
    """
    ipdb = __import__("ipdb")

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        ipdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        ipdb.post_mortem(traceback)

    sys.excepthook = excepthook


def get_git_commit_hash(cwd: Path | None = None) -> str:
    """Returns the current Git commit hash."""
    if cwd is None:
        cwd = Path.cwd()
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )


def get_git_diff(cwd: Path | None = None) -> str:
    """Returns the output of `git diff HEAD`."""
    if cwd is None:
        cwd = Path.cwd()
    try:
        output_bytes = subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd.as_posix())
        output = output_bytes.decode("ascii")
        return output.strip()
    except UnicodeDecodeError as e:
        # Try decoding as utf-8 to find non-ascii chars and print their locations
        output_bytes = subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd.as_posix())
        try:
            output = output_bytes.decode("utf-8")
        except UnicodeDecodeError as e_utf8:
            # Even utf-8 failed, so fallback to replace mode
            output = output_bytes.decode("utf-8", errors="replace")
        # Print out the non-ascii characters or their positions
        non_ascii = [c for c in output if ord(c) > 127]
        if non_ascii:
            print("Non-ASCII characters in git diff detected:")
            print("".join(non_ascii))
        else:
            print("Non-ASCII characters could not be decoded but none found via utf-8 fallback.")
        # Return the output still, possibly with ? in place of invalid chars
        return output.strip()