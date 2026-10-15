from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_sets.motion_representation import *
import pickle as pkl
from utils.representation_registry import RepresentationRegistry
import torch.distributed as dist
from typing import Literal
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from .dataset_amass_like import DatasetAMASSLike

RICH_TRAIN_SEQ_NAMES = [
    'BBQ_001_guitar',
    'BBQ_001_juggle',
    'LectureHall_018_wipingchairs1',
    'LectureHall_018_wipingspray1',
    'LectureHall_020_wipingtable1',
    'ParkingLot1_002_burpee3',
    'ParkingLot1_002_overfence1',
    'ParkingLot1_002_overfence2',
    'ParkingLot1_002_pushup1',
    'ParkingLot1_002_stretching1',
    'ParkingLot1_004_005_greetingchattingeating1',
    'ParkingLot1_004_burpeejump1',
    'ParkingLot1_004_eating1',
    'ParkingLot1_004_phonetalk1',
    'ParkingLot1_004_pushup2',
    'ParkingLot1_004_takingphotos1',
    'ParkingLot1_005_burpeejump2',
    'ParkingLot1_005_overfence1',
    'ParkingLot1_005_pushup2',
    'ParkingLot1_005_pushup3',
    'ParkingLot1_007_eating1',
    'ParkingLot1_007_eating2',
    'ParkingLot1_007_overfence2',
    'ParkingLot2_008_burpeejump1',
    'ParkingLot2_008_eating1',
    'ParkingLot2_008_overfence1',
    'ParkingLot2_008_overfence2',
    'ParkingLot2_008_overfence3',
    'ParkingLot2_008_phonetalk1',
    'ParkingLot2_008_pushup1',
    'ParkingLot2_008_pushup2',
    'ParkingLot2_014_burpeejump1',
    'ParkingLot2_014_burpeejump2',
    'ParkingLot2_014_overfence3',
    'ParkingLot2_014_phonetalk2',
    'ParkingLot2_014_pushup2',
    'ParkingLot2_014_takingphotos2',
    'ParkingLot2_015_burpeejump2',
    'ParkingLot2_015_eating2',
    'ParkingLot2_015_overfence1',
    'ParkingLot2_015_pushup1',
    'ParkingLot2_016_burpeejump2',
    'ParkingLot2_016_overfence2',
    'ParkingLot2_016_pushup1',
    'ParkingLot2_016_pushup2',
    'ParkingLot2_016_stretching1',
    'Pavallion_000_phonesiteat',
    'Pavallion_000_plankjack',
    'Pavallion_000_sidebalancerun',
    'Pavallion_000_yoga2',
    'Pavallion_002_phonesiteat',
    'Pavallion_002_plankjack',
    'Pavallion_003_018_tossball',
    'Pavallion_003_phonesiteat',
    'Pavallion_003_plankjack',
    'Pavallion_003_sidebalancerun',
    'Pavallion_006_phonesiteat',
    'Pavallion_006_plankjack',
    'Pavallion_006_sidebalancerun',
    'Pavallion_013_phonesiteat',
    'Pavallion_013_plankjack',
    'Pavallion_013_yoga2',
]

RICH_TEST_SEQ_NAMES = [
'Gym_010_cooking1',
'Gym_010_dips1',
'Gym_010_dips2',
'Gym_010_lunge1',
'Gym_010_lunge2',
'Gym_010_mountainclimber1',
'Gym_010_mountainclimber2',
'Gym_010_pushup1',
'Gym_010_pushup2',
'Gym_011_burpee2',
'Gym_011_cooking1',
'Gym_011_cooking2',
'Gym_011_dips1',
'Gym_011_dips2',
'Gym_011_dips3',
'Gym_011_dips4',
'Gym_011_pushup1',
'Gym_011_pushup2',
'Gym_012_cooking2',
'Gym_012_lunge1',
'Gym_012_lunge2',
'Gym_012_pushup2',
'Gym_013_burpee4',
'Gym_013_dips1',
'Gym_013_dips2',
'Gym_013_dips3',
'Gym_013_lunge1',
'Gym_013_lunge2',
'Gym_013_pushup1',
'Gym_013_pushup2',
'LectureHall_009_021_reparingprojector1',
'LectureHall_009_sidebalancerun1',
'LectureHall_010_plankjack1',
'LectureHall_010_sidebalancerun1',
'LectureHall_019_wipingchairs1',
'LectureHall_021_plankjack1',
'LectureHall_021_sidebalancerun1',
'ParkingLot2_009_burpeejump1',
'ParkingLot2_009_burpeejump2',
'ParkingLot2_009_impro1',
'ParkingLot2_009_impro2',
'ParkingLot2_009_impro5',
'ParkingLot2_009_overfence1',
'ParkingLot2_009_overfence2',
'ParkingLot2_009_spray1',
'ParkingLot2_017_burpeejump1',
'ParkingLot2_017_burpeejump2',
'ParkingLot2_017_eating1',
'ParkingLot2_017_overfence1',
'ParkingLot2_017_overfence2',
'ParkingLot2_017_pushup1',
'ParkingLot2_017_pushup2',
]

class DatasetRICH(DatasetAMASSLike):
    homogenous_transform_align_ = np.array([[1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]])
    def __init__(self,
                 *args,
                 repr_registry: RepresentationRegistry = None,
                 mask_scheme_registry=None,
                 **kwargs
                 ):
        # NOTE: self.num_joints has to be set before super().__init__() call sicne DatasetAMASSLike calls overriden `create_body_repr` method.
        self.num_joints = 22
        super(DatasetRICH, self).__init__(*args,
                           repr_registry=repr_registry,
                           mask_scheme_registry=mask_scheme_registry,
                           **kwargs)

    def _canonicalize_seq(self, source_data_joints, source_data_smplx):
        smplx_noise_dict = {}
        smplx_params_dict = {'global_orient': source_data_smplx[:, 0:3],
                                'transl': source_data_smplx[:, 3:6],
                                'betas': source_data_smplx[:, 6:16],
                                'body_pose': source_data_smplx[:, 16:(16+63)],
                                }  # for the clip
        T, *_ = smplx_params_dict['transl'].shape
        smplx_params_tensor_dict = {k: torch.from_numpy(v).to(self.device, dtype=self.dtype) for k,v in smplx_params_dict.items()}
        smplx_params_tensor_dict = {**smplx_params_tensor_dict,
                "jaw_pose": torch.zeros((T,3)).to(self.device),
                "leye_pose": torch.zeros((T, 3)).to(self.device),
                "reye_pose": torch.zeros((T, 3)).to(self.device),
                "left_hand_pose": torch.zeros((T, 45)).to(self.device),
                "right_hand_pose": torch.zeros((T, 45)).to(self.device),
                "expression": torch.zeros((T, 10)).to(self.device),
                }

        # First align to the inertial frame with homogenous_transform.
        J = self.smplx_neutral(**smplx_params_tensor_dict).joints[:, 0:22, :]  # Shape: (T, 22, 3)
        repr_dict_clean = {
            "smplx_trans": smplx_params_tensor_dict['transl'],
            "smplx_rot_6d": rotmat_to_rot6d(aa_to_rotmat(smplx_params_tensor_dict['global_orient'].reshape(-1, 3))).reshape(smplx_params_tensor_dict['global_orient'].shape[0], 1, 6),
            "smplx_body_pose_6d": rotmat_to_rot6d(aa_to_rotmat(smplx_params_tensor_dict['body_pose'].reshape(-1, 3))).reshape(smplx_params_tensor_dict['body_pose'].shape[0], 21, 6),
            "smplx_betas": smplx_params_tensor_dict['betas'],
            "local_positions": J.reshape(J.shape[0], -1),
        }

        rot_repr_dict = rotate_repr_smpl(repr_dict_clean, rotation_matrix=torch.from_numpy(self.homogenous_transform_align_[:3, :3]).float().to(smplx_params_tensor_dict['transl'].device))


        _floor_height_offset = 0.00
        floor_height_offset = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[..., 2].min() + _floor_height_offset
        first_frame_root_xy = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[0:1, 0:1, :2] # Shape: (1, 1, 2)

        # Option1: use constant floor height offset.
        transform = torch.tensor([[1, 0, 0, -first_frame_root_xy[0,0,0]],       
                                [0, 1, 0, -first_frame_root_xy[0,0,1]],       
                                [0, 0, 1, -floor_height_offset],       
                                [0, 0, 0, 1]], dtype=torch.float32).to(smplx_params_tensor_dict['transl'].device)
        # Option2: use per-timestamp floor height offset
        floor_height_offset = rot_repr_dict['local_positions'].reshape(-1, self.num_joints, 3)[..., 2].min(dim=1, keepdim=True)[0] 
        transform = torch.eye(4).unsqueeze(0).to(smplx_params_tensor_dict['transl'].device).repeat(T, 1, 1)
        transform[:, 0, 3] = -first_frame_root_xy[0,0,0]
        transform[:, 1, 3] = -first_frame_root_xy[0,0,1]
        transform[:, 2, 3] = -floor_height_offset[:,0]

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

        if self.input_noise:
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
        cano_repr_dict = get_repr_smplx(positions=cano_repr_dict['local_positions'].reshape(-1, self.num_joints, 3).numpy(force=True),
                                        cano_jts_pos=False,
                                    smplx_params_dict=cano_smplx_params_dict,
                                    feet_vel_thre=5e-5,
                                    smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                    device=self.device,
                                    modality_names=self._repr_list)  # a dict of reprs
        if self.input_noise:
            repr_dict_noisy = get_repr_smplx(positions=cano_positions_noisy,
                                            cano_jts_pos=False,
                                            smplx_params_dict=cano_smplx_params_noisy_dict,
                                            feet_vel_thre=5e-5,
                                            smplx_model=self.smplx_neutral if 'R_world_joints' in self._repr_list else None,
                                            device=self.device,
                                            modality_names=self._repr_list)  # a dict of reprs
        dbg = False
        if dbg:
            from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
            joints = cano_repr_dict['local_positions'].reshape(-1, self.joints_num, 3)
            verts = self.smplx_neutral(**cano_smplx_params_tensor_dict).vertices.detach().cpu()
            cfg = SmplxVisConfig(window_title="RICH Debug", num_repeats=1, fixed_cam=False, camera_follow=True)
            visualize_smplx_sequence(
                joints=torch.from_numpy(joints).unsqueeze(0),
                vertices=verts.unsqueeze(0),
                faces=self.smplx_neutral.faces,
                parent_indices=SMPLH_KINTREE,
                batch_idx=0,
                cfg=cfg,
            )

        output = {
            "cano_positions": cano_repr_dict['local_positions'].reshape(-1, self.num_joints, 3),
            "cano_smplx_params_dict": cano_smplx_params_dict,
            "repr_dict": cano_repr_dict,
        }

        if self.input_noise:
            output["cano_positions_noisy"] = cano_positions_noisy
            output["cano_smplx_params_dict_noisy"] = cano_smplx_params_noisy_dict
            output["repr_dict_noisy"] = repr_dict_noisy
            output["smplx_noise_dict"] = smplx_noise_dict
        return output

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from omegaconf import OmegaConf
    from utils.registry_builders import build_mask_scheme_registry, build_representation_registry
    from torch.utils.data import DataLoader
    from ipdb_safety_net import ipdb_safety_net
    ipdb_safety_net()

    repo_root = Path(__file__).resolve().parents[1]
    repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/xyz_only.yaml")
    mask_cfg = OmegaConf.load(repo_root / "cfg_files/conf/mask_registry/custom_body_parts_xyz.yaml")
    repr_registry = build_representation_registry(OmegaConf.to_container(repr_cfg, resolve=True))
    mask_scheme_registry = build_mask_scheme_registry(OmegaConf.to_container(mask_cfg, resolve=True))

    split = 'train'
    if split == 'train':
        datasets = RICH_TRAIN_SEQ_NAMES
        rich_dataset_root= "datasets/RICH_smplx_preprocessed_train_split"
    elif split == 'test':
        datasets = RICH_TEST_SEQ_NAMES
        rich_dataset_root= "datasets/RICH_smplx_preprocessed_test_split"
    else:
        raise ValueError(f"Invalid split '{split}'.")
    ds = DatasetRICH(preprocessed_dataset_root=rich_dataset_root, split=split,
                                    datasets=datasets,
                                    cano_jts_pos=False,
                                    body_model_path="data/body_models/smplx_model",
                                    input_noise=True,
                                    noise_std_smplx_global_rot=2.0,
                                    noise_std_smplx_body_rot=2.0,
                                    noise_std_smplx_trans=0.03,
                                    noise_std_smplx_betas=0.00,
                                    clip_len=145,
                                    slice_strategy='random_uniform_len',
                                    mean_std=True,
                                    norm_strategy="meanstd",
                                    joints_num=22,
                                    disable_cache=True,
                                    logdir=tempfile.mkdtemp(),
                                    device=dist_util.dev(),
                                    dtype=torch.float32,
                                    use_ddp=False,
                                    dist_rank=0,
                                    repr_registry=repr_registry,
                                    mask_scheme_registry=mask_scheme_registry)
    
    train_dl = DataLoader(ds, batch_size=1, sampler=None, drop_last=True, num_workers=0)
    for batch_idx, batch_data in enumerate(train_dl):
        if batch_idx > 3:
            break
        pass
