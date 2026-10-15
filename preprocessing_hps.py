import sys, os
import torch
import numpy as np
from tqdm import tqdm
# from utils import *
import glob
import argparse
import smplx

from data_sets.motion_representation import *
from ipdb_safety_net import ipdb_safety_net
import pickle as p
import json
from pathlib import Path

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

############################
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ex_fps = 30


def hps_to_pose(hps_dataset_root_path, seq_name, subject_name, save_path_joints, save_path_smplx_params, smplx_model_neutral):
    beta_file = os.path.join(hps_dataset_root_path, "hps_betas", f"{subject_name}.json")
    body_pose_file = os.path.join(hps_dataset_root_path, "hps_smpl", f"{seq_name}.pkl")
    beta_file = json.load(open(beta_file, "r"))
    beta = np.array(beta_file['betas'])
    smplx_body_pose = p.load(open(body_pose_file, "rb"))
    transes = smplx_body_pose['transes']
    poses = smplx_body_pose['poses']
    frame_number = transes.shape[0]

    process = True

    pose_seq = []
    smplx_params_seq = []
    if process:
        # print('process:', process, 'fps:', fps, 'down_sample', down_sample, 'total_frame_number', frame_number)
        with torch.no_grad():
            for fId in range(0, frame_number, 1):
                global_orient = torch.Tensor(poses[fId:fId + 1, :3]).to(comp_device)  # [1, 3]
                trans = torch.Tensor(transes[fId:fId + 1]).to(comp_device)  # [1, 3]
                betas = torch.Tensor(beta[:10][np.newaxis]).to(comp_device)  # [1, 10]
                body_pose = torch.Tensor(poses[fId:fId + 1, 3:66]).to(comp_device)  # [1, 63]
                hand_pose = torch.zeros((1, 90)).to(comp_device)  # [1, 90]
                jaw_pose = torch.zeros((1,3)).to(comp_device)  # [1, 3]
                leye_pose = torch.zeros((1,3)).to(comp_device)  # [1, 3]
                reye_pose = torch.zeros((1,3)).to(comp_device)  # [1, 3]

                body_params = {'global_orient': global_orient,
                               'transl': trans,
                               'betas': betas,
                               'body_pose': body_pose,
                               'hand_pose': hand_pose,
                               'jaw_pose': jaw_pose,
                               'leye_pose': leye_pose,
                               'reye_pose': reye_pose,}
                smplx_output = smplx_model_neutral(**body_params)
                smplx_joints = smplx_output.joints[:, 0:25]  # [1, 25, 3]

                smplx_params_seq.append(torch.cat([global_orient, trans, betas, body_pose, hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1))  # [1, 169]
                pose_seq.append(smplx_joints)

        pose_seq = torch.cat(pose_seq, dim=0)
        pose_seq_np = pose_seq.detach().cpu().numpy()  # [seq_len, 52, 3], position of 52 body joints?

        dbg = False
        if dbg:
            from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
            joints = pose_seq_np[::10, :22, :]
            T, *_ = joints.shape
            cfg = SmplxVisConfig(window_title="HPS Debug", num_repeats=1, fixed_cam=False, camera_follow=True, show_mesh=False, show_skeleton=True)
            visualize_smplx_sequence(
                joints=torch.from_numpy(joints).unsqueeze(0),
                vertices=None,
                faces=None,
                parent_indices=SMPLH_KINTREE,
                batch_idx=0,
                cfg=cfg,
            )


        smplx_params_seq = torch.cat(smplx_params_seq, dim=0)  # [seq_len, 169]
        smplx_params_seq_np = smplx_params_seq.detach().cpu().numpy()  # [seq_len, 169],

        # np.save(save_path_joints, pose_seq_np)
        # np.save(save_path_smplx_params, smplx_params_seq_np)

def main(args):
    smplx_model_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx", gender='neutral').to(comp_device)

    pickle_path_list = glob.glob(os.path.join(args.hps_root, "hps_smpl", '*.pkl'))
    pickle_path_list = sorted(pickle_path_list)
    save_root_joints_path = os.path.join(args.save_root, f"pose_data_fps_{ex_fps}")
    save_root_smpl_params_path = os.path.join(args.save_root, f"smpl_data_fps_{ex_fps}")
    for pickle_path in tqdm(pickle_path_list):
        subject_name = Path(pickle_path).name[:4]
        recording_name = Path(pickle_path).name[:-4]
        ######## process sequence
        save_path_joints = os.path.join(save_root_joints_path, subject_name, recording_name + '.npy')
        save_path_smpl_params = os.path.join(save_root_smpl_params_path, subject_name, recording_name + '.npy')
        # os.makedirs(os.path.join(save_root_joints_path, subject_name), exist_ok=True)
        # os.makedirs(os.path.join(save_root_smpl_params_path, subject_name), exist_ok=True)
        hps_to_pose(hps_dataset_root_path=args.hps_root, seq_name=recording_name, subject_name=subject_name, save_path_joints=save_path_joints, save_path_smplx_params=save_path_smpl_params, smplx_model_neutral=smplx_model_neutral)

    print('finished.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_model_path', type=str, default='data/body_models/smplx_model', help='path to smplx model')
    parser.add_argument('--hps_root', type=str, default='data/hps/smplx_neutral', help='Root dir of raw HPS data (smplx neutral body)')
    # HPS_preprocessed
    parser.add_argument('--save_root', type=str, default='data/hps/preprocessed', help='Root directory to save preprocessed data to.')

    config = parser.parse_known_args()
    config = config[0]

    ipdb_safety_net()
    main(config)




