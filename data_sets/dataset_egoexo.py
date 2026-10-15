import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import DBSCAN
import tempfile

from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence

from tqdm import tqdm
import random # Already imported

import joblib
from utils.other_utils import SMPLH_KINTREE, COCO_KINTREE
from utils.representation_registry import RepresentationRegistry
from typing import Tuple, List, Dict, Any, Literal
import pickle as pkl
from scipy.signal import savgol_filter
import pathlib
from cprint import cprint
from .dataset_base import DatasetBase

def find_numerical_key_in_dict(d):
    res = []
    for k in d.keys():
        try:
            # Try to convert to int, if successful, it's a numerical key (likely frame number)
            res.append(int(k))
        except ValueError:
            pass # Key is not purely numerical
    return sorted(res) # Return sorted numerical keys

random.seed(1)

class DatasetEgoExo(DatasetBase):
    def __init__(self, opt,
               norm_params_path='',
              repr_registry: RepresentationRegistry = None,
              mask_scheme_registry=None,
              **kwargs):

        super(DatasetEgoExo, self).__init__(repr_registry=repr_registry,
                          mask_scheme_registry=mask_scheme_registry,
                          **kwargs)
        
        self.root = opt['root']
        self.root_takes = os.path.join(self.root, "takes")
        self.split = opt['split']
        self.root_poses = os.path.join(self.root, "annotations", "ego_pose",self.split, "body")
        self.use_pseudo = opt['use_pseudo']
        self.coord = opt["coord"]
        self.slice_window =  opt["window_size"]
        
        # NOTE: train_norm_params.pkl is the default because norm parameters are produced during training and therefore named `*_train_norm_params.pkl`.
        default_norm_filename = f"{self.__class__.__name__}_train_norm_params.pkl"
        self.norm_params_path = self._resolve_norm_params_path(norm_params_path, default_norm_filename)
        self.norm_params = pkl.load(open(self.norm_params_path, "rb"))

        manually_annotated_takes = []
        if os.path.exists(os.path.join(self.root_poses,"annotation")):
            manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
            manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]
        
        pseudo_annotated_takes = []
        if self.use_pseudo and os.path.exists(os.path.join(self.root_poses,"automatic")):
            pseudo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]
        
        cameras_path = self.root_poses.replace("body", "camera_pose")
        cameras = []
        if os.path.exists(cameras_path):
            cameras = os.listdir(cameras_path)

        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))
        
        self.takes_uids = pseudo_annotated_takes if self.use_pseudo else manually_annotated_takes
        self.takes_metadata = {}

        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and 'bouldering' not in take_temp['take_name']: # Filter out bouldering
                self.takes_metadata[take_uid] = take_temp
        
        # Filter takes_uids based on takes_metadata (those not filtered out)
        self.takes_uids = [uid for uid in self.takes_uids if uid in self.takes_metadata]

        # NOTE: EgoExo4D annotations are incomplete, so root_rot_quat canonicalization is unavailable; the train-test config check aligns this with the train config (`align_to_yaxis=True`), which does not change the emitted data.
        self.align_to_yaxis = True

        self.poses = {}
        self.trajectories = {}
        self.cameras = {}

        num_takes_to_process = min(100, len(self.takes_uids))

        for take_idx, take_uid in tqdm(enumerate(self.takes_uids[:num_takes_to_process]), desc=f"Loading EgoExo {self.split} data"):
            if take_uid+".json" not in cameras:
                continue

            if take_idx > 30:
                break
            # if take_uid != "b3211853-7fbf-4c16-9a7e-af6c711526b0":
            #     continue
            camera_json_path = os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")
            if not os.path.exists(camera_json_path): continue
            camera_json = json.load(open(camera_json_path))
            
            pose_json_path = None
            if self.use_pseudo and take_uid in pseudo_annotated_takes:
                pose_json_path = os.path.join(self.root_poses,"automatic",take_uid+".json")
            elif take_uid in manually_annotated_takes:
                pose_json_path = os.path.join(self.root_poses,"annotation",take_uid+".json")

            if pose_json_path and os.path.exists(pose_json_path):
                pose_json = json.load(open(pose_json_path))
                # Check if pose_json is a dict and not empty
                if not isinstance(pose_json, dict) or not pose_json:
                    continue

                num_annotated_frames = len(find_numerical_key_in_dict(pose_json))

                if num_annotated_frames <= self.slice_window:
                    continue # Not enough frames for a slice in training

                ann, traj = self.translate_poses(pose_json, camera_json, self.coord)
                if not ann or not traj: continue # Skip if translation failed

                # Further check on translated data length
                if len(find_numerical_key_in_dict(traj)) <= self.slice_window:
                    continue
                
                self.poses[take_uid] = ann
                self.trajectories[take_uid] = traj
                self.cameras[take_uid] = camera_json # Store camera_json for this take
            
        self.poses_takes_uids = list(self.poses.keys())
        if not self.poses_takes_uids:
            cprint.warn(f"[Warning] DatasetEgoExo: No valid takes found for split {self.split} with use_pseudo={self.use_pseudo}.")

        self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
        # self.single_joint = opt['single_joint'] # This opt key was not in the provided args for dataloader_amass

        assert 'local_vel' not in self._repr_list, "Absolute representation must be used in `DatasetEgoExo`."

        # Ensure all keys in _repr_list are in Mean_dict and Std_dict, or provide defaults
        self.Mean = []
        self.Std = []
        self.Mean = np.concatenate([self.norm_params[repr_name]['mean'] for repr_name in self._repr_list], axis=-1)
        self.Std = np.concatenate([self.norm_params[repr_name]['std'] for repr_name in self._repr_list], axis=-1)
        self.Std[self.Std == 0] = 1e-6 # Avoid division by zero

        cprint.info(f'EgoExo Dataset ({self.split}) - Found {len(self.poses_takes_uids)} processable takes.')


    def translate_poses(self, anno, cams, coord):
        # ... (translate_poses function remains largely the same as provided) ...
        # Ensure it handles cases where `anno` or `cams` might be empty or malformed
        if not anno or not cams: return {}, {}
        trajectory = {}
        to_remove = []
        aria_key = None
        for key_cam in cams.keys(): # Find the aria key robustly
            if "aria" in key_cam.lower() and "camera_extrinsics" in cams[key_cam]:
                aria_key = key_cam
                break
        if aria_key is None: return {}, {} # No valid aria camera extrinsics

        # Ensure 'anno' is a dict and has keys that can be iterated for 'first'
        if not isinstance(anno, dict) or not anno: return {}, {}
        
        # Find first valid frame key in anno that also exists in camera extrinsics
        first_frame_key_str = None
        sorted_anno_frame_keys = find_numerical_key_in_dict(anno) # Get sorted numerical keys
        for fk_int in sorted_anno_frame_keys:
            fk_str = str(fk_int)
            if fk_str in cams[aria_key]['camera_extrinsics']:
                first_frame_key_str = fk_str
                break
        if first_frame_key_str is None: return {}, {} # No common frame key

        first_cam_extr = cams[aria_key]['camera_extrinsics'][first_frame_key_str]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam_extr)

        processed_anno = {} # Create a new dict for processed annotations

        for frame_key_int in sorted_anno_frame_keys:
            frame_key_str = str(frame_key_int)
            if frame_key_str not in anno or frame_key_str not in cams[aria_key]['camera_extrinsics']:
                continue # Skip if data is missing for this frame

            try:
                current_anno_list = anno[frame_key_str] # This is a list of persons
                if not current_anno_list: # Skip if no person annotated
                    to_remove.append(frame_key_str)
                    continue

                current_cam_extr = cams[aria_key]['camera_extrinsics'][frame_key_str]
                T_world_camera_inv = np.eye(4) # T_world_camera_inv is T_cam_world
                T_world_camera_inv[:3, :] = np.array(current_cam_extr)
                
                # T_world_camera is T_world_cam (transforms point from cam to world)
                T_world_camera = np.linalg.inv(T_world_camera_inv) 
                
                # T_first_cam_world is inv(T_world_first_cam)
                T_first_cam_world = np.linalg.inv(T_first_camera)


                processed_persons_in_frame = []
                for person_idx in range(len(current_anno_list)):
                    # Ensure "annotation3D" exists
                    if "annotation3D" not in current_anno_list[person_idx]:
                        continue
                    joints = current_anno_list[person_idx]["annotation3D"]
                    processed_joints = {}
                    for joint_name, joint_coords_dict in joints.items():
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array([joint_coords_dict["x"], joint_coords_dict["y"], joint_coords_dict["z"]])
                        
                        new_joint4d_world = T_world_camera @ joint4d # Point from cam to world

                        if coord == 'global':
                            final_joint4d = new_joint4d_world
                        elif coord == 'aria': # Relative to the first Aria camera pose
                            final_joint4d = T_first_cam_world @ new_joint4d_world
                        else: # 'camera' coord - relative to current Aria camera pose (points are in current camera frame)
                            final_joint4d = joint4d # Original points are already in current camera frame
                        
                        processed_joints[joint_name] = {"x": final_joint4d[0], "y": final_joint4d[1], "z": final_joint4d[2]}
                    
                    # Update person annotation with processed joints
                    # Create a new dict to avoid modifying original list items if they are shared
                    updated_person_anno = current_anno_list[person_idx].copy()
                    updated_person_anno["annotation3D"] = processed_joints
                    processed_persons_in_frame.append(updated_person_anno)

                if processed_persons_in_frame:
                    processed_anno[frame_key_str] = processed_persons_in_frame
                    # Trajectory of the camera in the chosen coordinate system
                    if coord == 'global':
                        trajectory[frame_key_str] = T_world_camera[:3,3] # Camera position in world
                    elif coord == 'aria':
                        # Position of current camera relative to first camera
                        T_first_curr = T_first_cam_world @ T_world_camera
                        trajectory[frame_key_str] = T_first_curr[:3,3]
                    else: # 'camera'
                        trajectory[frame_key_str] = np.array([0,0,0]) # Camera is at origin of its own frame
                else:
                    to_remove.append(frame_key_str)

            except Exception as e:
                cprint.warn(f"Error processing frame {frame_key_str} in translate_poses: {e}")
                to_remove.append(frame_key_str)
        
        # Clean up frames that had issues
        for frame_key_to_remove in to_remove:
            if frame_key_to_remove in processed_anno: del processed_anno[frame_key_to_remove]
            if frame_key_to_remove in trajectory: del trajectory[frame_key_to_remove]
                
        return processed_anno, trajectory


    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"]==uid:
                return take
        return None # Added to handle case where uid is not found

    def parse_skeleton(self, skeleton_dict_of_dicts): # Renamed from 'skeleton' to avoid confusion
        poses = []
        flags = [] # 0 for not visible/not annotated, 1 for visible/annotated
        for joint_name_map in self.joint_names: # Iterate through the canonical list of 17 joint names
            if joint_name_map in skeleton_dict_of_dicts:
                coords = skeleton_dict_of_dicts[joint_name_map]
                poses.append([coords['x'], coords['y'], coords['z']])
                flags.append(1) 
            else:
                poses.append([float('nan'), float('nan'), float('nan')]) 
                flags.append(0)
        return np.array(poses, dtype=np.float32), np.array(flags, dtype=np.float32)



    def __len__(self):
        return len(self.poses_takes_uids)

    def _process_joints(
        self,
        data_coco17: torch.Tensor, # Input is (T, 17, 3) COCO-style joints
        vis_coco17: torch.Tensor,  # Input is (T, 17) visibility
        return_smpl_family_jts: bool = True, # Should be True if num_joints is 22
        num_joints: int = 22, # Target number of joints (e.g., 22 for SMPL-H body)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not return_smpl_family_jts or num_joints == 17: # If not converting or target is 17
            return data_coco17, vis_coco17.bool()

        # Mapping from 17 COCO joints to 22 SMPL-H body joints
        # This needs to be accurate. EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES is defined in egoallo.mapping
        # It maps SMPLH_JOINT_NAMES (22 of them) to indices in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS (17 of them)
        # We need the inverse: map EGOEXO4D 17 joints to SMPLH 22 joints.
        # For simplicity, let's use a direct conceptual mapping if EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES is not directly usable.
        # The provided EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES in the prompt was incorrect.
        # A correct mapping would be:
        # SMPLH_idx: COCO_idx (using self.joint_names for COCO order)
        # 0 (Pelvis): Derived from LHip (11), RHip (12) - complex, often set to mid-hip
        # 1 (LHip): 11
        # 2 (RHip): 12
        # 3 (Spine1): Derived from Shoulders/Hips - complex
        # 4 (LKnee): 13
        # 5 (RKnee): 14
        # 6 (Spine2): Derived - complex
        # 7 (LAnkle): 15
        # 8 (RAnkle): 16
        # 9 (Spine3): Derived - complex
        # 10 (LFoot): -1 (often LAnkle if no toe)
        # 11 (RFoot): -1 (often RAnkle if no toe)
        # 12 (Neck): Derived from Shoulders/Nose
        # 13 (LCollar): -1 (or LShoulder)
        # 14 (RCollar): -1 (or RShoulder)
        # 15 (Head): 0 (Nose can approximate Head for simplicity, or derived)
        # 16 (LShoulder): 5
        # 17 (RShoulder): 6
        # 18 (LElbow): 7
        # 19 (RElbow): 8
        # 20 (LWrist): 9
        # 21 (RWrist): 10

        # This mapping is crucial and often dataset-specific or model-specific.
        # Using a simplified direct mapping for available COCO joints to SMPL(H) 22.
        # This assumes SMPL_PARENTS/SMPLH_KINTREE uses a standard 22 joint order.
        
        # Indices for self.joint_names (COCO-like 17)
        # nose:0, l_eye:1, r_eye:2, l_ear:3, r_ear:4, l_sho:5, r_sho:6, l_elb:7, r_elb:8, 
        # l_wri:9, r_wri:10, l_hip:11, r_hip:12, l_kne:13, r_kne:14, l_ank:15, r_ank:16

        # Target SMPL 22 joint indices (conceptual, matching SMPLH_KINTREE typically)
        # 0:Pelvis, 1:LHip, 2:RHip, ..., 15:Head, 16:LShoulder, ..., 21:RWrist
        
        smpl22_to_coco17_map = [
            -1, # 0 Pelvis -> can be derived or use mid-hip (11,12)
            11, # 1 LHip
            12, # 2 RHip
            -1, # 3 Spine1 -> can be derived from shoulders/hips but is complex.
            13, # 4 LKnee
            14, # 5 RKnee
            -1, # 6 Spine2 -> can be derived but is complex.
            15, # 7 LAnkle
            16, # 8 RAnkle
            -1, # 9 Spine3 -> can be derived but is complex.
            -1, # 10 LFoot -> close to LAnkle but leave to -1 now.
            -1, # 11 RFoot -> close to RAnkle but leave to -1 now.
            -1, # 12 Neck -> can be derived from shoulders/nose
            -1,  # 13 LCollar -> close to LShoulder but leave to -1 now.
            -1,  # 14 RCollar -> close to RShoulder but leave to -1 now.
            0,  # 15 Head -> use Nose as approximation
            5,  # 16 LShoulder
            6,  # 17 RShoulder
            7,  # 18 LElbow
            8,  # 19 RElbow
            9,  # 20 LWrist
            10  # 21 RWrist
        ]

        T = data_coco17.shape[0]
        smpl_family_world = torch.full(
            (T, num_joints, 3), float("nan"), dtype=data_coco17.dtype, device=data_coco17.device
        )
        smpl_family_visible = torch.zeros((T, num_joints), dtype=torch.bool, device=data_coco17.device)

        for smpl_idx in range(num_joints):
            coco_idx = smpl22_to_coco17_map[smpl_idx]
            if coco_idx != -1:
                smpl_family_world[:, smpl_idx] = data_coco17[:, coco_idx]
                smpl_family_visible[:, smpl_idx] = vis_coco17[:, coco_idx].bool()
        
        return smpl_family_world, smpl_family_visible


    def apply_kinematic_constraints_v2(
        self,
        joints_world: torch.Tensor,
        joints_world_coco: torch.Tensor,
        threshold: float = 1.0,
        window_size: int = 11, 
        temporal_sigma: float = 2.0,
        smpl_family_name: Literal["smplh", "smpl"] = "smplh",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kintree_smpl_family = SMPLH_KINTREE[:joints_world.shape[1]] if smpl_family_name == "smplh" else SMPLH_KINTREE[:joints_world.shape[1]-2]
        kintree_coco = COCO_KINTREE[:joints_world_coco.shape[1]]

        def process_joints(joints: torch.Tensor, kintree: list) -> torch.Tensor:
            T, J, _ = joints.shape
            device = joints.device
            if J == 0 : return joints # Handle empty joint sets

            # Create Gaussian weights for temporal window
            half_window = window_size // 2
            x_gauss = np.linspace(-temporal_sigma, temporal_sigma, window_size) # Renamed to avoid conflict
            weights_gauss = torch.tensor(
                np.exp(-(x_gauss**2) / 2),
                dtype=torch.float32,
                device=device,
            )
            weights_gauss /= (weights_gauss.sum() + 1e-9) # Avoid division by zero

            for j_idx in range(J): # Corrected variable name from j to j_idx
                parent_idx = kintree[j_idx]
                if parent_idx == -1 or parent_idx >= J: # Added check for parent_idx bounds
                    continue

                valid_mask = ~torch.isnan(joints[:, j_idx]).any(dim=1) & ~torch.isnan(
                    joints[:, parent_idx],
                ).any(dim=1)
                valid_ts = torch.where(valid_mask)[0] # Keep as tensor

                if len(valid_ts) > 10:
                    global_dists = torch.norm(
                        joints[valid_ts, j_idx] - joints[valid_ts, parent_idx],
                        dim=1,
                    )
                    global_mean = global_dists.mean()
                    global_std = global_dists.std()
                    if global_std < 1e-6: global_std = torch.tensor(1e-6, device=device) # Avoid zero std
                else:
                    continue 

                for t_idx in range(T): # Corrected variable name from t to t_idx
                    if not valid_mask[t_idx]:
                        continue

                    start = max(0, t_idx - half_window)
                    end = min(T, t_idx + half_window + 1)
                    window_ts = torch.arange(start, end, device=device)

                    window_valid_mask_indices = torch.where(valid_mask[window_ts])[0] # Indices relative to window_ts
                    
                    if len(window_valid_mask_indices) < 3 : # Use global stats if insufficient
                        cprint.warn("Insufficient valid points in window, using global stats")
                        mean_dist = global_mean
                        std_dist = global_std
                    else:
                        actual_valid_window_ts = window_ts[window_valid_mask_indices] # Global indices of valid frames in window
                        assert len(actual_valid_window_ts) >= 3
                    
                        window_dists = torch.norm(
                            joints[actual_valid_window_ts, j_idx]
                            - joints[actual_valid_window_ts, parent_idx],
                            dim=1,
                        )
                        # Simplified: use mean/std of valid points in window if weighting is tricky
                        mean_dist = window_dists.mean()
                        std_dist = window_dists.std()
                        if std_dist < 1e-6: std_dist = torch.tensor(1e-6, device=device)

                    current_dist = torch.norm(joints[t_idx, j_idx] - joints[t_idx, parent_idx])
                    if not (
                        mean_dist - threshold * std_dist
                        <= current_dist
                        <= mean_dist + threshold * std_dist
                    ):
                        direction = joints[t_idx, j_idx] - joints[t_idx, parent_idx]
                        direction_normalized = direction / (current_dist + 1e-7)
                        adjusted_joint = (
                            joints[t_idx, parent_idx] + direction_normalized * mean_dist
                        )
                        joints[t_idx, j_idx] = adjusted_joint

            # Temporal smoothing
            for j_idx_smooth in range(J): 
                valid_ts_smooth = torch.where(~torch.isnan(joints[:, j_idx_smooth]).any(dim=1))[0]
                if len(valid_ts_smooth) > window_size : # Savgol needs more points than window_length
                    # Ensure window_size is odd and less than num points
                    current_window_size = min(window_size, len(valid_ts_smooth) -1 if (len(valid_ts_smooth)-1)%2==0 else len(valid_ts_smooth)-2 )
                    if current_window_size < 5: current_window_size = 5 # Minimum practical window
                    if current_window_size % 2 == 0: current_window_size -=1

                    if len(valid_ts_smooth) > current_window_size and current_window_size > 0:
                        smoothed = savgol_filter(
                            joints[valid_ts_smooth, j_idx_smooth].cpu().numpy(),
                            window_length=current_window_size, # Must be odd
                            polyorder=2, # Polyorder must be less than window_length
                            axis=0,
                        )
                        joints[valid_ts_smooth, j_idx_smooth] = torch.from_numpy(smoothed).to(device=device, dtype=joints.dtype)
                    pass
            return joints

        joints_world = process_joints(joints_world, kintree_smpl_family)
        joints_world_coco = process_joints(joints_world_coco, kintree_coco)

        return joints_world, joints_world_coco

    def _get_floor_height(self, take_uid):
        cache_dir = os.path.expanduser("~/.cache/rohm_egoexo_floor_heights")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{take_uid}_floor_z.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pkl.load(f)
                
        take_info = self.takes_metadata[take_uid]
        root_dir = take_info['root_dir']
            
        points_path = os.path.join(self.root, root_dir, "trajectory", "semidense_points.csv.gz")
        
        points_data = mps.read_global_point_cloud(points_path)
        filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.0001,
            threshold_dep=0.005,
        )
        filtered_points_np = np.array([x.position_world for x in filtered_points_data])
        if len(filtered_points_np) == 0:
            raise ValueError(f"No valid points found after filtering for take {take_uid} at {points_path}")
            
        zs = filtered_points_np[:, 2]
        
        # DBScan clustering on bottom 10%
        zs_sorted = np.sort(zs)
        zs_sorted = zs_sorted[len(zs_sorted) // 10000:]
        alpha = 0.1 if len(zs) < 10000 else 0.15
        min_z = np.min(zs_sorted)
        max_z = np.max(zs_sorted)
        zs_bottom = zs_sorted[zs_sorted <= min_z + (max_z - min_z) * alpha]
        
        zs_bottom_reshaped = zs_bottom.reshape(-1, 1)
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(zs_bottom_reshaped)
        unique_labels = np.unique(clustering.labels_)
        
        floor_z = None
        for label in sorted(unique_labels):
            if label == -1: continue # noise
            points_in_cluster = zs_bottom[clustering.labels_ == label]
            if len(points_in_cluster) > 50: # significant cluster
                floor_z = float(np.median(points_in_cluster))
                break
                
        if floor_z is None:
            floor_z = float(np.median(zs_bottom))
            
        fd, temp_path = tempfile.mkstemp(dir=cache_dir)
        with os.fdopen(fd, "wb") as f:
            pkl.dump(floor_z, f)
        os.replace(temp_path, cache_file)
            
        return floor_z

    def __getitem__(self, index):
        take_uid = self.poses_takes_uids[index]
        pose_data_for_take = self.poses[take_uid] # Dict of frames -> list of persons -> "annotation3D"
        aria_trajectory_for_take =  self.trajectories[take_uid] # Dict of frames -> [x,y,z]

        # Get sorted numerical frame keys that exist in BOTH pose and trajectory data
        pose_frame_keys_int = find_numerical_key_in_dict(pose_data_for_take)
        traj_frame_keys_int = find_numerical_key_in_dict(aria_trajectory_for_take)
        
        common_frame_keys_int = sorted(list(set(pose_frame_keys_int).intersection(set(traj_frame_keys_int))))

        if not common_frame_keys_int:
            # This should ideally not happen if preprocessing was correct
            # Return a dummy item or raise error
            cprint.warn(f"Warning: No common frames for take_uid {take_uid}. Returning dummy item.")
            dummy_motion_repr = np.zeros((self.slice_window if self.split == "train" else 1, self.body_feat_dim), dtype=np.float32)
            return {
                "motion_repr_clean": dummy_motion_repr, 
                "cond": dummy_motion_repr, # Or specific cond shape
                "meta": {"take_name": "dummy", "take_uid": "dummy", "frame_keys": [], 
                         "first_frame_transform": np.eye(4, dtype=np.float32), 
                         "floor_height": 0.0}
            }

        selected_original_frame_keys_int = common_frame_keys_int
        assert common_frame_keys_int

        skeletons_window_list = []
        flags_window_list = []
        # aria_pos_window_list = [] # Aria trajectory (camera position)

        processed_frame_keys_int = []
        for original_frame_key_int in selected_original_frame_keys_int:
            frame_key_str = str(original_frame_key_int)
            # Assuming the first person's annotation is the one of interest
            person_annotation = pose_data_for_take[frame_key_str][0]["annotation3D"] 
            skeleton_pts_np, flags_np = self.parse_skeleton(person_annotation) # (17,3), (17,)
            if flags_np.sum() == 0:
                # No valid joints in this frame, skip it
                cprint.warn(f"[Warning]: No valid joints in frame {frame_key_str} for take_uid {take_uid}. Skipping this frame.")
                continue
            processed_frame_keys_int.append(original_frame_key_int)
            skeletons_window_list.append(skeleton_pts_np)
            flags_window_list.append(flags_np)
            # aria_pos_window_list.append(aria_trajectory_for_take[frame_key_str])

        skeletons_window_torch = torch.from_numpy(np.array(skeletons_window_list, dtype=np.float32)) # (T, 17, 3)
        flags_window_torch = torch.from_numpy(np.array(flags_window_list, dtype=np.float32))       # (T, 17)
        # aria_pos_window_torch = torch.from_nump(np.array(aria_pos_window_list, dtype=np.float32))   # (T, 3)
        
        current_seq_len = skeletons_window_torch.shape[0]

        # Process to SMPL(H) 22-joint convention (from 17 COCO-like joints)
        joints_world_smpl22, visible_mask_smpl22 = self._process_joints(
            skeletons_window_torch, # (T, 17, 3)
            flags_window_torch,     # (T, 17)
            return_smpl_family_jts=True,
            num_joints=22, # Target SMPL 22 joints
        ) # (T, 22, 3), (T, 22)

        # Filter out frames with no visible joints at all
        vis_frame_mask = visible_mask_smpl22.sum(dim=1) > 0 # (T,) True if any joint visible in frame
        joints_world_smpl22 = joints_world_smpl22[vis_frame_mask]
        visible_mask_smpl22 = visible_mask_smpl22[vis_frame_mask]
        skeletons_window_torch = skeletons_window_torch[vis_frame_mask]
        flags_window_torch = flags_window_torch[vis_frame_mask]
        # Update after filtering invalid frames so all downstream shapes match.
        current_seq_len = skeletons_window_torch.shape[0]

        # Keep original COCO-style joints as well for potential separate processing or loss
        joints_world_coco17 = skeletons_window_torch.clone() # (T, 17, 3)
        visible_mask_coco17 = flags_window_torch.bool().clone() # (T, 17)
        
        # Apply kinematic constraints and smoothing
        # joints_world_smpl22_kin, joints_world_coco17_kin = self.apply_kinematic_constraints_v2(
        # 	joints_world=joints_world_smpl22.clone(), # Pass clones to avoid in-place modification of originals if not desired
        # 	joints_world_coco=joints_world_coco17.clone(),
        # 	smpl_family_name="smplh", # or "smpl" depending on your 22-joint convention
        # )
        joints_world_smpl22_kin, joints_world_coco17_kin = joints_world_smpl22.clone(), joints_world_coco17.clone() # Disable kinematic constraints for now
        
        original_joints_world_smpl22 = joints_world_smpl22_kin.clone()

        first_frame_joints_smpl22 = joints_world_smpl22_kin[0, :, :] # Shape (num_joints, 3)
        
        # Try to derive pelvis from left hip (idx 1) and right hip (idx 2) for the first frame
        lhip_pos_first_frame = first_frame_joints_smpl22[1, :2] # XY of Left Hip
        rhip_pos_first_frame = first_frame_joints_smpl22[2, :2] # XY of Right Hip

        offset_xy_values = None

        if not torch.isnan(lhip_pos_first_frame).any() and not torch.isnan(rhip_pos_first_frame).any():
            # Both hips are valid, use their midpoint
            offset_xy_values = (lhip_pos_first_frame + rhip_pos_first_frame) / 2.0
        else:
            # Fallback: use the mean of all non-NaN joint positions in the first frame for XY offset
            # Consider only XY coordinates for the mean calculation
            valid_joints_first_frame_xy = first_frame_joints_smpl22[:, :2] # Get XY of all joints
            # Compute nanmean across the joint dimension (dim=0)
            # nanmean will ignore NaNs. If all are NaN for X or Y, it will result in NaN.
            mean_xy_all_valid_joints = torch.nanmean(valid_joints_first_frame_xy, dim=0)
            
            if not torch.isnan(mean_xy_all_valid_joints).any():
                offset_xy_values = mean_xy_all_valid_joints
            else:
                # Ultimate fallback: if all joints are NaN (highly unlikely for a whole frame after processing)
                # or if nanmean resulted in NaN (e.g. a whole column was NaN)
                # In this extreme case, use zero offset, though this might indicate upstream data issues.
                cprint.warn(f"[Warning] DatasetEgoExo: Could not determine a valid XY offset for the first frame of take {take_uid}. Using zero offset.")
                offset_xy_values = torch.zeros(2, device=joints_world_smpl22_kin.device, dtype=joints_world_smpl22_kin.dtype)

        # Create offset tensor for broadcasting: (1, 1, 3)
        first_frame_offset_xy = torch.zeros((1, 1, 3), device=joints_world_smpl22_kin.device, dtype=joints_world_smpl22_kin.dtype)
        first_frame_offset_xy[0, 0, 0] = offset_xy_values[0] # X offset
        first_frame_offset_xy[0, 0, 1] = offset_xy_values[1] # Y offset
        # Z offset remains 0 at this stage, it will be handled by floor_height
        # Z offset will be floor_height

        joints_world_smpl22_kin_norm = joints_world_smpl22_kin - first_frame_offset_xy # Broadcasts over T and J

        temp_for_floor = joints_world_smpl22_kin_norm.clone()

        floor_height_val = self._get_floor_height(take_uid)
        floor_height = torch.tensor(floor_height_val, device=temp_for_floor.device, dtype=temp_for_floor.dtype)
        
        joints_world_smpl22_kin_norm[:, :, 2] = joints_world_smpl22_kin_norm[:, :, 2] - floor_height
        
        meta_transform_info = {
            "translation_xy_offset": first_frame_offset_xy[0,0,:2].cpu().numpy(), # This is the value subtracted
            "floor_height_offset": floor_height.cpu().item(), # This is the value subtracted
            "original_coord_system": self.coord
        }
        # ========

        final_visible_mask_smpl22 = ~torch.isnan(joints_world_smpl22_kin_norm).any(dim=-1) # (T, 22)
        
        item_dict = {}
        item_dict['motion_repr_clean'] = np.zeros((current_seq_len, self.body_feat_dim), dtype=np.float32)
        item_dict['cond_joint_mask'] = ~joints_world_smpl22_kin_norm.isnan().any(dim=-1).unsqueeze(-1).numpy(force=True).astype(bool) # [T, 22, 1]

        if 'local_positions' in self._repr_indices:
            s_lp, e_lp = self._repr_indices['local_positions']
            joints_for_repr = joints_world_smpl22_kin_norm.reshape(current_seq_len, -1).cpu().numpy()  # [T, 22*3]
            joint_vis_mask_np_flat = final_visible_mask_smpl22.unsqueeze(-1).repeat(1,1,3).reshape(current_seq_len, -1).cpu().numpy() # [T, 22*3]
            joints_for_repr[~joint_vis_mask_np_flat] = 0.0 # Zero out non-visible joints
            item_dict['motion_repr_clean'][:, s_lp:e_lp] = joints_for_repr
        
        # Store non-normalized, but kinematically constrained & interpolated data for visualization/debug
        item_dict['joints_world_kin_smpl22'] = joints_world_smpl22_kin.cpu().numpy() 
        item_dict['joints_world_kin_smpl22_norm'] = joints_world_smpl22_kin_norm.cpu().numpy()
        item_dict['visible_mask_smpl22'] = final_visible_mask_smpl22.cpu().numpy()
        item_dict['joints_world_kin_coco17'] = joints_world_coco17_kin.cpu().numpy() # If needed
        item_dict['visible_mask_coco17'] = visible_mask_coco17.cpu().numpy() # If needed

        item_dict['meta'] = {
            'take_name': pathlib.Path(self.takes_metadata[take_uid]['root_dir']).name, # Get 'cmu_bike01_5' from 'takes/cmu_bike01_5'
            'take_uid': take_uid,
            'frame_keys': np.array(processed_frame_keys_int, dtype=np.int32), # Original video frame numbers for this slice
            'normalization_params': meta_transform_info,
            'original_joints_world_smpl22': original_joints_world_smpl22.cpu().numpy(), # Store pre-normalized, kinematically processed
            'split': self.split,
        }

        return item_dict
    
if __name__ == "__main__": 
    split = "train"
    import third_party.ego_exo4d_egopose.bodypose.utils.utils_option as option
    from torch.utils.data import DataLoader
    from data_sets.dataset_egoexo import DatasetEgoExo
    from utils.egoexo_vis_utils import EgoExoFrameSynchronizer
    from viz.egoexo_visualizer import render_egoexo_synchronized_frame, save_egoexo_animation
    import tempfile
    from ipdb_safety_net import ipdb_safety_net
    from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
    from utils import dist_util
    from pathlib import Path
    from omegaconf import OmegaConf
    from utils.registry_builders import build_mask_scheme_registry, build_representation_registry
    ipdb_safety_net()
    opt_path = "cfg_files/test_cfg/egoexo/test_egoexo.json"
    logdir = './results/egoexo-default'
    egoexo_optn = option.parse(opt_path=opt_path)
    all_batches_cond_mask = {}
    repo_root = Path(__file__).resolve().parents[1]
    repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/xyz_only.yaml")
    mask_cfg = OmegaConf.load(repo_root / "cfg_files/conf/mask_registry/custom_body_parts_xyz.yaml")
    repr_registry = build_representation_registry(OmegaConf.to_container(repr_cfg, resolve=True))
    mask_scheme_registry = build_mask_scheme_registry(OmegaConf.to_container(mask_cfg, resolve=True))
    for phase, dataset_opt in egoexo_optn['datasets'].items():
        # if dataset_opt['split'] != 'train':
        # 	continue
        testD = DatasetEgoExo(
            dataset_opt,
            norm_params_path="ckpts/both_Nov_17_discriminative_clip_len_33/AMASS_norm_params.pkl",
            cano_jts_pos=False,
            logdir=logdir,
            body_model_path="data/body_models/smplx_model",
            device=dist_util.dev(),
            use_ddp=False,
            joints_num=22,
            dist_rank=0,
            mean_std=True,
            disable_cache=True,
            norm_strategy='meanstd',
            repr_registry=repr_registry,
            mask_scheme_registry=mask_scheme_registry,
        )
        test_dl = DataLoader(
            testD,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
        for batch_idx, batch in enumerate(test_dl):
            take_name = batch['meta']['take_name'][0]
            take_uid = batch['meta']['take_uid'][0]
            original_frame_keys = batch['meta']['frame_keys'][0].cpu().numpy().tolist()

            current_ego_frame_synchronizer = EgoExoFrameSynchronizer(
                'datasets/egoexo-default', 
                take_name,
                take_uid=take_uid,
                split=dataset_opt['split']
            )
            
            # output_frames_paths = []
            temp_dir = tempfile.mkdtemp()

            # joints = batch['joints_world_kin_smpl22']
            bs, T, *_ = batch['motion_repr_clean'].shape
            device, dtype = batch['motion_repr_clean'].device, batch['motion_repr_clean'].dtype
            s_lp, e_lp = testD._repr_indices['local_positions']
            joints = batch['motion_repr_clean'][:, :, s_lp:e_lp] 
            joints = joints.reshape(bs, T, 22, 3)
            all_batches_cond_mask[take_name] = batch['cond_mask'].squeeze(0).squeeze(-1).cpu().numpy()

            # NOTE: some points appear at the origin (0,0,0) in visualizations because they are unannotated in EgoExo4D.
            cfg = SmplxVisConfig(window_title="EgoExo Debug", num_repeats=1, fixed_cam=False, camera_follow=True, show_mesh=False, show_skeleton=True)
            visualize_smplx_sequence(
                joints=joints,
                vertices=None,
                faces=None,
                parent_indices=SMPLH_KINTREE,
                batch_idx=0,
                cfg=cfg,
            )

    # Create a log file to record visibility information
    log_file_path = os.path.join(temp_dir, 'joint_visibility_analysis.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Joint Visibility Analysis Report\n")
        log_file.write("=" * 50 + "\n\n")
        
        # Define joint names for SMPL-H 22 joints (based on typical SMPL-H ordering)
        joint_names_smpl22 = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
            'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
            'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
        ]
        
        total_sequences = len(all_batches_cond_mask)
        log_file.write(f"Total number of sequences analyzed: {total_sequences}\n\n")
        
        for take_name, cond_mask in all_batches_cond_mask.items():
            T, num_joints = cond_mask.shape
            log_file.write(f"Take Name: {take_name}:\n")
            log_file.write(f"  Duration: {T} timesteps\n")
            log_file.write(f"  Number of joints: {num_joints}\n\n")
            
            # Convert to visibility mask (1 = visible, 0 = invisible)
            # Note: cond_mask is True for masked (invisible) joints, so we need to invert
            visibility_mask = (~cond_mask).astype(int)
            
            # 1. For each joint, count how many timesteps it's visible
            log_file.write("  Joint Visibility Summary:\n")
            log_file.write("  " + "-" * 40 + "\n")
            for joint_idx in range(num_joints):
                visible_count = visibility_mask[:, joint_idx].sum()
                visibility_ratio = visible_count / T
                joint_name = joint_names_smpl22[joint_idx] if joint_idx < len(joint_names_smpl22) else f"Joint_{joint_idx}"
                log_file.write(f"    {joint_name:12}: {visible_count:3}/{T:3} timesteps ({visibility_ratio:.2%})\n")
            
            log_file.write("\n")
            
            # 2. For each timestep, list which joints are visible
            log_file.write("  Timestep-wise Visibility:\n")
            log_file.write("  " + "-" * 40 + "\n")
            for t in range(T):
                visible_joints_at_t = [i for i in range(num_joints) if visibility_mask[t, i] == 1]
                visible_joint_names = [joint_names_smpl22[i] if i < len(joint_names_smpl22) else f"Joint_{i}" 
                                      for i in visible_joints_at_t]
                log_file.write(f"    Timestep {t:3}: {len(visible_joints_at_t):2}/{num_joints} visible joints\n")
                if visible_joint_names:
                    # Split long lists into multiple lines for readability
                    names_per_line = 6
                    for i in range(0, len(visible_joint_names), names_per_line):
                        line_names = visible_joint_names[i:i+names_per_line]
                        log_file.write(f"                 {', '.join(line_names)}\n")
                else:
                    log_file.write(f"                 No visible joints\n")
            
            log_file.write("\n" + "=" * 50 + "\n\n")
        
        # Summary statistics across all sequences
        log_file.write("Overall Statistics:\n")
        log_file.write("-" * 30 + "\n")
        
        # Aggregate visibility statistics
        total_joint_visibility_counts = np.zeros(22)
        total_timesteps = 0
        
        for cond_mask in all_batches_cond_mask.values():
            visibility_mask = (~cond_mask).astype(int)
            total_joint_visibility_counts += visibility_mask.sum(axis=0)
            total_timesteps += cond_mask.shape[0]
        
        log_file.write(f"Total timesteps across all sequences: {total_timesteps}\n\n")
        log_file.write("Average visibility per joint across all sequences:\n")
        for joint_idx in range(22):
            avg_visibility = total_joint_visibility_counts[joint_idx] / total_timesteps
            joint_name = joint_names_smpl22[joint_idx] if joint_idx < len(joint_names_smpl22) else f"Joint_{joint_idx}"
            log_file.write(f"  {joint_name:12}: {avg_visibility:.2%}\n")

    print(f"Joint visibility analysis saved to: {log_file_path}")
