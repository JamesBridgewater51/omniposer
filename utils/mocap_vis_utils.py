#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for AIST, TotalCapture, and 3DPW datasets.

This module provides tools to load original mocap videos and camera parameters,
project 3D skeleton predictions (from a model) onto the 2D video frames,
and render synchronized 3D and 2D views into an output video.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
from tqdm import tqdm
import imageio
import warnings
from typing import List, Dict, Optional, Any, Tuple

# --- Constants ---

# SMPL 22-joint topology (parents for joints 0-21)
SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
LIMBS_SMPL_22 = [(i, parent) for i, parent in enumerate(SMPL_22_PARENTS) if parent != -1]

# Coordinate transformations from preprocessing scripts
AIST_ROT = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float()
TOTALCAP_ROT = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float()
PW3D_ROT = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).float()

# Floor offsets from dataloaders
DATASET_FLOOR_OFFSETS = {
    'AIST': 0.10,
    'TotalCapture': 0.00,
    '3DPW': 0.00,
}

# --- Core Projection & Drawing Utilities ---

def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   T_cam_world: np.ndarray,
                   dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Projects 3D points from world coordinates to 2D image coordinates.
    
    Args:
        points_3d (np.ndarray): 3D points in world space, shape (N, 3).
        K (np.ndarray): Camera intrinsics matrix, shape (3, 3).
        T_cam_world (np.ndarray): Camera extrinsics (world-to-cam), shape (4, 4) or (3, 4).
        dist_coeffs (Optional[np.ndarray]): Distortion coefficients (k1, k2, p1, p2, k3).
    
    Returns:
        np.ndarray: 2D projected points in image space, shape (N, 2).
    """
    if points_3d.size == 0:
        return np.empty((0, 2))
        
    # Ensure T_cam_world is (3, 4)
    if T_cam_world.shape == (4, 4):
        T_cam_world = T_cam_world[:3, :] # R | t

    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # Project points
    projected_points, _ = cv2.projectPoints(points_3d, R, t, K, dist_coeffs)
    
    return projected_points.squeeze(1) # Shape (N, 2)

def draw_skeleton_3d(ax: plt.Axes,
                       joints_3d: np.ndarray,
                       limbs: List[Tuple[int, int]],
                       color: str = 'r',
                       alpha: float = 0.7):
    """Draws a 3D skeleton on a matplotlib Axes3D object."""
    if joints_3d.size == 0:
        return
        
    # Plot limbs
    for i, j in limbs:
        if i < len(joints_3d) and j < len(joints_3d):
            ax.plot(
                [joints_3d[i, 0], joints_3d[j, 0]],
                [joints_3d[i, 1], joints_3d[j, 1]],
                [joints_3d[i, 2], joints_3d[j, 2]],
                color=color,
                alpha=alpha
            )
    # Plot joints
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='b', s=10, alpha=0.5)

def set_3d_axes_limits(ax: plt.Axes, joints_3d: np.ndarray, range_scale: float = 1.0):
    """Sets consistent axis limits for a 3D plot based on joint positions."""
    if joints_3d.size == 0:
        return
        
    max_range = np.array([
        joints_3d[:, 0].max() - joints_3d[:, 0].min(),
        joints_3d[:, 1].max() - joints_3d[:, 1].min(),
        joints_3d[:, 2].max() - joints_3d[:, 2].min()
    ]).max() / 2.0 * range_scale
    
    mid_x = (joints_3d[:, 0].max() + joints_3d[:, 0].min()) * 0.5
    mid_y = (joints_3d[:, 1].max() + joints_3d[:, 1].min()) * 0.5
    mid_z = (joints_3d[:, 2].max() + joints_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (world)'); ax.set_ylabel('Y (world)'); ax.set_zlabel('Z (world)')
    ax.view_init(elev=15., azim=-45)


def save_animation(frame_paths: List[str], output_path: str, fps: int = 30):
    """Compiles a list of image frames into an MP4 video."""
    if not frame_paths:
        print("Warning: No frames to animate.")
        return
        
    print(f"🎞️  Saving animation to {output_path} at {fps} fps...")
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame_path in tqdm(frame_paths, desc="Compiling video"):
            writer.append_data(imageio.imread(frame_path))
    print(f"✅ Animation saved successfully.")


# --- MocapFrameSynchronizer Class ---

class MocapFrameSynchronizer:
    """
    Handles loading of synchronized video frames and camera parameters
    for a specific sequence from AIST, TotalCapture, or 3DPW.
    """
    def __init__(self,
                 mocap_raw_root: str,
                 preprocessed_data_root: str,
                 dataset_name: str,
                 split: str):
        """
        Args:
            mocap_raw_root: Path to the raw dataset directory (e.g., './datasets/mocap_raw')
            preprocessed_data_root: Path to the preprocessed .pt files (e.g., './datasets/RobustCap/dataset_work')
            dataset_name: 'AIST', 'TotalCapture', or '3DPW'
            split: 'train' or 'test'
        """
        self.mocap_raw_root = mocap_raw_root
        self.preprocessed_data_root = preprocessed_data_root
        self.dataset_name = dataset_name
        self.split = split
        
        if self.dataset_name not in ['AIST', 'TotalCapture', '3DPW']:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

        self.data = self._load_preprocessed_data()
        self.name_to_index = {name: i for i, name in enumerate(self.data['name'])}
        
        self.sequence_name = None
        self.seq_index = -1
        self.video_paths: List[Optional[str]] = []
        self.video_caps: List[Optional[cv2.VideoCapture]] = []
        self.cam_Ks = None
        self.cam_Ts = None
        self.norm_offset = np.zeros((1, 1, 3), dtype=np.float32)
        self.frame_indices_to_load: List[int] = []
        self.num_cams = 0
        self.fps = 30

    def _load_preprocessed_data(self) -> Dict[str, Any]:
        """Loads the master .pt file for the specified dataset and split."""
        data_file = f'{self.split}.pt'
            
        data_path = os.path.join(self.preprocessed_data_root, self.dataset_name, data_file)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Could not find preprocessed data file: {data_path}\n"
                f"Please check 'preprocessed_data_root' and 'dataset_name'."
            )
        
        print(f"Loading preprocessed data from {data_path}...")
        return torch.load(data_path, map_location='cpu')

    def load_sequence_data(self, sequence_name: str):
        """
        Loads all necessary data (cams, videos, norm params) for a given sequence.
        """
        if sequence_name not in self.name_to_index:
            raise ValueError(f"Sequence '{sequence_name}' not found in {self.dataset_name} {self.split} split.")
        
        self.release() # Release any previously opened videos
        
        self.sequence_name = sequence_name
        self.seq_index = self.name_to_index[sequence_name]
        
        # 1. Get camera parameters
        self.cam_Ks = self.data['cam_K'][self.seq_index]
        self.cam_Ts = self.data['cam_T'][self.seq_index]
        if not isinstance(self.cam_Ks, np.ndarray):
            self.cam_Ks = self.cam_Ks.numpy()
        if not isinstance(self.cam_Ts, np.ndarray):
            self.cam_Ts = self.cam_Ts.numpy()

        # 2. Find video paths
        self.video_paths = self._find_video_paths(sequence_name)
        self.num_cams = len(self.video_paths) if self.dataset_name != '3DPW' else 1
        
        # 3. Open video captures
        self.video_caps = []
        for p in self.video_paths:
            if p and os.path.exists(p):
                self.video_caps.append(cv2.VideoCapture(p))
            else:
                self.video_caps.append(None)

        # 4. Calculate normalization parameters and frame indices
        self.norm_offset, self.frame_indices_to_load, self.fps = self._calculate_normalization_params()
        
        print(f"Loaded sequence '{sequence_name}'. Found {self.num_cams} cameras and {len(self.frame_indices_to_load)} frames.")

    def _find_video_paths(self, sequence_name: str) -> List[Optional[str]]:
        """Finds the raw video file paths based on the sequence name."""
        paths = []
        if self.dataset_name == 'AIST':
            # sequence_name is like 'gJB_sBM_cAll_d07_mJB2_ch10'
            # We need to find 'gJB_sBM_c01_d07_mJB2_ch10.mp4', ... 'gJB_sBM_c09_d07_mJB2_ch10.mp4'
            base_path = os.path.join(self.mocap_raw_root, 'AIST')
            for i in range(9): # 9 cameras
                cam_name = f'c0{i+1}'
                video_name = sequence_name.replace('cAll', cam_name) + '.mp4'
                paths.append(os.path.join(base_path, video_name))
                
        elif self.dataset_name == 'TotalCapture':
            # sequence_name is like 'TC_S1_acting1'
            # We need '.../totalcapture/s1/acting1/TC_S1_acting1_cam1.mp4', ...
            try:
                parts = sequence_name.split('_')
                subject = parts[1].lower() # 's1'
                motion = parts[2]         # 'acting1'
                base_path = os.path.join(self.mocap_raw_root, 'totalcapture', subject, motion)
                for i in range(8): # 8 cameras
                    video_name = f'{sequence_name}_cam{i+1}.mp4'
                    paths.append(os.path.join(base_path, video_name))
            except Exception as e:
                print(f"Warning: Could not parse TotalCapture name '{sequence_name}': {e}")
                paths = [None] * 8
                
        elif self.dataset_name == '3DPW':
            # 3DPW videos are not typically provided in this format.
            # We will render without video backgrounds.
            # We return [None] to signify 1 "camera" with no video.
            return [None]
            
        return [p if os.path.exists(p) else None for p in paths]

    def _calculate_normalization_params(self) -> Tuple[np.ndarray, List[int], int]:
        """
        Re-implements the dataloader's normalization logic to get the
        offset required to un-normalize predictions back to world space.
        """
        # 1. Get raw 3D joints and apply dataloader transforms
        joints_3d_raw = self.data['joint3d'][self.seq_index]
        if not isinstance(joints_3d_raw, torch.Tensor):
            joints_3d_raw = torch.from_numpy(joints_3d_raw).float()
        
        downsample_rate = 1
        fps = 60 # Default for 3DPW
        
        if self.dataset_name == 'AIST':
            joints_3d_raw = torch.einsum('ij,klj->kli', AIST_ROT, joints_3d_raw)
            downsample_rate = 2
            fps = 30
        elif self.dataset_name == 'TotalCapture':
            joints_3d_raw = torch.einsum('ij,klj->kli', TOTALCAP_ROT, joints_3d_raw)
            downsample_rate = 2
            fps = 30
        elif self.dataset_name == '3DPW':
            joints_3d_raw = torch.einsum('ij,klj->kli', PW3D_ROT, joints_3d_raw)
            # 3DPW is 60fps in preprocessing, and dataloader does not downsample.
            downsample_rate = 1
            fps = 60
            
        # 2. Get frame indices that the dataloader would use
        original_frame_indices = list(range(0, joints_3d_raw.shape[0], downsample_rate))
        joints_3d_sampled = joints_3d_raw[original_frame_indices]
        
        # 3. Standardize to 22 joints
        joints_world = joints_3d_sampled[:, :22, :].clone() # (T_sampled, 22, 3)

        # 4. Calculate normalization offset
        # a) Center XY at first frame root
        first_frame_root_xy = joints_world[0:1, 0:1, :2] # (1, 1, 2)
        offset = torch.zeros((1, 1, 3), dtype=joints_world.dtype)
        offset[..., :2] = first_frame_root_xy
        
        # b) Calculate floor height
        floor_height_offset = DATASET_FLOOR_OFFSETS[self.dataset_name]
        joints_world_norm = joints_world - offset
        floor_height = joints_world_norm[..., 2].min() + floor_height_offset
        
        # c) Combine offsets
        offset[..., 2] = floor_height
        
        return offset.numpy(), original_frame_indices, fps

    def get_synchronized_frames(self) -> List[Dict[str, Any]]:
        """
        Fetches all video frames and camera parameters for the loaded sequence.
        """
        output_data_list = []
        
        for t, original_frame_idx in enumerate(tqdm(self.frame_indices_to_load, desc="Loading frames")):
            
            exo_frames_data: Dict[str, Dict[str, Any]] = {}
            for cam_idx in range(self.num_cams):
                frame = None
                cap = self.video_caps[cam_idx]
                if cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
                    ret, bgr_frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    else:
                        warnings.warn(f"Failed to read frame {original_frame_idx} from {self.video_paths[cam_idx]}")
                
                cam_name = f'cam_{cam_idx+1}'
                K, T = None, None
                
                if self.dataset_name == '3DPW':
                    # 3DPW has one moving camera
                    cam_name = 'cam_1'
                    K = self.cam_Ks         # Shape (3, 3)
                    if t < len(self.cam_Ts):
                        T = self.cam_Ts[t] # Shape (N, 4, 4), pick frame t
                    else:
                        warnings.warn(f"Frame index {t} out of bounds for 3DPW camera poses.")
                        T = np.eye(4)
                else:
                    # AIST/TotalCapture have static multi-cameras
                    K = self.cam_Ks[cam_idx] # Shape (num_cams, 3, 3)
                    T = self.cam_Ts[cam_idx] # Shape (num_cams, 4, 4)

                exo_frames_data[cam_name] = {
                    'frame': frame,
                    'params': {'K': K, 'T_cam_world': T, 'dist_coeffs': None}
                }
            
            output_data_list.append({
                'frame_original_index': original_frame_idx,
                'exo_frames_data': exo_frames_data
            })
            
        return output_data_list

    def release(self):
        """Releases all open video capture objects."""
        for cap in self.video_caps:
            if cap:
                cap.release()
        self.video_caps = []

# --- Main Rendering Function ---

def render_mocap_synchronized_frame(output_path: str,
                                      frame_idx: int,
                                      smpl_joints_world: np.ndarray, # (J, 3) in WORLD coords
                                      exo_frames_data: Dict[str, Dict[str, Any]],
                                      smpl_verts_world: Optional[np.ndarray] = None, # (V, 3) in WORLD
                                      limbs: List[Tuple[int, int]] = LIMBS_SMPL_22):
    """
    Renders a single, synchronized frame with 3D view and all exocentric camera views.
    """
    num_exo_cams = len(exo_frames_data)
    num_subplots = 1 + num_exo_cams # 1 3D view + N exo views
    
    n_cols = 3
    n_rows = int(np.ceil(num_subplots / n_cols))
    
    fig = plt.figure(figsize=(7 * n_cols, 7 * n_rows))
    fig.suptitle(f'Frame {frame_idx}', fontsize=16)

    # --- 1. 3D SMPL View ---
    ax_3d = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    if smpl_joints_world is not None:
        draw_skeleton_3d(ax_3d, smpl_joints_world, limbs, color='r')
        if smpl_verts_world is not None:
             ax_3d.scatter(
                smpl_verts_world[::5, 0], smpl_verts_world[::5, 1], smpl_verts_world[::5, 2],
                s=1, c='c', alpha=0.1
             )
        set_3d_axes_limits(ax_3d, smpl_joints_world, range_scale=1.2)
    ax_3d.set_title("3D View (Predicted, World Coords)")

    # --- 2. Exocentric Views ---
    subplot_idx_offset = 2
    for i, (cam_name, exo_data) in enumerate(exo_frames_data.items()):
        if subplot_idx_offset + i > n_cols * n_rows:
            break
            
        ax_exo = fig.add_subplot(n_rows, n_cols, subplot_idx_offset + i)
        exo_frame_rgb = exo_data.get('frame')
        cam_params = exo_data.get('params')

        if exo_frame_rgb is not None and exo_frame_rgb.size > 0:
            ax_exo.imshow(exo_frame_rgb)
        else:
            # Show black background if no video
            if exo_frame_rgb is not None:
                h, w, _ = exo_frame_rgb.shape
            else:
                h, w = 1080, 1920 # Default size
            ax_exo.imshow(np.zeros((h, w, 3), dtype=np.uint8))
            ax_exo.set_title(f"Exo: {cam_name} (No Video)")
            
        if smpl_joints_world is not None and cam_params:
            K = cam_params.get('K')
            T_cam_world = cam_params.get('T_cam_world')
            dist_coeffs = cam_params.get('dist_coeffs')
            
            if K is not None and T_cam_world is not None:
                # Project and draw skeleton
                projected_smpl_2d = project_points(smpl_joints_world, K, T_cam_world, dist_coeffs)
                if projected_smpl_2d is not None:
                    valid_proj = ~np.isnan(projected_smpl_2d).any(axis=1)
                    ax_exo.plot(
                        projected_smpl_2d[valid_proj, 0], projected_smpl_2d[valid_proj, 1],
                        'ro', markersize=2, alpha=0.5, label='Pred. Proj.'
                    )
                    for p_idx, c_idx in limbs:
                        if p_idx < len(projected_smpl_2d) and c_idx < len(projected_smpl_2d) and valid_proj[p_idx] and valid_proj[c_idx]:
                            ax_exo.plot(
                                [projected_smpl_2d[p_idx, 0], projected_smpl_2d[c_idx, 0]],
                                [projected_smpl_2d[p_idx, 1], projected_smpl_2d[c_idx, 1]],
                                'r-', alpha=0.4, linewidth=1.0
                            )

                # Project and draw mesh vertices
                if smpl_verts_world is not None:
                    projected_verts_2d = project_points(smpl_verts_world, K, T_cam_world, dist_coeffs)
                    if projected_verts_2d is not None:
                        valid_verts = ~np.isnan(projected_verts_2d).any(axis=1)
                        ax_exo.scatter(
                            projected_verts_2d[valid_verts, 0], projected_verts_2d[valid_verts, 1],
                            s=1, c='cyan', alpha=0.1, label='Mesh Proj.'
                        )
            
            if exo_frame_rgb is not None: # Only set title if video exists
                ax_exo.set_title(f"Exo: {cam_name}")

        ax_exo.axis('off')
        if i == 0:
             handles, labels = ax_exo.get_legend_handles_labels()
             if labels:
                 by_label = dict(zip(labels, handles))
                 ax_exo.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    fig.savefig(output_path)
    plt.close(fig)

# --- Top-Level Pipeline Function ---

def visualize_mocap_prediction(
    sequence_name: str,
    predicted_joints_norm: np.ndarray,
    preprocessed_data_root: str,
    mocap_raw_root: str,
    output_video_path: str,
    dataset_name: str,
    split: str,
    predicted_verts_norm: Optional[np.ndarray] = None
):
    """
    Main pipeline to generate a visualization video for a mocap sequence.
    
    Args:
        sequence_name (str): The name of the sequence (e.g., 'gJB_sBM_cAll_d07_mJB2_ch10').
        predicted_joints_norm (np.ndarray): (T, 22, 3) array of predicted joints
            in the *normalized* space (as output by the dataloader).
        preprocessed_data_root (str): Path to the preprocessed .pt files.
        mocap_raw_root (str): Path to the raw dataset videos.
        output_video_path (str): Path to save the final .mp4 video.
        dataset_name (str): 'AIST', 'TotalCapture', or '3DPW'.
        split (str): 'train' or 'test'.
        predicted_verts_norm (Optional[np.ndarray]): (T, V, 3) array of predicted
            vertices in *normalized* space.
    """
    
    # 1. Initialize and load sequence data
    synchronizer = MocapFrameSynchronizer(
        mocap_raw_root=mocap_raw_root,
        preprocessed_data_root=preprocessed_data_root,
        dataset_name=dataset_name,
        split=split
    )
    synchronizer.load_sequence_data(sequence_name)


    # 2. Check for sequence length mismatch
    num_pred_frames = predicted_joints_norm.shape[0]
    num_sync_frames = len(synchronizer.frame_indices_to_load)
    
    if num_pred_frames != num_sync_frames:
        warnings.warn(
            f"Prediction length mismatch for sequence '{sequence_name}':\n"
            f"Model prediction has {num_pred_frames} frames.\n"
            f"Synchronizer found {num_sync_frames} frames based on dataset logic.\n"
            f"Will truncate to the shorter length ({min(num_pred_frames, num_sync_frames)})."
        )
        T = min(num_pred_frames, num_sync_frames)
    else:
        T = num_pred_frames
        
    # 3. Un-normalize predictions to get them into world space
    # (T, 22, 3) + (1, 1, 3) -> (T, 22, 3)
    norm_offset = synchronizer.norm_offset
    predicted_joints_world = predicted_joints_norm[:T] + norm_offset
    
    predicted_verts_world = None
    if predicted_verts_norm is not None:
        predicted_verts_world = predicted_verts_norm[:T] + norm_offset

    # 4. Get all synchronized frame data (video frames, cam params)
    sync_data_list = synchronizer.get_synchronized_frames()
    
    # 5. Render each frame
    output_frames_paths = []
    temp_dir = tempfile.mkdtemp()
    
    print(f"🎨 Rendering {T} frames for '{sequence_name}'...")
    for t in tqdm(range(T), desc="Rendering frames"):
        synced_data_for_frame = sync_data_list[t]
        output_path = os.path.join(temp_dir, f"frame_{t:05d}.png")
        
        render_mocap_synchronized_frame(
            output_path=output_path,
            frame_idx=synced_data_for_frame['frame_original_index'],
            smpl_joints_world=predicted_joints_world[t],
            exo_frames_data=synced_data_for_frame['exo_frames_data'],
            smpl_verts_world=predicted_verts_world[t] if predicted_verts_world is not None else None,
            limbs=LIMBS_SMPL_22
        )
        output_frames_paths.append(output_path)

    # 6. Compile video
    if output_frames_paths:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        save_animation(output_frames_paths, output_video_path, fps=synchronizer.fps)
        
    # 7. Cleanup
    try:
        for p in output_frames_paths:
            os.remove(p)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temp files in {temp_dir}: {e}")
        
    synchronizer.release()
    print("Visualization complete.")


# --- Example Usage ---

if __name__ == '__main__':
    """
    Example of how to use the visualization pipeline.
    
    This example requires you to have:
    1. The preprocessed .pt files (e.g., in './datasets/RobustCap/dataset_work')
    2. The raw video files (e.g., in './datasets/mocap_raw')
    3. The dataloader code from your prompt (saved as e.g., 'mocap_dataloader.py')
       to fetch a ground truth sample to visualize.
    """
    from data_sets.dataset_mocap import DatasetMocap, DatasetAIST, DatasetTotalCapture, Dataset3DPW
    from utils import dist_util
    from torch.utils.data import Dataset, DataLoader
    from cprint import cprint

    base_data_dir = 'datasets/RobustCap/dataset_work' 
    
    datasets_to_test = {
        "AIST": {
            'class': DatasetAIST,
            'opt': {'root': os.path.join(base_data_dir, 'AIST'), 'split': 'test'}
        },
        "TotalCapture": {
            'class': DatasetTotalCapture,
            'opt': {'root': os.path.join(base_data_dir, 'TotalCapture'), 'split': 'test'}
        },
        "3DPW": {
            'class': Dataset3DPW,
            'opt': {'root': os.path.join(base_data_dir, '3DPW'), 'split': 'test'}
        }
    }
    print("Starting Mocap Visualization Example...")

    for name, config in datasets_to_test.items():
        print("-" * 50)
        print(f"🚀 Testing Dataset: {name}")
        
        # Initialize the dataset
        dataset = config['class'](opt=config['opt'],
                                logdir=config['opt']['root'], 
                                device=dist_util.dev(),
                                body_model_path="data/body_models/smplx_model",
                                joints_num=22,
                                use_ddp=False,
                                disable_cache=False,
                                mean_std=True,
                                norm_strategy='meanstd',
                                repr_abs_only=True,
                                include_global_joint_rotations=False,)
        
        if len(dataset) == 0:
            cprint.warn(f"⚠️  Dataset {name} is empty. Skipping.")
            continue

        # NOTE: batch_size=1 since sequences may have variable lengths
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Fetch and inspect one batch
        first_batch = next(iter(dataloader))
        
        print(f"    Successfully loaded a batch of size: {len(first_batch['meta']['name'])}")
        
        # --- Inspecting the batch content ---
        motion_repr = first_batch['motion_repr_clean']
        cond_mask = first_batch['cond_mask']
        meta = first_batch['meta']
        
        # --- 1. Configuration ---
        # !!! UPDATE THESE PATHS to match your system !!!
        MOCAP_RAW_ROOT = './datasets/mocap_raw'
        PREPROCESSED_DATA_ROOT = './datasets/RobustCap/dataset_work'
        OUTPUT_DIR = './visualizations'
        
        # --- 2. Get a Sample to Visualize (e.g., Ground Truth from Dataset) ---
        # To run this, you need to import your dataloader classes
        # (DatasetAIST, DatasetTotalCapture, Dataset3DPW)
        # and their base class (DatasetBase) and dependencies.
        
        # Since importing the full dataloader setup is complex,
        # we will show the *logic* of how you'd call the visualizer
        # from your existing evaluation script.

        print("-" * 50)
        print("Simulating a call from an evaluation loop (visualizing GT)...")
        
        # --- Mockup: Imagine this data came from your dataloader ---
        # batch = next(iter(your_aist_dataloader))
        # 'batch' is a dictionary. We extract the relevant parts.
        
        # Example 1: AIST
        # --- This is what you'd do in your script ---
        # predicted_joints_norm = batch['joints_world_kin_smpl22'][0].numpy()
        # sequence_name = batch['meta']['name'][0]
        # split = batch['meta']['split'][0]
        # dataset_name = "AIST" # You know this when you create the dataloader
        
        # --- For this demo, we'll hardcode values for one AIST sequence ---
        print("Running AIST Example (Visualizing GT as 'prediction')")
        dataset_name = name
        split = meta['split'][0]
        sequence_name = meta['name'][0]
        output_video_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_{sequence_name}.mp4')

        # To get the GT joints, we temporarily use the Synchronizer
        # In your code, you'd just get this from your dataloader batch
        temp_sync = MocapFrameSynchronizer(MOCAP_RAW_ROOT, PREPROCESSED_DATA_ROOT, dataset_name, split)
        temp_sync.load_sequence_data(sequence_name)
        gt_joints_3d_raw = temp_sync.data['joint3d'][temp_sync.seq_index].float()
        gt_joints_3d_rot = torch.einsum('ij,klj->kli', AIST_ROT, gt_joints_3d_raw)
        gt_joints_sampled = gt_joints_3d_rot[temp_sync.frame_indices_to_load]
        gt_joints_world = gt_joints_sampled[:, :22, :].clone()
        # Apply normalization
        gt_joints_norm = (gt_joints_world - temp_sync.norm_offset).numpy()
        temp_sync.release()
        
        # This 'gt_joints_norm' is what your dataloader provides.
        # Your model's output 'predicted_joints_norm' will be in this same space.
        
        visualize_mocap_prediction(
            sequence_name=sequence_name,
            predicted_joints_norm=gt_joints_norm, # Using GT as the prediction
            preprocessed_data_root=PREPROCESSED_DATA_ROOT,
            mocap_raw_root=MOCAP_RAW_ROOT,
            output_video_path=output_video_path,
            dataset_name=dataset_name,
            split=split,
            predicted_verts_norm=None # Pass (T, V, 3) verts here if you have them
        )
