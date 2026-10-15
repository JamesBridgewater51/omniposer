import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import numpy as np
from utils.other_utils import LIMBS_BODY_SMPL # Or your preferred kinematic tree
from projectaria_tools.core.sophus import SE3 # If poses are SE3
import torch # For tensor operations if needed
import cv2
from typing import List, Dict, Any, Optional
from pathlib import Path
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Helper to project 3D points to 2D (simplified, no distortion handling here)
def project_points(points_3d_world, K, T_cam_world_matrix):
    """
    Projects 3D points from world to 2D image plane.
    points_3d_world: (N, 3)
    K: (3, 3) camera intrinsics
    T_cam_world_matrix: (4, 4) or (3,4) world to camera transformation matrix
    """
    if points_3d_world is None or K is None or T_cam_world_matrix is None:
        return None
    if points_3d_world.shape[0] == 0:
        return np.array([])

    points_3d_world_h = np.concatenate([points_3d_world, np.ones((points_3d_world.shape[0], 1))], axis=1) # (N, 4)
    
    # Ensure T_cam_world_matrix is 3x4 for projection
    if T_cam_world_matrix.shape == (4,4):
        P = K @ T_cam_world_matrix[:3, :] # K @ [R|t]
    elif T_cam_world_matrix.shape == (3,4):
        P = K @ T_cam_world_matrix
    else:
        print(f"[Viz] Invalid T_cam_world_matrix shape: {T_cam_world_matrix.shape}")
        return None

    points_cam_h = (P @ points_3d_world_h.T).T # (N, 3) where last col is depth
    
    # Perspective divide, handle near-zero depth
    depths = points_cam_h[:, 2]
    # Create a mask for points with positive depth (in front of camera)
    valid_depth_mask = depths > 1e-3 # Small epsilon
    
    points_img = np.zeros_like(points_cam_h[:, :2]) # Initialize with zeros or NaNs
    if np.any(valid_depth_mask):
        points_img[valid_depth_mask] = points_cam_h[valid_depth_mask, :2] / depths[valid_depth_mask, np.newaxis]
    
    # Mark points behind camera or with near-zero depth as invalid for plotting (e.g., by setting to NaN)
    points_img[~valid_depth_mask] = np.nan 
    
    return points_img # (N, 2)

def undistort_exocam(image, intrinsics, distortion_coeffs, balance=0.8):
    if intrinsics is None or distortion_coeffs is None:
        return image, None
    dim1 = image.shape[:2][::-1]
    scaled_K = intrinsics.copy()
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim1, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim1, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted_image, new_K

def render_egoexo_synchronized_frame(
    output_path: str,
    frame_idx_in_seq: int,
    smpl_joints_world: np.ndarray, # (J, 3) current SMPL pose in original world coords
    ego_rgb_frame: Optional[np.ndarray],
    ego_frame_original_index: int,
    exo_frames_data: Dict[str, Dict[str, Any]], # {cam_name: {'frame': img, 'params': cam_params, 'annos_2d': annos}}
    point_cloud_world: Optional[np.ndarray] = None, # (P, 3)
    smpl_verts_world: Optional[np.ndarray] = None # (V, 3)
):
    num_exo_cams = len(exo_frames_data)
    num_subplots = 1 + 1 + num_exo_cams
    
    n_cols = 3
    n_rows = int(np.ceil(num_subplots / n_cols))

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # 1. Egocentric View
    ax_ego = fig.add_subplot(n_rows, n_cols, 1)
    if ego_rgb_frame is not None and ego_rgb_frame.size > 0 :
        ax_ego.imshow(ego_rgb_frame)
    ax_ego.set_title(f"Ego Frame (Orig Idx: {ego_frame_original_index})")
    ax_ego.axis('off')

    # 2. 3D SMPL View
    ax_3d = fig.add_subplot(n_rows, n_cols, 2, projection='3d')
    if smpl_joints_world is not None:
        for i, j in LIMBS_BODY_SMPL:
            if i < smpl_joints_world.shape[0] and j < smpl_joints_world.shape[0]:
                ax_3d.plot(
                    [smpl_joints_world[i, 0], smpl_joints_world[j, 0]],
                    [smpl_joints_world[i, 1], smpl_joints_world[j, 1]],
                    [smpl_joints_world[i, 2], smpl_joints_world[j, 2]],
                    'r-'
                )
        ax_3d.scatter(smpl_joints_world[:, 0], smpl_joints_world[:, 1], smpl_joints_world[:, 2], c='b', s=10)
        if smpl_verts_world is not None:
             ax_3d.scatter(smpl_verts_world[::2,0], smpl_verts_world[::2,1], smpl_verts_world[::2,2], s=1, c='c', alpha=0.3)

    ax_3d.set_title("3D View (SMPL)")
    if smpl_joints_world is not None:
        max_range = np.array([smpl_joints_world[:,0].max()-smpl_joints_world[:,0].min(), 
                              smpl_joints_world[:,1].max()-smpl_joints_world[:,1].min(), 
                              smpl_joints_world[:,2].max()-smpl_joints_world[:,2].min()]).max() / 2.0
        mid_x = (smpl_joints_world[:,0].max()+smpl_joints_world[:,0].min()) * 0.5
        mid_y = (smpl_joints_world[:,1].max()+smpl_joints_world[:,1].min()) * 0.5
        mid_z = (smpl_joints_world[:,2].max()+smpl_joints_world[:,2].min()) * 0.5
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')

 # 3. Exocentric Views
    subplot_idx_offset = 3
    for i, (cam_name, exo_data) in enumerate(exo_frames_data.items()):
        if subplot_idx_offset + i > n_cols * n_rows : break
        ax_exo = fig.add_subplot(n_rows, n_cols, subplot_idx_offset + i)
        exo_frame_rgb = exo_data.get('frame')
        cam_params = exo_data.get('params')
        annos_2d = exo_data.get('annos_2d')

        if exo_frame_rgb is not None and exo_frame_rgb.size > 0:
            K = cam_params.get('K')
            dist_coeffs = cam_params.get('dist_coeffs')
            undistorted_frame, new_K = undistort_exocam(exo_frame_rgb, K, dist_coeffs)
            ax_exo.imshow(undistorted_frame)

            if annos_2d:
                for joint_name_egoexo, coords in annos_2d.items():
                    ax_exo.plot(coords['x'], coords['y'], 'go', markersize=3, alpha=0.7, label='GT 2D')
            
            if smpl_joints_world is not None and new_K is not None:
                T_cam_world_se3 = cam_params.get('T_cam_world')
                if T_cam_world_se3 is not None:
                    T_cam_world_matrix = T_cam_world_se3.to_matrix() if isinstance(T_cam_world_se3, SE3) else np.array(T_cam_world_se3)
                    
                    # Project and draw skeleton
                    projected_smpl_2d = project_points(smpl_joints_world, new_K, T_cam_world_matrix)
                    if projected_smpl_2d is not None:
                        valid_proj = ~np.isnan(projected_smpl_2d).any(axis=1)
                        ax_exo.plot(projected_smpl_2d[valid_proj, 0], projected_smpl_2d[valid_proj, 1], 'ro', markersize=2, alpha=0.5, label='Pred SMPL Proj.')
                        for p_idx, c_idx in LIMBS_BODY_SMPL[1:]:
                            if p_idx < len(projected_smpl_2d) and c_idx < len(projected_smpl_2d) and valid_proj[p_idx] and valid_proj[c_idx]:
                                ax_exo.plot([projected_smpl_2d[p_idx,0], projected_smpl_2d[c_idx,0]],
                                            [projected_smpl_2d[p_idx,1], projected_smpl_2d[c_idx,1]], 'r-', alpha=0.4, linewidth=0.5)

                    # Project and draw mesh vertices
                    if smpl_verts_world is not None:
                        projected_smpl_verts_2d = project_points(smpl_verts_world, new_K, T_cam_world_matrix)
                        if projected_smpl_verts_2d is not None:
                            valid_verts_proj = ~np.isnan(projected_smpl_verts_2d).any(axis=1)
                            ax_exo.scatter(
                                projected_smpl_verts_2d[valid_verts_proj, 0], 
                                projected_smpl_verts_2d[valid_verts_proj, 1], 
                                s=1, c='cyan', alpha=0.2, label='Pred Mesh Proj.'
                            )

        ax_exo.set_title(f"Exo: {cam_name}")
        ax_exo.axis('off')
        if i == 0 and (annos_2d or (smpl_joints_world is not None and cam_params)):
             handles, labels = ax_exo.get_legend_handles_labels()
             by_label = dict(zip(labels, handles))
             ax_exo.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='xx-small')

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def save_egoexo_animation(output_frames_list: List[str], video_output_path: str, fps: int = 30):
    if not output_frames_list:
        print("[Viz] No frames to create animation.")
        return
    
    Path(video_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    clip = ImageSequenceClip(output_frames_list, fps=fps)
    clip.write_videofile(video_output_path, codec='libx264')
    print(f"[Viz] Saved animation to {video_output_path}")