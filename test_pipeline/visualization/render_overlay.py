"""
Video overlay renderer for model predictions on real dataset videos.
Integrates with test_pipeline for seamless rendering during evaluation.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


@dataclass
class VideoConfig:
    """Video rendering configuration."""
    fps: float = 30.0
    codec: str = 'mp4v'
    overlay_alpha: float = 0.6
    point_radius: int = 2
    line_thickness: int = 2


class VideoOverlayRenderer(BaseVisualizer):
    """
    Render SMPL predictions as overlay on original dataset videos.
    Handles coordinate transforms and camera projections.
    """
    
    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.video_config = VideoConfig()
        self.video_writer = None
        self.video_cap = None
        self.current_seq = None
    
    def setup(self) -> None:
        """Initialize video resources."""
        self._is_initialized = True
    
    def visualize(self, data: VisualizationData) -> None:
        """Render predictions onto video frames."""
        if not self.should_visualize():
            return
        
        batch = data.batch
        seq_name = self._get_sequence_name(batch)
        
        # Initialize video for new sequence
        if seq_name != self.current_seq:
            self._init_video(batch, seq_name)
        
        # Render each frame
        self._render_batch(data, batch)
    
    def _get_sequence_name(self, batch: Dict) -> str:
        """Extract sequence name from batch."""
        if 'name' in batch:
            return batch['name'][0] if isinstance(batch['name'], list) else batch['name']
        return "unknown"
    
    def _init_video(self, batch: Dict, seq_name: str):
        """Initialize video capture and writer for sequence."""
        self._cleanup_video()
        
        video_path = self._find_video_path(batch, seq_name)
        if video_path is None or not video_path.exists():
            print(f"[VideoOverlay] Video not found: {video_path}")
            return
        
        # Open input video
        self.video_cap = cv2.VideoCapture(str(video_path))
        if not self.video_cap.isOpened():
            print(f"[VideoOverlay] Failed to open: {video_path}")
            return
        
        # Setup output video
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or self.video_config.fps
        
        output_dir = Path(self.test_config.save_root) / "video_overlays"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*self.video_config.codec)
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        self.current_seq = seq_name
        
        print(f"[VideoOverlay] Rendering: {seq_name} -> {output_path}")
    
    def _find_video_path(self, batch: Dict, seq_name: str) -> Optional[Path]:
        """Find video file for sequence."""
        dataset_name = self.test_config.dataset_name
        
        if dataset_name == "aist":
            return self._find_aist_video(seq_name, batch)
        elif dataset_name == "totalcapture":
            return self._find_totalcapture_video(seq_name)
        elif dataset_name == "3dpw":
            return self._find_3dpw_video(seq_name)
        return None
    
    def _find_aist_video(self, seq_name: str, batch: Dict) -> Optional[Path]:
        """Find AIST video file."""
        cam_id = batch.get('camera_id', [0])[0] if 'camera_id' in batch else 0
        video_name = seq_name.replace('cAll', f'c0{cam_id + 1}')
        video_path = Path("datasets/mocap_raw/AIST/video") / f"{video_name}.mp4"
        return video_path if video_path.exists() else None
    
    def _find_totalcapture_video(self, seq_name: str) -> Optional[Path]:
        """Find TotalCapture video file."""
        parts = seq_name.split('_')
        if len(parts) < 3:
            return None
        subject, motion, cam = parts[0], parts[1], parts[2]
        video_dir = Path(f"datasets/mocap_raw/TotalCapture/video/{subject.upper()}/{motion}")
        videos = list(video_dir.glob(f"{cam}_*.mp4"))
        return videos[0] if videos else None
    
    def _find_3dpw_video(self, seq_name: str) -> Optional[Path]:
        """Find 3DPW video file (image sequence)."""
        return Path(f"datasets/mocap_raw/3DPW/imageFiles/{seq_name}")
    
    def _render_batch(self, data: VisualizationData, batch: Dict):
        """Render all frames in batch."""
        if self.video_cap is None or self.video_writer is None:
            return
        
        bs, T = data.jts_rec_tries[0].shape[:2]
        cam_K, cam_T = self._get_camera_params(batch)
        
        for t in range(T):
            ret, frame = self.video_cap.read()
            if not ret:
                break
            
            # Render GT
            if data.jts_clean is not None:
                jts_gt = data.jts_clean[0, t].cpu().numpy()
                frame = self._draw_skeleton(frame, jts_gt, cam_K, cam_T, (0, 0, 255))
            
            # Render predictions (contract guarantees aligned try lists)
            for try_idx, (jts_rec, color) in enumerate(zip(data.jts_rec_tries, data.color_vis_tries)):
                jts_pred = jts_rec[0, t].cpu().numpy()
                color_bgr = tuple(int(c * 255) for c in reversed(color))
                frame = self._draw_skeleton(frame, jts_pred, cam_K, cam_T, color_bgr)
            
            self.video_writer.write(frame)
    
    def _get_camera_params(self, batch: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract camera intrinsics and extrinsics from batch."""
        cam_K = batch.get('cam_K', None)
        cam_T = batch.get('cam_T', None)
        
        if cam_K is not None:
            cam_K = cam_K[0].cpu().numpy() if isinstance(cam_K, torch.Tensor) else cam_K
        if cam_T is not None:
            cam_T = cam_T[0].cpu().numpy() if isinstance(cam_T, torch.Tensor) else cam_T
        
        return cam_K, cam_T
    
    def _draw_skeleton(
        self,
        image: np.ndarray,
        joints_3d: np.ndarray,
        cam_K: Optional[np.ndarray],
        cam_T: Optional[np.ndarray],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw skeleton on image."""
        if cam_K is None:
            return image
        
        # Transform to camera coords if needed
        if cam_T is not None and cam_T.shape == (4, 4):
            cam_R, cam_t = cam_T[:3, :3], cam_T[:3, 3]
            joints_cam = (cam_R @ joints_3d.T).T + cam_t
        else:
            joints_cam = joints_3d
        
        # Project to image
        joints_2d = (cam_K @ joints_cam.T).T
        joints_2d = joints_2d[:, :2] / joints_2d[:, 2:3]
        
        # Draw joints
        for j2d in joints_2d:
            x, y = int(j2d[0]), int(j2d[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), self.video_config.point_radius, color, -1)
        
        # Draw bones
        from .base import SMPL_SKELETON_CONNECTIONS
        for conn in SMPL_SKELETON_CONNECTIONS:
            if conn[0] < len(joints_2d) and conn[1] < len(joints_2d):
                p1 = tuple(joints_2d[conn[0]].astype(int))
                p2 = tuple(joints_2d[conn[1]].astype(int))
                if (0 <= p1[0] < image.shape[1] and 0 <= p1[1] < image.shape[0] and
                    0 <= p2[0] < image.shape[1] and 0 <= p2[1] < image.shape[0]):
                    cv2.line(image, p1, p2, color, self.video_config.line_thickness)
        
        return image
    
    def _cleanup_video(self):
        """Release video resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
    
    def cleanup(self) -> None:
        """Release all resources."""
        self._cleanup_video()
        self.current_seq = None
        self._is_initialized = False


# Register for MoCap datasets
@VisualizerRegistry.register('aist_overlay')
@VisualizerRegistry.register('totalcapture_overlay')
@VisualizerRegistry.register('3dpw_overlay')
class MoCaPVideoOverlayVisualizer(VideoOverlayRenderer):
    """Video overlay visualizer for MoCap datasets."""
    pass
