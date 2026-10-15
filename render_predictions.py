"""
Render model predictions onto real dataset sequences.
Handles coordinate transforms, alignment, and visualization.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# from smpl_based_model.smplx import SMPLX
from smplx import SMPLX
from data_sets.motion_representation import recover_from_repr_smpl


@dataclass
class RenderConfig:
    """Rendering configuration."""
    dataset_name: str  # 'aist', 'totalcapture', '3dpw'
    sequence_name: str
    camera_id: int = 0
    output_dir: str = "render_output"
    overlay_alpha: float = 0.6
    render_gt: bool = True
    render_pred: bool = True


class CoordinateTransform:
    """Handle coordinate system transforms between datasets."""
    
    @staticmethod
    def world_to_camera(points: torch.Tensor, cam_R: torch.Tensor, cam_t: torch.Tensor) -> torch.Tensor:
        """Transform points from world to camera coordinates."""
        return torch.matmul(cam_R, points.unsqueeze(-1)).squeeze(-1) + cam_t
    
    @staticmethod
    def camera_to_world(points: torch.Tensor, cam_R: torch.Tensor, cam_t: torch.Tensor) -> torch.Tensor:
        """Transform points from camera to world coordinates."""
        return torch.matmul(cam_R.transpose(-2, -1), (points - cam_t).unsqueeze(-1)).squeeze(-1)
    
    @staticmethod
    def project_to_image(points_3d: torch.Tensor, cam_K: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates."""
        points_2d = torch.matmul(cam_K, points_3d.unsqueeze(-1)).squeeze(-1)
        return points_2d[..., :2] / points_2d[..., 2:3]


class DatasetLoader:
    """Load preprocessed dataset sequences."""
    
    def __init__(self, dataset_name: str, data_root: str = "datasets"):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
    
    def load_sequence(self, seq_name: str, split: str = "test") -> Dict:
        """Load sequence data."""
        if self.dataset_name == "aist":
            return self._load_aist(seq_name, split)
        elif self.dataset_name == "totalcapture":
            return self._load_totalcapture(seq_name, split)
        elif self.dataset_name == "3dpw":
            return self._load_3dpw(seq_name, split)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_aist(self, seq_name: str, split: str) -> Dict:
        """Load AIST sequence."""
        data_file = self.data_root / "AIST" / f"{split}.pt"
        data = torch.load(data_file)
        idx = data['name'].index(seq_name)
        return {
            'pose': data['pose'][idx],  # (T, 72)
            'tran': data['tran'][idx],  # (T, 3)
            'cam_K': data['cam_K'][idx],  # (9, 3, 3)
            'cam_T': data['cam_T'][idx],  # (9, 4, 4)
            'video_path': self.data_root / "mocap_raw" / "AIST" / "video" / f"{seq_name}.mp4"
        }
    
    def _load_totalcapture(self, seq_name: str, split: str) -> Dict:
        """Load TotalCapture sequence."""
        data_file = self.data_root / "TotalCapture-Toolbox" / f"{split}.pt"
        data = torch.load(data_file)
        idx = data['name'].index(seq_name)
        return {
            'pose': data['pose'][idx],  # (T, 24, 3)
            'tran': data['tran'][idx],  # (T, 3)
            'cam_K': data['cam_K'][idx],  # (8, 3, 3)
            'cam_T': data['cam_T'][idx],  # (8, 4, 4)
            'video_path': self._find_totalcapture_video(seq_name)
        }
    
    def _load_3dpw(self, seq_name: str, split: str) -> Dict:
        """Load 3DPW sequence."""
        data_file = self.data_root / "3DPW" / f"{split}.pt"
        data = torch.load(data_file)
        idx = data['name'].index(seq_name)
        return {
            'posec': data['posec'][idx],  # (T, 24, 3, 3) in camera coords
            'tranc': data['tranc'][idx],  # (T, 3) in camera coords
            'cam_K': data['cam_K'][idx],  # (3, 3)
            'cam_T': data['cam_T'][idx],  # (T, 4, 4)
            'video_path': self._find_3dpw_video(seq_name)
        }
    
    def _find_totalcapture_video(self, seq_name: str) -> Path:
        """Find TotalCapture video file."""
        parts = seq_name.split('_')
        subject, motion, cam = parts[0], parts[1], parts[2]
        video_dir = self.data_root / "mocap_raw" / "TotalCapture" / "video" / subject.upper() / motion
        videos = list(video_dir.glob(f"{cam}_*.mp4"))
        return videos[0] if videos else None
    
    def _find_3dpw_video(self, seq_name: str) -> Path:
        """Find 3DPW video file."""
        video_path = self.data_root / "mocap_raw" / "3DPW" / "imageFiles" / seq_name / "image_00000.jpg"
        return video_path.parent if video_path.parent.exists() else None


class SMPLRenderer:
    """Render SMPL meshes onto images."""
    
    def __init__(self, smplx_model: SMPLX):
        self.smplx_model = smplx_model
        self.coord_transform = CoordinateTransform()
    
    def render_frame(
        self,
        image: np.ndarray,
        pose: torch.Tensor,
        tran: torch.Tensor,
        cam_K: torch.Tensor,
        cam_T: Optional[torch.Tensor] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        in_camera_coords: bool = False
    ) -> np.ndarray:
        """Render SMPL on single frame."""
        # Get SMPL vertices
        if pose.shape[-1] == 3:  # axis-angle
            pose_mat = self._axis_angle_to_matrix(pose)
        else:
            pose_mat = pose
        
        with torch.no_grad():
            output = self.smplx_model(
                body_pose=pose_mat[1:],
                global_orient=pose_mat[0:1],
                transl=tran.unsqueeze(0)
            )
            verts = output.vertices[0]  # (V, 3)
        
        # Transform to camera coordinates if needed
        if not in_camera_coords and cam_T is not None:
            cam_R, cam_t = cam_T[:3, :3], cam_T[:3, 3]
            verts = self.coord_transform.world_to_camera(verts, cam_R, cam_t)
        
        # Project to image
        verts_2d = self.coord_transform.project_to_image(verts, cam_K)
        
        # Draw on image
        overlay = image.copy()
        for v2d in verts_2d:
            x, y = int(v2d[0].item()), int(v2d[1].item())
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(overlay, (x, y), 2, color, -1)
        
        return overlay
    
    @staticmethod
    def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix."""
        from utils.konia_transform import axis_angle_to_matrix
        return axis_angle_to_matrix(axis_angle.reshape(-1, 3)).reshape(*axis_angle.shape[:-1], 3, 3)


class PredictionRenderer:
    """Main rendering pipeline."""
    
    def __init__(self, config: RenderConfig, smplx_model: SMPLX):
        self.config = config
        self.loader = DatasetLoader(config.dataset_name)
        self.renderer = SMPLRenderer(smplx_model)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def render_sequence(
        self,
        gt_data: Dict,
        pred_repr_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Render full sequence with GT and predictions."""
        # Load video
        video_path = gt_data['video_path']
        if video_path is None or not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video
        out_path = self.output_dir / f"{self.config.sequence_name}_cam{self.config.camera_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        
        # Get camera params
        cam_id = self.config.camera_id
        cam_K = gt_data['cam_K'][cam_id] if gt_data['cam_K'].ndim > 2 else gt_data['cam_K']
        cam_T = gt_data['cam_T'][cam_id] if gt_data['cam_T'].ndim > 2 else None
        
        # Convert predictions to SMPL if provided
        if pred_repr_dict is not None:
            pred_pose, pred_tran = self._repr_to_smpl(pred_repr_dict)
        else:
            pred_pose, pred_tran = None, None
        
        # Render frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= len(gt_data['pose']):
                break
            
            # Render GT
            if self.config.render_gt:
                gt_pose = gt_data['pose'][frame_idx]
                gt_tran = gt_data['tran'][frame_idx]
                in_cam_coords = 'posec' in gt_data
                frame = self.renderer.render_frame(
                    frame, gt_pose, gt_tran, cam_K, cam_T,
                    color=(0, 0, 255), in_camera_coords=in_cam_coords
                )
            
            # Render prediction
            if self.config.render_pred and pred_pose is not None:
                frame = self.renderer.render_frame(
                    frame, pred_pose[frame_idx], pred_tran[frame_idx],
                    cam_K, cam_T, color=(0, 255, 0)
                )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"Saved: {out_path}")
    
    def _repr_to_smpl(self, repr_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert representation dict to SMPL pose and translation."""
        pose, tran, _ = recover_from_repr_smpl(
            repr_dict,
            smplx_model=self.renderer.smplx_model,
            return_joints=False
        )
        return pose, tran


def render_predictions_on_dataset(
    dataset_name: str,
    sequence_name: str,
    pred_repr_dict: Optional[Dict[str, torch.Tensor]] = None,
    camera_id: int = 0,
    output_dir: str = "render_output"
):
    """
    Main entry point for rendering predictions.
    
    Args:
        dataset_name: 'aist', 'totalcapture', or '3dpw'
        sequence_name: Name of sequence to render
        pred_repr_dict: Model predictions (optional, renders GT only if None)
        camera_id: Camera index for multi-camera datasets
        output_dir: Output directory for rendered videos
    """
    # Load SMPL model
    smplx_model = SMPLX("data/body_models/smplx_model/smplx", gender="neutral")
    
    # Setup config
    config = RenderConfig(
        dataset_name=dataset_name,
        sequence_name=sequence_name,
        camera_id=camera_id,
        output_dir=output_dir
    )
    
    # Load GT data
    loader = DatasetLoader(dataset_name, data_root = "datasets/RobustCap/dataset_work/")
    gt_data = loader.load_sequence(sequence_name)
    
    # Render
    renderer = PredictionRenderer(config, smplx_model)
    renderer.render_sequence(gt_data, pred_repr_dict)


if __name__ == "__main__":
    # Example usage
    render_predictions_on_dataset(
        dataset_name="aist",
        sequence_name="gJS_sBM_cAll_d03_mJS2_ch03",
        camera_id=0
    )
