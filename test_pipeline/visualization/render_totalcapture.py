"""
TotalCapture video mesh renderer.
Renders SMPLX mesh predictions onto TotalCapture dataset videos.
"""
import os
import torch
import cv2
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from test_pipeline.postprocessing import recover_world_from_repr

from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


# Material definitions for rendering
MATERIAL_PRED = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(66/255, 149/255, 245/255, 1.0)  # Light blue for predictions
)

MATERIAL_GT = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255/255, 102/255, 102/255, 1.0)  # Light red for ground truth
)


@VisualizerRegistry.register('totalcapture')
class TotalCaptureMeshRenderer(BaseVisualizer):
    """
    Render SMPLX meshes onto TotalCapture video frames using pyrender.

    Supports rendering both predicted meshes and ground truth meshes with
    proper camera projections using the dataset's calibration parameters.
    """

    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.renderer = None
        self.video_writer = None
        self.video_cap = None
        self.current_seq = None
        self.camera_id = getattr(test_config, 'camera_id', 1)
        self.render_gt = getattr(test_config, 'render_gt', True)
        self.overlay_alpha = getattr(test_config, 'overlay_alpha', 0.7)
        self.debug_visualize = getattr(test_config, 'debug_visualize', False)

        # TotalCapture raw data path
        self.tc_raw_dir = Path("datasets/TotalCapture")
    
    def setup(self) -> None:
        """Initialize pyrender offscreen renderer."""
        if self._is_initialized:
            return
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self._is_initialized = True
    
    def visualize(self, data: VisualizationData) -> None:
        """Render predictions onto video frames."""
        if not self.should_visualize():
            return
        
        if not self._is_initialized:
            self.setup()
        
        batch = data.batch
        seq_name = self._get_sequence_name(batch)
        
        if seq_name != self.current_seq:
            self._init_video(batch, seq_name)
        
        if self.video_cap is None or self.video_writer is None:
            return
        
        self._render_batch(data, batch)
    
    def _get_sequence_name(self, batch: Dict) -> str:
        """Extract sequence name from batch."""
        if 'meta' in batch and 'name' in batch['meta']:
            name = batch['meta']['name']
            return name[0] if isinstance(name, (list, tuple)) else name
        if 'meta' in batch and 'name' in batch['meta']:
            name = batch['meta']['name']
            return name[0] if isinstance(name, (list, tuple)) else name
        raise ValueError("Sequence name not found in batch")
    
    def _find_video_path(self, seq_name: str) -> Optional[Path]:
        """
        Find TotalCapture video file from sequence name.
        Sequence name format: {subject}_{motion}
        Video format: TC_S{n}_{motion}_cam{id}.mp4
        """
        if seq_name.startswith("TC_"):
            seq_name = seq_name[3:]

        parts = seq_name.split('_')
        if len(parts) < 2:
            return None

        subject = parts[0].lower()  # e.g., "s1"
        motion = parts[1]  # e.g., "acting1"
        cam_id = self.camera_id + 1

        video_dir = self.tc_raw_dir / "video" / subject / motion

        if not video_dir.exists():
            return None

        # TotalCapture video naming: TC_S1_acting1_cam1.mp4
        pattern = f"TC_{subject.upper()}_{motion}_cam{cam_id}.mp4"
        video_path = video_dir / pattern
        if video_path.exists():
            return video_path

        # Fallback: search for any matching pattern
        for video in video_dir.glob(f"*cam{cam_id}*.mp4"):
            return video

        return None
    
    def _init_video(self, batch: Dict, seq_name: str):
        """Initialize video capture and writer for new sequence."""
        self._cleanup_video()
        
        video_path = self._find_video_path(seq_name)
        if video_path is None or not video_path.exists():
            print(f"[TotalCaptureMeshRenderer] Video not found for: {seq_name}")
            breakpoint()
        
        self.video_cap = cv2.VideoCapture(str(video_path))
        if not self.video_cap.isOpened():
            print(f"[TotalCaptureMeshRenderer] Failed to open: {video_path}")
            self.video_cap = None
            breakpoint()
        
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 60.0
        
        # Initialize pyrender renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )
        
        # Setup output directory and video writer
        output_dir = Path(self.test_config.save_root) / "video_renders"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}_mesh.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        self.current_seq = seq_name
        
        print(f"[TotalCaptureMeshRenderer] Rendering: {seq_name} -> {output_path}")
    
    def _load_calibration(self) -> None:
        """Load camera calibration from TotalCapture calibration.cal file."""
        if hasattr(self, '_cam_params'):
            return

        cal_file = self.tc_raw_dir / "calibration.cal"
        if not cal_file.exists():
            self._cam_params = None
            return

        cams = []
        with open(cal_file) as f:
            f.readline()  # Skip header
            for _ in range(8):  # 8 cameras
                f.readline()  # bounds line
                intrinsics = f.readline().strip().split()
                fx, fy, cx, cy = float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])
                cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

                f.readline()  # distortion
                r1 = [float(x) for x in f.readline().strip().split()]
                r2 = [float(x) for x in f.readline().strip().split()]
                r3 = [float(x) for x in f.readline().strip().split()]
                cam_R = np.array([r1, r2, r3], dtype=np.float32)

                t = [float(x) for x in f.readline().strip().split()]
                cam_t = np.array(t, dtype=np.float32)

                cam_T = np.eye(4, dtype=np.float32)
                cam_T[:3, :3] = cam_R
                cam_T[:3, 3] = cam_t

                cams.append((cam_K, cam_T))

        self._cam_params = cams

    def _get_camera_params(self, batch: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera K and T for configured camera ID."""
        self._load_calibration()

        if self._cam_params is None or self.camera_id >= len(self._cam_params):
            return None, None

        return self._cam_params[self.camera_id]
    
    def _create_scene(self, cam_K: np.ndarray, width: int, height: int):
        """Create pyrender scene with camera and lighting."""
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        
        # Create camera from intrinsics
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
        
        # Camera pose - pyrender uses OpenGL convention
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)
        
        return scene

    def _transform_verts_to_camera(self, verts: np.ndarray, cam_T: np.ndarray) -> np.ndarray:
        """Transform vertices from world to camera coordinates."""
        if cam_T.shape == (4, 4):
            R = cam_T[:3, :3]
            t = cam_T[:3, 3]
            verts_cam = (R @ verts.T).T + t
        else:
            verts_cam = verts
        return verts_cam

    def _undo_preprocessing_trans_adjust(self, verts: np.ndarray, meta: Dict, frame_idx: int) -> np.ndarray:
        return verts
    
    def _render_mesh(self, scene: pyrender.Scene, verts: np.ndarray, faces: np.ndarray,
                     material: pyrender.MetallicRoughnessMaterial) -> np.ndarray:
        """Render a mesh and return RGBA image."""
        body_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        render_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=material)
        
        mesh_node = scene.add(render_mesh)
        color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(mesh_node)
        
        return color
    
    def _blend_overlay(self, background: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """Blend RGBA overlay onto background image."""
        alpha = (overlay_rgba[:, :, 3:4] > 0).astype(np.float32) * self.overlay_alpha
        overlay_rgb = overlay_rgba[:, :, :3][:, :, ::-1]  # Convert RGB to BGR for cv2
        
        result = background.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha
        return result.astype(np.uint8)
    
    def _render_batch(self, data: VisualizationData, batch: Dict):
        """Render all frames in batch onto video."""
        cam_K, cam_T = self._get_camera_params(batch)
        if cam_K is None:
            print("[TotalCaptureMeshRenderer] Missing camera parameters")
            return

        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        meta = batch.get('meta', {})
        repr_pred = data.repr_dict_rec_tries[0]
        _, verts_pred_world = recover_world_from_repr(
            repr_pred,
            meta,
            self.smplx_model,
            return_full_joints=False,
        )
        if isinstance(verts_pred_world, torch.Tensor):
            verts_pred_world = verts_pred_world.detach().cpu().numpy()

        verts_gt_world = None
        if self.render_gt and data.repr_dict_clean is not None:
            _, verts_gt_world = recover_world_from_repr(
                data.repr_dict_clean,
                meta,
                self.smplx_model,
                return_full_joints=False,
            )
            if isinstance(verts_gt_world, torch.Tensor):
                verts_gt_world = verts_gt_world.detach().cpu().numpy()

        faces = self.smplx_model.faces.astype(np.int32)
        _bs, T = verts_pred_world.shape[:2]

        for t in range(T):
            ret, frame = self.video_cap.read()
            if not ret:
                break

            scene = self._create_scene(cam_K, width, height)

            # Transform and render GT
            if verts_gt_world is not None:
                verts_gt_world_t = self._undo_preprocessing_trans_adjust(verts_gt_world[0, t], meta, t)
                verts_gt_cam = self._transform_verts_to_camera(verts_gt_world_t, cam_T)
                gt_overlay = self._render_mesh(scene, verts_gt_cam, faces, MATERIAL_GT)
                frame = self._blend_overlay(frame, gt_overlay)

            # Transform and render prediction
            verts_pred_world_t = self._undo_preprocessing_trans_adjust(verts_pred_world[0, t], meta, t)
            verts_pred_cam = self._transform_verts_to_camera(verts_pred_world_t, cam_T)
            pred_overlay = self._render_mesh(scene, verts_pred_cam, faces, MATERIAL_PRED)
            frame = self._blend_overlay(frame, pred_overlay)

            if self.debug_visualize:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

            self.video_writer.write(frame)
    
    def _cleanup_video(self):
        """Release video resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
    
    def cleanup(self) -> None:
        """Release all resources."""
        self._cleanup_video()
        self.current_seq = None
        self._is_initialized = False
