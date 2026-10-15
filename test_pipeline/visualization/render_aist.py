"""
AIST++ video mesh renderer.
Renders SMPLX mesh predictions onto AIST++ dataset videos.
"""
import os
import json
import torch
import cv2
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from typing import Dict, Optional, Tuple

from test_pipeline.postprocessing import recover_world_from_repr

from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


MATERIAL_PRED = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(66/255, 149/255, 245/255, 1.0)
)

MATERIAL_GT = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255/255, 102/255, 102/255, 1.0)
)


@VisualizerRegistry.register('aist')
class AISTMeshRenderer(BaseVisualizer):
    """Render SMPLX meshes onto AIST++ video frames."""

    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.renderer = None
        self.video_writer = None
        self.video_cap = None
        self.current_seq = None
        self.camera_id = getattr(test_config, 'camera_id', 0)
        self.render_gt = getattr(test_config, 'render_gt', True)
        self.overlay_alpha = getattr(test_config, 'overlay_alpha', 0.7)
        self.debug_visualize = getattr(test_config, 'debug_visualize', False)

        # AIST raw data path
        self.aist_raw_dir = Path("datasets/AIST/aist_plusplus_final")

        # Load camera mapping
        self.camera_mapping = self._load_camera_mapping()

    def _load_camera_mapping(self) -> Dict[str, str]:
        """Load sequence to camera setting mapping."""
        mapping_file = self.aist_raw_dir / "cameras" / "mapping.txt"
        if not mapping_file.exists():
            return {}
        mapping = {}
        with open(mapping_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
        return mapping

    def setup(self) -> None:
        if self._is_initialized:
            return
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self._is_initialized = True

    def visualize(self, data: VisualizationData) -> None:
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
        if 'meta' in batch and 'name' in batch['meta']:
            name = batch['meta']['name']
            return name[0] if isinstance(name, (list, tuple)) else name
        raise ValueError("Sequence name not found in batch")

    def _find_video_path(self, seq_name: str) -> Optional[Path]:
        """Find AIST video file. Sequence uses cAll, video uses c0X."""
        cam_str = f"c0{self.camera_id + 1}"
        video_name = seq_name.replace('cAll', cam_str) + '.mp4'
        video_path = self.aist_raw_dir / "video" / video_name
        if video_path.exists():
            return video_path
        return None

    def _load_camera_params(self, seq_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Load camera K and T from raw AIST data, applying scale from SMPL data."""
        setting = self.camera_mapping.get(seq_name)
        if not setting:
            return None, None, 1.0

        cam_file = self.aist_raw_dir / "cameras" / f"{setting}.json"
        if not cam_file.exists():
            return None, None, 1.0

        # Load scale from SMPL motion data
        import pickle
        smpl_file = self.aist_raw_dir / "motions" / f"{seq_name}.pkl"
        scale = 1.0
        if smpl_file.exists():
            smpl_data = pickle.load(open(smpl_file, 'rb'), encoding='latin1')
            scale = smpl_data['smpl_scaling'].item()

        with open(cam_file) as f:
            cam_data = json.load(f)

        cam = cam_data[self.camera_id]
        cam_K = np.array(cam['matrix'], dtype=np.float32)

        # Build extrinsic matrix from rotation (axis-angle) and translation
        # Apply scale to translation as done in preprocessing
        from scipy.spatial.transform import Rotation as R
        rot = R.from_rotvec(cam['rotation']).as_matrix()
        trans = np.array(cam['translation'], dtype=np.float32) / scale

        cam_T = np.eye(4, dtype=np.float32)
        cam_T[:3, :3] = rot
        cam_T[:3, 3] = trans

        return cam_K, cam_T, scale

    def _init_video(self, batch: Dict, seq_name: str):
        self._cleanup_video()

        video_path = self._find_video_path(seq_name)
        if video_path is None:
            print(f"[AISTMeshRenderer] Video not found for: {seq_name}")
            return

        self.video_cap = cv2.VideoCapture(str(video_path))
        if not self.video_cap.isOpened():
            print(f"[AISTMeshRenderer] Failed to open: {video_path}")
            self.video_cap = None
            return

        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 60.0

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )

        output_dir = Path(self.test_config.save_root) / "video_renders"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}_c0{self.camera_id+1}_mesh.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        self.current_seq = seq_name

        print(f"[AISTMeshRenderer] Rendering: {seq_name} -> {output_path}")

    def _transform_verts_to_camera(self, verts: np.ndarray, cam_T: np.ndarray) -> np.ndarray:
        """Transform vertices from world to camera coordinates."""
        R = cam_T[:3, :3]
        t = cam_T[:3, 3]
        return (R @ verts.T).T + t

    def _create_scene(self, cam_K: np.ndarray, width: int, height: int):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        return scene

    def _render_mesh(self, scene: pyrender.Scene, verts: np.ndarray, faces: np.ndarray,
                     material: pyrender.MetallicRoughnessMaterial) -> np.ndarray:
        body_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        render_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=material)

        mesh_node = scene.add(render_mesh)
        color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(mesh_node)

        return color

    def _blend_overlay(self, background: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        alpha = (overlay_rgba[:, :, 3:4] > 0).astype(np.float32) * self.overlay_alpha
        overlay_rgb = overlay_rgba[:, :, :3][:, :, ::-1]  # Convert RGB to BGR for cv2
        result = background.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha
        return result.astype(np.uint8)

    def _render_batch(self, data: VisualizationData, batch: Dict):
        seq_name = self._get_sequence_name(batch)
        cam_K, cam_T, scale = self._load_camera_params(seq_name)
        if cam_K is None:
            print(f"[AISTMeshRenderer] Missing camera params for {seq_name}")
            return

        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        meta = batch.get('meta')
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
                verts_gt_cam = self._transform_verts_to_camera(verts_gt_world[0, t], cam_T)
                gt_overlay = self._render_mesh(scene, verts_gt_cam, faces, MATERIAL_GT)
                frame = self._blend_overlay(frame, gt_overlay)

            # Transform and render prediction
            verts_pred_cam = self._transform_verts_to_camera(verts_pred_world[0, t], cam_T)
            pred_overlay = self._render_mesh(scene, verts_pred_cam, faces, MATERIAL_PRED)
            frame = self._blend_overlay(frame, pred_overlay)

            if self.debug_visualize:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

            self.video_writer.write(frame)

    def _cleanup_video(self):
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
        self._cleanup_video()
        self.current_seq = None
        self._is_initialized = False
