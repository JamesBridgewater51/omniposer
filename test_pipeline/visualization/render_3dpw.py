"""
3DPW image renderer.
Renders SMPLX mesh predictions onto 3DPW image frames.
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyrender
import torch
import trimesh

from test_pipeline.postprocessing import recover_world_from_repr
from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


MATERIAL_PRED = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(66 / 255, 149 / 255, 245 / 255, 1.0),
)

MATERIAL_GT = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255 / 255, 102 / 255, 102 / 255, 1.0),
)


@VisualizerRegistry.register('3dpw')
class PW3DMeshRenderer(BaseVisualizer):
    """Render SMPLX meshes onto 3DPW image frames."""

    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_seq: Optional[str] = None
        self.render_gt = getattr(test_config, 'render_gt', True)
        self.overlay_alpha = getattr(test_config, 'overlay_alpha', 0.7)
        self.apply_extrinsics = getattr(test_config, 'apply_extrinsics', False)
        self.default_fps = getattr(test_config, 'fps', 30.0)
        self.save_root = Path(getattr(test_config, 'save_root', 'test_results'))
        self.pw3d_raw_dir = Path(getattr(test_config, 'pw3d_raw_dir', 'datasets/3DPW'))
        self.debug_visualize = getattr(test_config, 'debug_visualize', False)
        self.seq_cache: Dict[str, Dict] = {}

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
        split = self._get_split(batch)

        seq_info = self._load_sequence_info(seq_name, split)
        if seq_info is None or not seq_info['image_paths']:
            print(f"[PW3DMeshRenderer] Missing assets for {seq_name}")
            return

        if seq_name != self.current_seq:
            self._init_video(seq_name, seq_info)

        if self.video_writer is None or self.renderer is None:
            return

        self._render_batch(data, seq_info)

    def _get_sequence_name(self, batch: Dict) -> str:
        if 'meta' in batch and 'name' in batch['meta']:
            name = batch['meta']['name']
            return name[0] if isinstance(name, (list, tuple)) else name
        raise ValueError("Sequence name not found in batch metadata.")

    def _get_split(self, batch: Dict) -> str:
        if 'meta' in batch and 'split' in batch['meta']:
            split = batch['meta']['split']
            return split[0] if isinstance(split, (list, tuple)) else split
        return getattr(self.test_config, 'dataset_split', 'test')

    def _load_sequence_info(self, seq_name: str, split: str) -> Optional[Dict]:
        cache_key = f"{split}:{seq_name}"
        if cache_key in self.seq_cache:
            return self.seq_cache[cache_key]

        seq_file = None
        split_dir = self.pw3d_raw_dir / "sequenceFiles" / "test"
        if (split_dir / f"{seq_name}.pkl").exists():
            seq_file = split_dir / f"{seq_name}.pkl"
        else:
            # Fallback: search all splits if split hint fails
            for sub in ("train", "validation", "test"):
                candidate = self.pw3d_raw_dir / "sequenceFiles" / sub / f"{seq_name}.pkl"
                if candidate.exists():
                    seq_file = candidate
                    break

        if seq_file is None or not seq_file.exists():
            print(f"[PW3DMeshRenderer] Sequence file not found for {seq_name}")
            return None

        with open(seq_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        image_dir = self.pw3d_raw_dir / "imageFiles" / seq_name
        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            print(f"[PW3DMeshRenderer] Images not found for {seq_name} at {image_dir}")
            return None

        frame_ids = np.array(data.get('img_frame_ids', []), dtype=np.int64)
        if frame_ids.size == 0:
            frame_ids = np.arange(len(image_paths), dtype=np.int64) * 2
        cam_K = np.array(data['cam_intrinsics'], dtype=np.float32)
        cam_T = np.repeat(np.array(data['cam_poses'], dtype=np.float32), 2, axis=0)

        seq_info = {
            "image_paths": image_paths,
            "frame_ids": frame_ids,
            "cam_K": cam_K,
            "cam_T": cam_T,
        }
        self.seq_cache[cache_key] = seq_info
        return seq_info

    def _init_video(self, seq_name: str, seq_info: Dict):
        self._cleanup_video()
        first_frame = cv2.imread(str(seq_info['image_paths'][0]))
        if first_frame is None:
            print(f"[PW3DMeshRenderer] Failed to read first frame for {seq_name}")
            return

        height, width = first_frame.shape[:2]
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

        output_dir = self.save_root / "video_renders"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{seq_name}_mesh.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, self.default_fps, (width, height))
        self.current_seq = seq_name

        print(f"[PW3DMeshRenderer] Rendering {seq_name} -> {output_path}")

    def _create_scene(self, cam_K: np.ndarray) -> pyrender.Scene:
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

        camera_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)
        return scene

    def _transform_verts_to_camera(self, verts: np.ndarray, cam_T: np.ndarray) -> np.ndarray:
        if cam_T.shape == (4, 4):
            R = cam_T[:3, :3]
            t = cam_T[:3, 3]
            return (R @ verts.T).T + t
        return verts

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

    def _render_batch(self, data: VisualizationData, seq_info: Dict):
        repr_pred = data.repr_dict_rec_tries[0]
        meta = data.batch.get('meta', {})
        _, verts_pred_world = recover_world_from_repr(repr_pred, meta, self.smplx_model, return_full_joints=False)
        if isinstance(verts_pred_world, torch.Tensor):
            verts_pred_world = verts_pred_world.detach().cpu().numpy()

        verts_gt_world = None
        if self.render_gt and data.repr_dict_clean is not None:
            _, verts_gt_world = recover_world_from_repr(data.repr_dict_clean, meta, self.smplx_model, return_full_joints=False)
            if isinstance(verts_gt_world, torch.Tensor):
                verts_gt_world = verts_gt_world.detach().cpu().numpy()

        faces = self.smplx_model.faces.astype(np.int32)
        total_frames = verts_pred_world.shape[1]

        cam_K = seq_info['cam_K']
        cam_T = seq_info['cam_T']
        frame_ids = seq_info['frame_ids']
        image_paths = seq_info['image_paths']

        for img_path, frame_id in zip(image_paths, frame_ids):
            if frame_id >= total_frames:
                break

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            scene = self._create_scene(cam_K)

            if verts_gt_world is not None and frame_id < verts_gt_world.shape[1]:
                verts_gt = verts_gt_world[0, frame_id]
                if self.apply_extrinsics:
                    verts_gt = self._transform_verts_to_camera(verts_gt, cam_T[frame_id])
                gt_overlay = self._render_mesh(scene, verts_gt, faces, MATERIAL_GT)
                frame = self._blend_overlay(frame, gt_overlay)

            verts_pred = verts_pred_world[0, frame_id]
            if self.apply_extrinsics:
                verts_pred = self._transform_verts_to_camera(verts_pred, cam_T[frame_id])
            pred_overlay = self._render_mesh(scene, verts_pred, faces, MATERIAL_PRED)
            frame = self._blend_overlay(frame, pred_overlay)

            if self.debug_visualize:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

            self.video_writer.write(frame)

    def _cleanup_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def cleanup(self) -> None:
        self._cleanup_video()
        self.current_seq = None
        self._is_initialized = False
