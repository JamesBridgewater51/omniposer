"""
EgoExo dataset visualization with synchronized ego/exo views + viser 3D mode.
"""
import os
import tempfile
import numpy as np
from typing import Optional
from tqdm import tqdm

from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


@VisualizerRegistry.register('ego_exo')
class EgoExoVisualizer(BaseVisualizer):
    """Visualizer for EgoExo dataset with synchronized views and viser 3D mode."""
    
    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.frame_synchronizer = None
    
    def setup(self) -> None:
        """Initialize EgoExo-specific resources."""
        if self._is_initialized:
            return
        self._is_initialized = True
    
    def visualize(self, data: VisualizationData) -> None:
        """Render EgoExo visualization.
        
        Dispatches to viser 3D mode or 2D synchronized frame mode
        based on test_config.vis_mode.
        """
        if not self.should_visualize():
            return
        
        vis_mode = getattr(self.test_config, 'vis_mode', '2d')
        vis_mode = '2d'
        if vis_mode == 'viser':
            self._visualize_viser(data)
        else:
            self._visualize_2d(data)

    def _visualize_viser(self, data: VisualizationData) -> None:
        """Render EgoExo visualization in viser 3D mode with splats + SMPL skeleton."""
        from .viser_smpl_viewer import ViserSmplViewer, find_splat_file
        from pathlib import Path

        if not data.jts_rec_tries:
            print("[Warning] EgoExo viser visualization requires reconstructed joints")
            return

        jts_rec = data.jts_rec_tries[0].cpu().numpy()  # (B, T, J, 3)
        joints = jts_rec[0]  # (T, J, 3)

        gt_joints = None
        if data.jts_clean is not None:
            gt_joints = data.jts_clean.cpu().numpy()[0]

        vertices = None
        faces = None
        if data.verts_rec_tries:
            verts_rec = data.verts_rec_tries[0].cpu().numpy()
            vertices = verts_rec[0]
            if hasattr(self.fk_model, 'faces'):
                faces = self.fk_model.faces
                if hasattr(faces, 'numpy'):
                    faces = faces.numpy()

        # Find splat file
        splat_path = None
        splat_search_dir = getattr(self.test_config, 'splat_dir')
        if splat_search_dir:
            splat_path = find_splat_file(Path(splat_search_dir))
        
        # Also search in egoexo dataset root if available
        if splat_path is None and data.batch is not None:
            meta = data.batch.get('meta', {})
            take_name = meta.get('take_name', [None])[0]
            egoexo_root = getattr(self.test_config, 'egoexo_root_path')
            if take_name and egoexo_root:
                train_dir = Path(egoexo_root) / f"{take_name}.train"
                if train_dir.exists():
                    splat_path = find_splat_file(train_dir)

        # Estimate floor height
        foot_indices = [i for i in [7, 8, 10, 11] if i < joints.shape[1]]
        if foot_indices:
            floor_z = float(np.nanmin(joints[:, foot_indices, 2]))
        else:
            floor_z = float(np.nanmin(joints[:, :, 2]))

        port = getattr(self.test_config, 'viser_port', 8080)
        viewer = ViserSmplViewer(port=port, floor_z=floor_z)
        viewer.set_joints(joints, gt_joints=gt_joints)
        if vertices is not None and faces is not None:
            viewer.set_mesh(vertices, faces)
        if splat_path:
            viewer.set_splat(splat_path)

        print(f"[EgoExoViser] Starting viewer at http://localhost:{port}")
        print("[EgoExoViser] Press Ctrl+C to stop and continue to next batch")
        viewer.run()

    def _visualize_2d(self, data: VisualizationData) -> None:
        """Render per-camera exo videos with SMPL mesh overlay (pyrender)."""
        if data.batch is None:
            print("[Warning] EgoExo visualization requires batch metadata")
            return
        
        batch = data.batch
        if 'meta' not in batch or 'take_name' not in batch['meta']:
            print("[Warning] Missing take metadata for EgoExo visualization")
            return
        
        take_name = batch['meta']['take_name'][0]
        take_uid = batch['meta']['take_uid'][0]
        frame_keys = batch['meta']['frame_keys'][0].cpu().numpy().tolist()
        
        if len(frame_keys) > 500:
            print(f"[Warning] Skipping visualization for {take_name}: {len(frame_keys)} frames exceeds limit")
            return
        
        # Lazy import EgoExo utilities
        try:
            from utils.egoexo_vis_utils import EgoExoFrameSynchronizer
            from viz.egoexo_visualizer import render_exo_per_camera_frame, save_egoexo_animation
        except ImportError as e:
            print(f"[Warning] EgoExo visualization dependencies not available: {e}")
            return
        
        # Resolve SMPL faces from the FK / SMPLX model
        faces = None
        if hasattr(self.fk_model, 'faces'):
            faces = self.fk_model.faces
            if hasattr(faces, 'numpy'):
                faces = faces.numpy()
        if faces is None:
            print("[Warning] Cannot resolve SMPL faces from fk_model — skipping pyrender exo visualization")
            return

        # Initialize frame synchronizer
        self.frame_synchronizer = EgoExoFrameSynchronizer(
            self.test_config.egoexo_root_path,
            take_name,
            take_uid=take_uid,
            split=batch['meta']['split'][0],
        )
        
        temp_dir = tempfile.mkdtemp()
        # Per-camera frame path accumulator:  {cam_name: [path1, path2, ...]}
        cam_frame_paths: dict[str, list[str]] = {}
        
        # Iterate synchronized frames lazily — only 1 frame-set in memory at a time
        sync_iter = self.frame_synchronizer.get_synchronized_frames_iter(
            ego_frame_indices=frame_keys,
        )
        
        # Convert tensors
        verts_rec = data.verts_rec_tries[0].cpu().numpy() if data.verts_rec_tries else None
        if verts_rec is None:
            print("[Warning] No reconstructed vertices available — skipping pyrender exo visualization")
            return

        for t, synced_data in enumerate(tqdm(sync_iter, total=len(frame_keys), desc="Rendering EgoExo exo frames (pyrender)")):
            saved = render_exo_per_camera_frame(
                output_dir=temp_dir,
                frame_idx_in_seq=frame_keys[t],
                smpl_verts_world=verts_rec[0, t],
                faces=faces,
                exo_frames_data=synced_data['exo_frames_data'],
            )
            
            for cam_name, saved_path in saved.items():
                if cam_name not in cam_frame_paths:
                    cam_frame_paths[cam_name] = []
                cam_frame_paths[cam_name].append(saved_path)
        
        # Compile one video per camera
        for cam_name, frame_paths in cam_frame_paths.items():
            if not frame_paths:
                continue
            video_path = os.path.join(
                self.test_config.save_root,
                f'egoexo_{take_uid}_{cam_name}.mp4',
            )
            save_egoexo_animation(frame_paths, video_path)
            print(f"EgoExo exo visualization saved to: {video_path}")
        
        # Cleanup temp frame images and per-camera subdirectories
        for cam_name, frame_paths in cam_frame_paths.items():
            for p in frame_paths:
                if os.path.exists(p):
                    os.remove(p)
            cam_sub_dir = os.path.join(temp_dir, cam_name)
            if os.path.isdir(cam_sub_dir):
                try:
                    os.rmdir(cam_sub_dir)
                except OSError as e:
                    print(f"[Warning] Could not remove temp subdir {cam_sub_dir}: {e}")
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"[Warning] Could not remove temp dir {temp_dir}: {e}")
    
    def cleanup(self) -> None:
        """Release EgoExo resources."""
        self.frame_synchronizer = None
        self._is_initialized = False
