"""
Nymeria dataset viser-based 3D visualization.

Registered as 'nymeria_viser' to avoid conflict with the existing 'nymeria'
registration in amass.py (which provides Open3D-based visualization).
"""
import numpy as np
from typing import Optional

from .base import BaseVisualizer, VisualizationData
from .registry import VisualizerRegistry


@VisualizerRegistry.register('nymeria_viser')
class NymeriaViserVisualizer(BaseVisualizer):
    """Viser-based 3D visualizer for Nymeria dataset.
    
    Launches a viser server with interactive SMPL skeleton animation
    and optional Gaussian splat scene.
    """

    def __init__(self, train_config, test_config, fk_model):
        super().__init__(train_config, test_config, fk_model)
        self._viewer = None

    def setup(self) -> None:
        """Initialize viser viewer (lazy, created on first visualize call)."""
        if self._is_initialized:
            return
        self._is_initialized = True

    def visualize(self, data: VisualizationData) -> None:
        """Render Nymeria visualization in viser.
        
        Extracts joint positions from VisualizationData and launches
        an interactive viser 3D viewer.
        """
        if not self.should_visualize():
            return

        from .viser_smpl_viewer import ViserSmplViewer, find_splat_file
        from pathlib import Path

        # Extract joints from visualization data
        if not data.jts_rec_tries:
            print("[Warning] Nymeria viser visualization requires reconstructed joints")
            return

        jts_rec = data.jts_rec_tries[0].cpu().numpy()  # (B, T, J, 3)
        joints = jts_rec[0]  # Take first batch: (T, J, 3)

        gt_joints = None
        if data.jts_clean is not None:
            gt_joints = data.jts_clean.cpu().numpy()[0]  # (T, J, 3)

        # Optionally load mesh
        vertices = None
        faces = None
        if data.verts_rec_tries:
            verts_rec = data.verts_rec_tries[0].cpu().numpy()
            vertices = verts_rec[0]  # (T, V, 3)
            if hasattr(self.fk_model, 'faces'):
                faces = self.fk_model.faces
                if hasattr(faces, 'numpy'):
                    faces = faces.numpy()

        # Try to find splat file if path is configured
        splat_path = None
        splat_search_dir = getattr(self.test_config, 'splat_dir', None)
        if splat_search_dir:
            splat_path = find_splat_file(Path(splat_search_dir))

        # Estimate floor height
        foot_indices = [i for i in [7, 8, 10, 11] if i < joints.shape[1]]
        if foot_indices:
            floor_z = float(np.nanmin(joints[:, foot_indices, 2]))
        else:
            floor_z = float(np.nanmin(joints[:, :, 2]))

        # Create viewer
        port = getattr(self.test_config, 'viser_port', 8080)
        viewer = ViserSmplViewer(port=port, floor_z=floor_z)
        viewer.set_joints(joints, gt_joints=gt_joints)
        if vertices is not None and faces is not None:
            viewer.set_mesh(vertices, faces)
        if splat_path:
            viewer.set_splat(splat_path)

        print(f"[NymeriaViser] Starting viewer at http://localhost:{port}")
        print("[NymeriaViser] Press Ctrl+C to stop and continue to next batch")

        # Run viewer (blocking - user must Ctrl+C to continue)
        viewer.run()

    def cleanup(self) -> None:
        """Release viser resources."""
        self._viewer = None
        self._is_initialized = False
