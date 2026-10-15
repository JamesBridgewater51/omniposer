"""
PyVista-based real-time visualization helpers for SMPL-X optimization scripts.

This module is intentionally lightweight and optional: PyVista is imported lazily
and only required when visualization is enabled.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

# Add scripts/mvnx to path for skeleton_defs import
_mvnx_dir = Path(__file__).parent.parent / "scripts" / "mvnx"
if str(_mvnx_dir) not in sys.path:
    sys.path.insert(0, str(_mvnx_dir))

from skeleton_defs import SKELETON_CONNECTIONS_SHORT as SKELETON_CONNECTIONS


def get_keyframe_indices(num_frames: int, num_keyframes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Evenly spaced keyframes in [0, num_frames-1].
    Returns a 1D long tensor on `device` (or CPU if None).
    """
    if num_frames <= 0 or num_keyframes <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    num_keyframes = int(min(num_keyframes, num_frames))
    idx = torch.linspace(0, num_frames - 1, steps=num_keyframes, device=device)
    idx = torch.round(idx).long()
    # Ensure strictly non-decreasing unique indices.
    idx = torch.unique_consecutive(idx)
    return idx


def get_downsample_indices(num_points: int, max_points: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Deterministic downsample indices for point clouds.
    """
    if max_points is None or max_points <= 0 or num_points <= max_points:
        return torch.arange(num_points, device=device)
    return torch.linspace(0, num_points - 1, steps=int(max_points), device=device).long()


@dataclass
class VisConfig:
    max_verts_points: int = 10000
    verts_point_size: float = 2.0
    joints_point_size: float = 8.0
    targets_point_size: float = 10.0
    verts_opacity: float = 0.5
    window_title: str = "SMPL-X Optimization Visualization"
    show_skeleton: bool = True
    skeleton_line_width: float = 3.0


class PyVistaSequenceVisualizer:
    """
    Real-time PyVista visualizer that overlays multiple keyframes in one view.

    - pred_vertices: (T, V, 3) torch tensor
    - pred_joints:   (T, J, 3) torch tensor (already selected joints)
    - targets:       (T, J, 3) torch tensor, optional (e.g., COCO/MVNX targets)
    - targets_valid: (T, J) bool torch tensor, optional
    - target_names:  list of J names, optional (labels only shown for valid targets)
    - keyframes:     1D long tensor of frame indices within current chunk
    """

    def __init__(
        self,
        pred_vertices: torch.Tensor,
        pred_joints: torch.Tensor,
        keyframes: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_valid: Optional[torch.Tensor] = None,
        target_names: Optional[Sequence[str]] = None,
        cfg: Optional[VisConfig] = None,
    ) -> None:
        try:
            import pyvista as pv  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyVista is required for visualization. Install it (e.g., `pip install pyvista vtk`)."
            ) from e

        self.pv = pv
        self.cfg = cfg or VisConfig()

        if keyframes.numel() == 0:
            raise ValueError("keyframes must be non-empty for visualization.")

        # Store indices on the same device as predictions for fast indexing.
        self.keyframes = keyframes.to(pred_vertices.device)
        self.num_keyframes = int(self.keyframes.numel())

        V = int(pred_vertices.shape[1])
        self.vert_idx = get_downsample_indices(V, self.cfg.max_verts_points, device=pred_vertices.device)
        self.num_verts_sampled = int(self.vert_idx.numel())

        # Build initial point clouds.
        verts_points, verts_scalars = self._build_pred_verts_points(pred_vertices)
        joints_points, joints_scalars = self._build_pred_joints_points(pred_joints)

        self.plotter = pv.Plotter(window_size=(1024, 768), title=self.cfg.window_title)
        self.plotter.add_axes()

        self.pred_verts_pd = pv.PolyData(verts_points)
        self.plotter.add_points(
            self.pred_verts_pd,
            scalars=verts_scalars,
            cmap="viridis" if self.num_keyframes > 1 else None,
            color=None if self.num_keyframes > 1 else (0.6, 0.6, 0.6),
            point_size=self.cfg.verts_point_size,
            opacity=self.cfg.verts_opacity,
            render_points_as_spheres=True,
        )

        self.pred_joints_pd = pv.PolyData(joints_points)
        self.plotter.add_points(
            self.pred_joints_pd,
            scalars=joints_scalars,
            cmap="coolwarm" if self.num_keyframes > 1 else None,
            color=None if self.num_keyframes > 1 else (0.0, 0.0, 1.0),
            point_size=self.cfg.joints_point_size,
            render_points_as_spheres=True,
        )

        self.targets_pd = None
        self.targets_labels_actor = None
        if targets is not None:
            targets_points, labels = self._build_targets_points(targets, targets_valid, target_names)
            if targets_points.size > 0:
                self.targets_pd = pv.PolyData(targets_points)
                self.plotter.add_points(
                    self.targets_pd,
                    color=(1.0, 0.0, 0.0),
                    point_size=self.cfg.targets_point_size,
                    render_points_as_spheres=True,
                )
                if labels is not None and len(labels) > 0:
                    self.targets_labels_actor = self.plotter.add_point_labels(
                        targets_points,
                        labels,
                        font_size=12,
                        text_color="black",
                        point_color="red",
                        shape=None,
                        always_visible=True,
                    )

        # Add skeleton lines for targets
        self.target_names = target_names
        self.skeleton_lines_pd = None
        if targets is not None and target_names is not None and self.cfg.show_skeleton:
            skeleton_lines = self._build_skeleton_lines(targets, targets_valid, target_names)
            if skeleton_lines is not None:
                self.skeleton_lines_pd = skeleton_lines
                self.plotter.add_mesh(
                    self.skeleton_lines_pd,
                    color=(1.0, 0.3, 0.3),
                    line_width=self.cfg.skeleton_line_width,
                )

        # Initialize camera to a sensible view.
        self.plotter.view_isometric()
        self.plotter.show(auto_close=False, interactive_update=True)

    def _build_pred_verts_points(self, pred_vertices: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        verts_k = pred_vertices.index_select(0, self.keyframes).index_select(1, self.vert_idx)
        verts_points = verts_k.detach().cpu().numpy().reshape(-1, 3)
        verts_scalars = np.repeat(np.arange(self.num_keyframes), self.num_verts_sampled)
        return verts_points, verts_scalars

    def _build_pred_joints_points(self, pred_joints: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        joints_k = pred_joints.index_select(0, self.keyframes)
        J = int(joints_k.shape[1])
        joints_points = joints_k.detach().cpu().numpy().reshape(-1, 3)
        joints_scalars = np.repeat(np.arange(self.num_keyframes), J)
        return joints_points, joints_scalars

    def _build_targets_points(
        self,
        targets: torch.Tensor,
        targets_valid: Optional[torch.Tensor],
        target_names: Optional[Sequence[str]],
    ) -> tuple[np.ndarray, Optional[List[str]]]:
        targets_k = targets.index_select(0, self.keyframes)
        valid_k = (
            targets_valid.index_select(0, self.keyframes)
            if targets_valid is not None
            else torch.ones(targets_k.shape[:2], dtype=torch.bool, device=targets_k.device)
        )
        finite_k = torch.isfinite(targets_k).all(dim=-1)
        valid_k = valid_k & finite_k

        targets_flat = targets_k.reshape(-1, 3)
        valid_flat = valid_k.reshape(-1)
        targets_points = targets_flat[valid_flat].detach().cpu().numpy()

        labels: Optional[List[str]] = None
        if target_names is not None:
            target_names_arr = np.asarray(list(target_names))
            if target_names_arr.size == targets_k.shape[1]:
                tiled = np.tile(target_names_arr, self.num_keyframes)
                labels = tiled[valid_flat.detach().cpu().numpy()].tolist()
        return targets_points, labels

    def _build_skeleton_lines(
        self,
        targets: torch.Tensor,
        targets_valid: Optional[torch.Tensor],
        target_names: Sequence[str],
    ):
        """Build skeleton lines connecting adjacent landmarks."""
        import pyvista as pv
        
        # Build name -> index mapping
        name_to_idx = {name: i for i, name in enumerate(target_names)}
        
        targets_k = targets.index_select(0, self.keyframes)  # (K, J, 3)
        valid_k = (
            targets_valid.index_select(0, self.keyframes)
            if targets_valid is not None
            else torch.ones(targets_k.shape[:2], dtype=torch.bool, device=targets_k.device)
        )
        finite_k = torch.isfinite(targets_k).all(dim=-1)
        valid_k = valid_k & finite_k
        
        all_points = []
        all_lines = []
        point_offset = 0
        
        for k_idx in range(self.num_keyframes):
            frame_points = targets_k[k_idx].detach().cpu().numpy()  # (J, 3)
            frame_valid = valid_k[k_idx].detach().cpu().numpy()  # (J,)
            
            for parent, child in SKELETON_CONNECTIONS:
                if parent not in name_to_idx or child not in name_to_idx:
                    continue
                p_idx, c_idx = name_to_idx[parent], name_to_idx[child]
                if not frame_valid[p_idx] or not frame_valid[c_idx]:
                    continue
                
                p_pos = frame_points[p_idx]
                c_pos = frame_points[c_idx]
                
                # Add two points and a line connecting them
                all_points.append(p_pos)
                all_points.append(c_pos)
                all_lines.append([2, point_offset, point_offset + 1])
                point_offset += 2
        
        if not all_points:
            return None
        
        points = np.array(all_points)
        lines = np.hstack(all_lines)
        return pv.PolyData(points, lines=lines)

    def update(self, pred_vertices: torch.Tensor, pred_joints: torch.Tensor) -> None:
        verts_points, _ = self._build_pred_verts_points(pred_vertices)
        joints_points, _ = self._build_pred_joints_points(pred_joints)
        self.pred_verts_pd.points = verts_points
        self.pred_joints_pd.points = joints_points
        self.plotter.update()

    def close(self) -> None:
        try:
            self.plotter.close()
        except Exception:  # pragma: no cover
            pass

