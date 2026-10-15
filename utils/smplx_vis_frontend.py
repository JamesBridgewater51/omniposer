from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class SmplxVisConfig:
    window_title: str = "SMPL-X Visualization"
    num_repeats: int = 1
    fixed_cam: bool = False
    camera_follow: bool = True
    orbit_period: int = 180
    orbit_radius: float = 2.5
    orbit_height: float = 1.2
    look_at_height: float = 0.3
    axis_size: float = 0.5
    joint_point_size: float = 8.0
    joint_color: Sequence[float] = (0.8, 0.1, 0.1)
    skeleton_color: Sequence[float] = (0.1, 0.8, 0.1)
    mesh_color: Sequence[float] = (0.7, 0.7, 0.7)
    show_skeleton: bool = True
    show_mesh: bool = True
    show_ground: bool = True
    ground_size: float = 5.0
    ground_color: Sequence[float] = (0.9, 0.9, 0.9)
    frame_time: float = 0.001


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Note the Open3D camera conventions.
    """
    direction = target - eye
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)

    right = np.cross(-up, direction)
    if np.linalg.norm(right) > 0:
        right = right / np.linalg.norm(right)

    new_up = np.cross(direction, right)
    if np.linalg.norm(new_up) > 0:
        new_up = new_up / np.linalg.norm(new_up)

    R = np.array([right, new_up, direction])
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    return M @ T


def _select_batch(tensor: Optional[torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        return tensor[batch_idx]
    return tensor


def _get_sequence_len(joints: Optional[torch.Tensor], vertices: Optional[torch.Tensor]) -> int:
    if joints is not None:
        return int(joints.shape[0])
    if vertices is not None:
        return int(vertices.shape[0])
    raise ValueError("Either joints or vertices must be provided for visualization.")


def _compute_center(
    joints_frame: Optional[np.ndarray],
    verts_frame: Optional[np.ndarray],
) -> np.ndarray:
    if joints_frame is not None and joints_frame.shape[0] > 0:
        return joints_frame[0]
    if verts_frame is not None and verts_frame.shape[0] > 0:
        return verts_frame.mean(axis=0)
    return np.zeros(3, dtype=np.float32)


def visualize_smplx_sequence(
    *,
    joints: Optional[torch.Tensor] = None,
    vertices: Optional[torch.Tensor] = None,
    faces: Optional[np.ndarray] = None,
    parent_indices: Optional[Sequence[int]] = None,
    batch_idx: int = 0,
    cfg: Optional[SmplxVisConfig] = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for visualization. Install it with `pip install open3d`.") from exc

    config = cfg or SmplxVisConfig()

    joints_seq = _select_batch(joints, batch_idx)
    verts_seq = _select_batch(vertices, batch_idx)

    if joints_seq is None and verts_seq is None:
        raise ValueError("visualize_smplx_sequence requires joints or vertices.")

    seq_len = _get_sequence_len(joints_seq, verts_seq)

    joints_np = joints_seq.detach().cpu().numpy() if joints_seq is not None else None
    verts_np = verts_seq.detach().cpu().numpy() if verts_seq is not None else None

    if config.show_mesh and verts_np is None:
        raise ValueError("show_mesh=True requires vertices to be provided.")
    if config.show_mesh and faces is None:
        raise ValueError("show_mesh=True requires faces to be provided.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=config.window_title):
        raise RuntimeError(
            "Open3D failed to create a window. Check DISPLAY/GLFW/OSMesa availability."
        )

    should_exit = {"flag": False}

    def _on_key_exit(vis_obj):
        should_exit["flag"] = True
        return False

    vis.register_key_callback(ord("Q"), _on_key_exit)
    vis.register_key_callback(ord("q"), _on_key_exit)

    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        raise RuntimeError("Open3D render options unavailable; window creation failed.")
    opt.point_size = float(config.joint_point_size)

    if config.show_ground:
        ground = o3d.geometry.TriangleMesh.create_box(
            width=config.ground_size, height=config.ground_size, depth=0.01
        )
        ground.translate(
            [-0.5 * config.ground_size, -0.5 * config.ground_size, -0.01]
        )
        ground.paint_uniform_color(list(config.ground_color))
        vis.add_geometry(ground)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=config.axis_size)
    vis.add_geometry(world_frame)

    skeleton_lines = None
    joint_points = None
    if config.show_skeleton and joints_np is not None:
        if parent_indices is None:
            raise ValueError("parent_indices must be provided when show_skeleton=True.")
        joints_init = joints_np[0]
        line_pairs = [
            [parent, idx]
            for idx, parent in enumerate(parent_indices)
            if parent != -1 and 0 <= parent < joints_init.shape[0]
        ]
        skeleton_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(joints_init),
            lines=o3d.utility.Vector2iVector(line_pairs),
        )
        skeleton_lines.paint_uniform_color(list(config.skeleton_color))
        vis.add_geometry(skeleton_lines)

        joint_points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(joints_init)
        )
        joint_colors = np.tile(
            np.asarray(config.joint_color, dtype=np.float32), (joints_init.shape[0], 1)
        )
        joint_points.colors = o3d.utility.Vector3dVector(joint_colors)
        vis.add_geometry(joint_points)

    mesh = None
    if config.show_mesh and verts_np is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_np[0])
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.paint_uniform_color(list(config.mesh_color))
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    vis.poll_events()
    vis.update_renderer()

    if config.fixed_cam:
        cam_param = o3d.camera.PinholeCameraParameters()
        cam_intr_param = o3d.camera.PinholeCameraIntrinsic()
        cam_intr_param.intrinsic_matrix = np.array(
            [
                [935, 0, 959.5],
                [0, 935, 539.5],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        cam_param.intrinsic = cam_intr_param
        cam_param.extrinsic = np.array(
            [
                [-0.99911475, -0.03019581, -0.02929056, 0.02251293],
                [0.01173269, 0.46862719, -0.88331812, 0.67892441],
                [0.04039886, -0.88287982, -0.46785805, 4.34331579],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        view_ctl = vis.get_view_control()
        view_ctl.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

    for _ in range(config.num_repeats):
        for frame_idx in range(seq_len):
            if should_exit["flag"]:
                break

            joints_frame = joints_np[frame_idx] if joints_np is not None else None
            verts_frame = verts_np[frame_idx] if verts_np is not None else None

            # After creating skeleton_lines / joint_points / mesh (once), cache numpy views:
            skel_pts_np = np.asarray(skeleton_lines.points) if skeleton_lines is not None else None
            joint_pts_np = np.asarray(joint_points.points) if joint_points is not None else None
            mesh_verts_np = np.asarray(mesh.vertices) if mesh is not None else None

            # In the frame loop:
            if skel_pts_np is not None and joints_frame is not None:
                skel_pts_np[:] = joints_frame  # in-place copy
                vis.update_geometry(skeleton_lines)

            if joint_pts_np is not None and joints_frame is not None:
                joint_pts_np[:] = joints_frame
                vis.update_geometry(joint_points)

            if mesh_verts_np is not None and verts_frame is not None:
                mesh_verts_np[:] = verts_frame
                vis.update_geometry(mesh)

            if config.camera_follow and not config.fixed_cam:
                center = _compute_center(joints_frame, verts_frame)
                orbit_period = max(1, int(config.orbit_period))
                angle = 2.0 * np.pi * (frame_idx % orbit_period) / float(
                    orbit_period
                )
                eye = center + np.array(
                    [
                        config.orbit_radius * np.cos(angle),
                        config.orbit_radius * np.sin(angle),
                        config.orbit_height,
                    ],
                    dtype=np.float32,
                )
                target = center + np.array([0.0, 0.0, config.look_at_height], dtype=np.float32)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam_param.extrinsic = _look_at(eye, target, up)
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    cam_param, allow_arbitrary=True
                )

            vis.poll_events()
            vis.update_renderer()
            if config.frame_time > 0:
                import time

                time.sleep(config.frame_time)

        if should_exit["flag"]:
            break

    vis.destroy_window()
