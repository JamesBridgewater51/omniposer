"""
Reusable viser-based SMPL motion viewer.

Provides interactive 3D visualization with:
- Gaussian splats (3DGS scene reconstruction)
- Animated SMPL skeleton / mesh
- Ground plane, playback controls, camera attach

Modeled after egoallo's vis_helpers.py but adapted for OmniPose output
(raw SMPL joint positions rather than EgoDenoiseTraj + fncsmpl body model).
"""
import time
import logging
from pathlib import Path
from typing import TypedDict, Callable

import numpy as np
import numpy.typing as npt
import viser
import viser.transforms as vtf
from plyfile import PlyData

logger = logging.getLogger(__name__)


# ── Splat loading utilities (ported from egoallo vis_helpers.py) ──────────────

class SplatArgs(TypedDict):
    centers: npt.NDArray[np.floating]     # (N, 3)
    rgbs: npt.NDArray[np.floating]        # (N, 3) in [0, 1]
    opacities: npt.NDArray[np.floating]   # (N, 1) in [0, 1]
    covariances: npt.NDArray[np.floating] # (N, 3, 3)


def load_splat_file(splat_path: Path, center: bool = False) -> SplatArgs:
    """Load an antimatter15-style .splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = 3 * 4 + 3 * 4 + 4 + 4  # position + scales + rgba + quat
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    logger.info(f"Splat file with {num_gaussians=} loaded in {time.time() - start_time:.2f}s")
    return {
        "centers": centers,
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatArgs:
    """Load Gaussians stored in a PLY file (splatfacto export format)."""
    start_time = time.time()
    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    logger.info(f"PLY file with {num_gaussians=} loaded in {time.time() - start_time:.2f}s")
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def add_splat_to_viser(
    splat_or_ply_path: Path,
    server: viser.ViserServer,
    z_offset: float = 0.0,
) -> None:
    """Add Gaussian splats from a .splat or .ply file to the viser server."""
    suffix = splat_or_ply_path.suffix.lower()
    if suffix == ".ply":
        splat_args = load_ply_file(splat_or_ply_path)
    elif suffix == ".splat":
        splat_args = load_splat_file(splat_or_ply_path)
    else:
        raise ValueError(f"Unsupported splat file format: {suffix}. Expected .ply or .splat")
    server.scene.add_gaussian_splats(
        "/gaussian_splats",
        centers=splat_args["centers"],
        rgbs=splat_args["rgbs"],
        opacities=splat_args["opacities"],
        covariances=splat_args["covariances"],
        position=(0.0, 0.0, z_offset),
    )


def find_splat_file(search_dir: Path) -> Path | None:
    """Search for a Gaussian splat file (.splat or .ply with Gaussian data) in a directory tree."""
    # 1. Check for scene.splat directly
    for name in ("scene.splat", "splat.ply"):
        candidate = search_dir / name
        if candidate.exists():
            return candidate

    # 2. Search for nerfstudio splatfacto exports
    # nerfstudio exports to outputs/{config_name}/splatfacto/{timestamp}/
    # We look for point_cloud.ply or similar
    for ply_path in sorted(search_dir.rglob("*.ply")):
        # Skip global_points.ply which is Aria MPS point cloud, not Gaussian
        if "global_points" in ply_path.name:
            continue
        # Check if it has Gaussian splat fields
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        prop_names = {p.name for p in v.properties}
        if "scale_0" in prop_names and "rot_0" in prop_names and "f_dc_0" in prop_names:
            return ply_path

    # 3. Search for .splat files
    splat_files = sorted(search_dir.rglob("*.splat"))
    if splat_files:
        return splat_files[0]

    return None


# ── SMPL skeleton definition ─────────────────────────────────────────────────

# SMPL-H 22-joint kinematic tree (parent indices, -1 = root)
SMPLH_KINTREE = [
    -1,  # 0  Pelvis
     0,  # 1  L_Hip
     0,  # 2  R_Hip
     0,  # 3  Spine1
     1,  # 4  L_Knee
     2,  # 5  R_Knee
     3,  # 6  Spine2
     4,  # 7  L_Ankle
     5,  # 8  R_Ankle
     6,  # 9  Spine3
     7,  # 10 L_Foot
     8,  # 11 R_Foot
     9,  # 12 Neck
     9,  # 13 L_Collar
     9,  # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
]

SMPLH_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

# Limb connections as (child, parent) pairs for skeleton drawing
SMPL_LIMBS = [(i, SMPLH_KINTREE[i]) for i in range(1, len(SMPLH_KINTREE))]


# ── Core viewer class ────────────────────────────────────────────────────────

class ViserSmplViewer:
    """Interactive viser-based viewer for SMPL motion sequences with optional Gaussian splat scenes.

    Usage:
        viewer = ViserSmplViewer()
        viewer.set_joints(joints_sequence)  # (T, J, 3)
        viewer.set_splat(splat_path)
        viewer.run()  # blocking loop
    """

    def __init__(
        self,
        port: int = 8080,
        floor_z: float = 0.0,
        body_color: tuple[int, int, int] = (152, 93, 229),
    ):
        self.port = port
        self.floor_z = floor_z
        self.body_color = body_color

        self._joints: np.ndarray | None = None  # (T, J, 3)
        self._verts: np.ndarray | None = None   # (T, V, 3)
        self._faces: np.ndarray | None = None   # (F, 3)
        self._gt_joints: np.ndarray | None = None  # (T, J, 3) optional GT

        self._splat_path: Path | None = None
        self._point_cloud: np.ndarray | None = None

        self._server: viser.ViserServer | None = None

    def set_joints(
        self,
        joints: np.ndarray,
        gt_joints: np.ndarray | None = None,
    ) -> None:
        """Set the joint sequence to visualize.
        
        Args:
            joints: (T, J, 3) predicted joint positions
            gt_joints: (T, J, 3) optional ground truth joints
        """
        self._joints = np.asarray(joints, dtype=np.float32)
        if gt_joints is not None:
            self._gt_joints = np.asarray(gt_joints, dtype=np.float32)

    def set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Set mesh data for SMPL body visualization.
        
        Args:
            vertices: (T, V, 3) vertex positions per frame
            faces: (F, 3) face indices (constant across frames)
        """
        self._verts = np.asarray(vertices, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.int32)

    def set_splat(self, splat_path: Path | None) -> None:
        """Set the path to the Gaussian splat file."""
        self._splat_path = splat_path

    def set_point_cloud(self, points: np.ndarray) -> None:
        """Set Aria MPS point cloud data."""
        self._point_cloud = np.asarray(points)

    def run(self, dry_run: bool = False) -> None:
        """Start the viser server and run the visualization loop."""
        assert self._joints is not None, "Must call set_joints() before run()"

        T, J, _ = self._joints.shape
        print(f"[ViserSmplViewer] {T} frames, {J} joints")

        server = viser.ViserServer(port=self.port)
        server.gui.configure_theme(dark_mode=True)
        self._server = server

        # ── Scene setup ──
        server.scene.add_grid(
            "/ground",
            plane="xy",
            cell_color=(80, 80, 80),
            section_color=(50, 50, 50),
            position=(0.0, 0.0, self.floor_z),
        )

        # Load Gaussian splats
        if self._splat_path is not None:
            print(f"[ViserSmplViewer] Loading splat: {self._splat_path}")
            add_splat_to_viser(self._splat_path, server)

        # Load point cloud
        if self._point_cloud is not None:
            server.scene.add_point_cloud(
                "/aria_points",
                points=self._point_cloud,
                colors=np.cos(self._point_cloud + np.arange(3)) / 3.0 + 0.7,
                point_size=0.01,
                point_shape="sparkle",
            )

        # ── Skeleton line handles (pre-create for update) ──
        # We create line segments that we update each frame
        joint_pc_handle = server.scene.add_point_cloud(
            "/skeleton/joints",
            points=self._joints[0],
            colors=np.full((J, 3), 255, dtype=np.uint8),
            point_size=0.015,
            point_shape="circle",
        )

        # Create limb lines
        limb_handles = []
        valid_limbs = [(c, p) for c, p in SMPL_LIMBS if c < J and p >= 0 and p < J]
        for idx, (child, parent) in enumerate(valid_limbs):
            pts = np.stack([self._joints[0, parent], self._joints[0, child]], axis=0)
            handle = server.scene.add_spline_catmull_rom(
                f"/skeleton/limb_{idx}",
                positions=pts,
                line_width=3.0,
                color=self.body_color,
            )
            limb_handles.append(handle)

        # GT skeleton (if provided)
        gt_joint_pc_handle = None
        gt_limb_handles = []
        if self._gt_joints is not None:
            gt_joint_pc_handle = server.scene.add_point_cloud(
                "/gt_skeleton/joints",
                points=self._gt_joints[0],
                colors=np.full((J, 3), [255, 50, 50], dtype=np.uint8),
                point_size=0.012,
                point_shape="diamond",
            )
            for idx, (child, parent) in enumerate(valid_limbs):
                pts = np.stack([self._gt_joints[0, parent], self._gt_joints[0, child]], axis=0)
                handle = server.scene.add_spline_catmull_rom(
                    f"/gt_skeleton/limb_{idx}",
                    positions=pts,
                    line_width=2.0,
                    color=(255, 80, 80),
                )
                gt_limb_handles.append(handle)

        # Mesh handle (if provided)
        mesh_handle = None
        if self._verts is not None and self._faces is not None:
            mesh_handle = server.scene.add_mesh_simple(
                "/body_mesh",
                vertices=self._verts[0],
                faces=self._faces,
                color=self.body_color,
            )

        # ── GUI controls ──
        gui_show_skeleton = server.gui.add_checkbox("Show skeleton", initial_value=True)
        gui_show_gt = server.gui.add_checkbox("Show GT skeleton", initial_value=self._gt_joints is not None)
        if self._verts is not None:
            gui_show_mesh = server.gui.add_checkbox("Show mesh", initial_value=True)
            gui_mesh_opacity = server.gui.add_slider("Mesh opacity", min=0.0, max=1.0, step=0.01, initial_value=0.7)
            gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
        gui_body_color = server.gui.add_rgb("Body color", initial_value=self.body_color)

        with server.gui.add_folder("Playback"):
            gui_timestep = server.gui.add_slider(
                "Timestep", min=0, max=T - 1, step=1, initial_value=0, disabled=True,
            )
            gui_start_end = server.gui.add_multi_slider(
                "Start/end", min=0, max=T - 1, initial_value=(0, T - 1), step=1,
            )
            gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
            gui_playing = server.gui.add_checkbox("Playing", True)
            gui_framerate = server.gui.add_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=15,
            )
            gui_framerate_options = server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60"),
            )

        gui_attach = server.gui.add_checkbox("Attach camera to head", initial_value=False)
        gui_attach_dist = server.gui.add_number("Attach distance", initial_value=1.5)

        # ── Callbacks ──
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % T

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % T

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        @gui_show_skeleton.on_update
        def _(_) -> None:
            joint_pc_handle.visible = gui_show_skeleton.value
            for h in limb_handles:
                h.visible = gui_show_skeleton.value

        if self._gt_joints is not None:
            @gui_show_gt.on_update
            def _(_) -> None:
                if gt_joint_pc_handle is not None:
                    gt_joint_pc_handle.visible = gui_show_gt.value
                for h in gt_limb_handles:
                    h.visible = gui_show_gt.value

        @gui_body_color.on_update
        def _(_) -> None:
            for h in limb_handles:
                h.color = gui_body_color.value
            if mesh_handle is not None:
                mesh_handle.color = gui_body_color.value

        # ── Update function ──
        def do_update() -> None:
            t = gui_timestep.value
            joints_t = self._joints[t]  # (J, 3)

            # Filter out NaN joints
            valid_mask = ~np.isnan(joints_t).any(axis=-1)

            # Update joint positions
            visible_joints = joints_t.copy()
            visible_joints[~valid_mask] = 0.0  # hide NaN joints
            joint_pc_handle.points = visible_joints

            # Update limb lines
            for idx, (child, parent) in enumerate(valid_limbs):
                if valid_mask[child] and valid_mask[parent]:
                    pts = np.stack([joints_t[parent], joints_t[child]], axis=0)
                    limb_handles[idx].positions = pts
                    limb_handles[idx].visible = gui_show_skeleton.value
                else:
                    limb_handles[idx].visible = False

            # Update GT
            if self._gt_joints is not None and gt_joint_pc_handle is not None:
                gt_t = self._gt_joints[t]
                gt_valid = ~np.isnan(gt_t).any(axis=-1)
                gt_visible = gt_t.copy()
                gt_visible[~gt_valid] = 0.0
                gt_joint_pc_handle.points = gt_visible
                for idx, (child, parent) in enumerate(valid_limbs):
                    if gt_valid[child] and gt_valid[parent]:
                        pts = np.stack([gt_t[parent], gt_t[child]], axis=0)
                        gt_limb_handles[idx].positions = pts
                        gt_limb_handles[idx].visible = gui_show_gt.value
                    else:
                        gt_limb_handles[idx].visible = False

            # Update mesh
            if mesh_handle is not None and self._verts is not None:
                mesh_handle.vertices = self._verts[t]

            # Camera attach (follow head joint, index 15)
            if gui_attach.value:
                head_idx = min(15, J - 1)
                if valid_mask[head_idx]:
                    head_pos = joints_t[head_idx]
                    for client in server.get_clients().values():
                        # Look at the head from behind
                        # Use pelvis-to-head direction for forward vector
                        pelvis_idx = 0
                        if valid_mask[pelvis_idx]:
                            fwd = head_pos - joints_t[pelvis_idx]
                            fwd_norm = np.linalg.norm(fwd[:2])
                            if fwd_norm > 1e-3:
                                fwd = fwd / np.linalg.norm(fwd)
                                cam_pos = head_pos - fwd * gui_attach_dist.value
                                cam_pos[2] = head_pos[2] + 0.3
                                client.camera.position = cam_pos
                                client.camera.look_at = head_pos

        # ── Main loop ──
        if dry_run:
            print("[ViserSmplViewer] Dry run: server started, performing one update then exit")
            do_update()
            print(f"[ViserSmplViewer] Server URL: http://localhost:{self.port}")
            return

        print(f"[ViserSmplViewer] Server started at http://localhost:{self.port}")

        prev_time = time.time()
        while True:
            start, end = gui_start_end.value
            duration = max(end - start, 1)

            now = time.time()
            sleepdur = 1.0 / gui_framerate.value - (now - prev_time)
            if sleepdur > 0.0:
                time.sleep(sleepdur)
            prev_time = time.time()

            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start

            do_update()


def visualize_omnipose_output(
    joints: np.ndarray,
    splat_path: Path | None = None,
    gt_joints: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    point_cloud: np.ndarray | None = None,
    floor_z: float = 0.0,
    port: int = 8080,
    dry_run: bool = False,
) -> None:
    """Convenience function: visualize OmniPose predictions in viser.
    
    Args:
        joints: (T, J, 3) predicted joint positions
        splat_path: path to .splat or .ply Gaussian splat file
        gt_joints: (T, J, 3) optional ground truth joints
        vertices: (T, V, 3) optional mesh vertices per frame
        faces: (F, 3) optional mesh face indices
        point_cloud: (N, 3) optional Aria MPS point cloud
        floor_z: z-coordinate of the ground plane
        port: viser server port
        dry_run: if True, start server but exit after one update
    """
    viewer = ViserSmplViewer(port=port, floor_z=floor_z)
    viewer.set_joints(joints, gt_joints=gt_joints)
    if vertices is not None and faces is not None:
        viewer.set_mesh(vertices, faces)
    viewer.set_splat(splat_path)
    if point_cloud is not None:
        viewer.set_point_cloud(point_cloud)
    viewer.run(dry_run=dry_run)


if __name__ == "__main__":
    """Self-test: create a simple bouncing skeleton animation."""
    print("Running self-test with synthetic skeleton data...")
    T = 120
    J = 22
    # Create a simple idle skeleton with slight head bob
    base_skeleton = np.zeros((J, 3), dtype=np.float32)
    # Pelvis at origin
    base_skeleton[0] = [0, 0, 0.9]       # Pelvis
    base_skeleton[1] = [-0.1, 0, 0.85]    # L_Hip
    base_skeleton[2] = [0.1, 0, 0.85]     # R_Hip
    base_skeleton[3] = [0, 0, 1.05]       # Spine1
    base_skeleton[4] = [-0.1, 0, 0.5]     # L_Knee
    base_skeleton[5] = [0.1, 0, 0.5]      # R_Knee
    base_skeleton[6] = [0, 0, 1.2]        # Spine2
    base_skeleton[7] = [-0.1, 0, 0.05]    # L_Ankle
    base_skeleton[8] = [0.1, 0, 0.05]     # R_Ankle
    base_skeleton[9] = [0, 0, 1.35]       # Spine3
    base_skeleton[10] = [-0.1, 0, 0.0]    # L_Foot
    base_skeleton[11] = [0.1, 0, 0.0]     # R_Foot
    base_skeleton[12] = [0, 0, 1.5]       # Neck
    base_skeleton[13] = [-0.12, 0, 1.42]  # L_Collar
    base_skeleton[14] = [0.12, 0, 1.42]   # R_Collar
    base_skeleton[15] = [0, 0, 1.65]      # Head
    base_skeleton[16] = [-0.2, 0, 1.35]   # L_Shoulder
    base_skeleton[17] = [0.2, 0, 1.35]    # R_Shoulder
    base_skeleton[18] = [-0.2, 0, 1.0]    # L_Elbow
    base_skeleton[19] = [0.2, 0, 1.0]     # R_Elbow
    base_skeleton[20] = [-0.2, 0, 0.7]    # L_Wrist
    base_skeleton[21] = [0.2, 0, 0.7]     # R_Wrist

    joints = np.tile(base_skeleton[None, :, :], (T, 1, 1))
    # Add walking motion
    for t in range(T):
        phase = t / T * 2 * np.pi * 2
        joints[t, :, 1] += t * 0.02  # Walk forward in Y
        joints[t, 15, 2] += np.sin(phase) * 0.02  # Head bob

    viewer = ViserSmplViewer(port=8080, floor_z=0.0)
    viewer.set_joints(joints)
    viewer.run()
