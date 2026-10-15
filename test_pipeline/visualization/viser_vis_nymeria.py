"""
Standalone viser visualization for Nymeria dataset predictions.

Usage:
    # With egoallo example trajectory (Nymeria sequences can be converted to this format):
    python -m test_pipeline.visualization.viser_vis_nymeria \\
        --traj_dir third_party/egoallo/egoallo_example_trajectories/jog

    # With Nymeria preprocessed data + prediction:
    python -m test_pipeline.visualization.viser_vis_nymeria \\
        --nymeria_seq datasets/Nymeria_smplx_preprocessed/smpl_data_fps_30/Nymeria/20230818_s1_daniel_gregory_act0_mfziu6 \\
        --joints_npz /path/to/prediction.npz

    # With custom joints + splat:
    python -m test_pipeline.visualization.viser_vis_nymeria \\
        --joints_npz /path/to/omnipose_joints.npz \\
        --splat_file /path/to/scene.splat

Coordinate System:
    Nymeria uses Aria glasses like EgoExo. When data is converted to the
    egoallo trajectory format, it shares the same Aria MPS world coordinate
    frame (gravity-aligned, Z-up). The preprocessed Nymeria data uses a
    Y-up coordinate convention in some cases, which we handle with auto-detection.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .viser_smpl_viewer import ViserSmplViewer, find_splat_file

logger = logging.getLogger(__name__)


def load_nymeria_preprocessed_sequence(
    seq_dir: Path,
    max_frames: int = 0,
) -> np.ndarray:
    """Load Nymeria preprocessed joint data from .npy files.
    
    The preprocessed data is stored as .npy files in:
        datasets/Nymeria_smplx_preprocessed/pose_data_fps_30/Nymeria/{seq_name}/
    
    Each .npy file contains (T, 25, 3) joint positions.
    
    Returns:
        (T, J, 3) concatenated joint positions
    """
    npy_files = sorted(seq_dir.glob("*.npy"))
    assert len(npy_files) > 0, f"No .npy files found in {seq_dir}"

    all_joints = []
    for npy_path in npy_files:
        joints = np.load(npy_path).astype(np.float32)  # (T, 25, 3)
        all_joints.append(joints)

    joints = np.concatenate(all_joints, axis=0)  # (total_T, 25, 3)

    if max_frames > 0 and joints.shape[0] > max_frames:
        joints = joints[:max_frames]

    # Map 25-joint format to 22-joint SMPL format
    # Nymeria preprocessed uses SMPL 25 joints: 22 body + 3 extra
    # We take the first 22
    if joints.shape[1] >= 22:
        joints = joints[:, :22, :]

    return joints


def detect_coordinate_convention(joints: np.ndarray) -> str:
    """Detect whether joints use Y-up or Z-up convention.
    
    Heuristic: If mean Z-position across joints is significantly larger than
    mean Y-position, it's likely Z-up (Aria convention). Otherwise Y-up.
    """
    mean_pos = np.nanmean(joints, axis=(0, 1))  # (3,)
    # In Z-up, Z values are larger (height); in Y-up, Y values are larger
    if abs(mean_pos[2]) > abs(mean_pos[1]) * 0.5:
        return "z_up"
    return "y_up"


def convert_yup_to_zup(joints: np.ndarray) -> np.ndarray:
    """Convert Y-up coordinates to Z-up (swap Y and Z)."""
    result = joints.copy()
    result[:, :, 1], result[:, :, 2] = joints[:, :, 2].copy(), joints[:, :, 1].copy()
    return result


def estimate_floor_z(joints: np.ndarray) -> float:
    """Estimate floor height from minimum Z of foot/ankle joints."""
    # Foot joints: 7 (L_Ankle), 8 (R_Ankle), 10 (L_Foot), 11 (R_Foot)
    J = joints.shape[1]
    foot_indices = [i for i in [7, 8, 10, 11] if i < J]
    if foot_indices:
        foot_z = joints[:, foot_indices, 2]
        floor_z = float(np.nanmin(foot_z))
    else:
        floor_z = float(np.nanmin(joints[:, :, 2]))
    return floor_z


def main():
    parser = argparse.ArgumentParser(description="Nymeria viser 3D visualization")
    parser.add_argument("--traj_dir", type=str, default=None,
                        help="egoallo-style trajectory directory")
    parser.add_argument("--nymeria_seq", type=str, default=None,
                        help="Nymeria preprocessed sequence directory (contains .npy files)")
    parser.add_argument("--joints_npz", type=str, default=None,
                        help="Path to OmniPose joints NPZ/NPY file")
    parser.add_argument("--splat_file", type=str, default=None,
                        help="Direct path to .splat or .ply file")
    parser.add_argument("--splat_dir", type=str, default=None,
                        help="Directory to search for splat files")
    parser.add_argument("--npz_index", type=int, default=-1,
                        help="Index of NPZ file to load from egoallo_outputs/ (-1 = latest)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum number of frames to load (0 = all)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--dry_run", action="store_true", help="Start server but exit after one update")
    parser.add_argument("--no_splat", action="store_true", help="Skip loading Gaussian splats")
    parser.add_argument("--no_points", action="store_true", help="Skip loading point cloud")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    joints = None
    vertices = None
    faces = None
    splat_path = None
    point_cloud = None
    floor_z = 0.0

    # ── Resolve splat file ──
    if not args.no_splat:
        if args.splat_file:
            splat_path = Path(args.splat_file)
            assert splat_path.exists(), f"Splat file not found: {splat_path}"
        elif args.splat_dir:
            splat_path = find_splat_file(Path(args.splat_dir))
        elif args.traj_dir:
            splat_path = find_splat_file(Path(args.traj_dir))

    if splat_path:
        print(f"[Nymeria] Found splat: {splat_path}")
    else:
        print("[Nymeria] No Gaussian splat file found")

    # ── Load joints ──
    if args.joints_npz:
        # Direct OmniPose joints
        from .viser_vis_egoexo import load_omnipose_joints
        joints = load_omnipose_joints(Path(args.joints_npz))
        print(f"[Nymeria] Loaded OmniPose joints: shape={joints.shape}")

    elif args.nymeria_seq:
        # Nymeria preprocessed format
        seq_dir = Path(args.nymeria_seq)
        joints = load_nymeria_preprocessed_sequence(seq_dir, max_frames=args.max_frames)
        print(f"[Nymeria] Loaded preprocessed joints: shape={joints.shape}")

        # Auto-detect and convert coordinate convention
        convention = detect_coordinate_convention(joints)
        print(f"[Nymeria] Detected coordinate convention: {convention}")
        if convention == "y_up":
            print("[Nymeria] Converting Y-up to Z-up for viser compatibility")
            joints = convert_yup_to_zup(joints)

    elif args.traj_dir:
        # egoallo-style trajectory directory
        traj_dir = Path(args.traj_dir)
        from .viser_vis_egoexo import discover_trajectory_data, _egoallo_fk, load_aria_point_cloud

        traj_data = discover_trajectory_data(traj_dir)

        if traj_data["npz_files"]:
            npz_path = traj_data["npz_files"][args.npz_index]
            print(f"[Nymeria] Loading egoallo output: {npz_path}")
            result = _egoallo_fk(np.load(npz_path))
            if isinstance(result, tuple):
                joints, vertices, faces = result
            else:
                joints = result
            print(f"[Nymeria] FK joints shape: {joints.shape}")

        # Load point cloud
        if not args.no_points and traj_data["points_path"]:
            print(f"[Nymeria] Loading point cloud: {traj_data['points_path']}")
            point_cloud, floor_z = load_aria_point_cloud(traj_data["points_path"])
            print(f"[Nymeria] Point cloud: {point_cloud.shape}, floor_z={floor_z:.3f}")

        if splat_path is None and traj_data["splat_path"]:
            splat_path = traj_data["splat_path"]
            print(f"[Nymeria] Found splat from trajectory: {splat_path}")

    if joints is None:
        print("[ERROR] No joint data to visualize.")
        print("  Provide one of: --joints_npz, --nymeria_seq, or --traj_dir")
        sys.exit(1)

    # Estimate floor height if not from point cloud
    if floor_z == 0.0:
        floor_z = estimate_floor_z(joints)
        print(f"[Nymeria] Estimated floor_z: {floor_z:.3f}")

    # ── Launch viewer ──
    viewer = ViserSmplViewer(port=args.port, floor_z=floor_z)
    viewer.set_joints(joints)
    if vertices is not None and faces is not None:
        viewer.set_mesh(vertices, faces)
    if splat_path:
        viewer.set_splat(splat_path)
    if point_cloud is not None:
        viewer.set_point_cloud(point_cloud)

    viewer.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
