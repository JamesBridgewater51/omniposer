"""
Standalone viser visualization for EgoExo dataset predictions.

Usage:
    # With egoallo example trajectory (uses egoallo NPZ output + scene.splat):
    python -m test_pipeline.visualization.viser_vis_egoexo \\
        --traj_dir third_party/egoallo/egoallo_example_trajectories/coffeemachine

    # With EgoExo dataset take (specify splat dir separately):
    python -m test_pipeline.visualization.viser_vis_egoexo \\
        --traj_dir datasets/egoexo-default.sample/cmu_bike14_5 \\
        --splat_dir datasets/egoexo-default.sample/cmu_bike14_5.train

    # With custom joints NPZ:
    python -m test_pipeline.visualization.viser_vis_egoexo \\
        --joints_npz /path/to/omnipose_joints.npz \\
        --splat_dir datasets/egoexo-default.sample/cmu_bike14_5.train

Coordinate System:
    Both nerfstudio splatfacto and Aria MPS trajectories use the same
    world coordinate frame (gravity-aligned, Z-up). OmniPose operates
    in the Aria coordinate system which is the same world frame when
    using the 'global' or 'aria' coordinate convention.

    For EgoExo, the annotation coordinate system (after translate_poses)
    is relative-to-first-Aria-frame when coord='aria', or world when coord='global'.
    The nerfstudio splatfacto model is trained with camera poses from Aria MPS
    which are in the same world frame.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .viser_smpl_viewer import ViserSmplViewer, find_splat_file

logger = logging.getLogger(__name__)


def load_egoallo_npz(npz_path: Path) -> dict:
    """Load egoallo-format inference output NPZ.
    
    Returns dict with 'Ts_world_root' (N_samples, T, 7) and other fields.
    We use Ts_world_root to position the skeleton via the SMPL body model.
    """
    data = np.load(npz_path)
    return {k: data[k] for k in data.keys()}


def joints_from_egoallo_npz(npz_path: Path) -> np.ndarray:
    """Extract approximate joint positions from egoallo NPZ output.
    
    Since we don't have the full SMPL-H FK pipeline available without egoallo deps,
    we use Ts_world_root translations as pelvis position and Ts_world_cpf as head.
    
    Returns:
        (T, 2, 3) array with pelvis (index 0) and head (index 1) positions.
    """
    # Try to use the full egoallo FK pipeline
    data = np.load(npz_path)
    
    # Attempt full FK with egoallo body model
    return _egoallo_fk(data)


def _egoallo_fk(data: dict) -> np.ndarray:
    """Run forward kinematics using egoallo's SMPL-H body model.
    
    Returns (T, 22, 3) joint positions.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "egoallo" / "src"))
    from egoallo import fncsmpl, fncsmpl_extensions
    from egoallo.transforms import SE3, SO3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]

    candidate_paths = [
        repo_root / "smpl_based_model" / "smplh" / "neutral" / "model.npz",
        repo_root / "third_party" / "egoallo" / "data" / "smplh" / "neutral" / "model.npz",
        repo_root / "data" / "body_models" / "smplh" / "neutral" / "model.npz",
    ]
    smplh_path = None
    for p in candidate_paths:
        if p.exists():
            smplh_path = p
            break
    assert smplh_path is not None, f"SMPL-H model not found. Searched: {[str(p) for p in candidate_paths]}"
    body_model = fncsmpl.SmplhModel.load(smplh_path).to(device)

    betas = torch.from_numpy(data["betas"]).to(device)
    body_quats = torch.from_numpy(data["body_quats"]).to(device)  # (N, T, 21, 4)
    
    left_hand_quats = None
    right_hand_quats = None
    if "left_hand_quats" in data:
        left_hand_quats = torch.from_numpy(data["left_hand_quats"]).to(device)
    if "right_hand_quats" in data:
        right_hand_quats = torch.from_numpy(data["right_hand_quats"]).to(device)

    Ts_world_cpf = torch.from_numpy(data["Ts_world_cpf"]).to(device)  # (T, 7)

    # Forward kinematics
    shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=body_quats.dtype).parameters(),
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )

    # Align to world using CPF poses
    timesteps = Ts_world_cpf.shape[0]
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        fk_outputs,
        Ts_world_cpf[None, ...],  # add sample dim
    )
    fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)

    # Extract joint positions: (N_samples, T, J, 7) -> take sample 0 -> (T, J, 3)
    joints_world = fk_outputs.Ts_world_joint[0, :, :22, 4:7].cpu().numpy()  # (T, 22, 3)

    # Also get vertices for mesh visualization
    mesh = fk_outputs.lbs()
    vertices = mesh.verts[0].cpu().numpy()  # (T, V, 3)
    faces = mesh.faces.cpu().numpy()  # (F, 3)

    return joints_world, vertices, faces


def load_omnipose_joints(npz_path: Path) -> np.ndarray:
    """Load OmniPose prediction joints from NPZ or NPY file.
    
    Expected format:
        - NPZ with key 'joints' or 'jts_rec': (T, J, 3) or (1, T, J, 3)
        - NPY: (T, J, 3) or (1, T, J, 3)
    
    Returns (T, J, 3) joint positions.
    """
    if npz_path.suffix == '.npy':
        joints = np.load(npz_path)
    else:
        data = np.load(npz_path)
        for key in ['joints', 'jts_rec', 'joint_positions', 'pred_joints']:
            if key in data:
                joints = data[key]
                break
        else:
            raise KeyError(f"No joint key found in {npz_path}. Available keys: {list(data.keys())}")

    # Remove batch dim if present
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]
    assert joints.ndim == 3, f"Expected (T, J, 3), got shape {joints.shape}"

    return joints.astype(np.float32)


def discover_trajectory_data(traj_dir: Path) -> dict:
    """Auto-discover data files in an egoallo-style trajectory directory.
    
    Expected structure:
        traj_dir/
            scene.splat or splat.ply
            egoallo_outputs/
                *.npz  (prediction files)
            slam/
                semidense_points.csv.gz or closed_loop_trajectory.csv
            video.vrs
    """
    result = {
        "splat_path": None,
        "npz_files": [],
        "points_path": None,
    }

    # Find splat
    result["splat_path"] = find_splat_file(traj_dir)

    # Find NPZ output files
    egoallo_outputs = traj_dir / "egoallo_outputs"
    if egoallo_outputs.exists():
        result["npz_files"] = sorted(egoallo_outputs.glob("*.npz"))

    # Find point cloud
    for points_name in ("semidense_points.csv.gz", "global_points.csv.gz"):
        candidates = list(traj_dir.rglob(points_name))
        if candidates:
            result["points_path"] = candidates[0]
            break

    return result


def load_aria_point_cloud(points_path: Path) -> np.ndarray | None:
    """Load Aria MPS semidense/global point cloud."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "egoallo" / "src"))
    from egoallo.data.aria_mps import load_point_cloud_and_find_ground
    
    points_data, floor_z = load_point_cloud_and_find_ground(points_path, "filtered")
    return points_data, floor_z


def main():
    parser = argparse.ArgumentParser(description="EgoExo viser 3D visualization")
    parser.add_argument("--traj_dir", type=str, default=None,
                        help="Trajectory directory (egoallo-style with scene.splat, egoallo_outputs/, slam/)")
    parser.add_argument("--splat_dir", type=str, default=None,
                        help="Separate directory to search for splat files (e.g., take_name.train/)")
    parser.add_argument("--splat_file", type=str, default=None,
                        help="Direct path to .splat or .ply file")
    parser.add_argument("--joints_npz", type=str, default=None,
                        help="Path to OmniPose joints NPZ/NPY file")
    parser.add_argument("--npz_index", type=int, default=-1,
                        help="Index of NPZ file to load from egoallo_outputs/ (-1 = latest)")
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
            # Also check .train directory
            if splat_path is None:
                train_dir = Path(str(args.traj_dir) + ".train")
                if train_dir.exists():
                    splat_path = find_splat_file(train_dir)

    if splat_path:
        print(f"[EgoExo] Found splat: {splat_path}")
    else:
        print("[EgoExo] No Gaussian splat file found")

    # ── Load joints ──
    if args.joints_npz:
        # Direct OmniPose joints
        joints = load_omnipose_joints(Path(args.joints_npz))
        print(f"[EgoExo] Loaded joints: shape={joints.shape}")

    elif args.traj_dir:
        traj_dir = Path(args.traj_dir)
        traj_data = discover_trajectory_data(traj_dir)

        # Use egoallo NPZ if available
        if traj_data["npz_files"]:
            npz_path = traj_data["npz_files"][args.npz_index]
            print(f"[EgoExo] Loading egoallo output: {npz_path}")
            result = _egoallo_fk(np.load(npz_path))
            if isinstance(result, tuple):
                joints, vertices, faces = result
            else:
                joints = result
            print(f"[EgoExo] FK joints shape: {joints.shape}")

        # Load point cloud
        if not args.no_points and traj_data["points_path"]:
            print(f"[EgoExo] Loading point cloud: {traj_data['points_path']}")
            point_cloud, floor_z = load_aria_point_cloud(traj_data["points_path"])
            print(f"[EgoExo] Point cloud: {point_cloud.shape}, floor_z={floor_z:.3f}")

        # Override splat from trajectory data if not already set
        if splat_path is None and traj_data["splat_path"]:
            splat_path = traj_data["splat_path"]
            print(f"[EgoExo] Found splat from trajectory: {splat_path}")

    if joints is None:
        print("[ERROR] No joint data to visualize. Provide --joints_npz or --traj_dir with egoallo outputs.")
        sys.exit(1)

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
