import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import pickle as p
import numpy as np
import torch
import smplx

# Limit thread usage in numerical libraries (helps with HPC / dataloaders)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def expand_hand_pose(src, target_len=45):
    """Expand or truncate source hand-pose vector to target_len.
    Strategy: place src in the beginning and pad with zeros.
    If src is longer than target_len, truncate.
    """
    src = np.asarray(src).reshape(-1)
    if src.size >= target_len:
        return src[:target_len].astype(np.float32)
    else:
        out = np.zeros((target_len,), dtype=np.float32)
        out[: src.size] = src.astype(np.float32)
        return out


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = p.load(f)
    return data


def compose_smplx_param_vector(d, expect_hand_per=45):
    """Given loaded pkl dict `d`, compose a single parameter vector for SMPL-X.
    Returns (param_vec, meta) where param_vec is 1D numpy array and meta is dict of lengths
    """
    # expected sizes
    sizes = {
        'global_orient': 3,
        'transl': 3,
        'betas': 10,
        'body_pose': 63,
        'left_hand_pose': expect_hand_per,
        'right_hand_pose': expect_hand_per,
        'jaw_pose': 3,
        'leye_pose': 3,
        'reye_pose': 3,
    }

    out_parts = []
    meta = {}

    # helper to get key or zeros
    def get_key(k, tgt_len):
        if k in d and d[k] is not None:
            arr = np.asarray(d[k]).reshape(-1)
            # sometimes arrays include batch dim
            if arr.size == tgt_len or arr.size > tgt_len:
                return arr[:tgt_len].astype(np.float32)
            elif arr.size < tgt_len:
                # for hands, use expand_hand_pose
                if 'hand' in k:
                    return expand_hand_pose(arr, tgt_len)
                # for betas, if shape is (1,10) — flatten
                out = np.zeros((tgt_len,), dtype=np.float32)
                out[:arr.size] = arr.astype(np.float32)
                return out
        # missing -> zeros
        return np.zeros((tgt_len,), dtype=np.float32)

    for k, tgt in sizes.items():
        part = get_key(k, tgt)
        meta[k] = part.shape[0]
        out_parts.append(part)

    param_vec = np.concatenate(out_parts, axis=0)
    return param_vec, meta

def process_sequence_multi(
    seq_dir,
    save_joints_path,
    save_params_path,
    smplx_model,
    device,
    save_pose_embedding=False,
    align_to_frames=False,
    expect_hand_per=45,
):
    """
    Like process_sequence but handles multiple objects (multiple .pkl files per frame folder).
    Signature and return behavior match process_sequence:
      - If exactly one object found, writes to save_joints_path and save_params_path exactly.
      - If multiple objects found, writes per-object files in the same directories as the provided paths,
        by inserting the object id before the .npy extension: e.g. recording.npy -> recording_004.npy
    Returns:
      True on success (at least one object processed and saved), False otherwise.
    """
    # gather frame folders
    frame_folders = sorted([d for d in glob.glob(os.path.join(seq_dir, "*")) if os.path.isdir(d)])
    if len(frame_folders) == 0:
        raise AssertionError(f"No frame subfolders found under {seq_dir}")

    # Build mapping: object_id -> list of (frame_idx, pkl_path)
    obj_map = {}
    for frame_idx, folder in enumerate(frame_folders):
        pkl_paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for p in pkl_paths:
            obj_id = os.path.splitext(os.path.basename(p))[0]
            obj_map.setdefault(obj_id, []).append((frame_idx, p))

    if len(obj_map) == 0:
        print(f"No .pkl files found in {seq_dir}")
        return False

    seqname = os.path.basename(os.path.normpath(seq_dir))

    # ensure directories exist (use same parent directories as provided paths)
    os.makedirs(os.path.dirname(save_joints_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_params_path), exist_ok=True)

    smplx_model.eval()

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)

    # Helper to create per-object file path based on the provided path.
    def _make_object_path(base_path, obj_id):
        # If base filename is "recording.npy" -> "recording_<obj>.npy"
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(base_dir, f"{name}_{obj_id}{ext}")

    any_saved = False

    # If only one object, keep the exact filenames (behave like process_sequence)
    single_object_mode = (len(obj_map) == 1)
    # If single, pick that object id so we can reuse same save paths
    if single_object_mode:
        only_obj = next(iter(obj_map.keys()))
        frame_entries = sorted(obj_map[only_obj], key=lambda t: t[0])

        param_list = []
        joints_list = []
        pose_embeddings = []

        with torch.no_grad():
            for (frame_idx, pkl_path) in tqdm(frame_entries, desc=f"Processing {seqname}"):
                d = load_pkl(pkl_path)
                param_vec, meta = compose_smplx_param_vector(d, expect_hand_per=expect_hand_per)

                global_orient = to_tensor(param_vec[0:3])
                transl = to_tensor(param_vec[3:6])
                betas = to_tensor(param_vec[6:6+10])
                body_pose = to_tensor(param_vec[16:16+63])
                hand_pose = to_tensor(param_vec[16+63:16+63+90])
                off = 16 + 63 + 90
                jaw_pose = to_tensor(param_vec[off:off+3]); off += 3
                leye_pose = to_tensor(param_vec[off:off+3]); off += 3
                reye_pose = to_tensor(param_vec[off:off+3])

                body_params = {
                    'global_orient': global_orient,
                    'transl': transl,
                    'betas': betas,
                    'body_pose': body_pose,
                    'hand_pose': hand_pose,
                    'jaw_pose': jaw_pose,
                    'leye_pose': leye_pose,
                    'reye_pose': reye_pose,
                }

                smplx_output = smplx_model(**body_params)
                smplx_joints = smplx_output.joints[:, :25, :].detach().cpu().numpy()

                param_list.append(param_vec.astype(np.float32))
                joints_list.append(smplx_joints.squeeze(0))
                if save_pose_embedding and ('pose_embedding' in d):
                    pose_embeddings.append(np.asarray(d['pose_embedding']).reshape(-1).astype(np.float32))

        if len(joints_list) == 0:
            print(f"No frames processed for {seq_dir}")
            return False

        joints_np = np.stack(joints_list, axis=0)
        params_np = np.stack(param_list, axis=0)

        np.save(save_joints_path, joints_np)
        np.save(save_params_path, params_np)

        if save_pose_embedding and len(pose_embeddings) > 0:
            pe_np = np.stack(pose_embeddings, axis=0)
            np.save(save_params_path.replace('.npy', '_pose_embedding.npy'), pe_np)

        any_saved = True
        return True

    # Multiple objects: save per-object files in same parent directories as save_joints_path/save_params_path,
    # using the provided filenames as base and inserting object id before extension.
    for obj_id, frame_entries in obj_map.items():
        frame_entries = sorted(frame_entries, key=lambda t: t[0])

        if align_to_frames:
            total_len = len(frame_folders)
            params_by_index = [None] * total_len
            joints_by_index = [None] * total_len
            pe_by_index = [None] * total_len

        else:
            param_list = []
            joints_list = []
            pose_embeddings = []

        with torch.no_grad():
            for (frame_idx, pkl_path) in tqdm(frame_entries, desc=f"Processing {seqname} object {obj_id}", leave=False):
                d = load_pkl(pkl_path)
                param_vec, meta = compose_smplx_param_vector(d, expect_hand_per=expect_hand_per)

                global_orient = to_tensor(param_vec[0:3])
                transl = to_tensor(param_vec[3:6])
                betas = to_tensor(param_vec[6:6+10])
                body_pose = to_tensor(param_vec[16:16+63])
                hand_pose = to_tensor(param_vec[16+63:16+63+90])
                off = 16 + 63 + 90
                jaw_pose = to_tensor(param_vec[off:off+3]); off += 3
                leye_pose = to_tensor(param_vec[off:off+3]); off += 3
                reye_pose = to_tensor(param_vec[off:off+3])

                body_params = {
                    'global_orient': global_orient,
                    'transl': transl,
                    'betas': betas,
                    'body_pose': body_pose,
                    'hand_pose': hand_pose,
                    'jaw_pose': jaw_pose,
                    'leye_pose': leye_pose,
                    'reye_pose': reye_pose,
                }

                smplx_output = smplx_model(**body_params)
                smplx_joints = smplx_output.joints[:, :25, :].detach().cpu().numpy()

                if align_to_frames:
                    params_by_index[frame_idx] = param_vec.astype(np.float32)
                    joints_by_index[frame_idx] = smplx_joints.squeeze(0)
                    if save_pose_embedding and ('pose_embedding' in d):
                        pe_by_index[frame_idx] = np.asarray(d['pose_embedding']).reshape(-1).astype(np.float32)
                else:
                    param_list.append(param_vec.astype(np.float32))
                    joints_list.append(smplx_joints.squeeze(0))
                    if save_pose_embedding and ('pose_embedding' in d):
                        pose_embeddings.append(np.asarray(d['pose_embedding']).reshape(-1).astype(np.float32))

        # assemble final arrays
        if align_to_frames:
            present_idx = [i for i, v in enumerate(joints_by_index) if v is not None]
            if len(present_idx) == 0:
                print(f"No frames for object {obj_id} in {seq_dir}, skipping")
                continue

            total_len = len(frame_folders)
            joints_np = np.full((total_len, 25, 3), np.nan, dtype=np.float32)
            params_dim = params_by_index[next(i for i, v in enumerate(params_by_index) if v is not None)].shape[0]
            params_np = np.full((total_len, params_dim), np.nan, dtype=np.float32)

            for i in range(total_len):
                if joints_by_index[i] is not None:
                    joints_np[i] = joints_by_index[i]
                    params_np[i] = params_by_index[i]

            if save_pose_embedding:
                pe_dim = None
                for v in pe_by_index:
                    if v is not None:
                        pe_dim = v.shape[0]
                        break
                if pe_dim is not None:
                    pe_np = np.full((total_len, pe_dim), np.nan, dtype=np.float32)
                    for i in range(total_len):
                        if pe_by_index[i] is not None:
                            pe_np[i] = pe_by_index[i]
                else:
                    pe_np = None

        else:
            if len(joints_list) == 0:
                print(f"No frames for object {obj_id} in {seq_dir}, skipping")
                continue

            joints_np = np.stack(joints_list, axis=0)
            params_np = np.stack(param_list, axis=0)
            pe_np = np.stack(pose_embeddings, axis=0) if (save_pose_embedding and len(pose_embeddings) > 0) else None

        # build per-object save paths using the provided base paths
        joints_path_obj = _make_object_path(save_joints_path, obj_id)
        params_path_obj = _make_object_path(save_params_path, obj_id)

        # ensure parent dirs exist (they will, but keep same behavior)
        os.makedirs(os.path.dirname(joints_path_obj), exist_ok=True)
        os.makedirs(os.path.dirname(params_path_obj), exist_ok=True)

        np.save(joints_path_obj, joints_np)
        np.save(params_path_obj, params_np)

        if save_pose_embedding and (('pe_np' in locals() and pe_np is not None) or ('pe_np' in globals() and pe_np is not None)):
            pe_path_obj = params_path_obj.replace('.npy', '_pose_embedding.npy')
            np.save(pe_path_obj, pe_np)

        any_saved = True

    return any_saved

def process_sequence(seq_dir, save_joints_path, save_params_path, smplx_model, device, save_pose_embedding=False):
    """Process one sequence directory which contains many numbered subfolders each with 001.pkl.
    We assume seq_dir has children like '00005/001.pkl', '00006/001.pkl', ...
    """
    # find all 001.pkl files under seq_dir's immediate subdirs
    frame_pkl_paths = sorted(glob.glob(os.path.join(seq_dir, "*/*.pkl")))
    assert len(frame_pkl_paths) > 0

    param_list = []
    joints_list = []
    pose_embeddings = []

    smplx_model.eval()
    with torch.no_grad():
        for pkl_path in tqdm(frame_pkl_paths, desc=f"Processing {os.path.basename(seq_dir)}"):
            d = load_pkl(pkl_path)
            # Compose param vector used for saving
            param_vec, meta = compose_smplx_param_vector(d, expect_hand_per=45)

            # Now build the inputs to smplx model. Convert to torch tensors with batch dim 1
            def to_tensor(x):
                return torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)

            global_orient = to_tensor(param_vec[0:3])
            transl = to_tensor(param_vec[3:6])
            betas = to_tensor(param_vec[6:6+10])
            body_pose = to_tensor(param_vec[16:16+63])
            hand_pose = to_tensor(param_vec[16+63:16+63+90])
            # remaining small poses
            off = 16 + 63 + 90
            jaw_pose = to_tensor(param_vec[off:off+3])
            off += 3
            leye_pose = to_tensor(param_vec[off:off+3])
            off += 3
            reye_pose = to_tensor(param_vec[off:off+3])

            body_params = {
                'global_orient': global_orient,
                'transl': transl,
                'betas': betas,
                'body_pose': body_pose,
                'hand_pose': hand_pose,
                'jaw_pose': jaw_pose,
                'leye_pose': leye_pose,
                'reye_pose': reye_pose,
            }

            smplx_output = smplx_model(**body_params)
            # select first 25 joints (like example). Some models return more joints.
            smplx_joints = smplx_output.joints[:, :25, :].detach().cpu().numpy()  # [1, 25, 3]

            param_list.append(param_vec.astype(np.float32))
            joints_list.append(smplx_joints.squeeze(0))

            if save_pose_embedding and ('pose_embedding' in d):
                pose_embeddings.append(np.asarray(d['pose_embedding']).reshape(-1).astype(np.float32))

    if len(joints_list) == 0:
        print(f"No frames processed for {seq_dir}")
        return False

    joints_np = np.stack(joints_list, axis=0)  # [seq_len, 25, 3]
    params_np = np.stack(param_list, axis=0)  # [seq_len, param_dim]

    os.makedirs(os.path.dirname(save_joints_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_params_path), exist_ok=True)
    np.save(save_joints_path, joints_np)
    np.save(save_params_path, params_np)

    if save_pose_embedding and len(pose_embeddings) > 0:
        pe_np = np.stack(pose_embeddings, axis=0)
        np.save(save_params_path.replace('.npy', '_pose_embedding.npy'), pe_np)

    return True


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    smplx_model = smplx.create(model_path=args.body_model_path, model_type='smplx', gender='neutral', use_pca=False).to(device)

    # Expect that args.rich_root contains many subject folders like BBQ_001_guitar (or higher-level root containing many such folders).
    seq_dirs = []
    if os.path.isdir(args.rich_root):
        # if rich_root directly points to one sample like BBQ_001_guitar, process it
        # Otherwise if rich_root contains multiple samples, iterate through them
        # We'll consider each leaf folder that contains '*/001.pkl' as a sequence
        # First check immediate subfolders named like BBQ_001_*
        candidate = sorted([str(p) for p in Path(args.rich_root).glob('*')])
        # If the directory name itself contains numbered subdirs (00005 etc.), treat rich_root as one sample
        has_numbered = any(Path(args.rich_root).joinpath(name).is_dir() for name in ['00005', '00006'])
        if has_numbered:
            seq_dirs = [args.rich_root]
        else:
            # otherwise each child that contains numbered subdirs is a sequence
            for c in candidate:
                if os.path.isdir(c):
                    # check if c has subfolders containing 001.pkl
                    if len(glob.glob(os.path.join(c, '*/*.pkl'))) > 0:
                        seq_dirs.append(c)
    else:
        raise RuntimeError(f"rich_root {args.rich_root} is not a directory")

    save_root_joints_path = os.path.join(args.save_root, f"pose_data_fps_{args.fps}")
    save_root_smpl_params_path = os.path.join(args.save_root, f"smpl_data_fps_{args.fps}")

    for seq in tqdm(seq_dirs, desc='All sequences'):
        subject_name = Path(seq).name
        recording_name = subject_name  # for single-sample directories we keep the name; if you want per-subject naming change here

        save_path_joints = os.path.join(save_root_joints_path, subject_name, recording_name + '.npy')
        save_path_smpl_params = os.path.join(save_root_smpl_params_path, subject_name, recording_name + '.npy')

        # success = process_sequence(seq, save_path_joints, save_path_smpl_params, smplx_model, device, save_pose_embedding=args.save_pose_embedding)
        success = process_sequence_multi(seq, save_path_joints, save_path_smpl_params, smplx_model, device)
        if not success:
            print(f"Failed to process {seq}")

    print('Finished preprocessing RICH dataset.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_model_path', type=str, required=True, help='path to smplx model directory')
    parser.add_argument('--rich_root', type=str, required=True, help='Root dir of RICH dataset (one sample like BBQ_001_guitar or parent containing many samples)')
    parser.add_argument('--save_root', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--fps', type=int, default=30, help='nominal fps for naming only')
    parser.add_argument('--save_pose_embedding', action='store_true', help='Also save pose_embedding if present in the pkl')
    args = parser.parse_args()
    main(args)

