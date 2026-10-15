# utils/egoexo_vis_utils.py
import cv2
import logging
import numpy as np
from pathlib import Path
import os
import json
from typing import Generator, List, Dict, Any, Optional, Tuple
from cprint import cprint

# Projectaria tools for VRS and MPS data
from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sophus import SE3 # For poses

class SimpleEgoExoPaths:
    def __init__(self, egoexo_root_path: str, take_name: str, take_uid: Optional[str] = None):
        self.root = Path(egoexo_root_path)
        self.take_name = take_name
        self.take_root = self.root / "takes" / self.take_name
        self.annotations_root = self.root / "annotations"
        self.take_uid = take_uid

    def get_ego_video_path(self) -> Optional[Path]:
        # Prefer preview.mp4 as it's usually smaller and processed
        for fname in ["ego_preview.mp4", "aria_rgb.mp4", f"{self.take_name}.mp4", "scene.mp4"]:
            p = self.take_root / fname
            if p.exists(): return p
            p_sub = self.take_root / "video" / fname # Common alternative
            if p_sub.exists(): return p_sub
        return None

    def get_exo_video_path(self, exo_cam_name: str) -> Optional[Path]:
        fav_dir = self.take_root / "frame_aligned_videos"
        if fav_dir.exists():
            p = fav_dir / f"{exo_cam_name}.mp4"
            if p.exists(): return p
        
        p_direct = self.take_root / f"{exo_cam_name}.mp4"
        if p_direct.exists(): return p_direct
        return None

    def get_camera_info_path(self, split: str) -> Optional[Path]:
        if not self.take_uid: return None
        p = self.annotations_root / "ego_pose" / split / "camera_pose" / f"{self.take_uid}.json"
        return p if p.exists() else None



    def get_egopose_annotation_path(self, split: str, annotation_type: str = "body") -> Optional[Path]:
        # Example: annotations/ego_pose/split/body/annotation/take_uid.json
        # Again, needs take_uid.
        # Simplified:
        p = self.take_root / f"egopose_{annotation_type}.json" # Highly simplified placeholder
        if p.exists(): return p
        return None
        
    def get_gopro_calibs_path(self) -> Optional[Path]:
        p = self.take_root / "trajectory" / "gopro_calibs.csv"
        if p.exists(): return p
        return None


class EgoExoFrameSynchronizer:
    def __init__(self, egoexo_root_path: str, take_name: str, take_uid: Optional[str] = None, split: Optional[str] = None):
        self.paths = SimpleEgoExoPaths(egoexo_root_path, take_name, take_uid=take_uid)
        self.take_name = take_name
        self.take_uid = take_uid # Important for finding some annotation files
        self.split = split       # Important for finding some annotation files

        self.ego_video_path = self.paths.get_ego_video_path()

        self.exo_cam_names = []
        self.exo_cam_intrinsics: Dict[str, np.ndarray] = {} # K matrix
        self.exo_cam_extrinsics_world_to_cam: Dict[str, Dict[Any, SE3]] = {} # T_cam_world, keyed by frame index or 'static'
        self.exo_cam_distortion_coeffs: Dict[str, np.ndarray] = {}
        self.ego_cam_intrinsics: Optional[np.ndarray] = None
        self.ego_cam_extrinsics_world_to_cam: Dict[int, SE3] = {} # T_cam_world for ego, keyed by frame index

        self.egopose_annos_2d: Dict[str, Dict[int, Dict[str, Any]]] = {} # cam_name -> frame_idx -> annotation_dict

        self._load_camera_parameters()
        self._load_2d_annotations()

        if self.ego_video_path:
            cprint.info(f"[EgoExoFrameSync] Initialized for take: {self.take_name}, ego_video: {self.ego_video_path}")
        else:
            cprint.warn(f"[EgoExoFrameSync] Warning: Egocentric video for take '{self.take_name}' not found during init.")

    def _load_camera_parameters(self):
        # Load static GoPro calibrations
        gopro_calibs_path = self.paths.get_gopro_calibs_path()
        if gopro_calibs_path and gopro_calibs_path.exists():
            static_calibrations = mps.read_static_camera_calibrations(str(gopro_calibs_path))
            for static_calib in static_calibrations:
                if static_calib.quality != 1.0:
                    continue
                cam_uid = static_calib.camera_uid
                if cam_uid not in self.exo_cam_names:
                    self.exo_cam_names.append(cam_uid)
                
                self.exo_cam_intrinsics[cam_uid] = np.array([
                    [static_calib.intrinsics[0], 0, static_calib.intrinsics[2]],
                    [0, static_calib.intrinsics[1], static_calib.intrinsics[3]],
                    [0, 0, 1]
                ])
                self.exo_cam_distortion_coeffs[cam_uid] = np.array(static_calib.intrinsics[4:])
                
                T_world_cam = static_calib.transform_world_cam
                if cam_uid not in self.exo_cam_extrinsics_world_to_cam:
                    self.exo_cam_extrinsics_world_to_cam[cam_uid] = {}
                self.exo_cam_extrinsics_world_to_cam[cam_uid]['static'] = T_world_cam.inverse()
        
        # Load dynamic camera info from JSON if available (overwrites static poses if needed)
        if self.take_uid and self.split:
            camera_info_path = self.paths.get_camera_info_path(self.split)
            if camera_info_path and camera_info_path.exists():
                with open(camera_info_path, 'r') as f:
                    camera_info = json.load(f)
                    for cam_name, cam_data in camera_info.items():
                        if 'camera_extrinsics' in cam_data:
                            camera_extrinsics = cam_data['camera_extrinsics']
                            
                            # Check if this is ego camera (dict with frame keys) or exo camera (list)
                            if isinstance(camera_extrinsics, dict):
                                self.ego_cam_intrinsics = np.array(cam_data['camera_intrinsics'])
                                # This is ego camera with per-frame poses
                                for frame_str, extrinsics_matrix in camera_extrinsics.items():
                                    frame_idx = int(frame_str)
                                    T_world_cam = SE3.from_matrix3x4(np.array(extrinsics_matrix))
                                    self.ego_cam_extrinsics_world_to_cam[frame_idx] = T_world_cam.inverse()

    def _load_2d_annotations(self):
        if self.take_uid and self.split:
            anno_path = self.paths.get_egopose_annotation_path(self.split, "body")
            if anno_path and anno_path.exists():
                with open(anno_path, 'r') as f:
                    all_annos = json.load(f)
                for frame_idx_str, persons_in_frame in all_annos.items():
                    frame_idx_int = int(frame_idx_str)
                    if not persons_in_frame: continue
                    person_anno = persons_in_frame[0]
                    if "annotation2D" in person_anno:
                        for cam_name, kpts_dict in person_anno["annotation2D"].items():
                            if cam_name not in self.egopose_annos_2d:
                                self.egopose_annos_2d[cam_name] = {}
                            self.egopose_annos_2d[cam_name][frame_idx_int] = kpts_dict

    @staticmethod
    def _read_single_frame(cap: cv2.VideoCapture, frame_idx: int, video_path: Path) -> np.ndarray:
        """Seek to *frame_idx* in an open VideoCapture and return the RGB frame.

        Raises:
            RuntimeError: if the seek or read fails.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"[EgoExoFrameSync] Failed to read frame {frame_idx} from {video_path}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_synchronized_frames_iter(
        self, ego_frame_indices: List[int]
    ) -> "Generator[Dict[str, Any], None, None]":
        """Lazily yield one synchronised frame-set at a time.

        For each egocentric frame index the generator reads exactly one ego
        frame and one frame per exo camera, composes the result dict, and
        yields it.  Memory usage is O(num_cameras) instead of
        O(num_cameras × num_frames).

        Video handles are opened once and closed when the generator is
        exhausted or explicitly ``.close()``d.

        Yields:
            Dict with keys ``ego_frame``, ``ego_frame_original_index``,
            ``exo_frames_data``.
        """
        _logger = logging.getLogger(__name__)

        if not self.ego_video_path:
            raise FileNotFoundError(
                f"[EgoExoFrameSync] Ego video not found for take '{self.take_name}'. "
                f"Searched paths rooted at {self.paths.take_root}"
            )

        # --- Open all video handles once ---
        ego_cap = cv2.VideoCapture(str(self.ego_video_path))
        if not ego_cap.isOpened():
            raise RuntimeError(
                f"[EgoExoFrameSync] Cannot open ego video: {self.ego_video_path}"
            )

        exo_caps: Dict[str, cv2.VideoCapture] = {}
        exo_video_paths: Dict[str, Path] = {}  # Cache validated paths for inner loop
        exo_params_per_cam: Dict[str, Dict[str, Any]] = {}

        for exo_cam_name in self.exo_cam_names:
            exo_video_path = self.paths.get_exo_video_path(exo_cam_name)
            if exo_video_path is None:
                _logger.warning(
                    "[EgoExoFrameSync] Exo video path not found for camera '%s' — skipping camera",
                    exo_cam_name,
                )
                continue

            cap = cv2.VideoCapture(str(exo_video_path))
            if not cap.isOpened():
                _logger.warning(
                    "[EgoExoFrameSync] Cannot open exo video '%s' at %s — skipping camera",
                    exo_cam_name, exo_video_path,
                )
                continue

            exo_caps[exo_cam_name] = cap
            exo_video_paths[exo_cam_name] = exo_video_path
            # Static per-camera params (shared dict reference across frames)
            exo_params_per_cam[exo_cam_name] = {
                'K': self.exo_cam_intrinsics[exo_cam_name],
                'dist_coeffs': self.exo_cam_distortion_coeffs[exo_cam_name],
                'T_cam_world': self.exo_cam_extrinsics_world_to_cam[exo_cam_name]['static'],
            }

        try:
            for ego_frame_idx in ego_frame_indices:
                ego_frame = self._read_single_frame(ego_cap, ego_frame_idx, self.ego_video_path)

                exo_frames_data: Dict[str, Dict[str, Any]] = {}
                for exo_cam_name, exo_cap in exo_caps.items():
                    exo_frame = self._read_single_frame(
                        exo_cap, ego_frame_idx,
                        exo_video_paths[exo_cam_name],
                    )
                    exo_frames_data[exo_cam_name] = {
                        'frame': exo_frame,
                        'params': exo_params_per_cam[exo_cam_name],
                        'annos_2d': self.egopose_annos_2d.get(exo_cam_name, {}).get(ego_frame_idx),
                    }

                yield {
                    "ego_frame": ego_frame,
                    "ego_frame_original_index": ego_frame_idx,
                    "exo_frames_data": exo_frames_data,
                }
        finally:
            # Guarantee all video handles are closed even on early exit / .close()
            ego_cap.release()
            for cap in exo_caps.values():
                cap.release()

if __name__ == '__main__':
    from data_sets.dataset_egoexo import DatasetEgoExo
    import third_party.ego_exo4d_egopose.bodypose.utils.utils_option as option
    from torch.utils.data import DataLoader
    from utils import dist_util
    from tqdm import tqdm

    egoexo_cfg_path = "cfg_files/test_cfg/egoexo/test_egoexo.json"
    egoexo_optn = option.parse(egoexo_cfg_path, is_train=True)
    for phase, dataset_opt in egoexo_optn['datasets'].items():
        testD = DatasetEgoExo(
            dataset_opt,
            mask_scheme='random',
            repr_abs_only=True,
            include_global_joint_rotations=False,
            logdir='test_results/results_amass_Jun_25',
            body_model_path="data/body_models/smplx_model",
            device=dist_util.dev(),
            use_ddp=False,
            joints_num=22,
            dist_rank=0,
            mean_std=True,
            disable_cache=True,
            norm_strategy='meanstd',
        )
        test_dl = DataLoader(
            testD,
            batch_size=dataset_opt['dataloader_batch_size'],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        for batch in test_dl:
            take_name = batch['meta']['take_name'][0]
            take_uid = batch['meta']['take_uid'][0]

            if take_uid != "cc8bfc86-777e-4537-b207-9c599e9a1f40":
                continue
            original_frame_keys = batch['meta']['frame_keys'][0].cpu().numpy().tolist()
            ego_frame_synchronizer = EgoExoFrameSynchronizer(
                egoexo_root_path='datasets/egoexo-default',
                take_name=take_name,
                take_uid=take_uid,
                split=batch['meta']['split'][0]
            )

            for synced_data in tqdm(
                ego_frame_synchronizer.get_synchronized_frames_iter(
                    ego_frame_indices=original_frame_keys,
                ),
                total=len(original_frame_keys),
                desc="Rendering frames",
            ):
                ego_rgb_frame = synced_data['ego_frame']
                cv2.imshow("ego_rgb_frame", ego_rgb_frame)
                cv2.waitKey(1)

                for cam_name, exo_data in synced_data['exo_frames_data'].items():
                    exo_frame_rgb = exo_data['frame']
                    cv2.imshow(f"exo_{cam_name}", exo_frame_rgb)
                    cv2.waitKey(1)

