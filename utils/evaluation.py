# utils/evaluation.py

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Iterable, Set
from data_sets.common.quaternion import rot6d_to_rotmat


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable
    baselines: Optional[Set[str]] = None
    requires_gt: bool = False
    requires_repr: bool = False
    requires_verts: bool = False


_METRIC_REGISTRY: Dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    baselines: Optional[Iterable[str]] = None,
    requires_gt: bool = False,
    requires_repr: bool = False,
    requires_verts: bool = False,
):
    """Register a metric computation function for specific baselines."""
    def decorator(fn):
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        baseline_set = set(baselines) if baselines is not None else None
        _METRIC_REGISTRY[name] = MetricSpec(
            name=name,
            fn=fn,
            baselines=baseline_set,
            requires_gt=requires_gt,
            requires_repr=requires_repr,
            requires_verts=requires_verts,
        )
        return fn
    return decorator


def get_metric_names_for_baseline(baseline_name: str) -> list:
    """Return metric names registered for a specific baseline."""
    names = []
    for spec in _METRIC_REGISTRY.values():
        if spec.baselines is None or baseline_name in spec.baselines:
            names.append(spec.name)
    return sorted(names)


def compute_metrics_for_baseline(
    baseline_name: str,
    pred_joints: np.ndarray,
    gt_joints: Optional[np.ndarray] = None,
    pred_verts: Optional[np.ndarray] = None,
    gt_verts: Optional[np.ndarray] = None,
    vis_joint_ids: Optional[Iterable[int]] = None,
    occ_joint_ids: Optional[Iterable[int]] = None,
    vis_joint_mask: Optional[np.ndarray] = None,
    occ_joint_mask: Optional[np.ndarray] = None,
    repr_dict_rec: Optional[Dict[str, torch.Tensor]] = None,
    repr_dict_clean: Optional[Dict[str, torch.Tensor]] = None,
    fps: int = 30,
) -> Dict[str, float]:
    """Compute metrics for a baseline using the registry."""
    evaluator = MotionEvaluator(fps=fps)
    metrics = {}
    vis_joint_ids = list(vis_joint_ids) if vis_joint_ids is not None else []
    occ_joint_ids = list(occ_joint_ids) if occ_joint_ids is not None else []

    for spec in _METRIC_REGISTRY.values():
        if spec.baselines is not None and baseline_name not in spec.baselines:
            continue
        if spec.requires_gt and gt_joints is None:
            continue
        if spec.requires_repr and (repr_dict_rec is None or repr_dict_clean is None):
            continue
        if spec.requires_verts and (pred_verts is None or gt_verts is None):
            continue
        value = spec.fn(
            evaluator=evaluator,
            pred_joints=pred_joints,
            gt_joints=gt_joints,
            pred_verts=pred_verts,
            gt_verts=gt_verts,
            vis_joint_ids=vis_joint_ids,
            occ_joint_ids=occ_joint_ids,
            vis_joint_mask=vis_joint_mask,
            occ_joint_mask=occ_joint_mask,
            repr_dict_rec=repr_dict_rec,
            repr_dict_clean=repr_dict_clean,
        )
        if value is None:
            continue
        metrics[spec.name] = float(value)

    if not metrics:
        raise ValueError(
            f"No metrics registered for baseline '{baseline_name}'. "
            f"Registered baselines: {sorted(_collect_registered_baselines())}"
        )
    return metrics


def _collect_registered_baselines() -> Set[str]:
    baselines: Set[str] = set()
    for spec in _METRIC_REGISTRY.values():
        if spec.baselines is None:
            baselines.add("ALL")
        else:
            baselines.update(spec.baselines)
    return baselines


_BASELINE_EGOEGO = "egoego"
_BASELINE_MOCAP_EVERYONE = "mocap_everyone_everywhere"
_BASELINE_HMD_POSER = "hmd_poser"
_BASELINE_EGOPOSER = "egoposer"
_BASELINE_EGOCAST = "egocast"
_BASELINE_EGOALLO = "egoallo"
_BASELINE_HEAD2BODY = "head2body"
_BASELINE_ROHM = "rohm"
_BASELINE_AVATARPOSER = "avatarposer"
_BASELINE_AGROL = "agrol"

_BASELINE_TRANSPOSE = "transpose"
_BASELINE_PIP = "pip"
_BASELINE_TIP = "tip"
_BASELINE_PNP = "pnp"
_BASELINE_DYNAIP = "dynaip"
_BASELINE_LIP = "lip"
_BASELINE_MAGSHIELD = "magshield"
_BASELINE_UMOTION = "umotion"
_BASELINE_GROUPIP = "groupip"

def procrustes_align(points_y, points_x):
    """
    Computes the similarity transform (s, R, t) that aligns points_x to points_y.
    Minimizes: || points_y - (s * R @ points_x + t) ||^2
    
    Args:
        points_y (np.ndarray): Target points of shape (*, N, 3).
        points_x (np.ndarray): Source points to be aligned, shape (*, N, 3).

    Returns:
        np.ndarray: Aligned source points of shape (*, N, 3).
    """
    # Using float64 for better precision in geometric computations
    points_y_t = torch.from_numpy(points_y).double()
    points_x_t = torch.from_numpy(points_x).double()
    
    # De-mean
    mu_y = points_y_t.mean(dim=-2, keepdim=True)
    mu_x = points_x_t.mean(dim=-2, keepdim=True)
    y0 = points_y_t - mu_y
    x0 = points_x_t - mu_x

    # Covariance
    C = torch.matmul(y0.transpose(-1, -2), x0)
    
    # SVD
    U, D, Vh = torch.linalg.svd(C)
    
    # Reflection correction
    s_diag = torch.ones_like(D)
    neg_det = torch.det(U) * torch.det(Vh) < 0
    s_diag[..., -1] = torch.where(neg_det, -1.0, 1.0)
    
    R = torch.matmul(U, torch.diag_embed(s_diag) @ Vh)
    
    # Scale
    var_x = (x0 ** 2).sum(dim=(-1, -2), keepdim=True)
    
    # --- START OF CORRECTED LINE ---
    
    s_numerator = (D * s_diag).sum(dim=-1, keepdim=True)
    
    # This is the critical fix. We unsqueeze the numerator to prevent faulty broadcasting.
    # (B, M, 1) / (B, M, 1, 1) -> Incorrectly broadcasts to (B, M, M, 1)
    # (B, M, 1, 1) / (B, M, 1, 1) -> Correctly broadcasts to (B, M, 1, 1)
    s = s_numerator.unsqueeze(-1) / var_x
    
    # --- END OF CORRECTED LINE ---

    # Translation
    t = mu_y - s * torch.matmul(R, mu_x.transpose(-1, -2)).transpose(-1, -2)
    
    # Apply transform
    aligned_x = s * torch.matmul(points_x_t, R.transpose(-1, -2)) + t
    
    return aligned_x.numpy()

def rotation_geodesic_distance(R1, R2):
    """
    Calculates the geodesic distance between two rotation matrices.
    Args:
        R1 (np.ndarray): Batch of rotation matrices of shape (*, 3, 3).
        R2 (np.ndarray): Batch of rotation matrices of shape (*, 3, 3).
    Returns:
        np.ndarray: The angular distance in degrees.
    """
    R_rel = np.einsum('...ij,...ik->...jk', R1, R2) # R1.T @ R2
    trace = np.trace(R_rel, axis1=-2, axis2=-1)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return angle_rad * 180.0 / np.pi


class MotionEvaluator:
    """A class to compute a comprehensive set of metrics for human motion."""
    
    def __init__(self, fps=30):
        self.fps = fps
        self.smpl_joint_names = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
            "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
            "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        ]
        self.joint_name_to_idx = {
            name: idx for idx, name in enumerate(self.smpl_joint_names)
        }
        self.head_idx = self.smpl_joint_names.index('head')
        self.foot_indices = [self.smpl_joint_names.index('left_foot'), self.smpl_joint_names.index('right_foot')]
        self.ankle_indices = [self.smpl_joint_names.index('left_ankle'), self.smpl_joint_names.index('right_ankle')]

    def _get_repr_by_name(self, motion_repr, name, repr_dim_dict, repr_list):
        """Helper to extract a specific feature from the motion representation array."""
        offset = 0
        for r_name in repr_list:
            dim = repr_dim_dict[r_name]
            if r_name == name:
                return motion_repr[..., offset:offset+dim]
            offset += dim
        raise ValueError(f"Representation '{name}' not found in repr_list.")

    def _compute_mpjpe(self, pred_joints, gt_joints, aligned=False):
        """
        Computes Mean Per Joint Position Error (MPJPE), with optional Procrustes alignment.
        """
        if aligned:
            # Align each frame's prediction to the ground truth
            pred_joints = procrustes_align(gt_joints, pred_joints)
        
        # L2 norm error in meters
        error = np.linalg.norm(pred_joints - gt_joints, axis=-1) # (n_seq, clip_len, 22)
        return error

    def _root_align(self, joints):
        """Root-align joints by subtracting pelvis (joint 0)."""
        return joints - joints[..., [0], :]

    def _compute_accel_error(self, pred_joints, gt_joints):
        """Computes the acceleration error."""
        accel_pred = (pred_joints[:, 2:] - 2 * pred_joints[:, 1:-1] + pred_joints[:, :-2]) * (self.fps ** 2)
        accel_gt = (gt_joints[:, 2:] - 2 * gt_joints[:, 1:-1] + gt_joints[:, :-2]) * (self.fps ** 2)
        return np.linalg.norm(accel_pred - accel_gt, axis=-1)

    def _compute_foot_contact_accuracy(self, repr_dict_rec, repr_dict_clean):
        """Computes foot contact label accuracy."""
        pred_contact = repr_dict_rec['foot_contact'].numpy(force=True)
        gt_contact = repr_dict_clean['foot_contact'].numpy(force=True)
        
        pred_contact_binary = (pred_contact > 0.5).astype(float)
        return np.mean(pred_contact_binary == gt_contact)

    def _compute_foot_skating(self, pred_joints, gt_joints=None):
        """
        Computes the foot skating ratio. If gt_joints is provided, it's used for the
        floor plane; otherwise, the floor is derived from the prediction itself.
        """
        foot_joint_indices = self.ankle_indices + self.foot_indices
        
        if gt_joints is not None:
            # Use GT floor plane for a more stable comparison
            min_height = gt_joints[..., 2].min(axis=(-1, -2), keepdims=True)
        else:
            # Estimate floor plane from the prediction
            min_height = pred_joints[..., 2].min(axis=(-1, -2), keepdims=True)
        
        pred_joints_normalized = pred_joints - min_height[..., None]
        
        foot_vel_horizontal = np.linalg.norm(pred_joints_normalized[:, 1:, foot_joint_indices, :2] - pred_joints_normalized[:, :-1, foot_joint_indices, :2], axis=-1) * self.fps
        foot_heights = pred_joints_normalized[:, :-1, foot_joint_indices, 2]
        
        vel_thresh = 0.10
        height_thresh = 1e-2
        
        is_skating = (foot_vel_horizontal > vel_thresh) & (foot_heights < height_thresh)
        return np.mean(is_skating)

    def _compute_foot_skating_legacy(self, pred_joints, gt_joints):
        """
        RoHM-compatible foot skating ratio used by the original AMASS scripts.

        This is a prediction metric: the floor height is taken from GT for a
        stable reference plane, but the velocities/heights are computed on
        ``pred_joints``. The event definition follows third_party/RoHM
        evaluation exactly: a foot skates only when both its ankle and toe move
        horizontally faster than 0.10 m/s while remaining near the GT floor.
        """
        foot_joint_indices = [7, 10, 8, 11]
        min_height_gt = gt_joints[:, :, :, 2].min(axis=-1).min(axis=-1)
        joints_foot = pred_joints[:, :, foot_joint_indices, :]
        foot_vel = (
            np.linalg.norm(
                joints_foot[:, 1:, :, [0, 1]] - joints_foot[:, :-1, :, [0, 1]],
                axis=-1,
            )
            * self.fps
        )
        foot_height = joints_foot[:, :-1, :, 2]
        foot_height = foot_height - np.tile(
            min_height_gt.reshape(len(min_height_gt), 1, 1),
            (1, foot_height.shape[-2], len(foot_joint_indices)),
        )

        thresh_height = 0.10
        thresh_vel = 0.10
        skating_left = (
            (foot_vel[:, :, 0] > thresh_vel)
            & (foot_vel[:, :, 1] > thresh_vel)
            & (foot_height[:, :, 0] < (thresh_height + 0.05))
            & (foot_height[:, :, 1] < thresh_height)
        )
        skating_right = (
            (foot_vel[:, :, 2] > thresh_vel)
            & (foot_vel[:, :, 3] > thresh_vel)
            & (foot_height[:, :, 2] < (thresh_height + 0.05))
            & (foot_height[:, :, 3] < thresh_height)
        )
        return np.mean(skating_left & skating_right)

    def _compute_foot_movement_speed(self, pred_joints):
        """Computes average horizontal foot speed in global space (mm/s)."""
        foot_joint_indices = self.ankle_indices + self.foot_indices
        foot_vel_horizontal = np.linalg.norm(
            pred_joints[:, 1:, foot_joint_indices, :2]
            - pred_joints[:, :-1, foot_joint_indices, :2],
            axis=-1,
        ) * self.fps
        return np.mean(foot_vel_horizontal) * 1000


        # NOTE: strictly, velocity in m/s should be computed with xfps, but the SIP paper reports foot movement in mm, not foot movement speed
        foot_vel_horizontal_gt = np.linalg.norm(
            gt_joints[:, 1:, foot_joint_indices, :2]
            - gt_joints[:, :-1, foot_joint_indices, :2],
            axis=-1,
        ) 
        is_grounded = foot_vel_horizontal_gt < 0.001 # less than 1mm/s
        
        foot_vel_horizontal = np.linalg.norm(
            pred_joints[:, 1:, foot_joint_indices, :2]
            - pred_joints[:, :-1, foot_joint_indices, :2],
            axis=-1,
        ) 

        foot_vel_horizontal[~is_grounded] = 0

        return np.mean(foot_vel_horizontal) * 1000


    
    def _compute_ground_penetration(self, pred_joints, gt_joints):
        """
        The looser criterion used in the RoHM codebase.
        """
        min_height_gt = gt_joints[..., 2].min(axis=(-1, -2), keepdims=True)
        n_seq, clip_len, _, _ = pred_joints.shape
        pene_dist = pred_joints[:, :, [10, 11], -1] - np.tile(min_height_gt.reshape(n_seq, 1, 1), (1, clip_len, 2))  # [n_seq, clip_len, 2]
        pene_freq = pene_dist < -0.05  # [clip_len]
        pene_freq = pene_freq.mean()
        pene_dist[pene_dist >= 0] = 0
        pene_dist = pene_dist.mean()
        return pene_freq, pene_dist

    def _compute_gnd(self, pred_joints):
        """
        Computes Grounding (GND) metric: percentage of sequences where feet ever touch the ground.
        """
        min_height_pred = pred_joints[..., 2].min(axis=(-1, -2))
        
        # We assume ground is at y=0 after normalization. A small threshold for contact.
        ground_contact_threshold = 0.05 # 5cm
        
        # Check if the lowest point of the body in each sequence is close to the ground
        is_grounded = min_height_pred < ground_contact_threshold
        
        return np.mean(is_grounded)

    def _compute_head_error(self, pred_joints, gt_joints, repr_dict_rec, repr_dict_clean):
        """Computes head translation and orientation error."""
        # Position error
        head_pos_error = np.linalg.norm(pred_joints[..., self.head_idx, :] - gt_joints[..., self.head_idx, :], axis=-1)
        
        # Rotation error
        try:
            pred_rot_6d, gt_rot_6d = repr_dict_rec['R_world_joints'], repr_dict_clean['R_world_joints']
        except ValueError:
            # If R_world_joints is not in the representation, return NaN
            return np.mean(head_pos_error), np.nan

        pred_head_rot_6d = pred_rot_6d.reshape(pred_rot_6d.shape[:-1] + (22, 6))[..., self.head_idx, :]
        gt_head_rot_6d = gt_rot_6d.reshape(gt_rot_6d.shape[:-1] + (22, 6))[..., self.head_idx, :]
        
        pred_head_rot_mat = rot6d_to_rotmat(pred_head_rot_6d.reshape(-1, 6)).reshape(pred_head_rot_6d.shape[:-1] + (3, 3))
        gt_head_rot_mat = rot6d_to_rotmat(gt_head_rot_6d.reshape(-1, 6)).reshape(gt_head_rot_6d.shape[:-1] + (3, 3))

        head_rot_error = rotation_geodesic_distance(pred_head_rot_mat.numpy(force=True), gt_head_rot_mat.numpy(force=True))
        
        return np.mean(head_pos_error), np.mean(head_rot_error)

    def _compute_mpjve(self, pred_joints, gt_joints):
        """Computes mean per joint velocity error."""
        vel_pred = (pred_joints[:, 1:] - pred_joints[:, :-1]) * self.fps
        vel_gt = (gt_joints[:, 1:] - gt_joints[:, :-1]) * self.fps
        return np.linalg.norm(vel_pred - vel_gt, axis=-1)

    def _compute_jitter(self, joints):
        """Computes mean jerk magnitude (third derivative) as jitter."""
        jerk = (
            joints[:, 3:]
            - 3 * joints[:, 2:-1]
            + 3 * joints[:, 1:-2]
            - joints[:, :-3]
        ) * (self.fps ** 3)
        return np.linalg.norm(jerk, axis=-1)

    def compute_self_metrics(self, pred_joints):
        """
        Computes and returns metrics that only depend on the predicted motion.
        """
        metrics = {}
        # Foot Skating (relative to its own predicted floor)
        metrics['skating_ratio_rec'] = float(self._compute_foot_skating(pred_joints))
        # GND (Grounding)
        metrics['gnd_percent'] = float(self._compute_gnd(pred_joints) * 100)
        return metrics

    def compute_comparison_metrics(
        self,
        pred_joints,
        gt_joints,
        vis_joint_ids=None,
        occ_joint_ids=None,
        repr_dict_rec: Optional[Dict[str, torch.Tensor]] = None,
        repr_dict_clean: Optional[Dict[str, torch.Tensor]] = None,
        vis_joint_mask: Optional[np.ndarray] = None,
        occ_joint_mask: Optional[np.ndarray] = None,
    ):
        """
        Computes and returns metrics that require ground truth for comparison.
        """
        metrics = {}

        # MPJPE
        mpjpe_error = self._compute_mpjpe(pred_joints, gt_joints, aligned=False)
        metrics['mpjpe_global_mm'] = float(np.mean(mpjpe_error) * 1000)
        if vis_joint_mask is not None:
            vis_joint_mask = np.asarray(vis_joint_mask, dtype=bool)
            if np.any(vis_joint_mask):
                metrics['mpjpe_visible_mm'] = float(np.mean(mpjpe_error[vis_joint_mask]) * 1000)
        elif vis_joint_ids:
            metrics['mpjpe_visible_mm'] = float(np.mean(mpjpe_error[..., vis_joint_ids]) * 1000)
        if occ_joint_mask is not None:
            occ_joint_mask = np.asarray(occ_joint_mask, dtype=bool)
            if np.any(occ_joint_mask):
                metrics['mpjpe_occluded_mm'] = float(np.mean(mpjpe_error[occ_joint_mask]) * 1000)
        elif occ_joint_ids:
            metrics['mpjpe_occluded_mm'] = float(np.mean(mpjpe_error[..., occ_joint_ids]) * 1000)

        # PA-MPJPE
        pampjpe_error = self._compute_mpjpe(pred_joints, gt_joints, aligned=True)
        metrics['pa_mpjpe_global_mm'] = float(np.mean(pampjpe_error) * 1000)
        if vis_joint_mask is not None:
            if np.any(vis_joint_mask):
                metrics['pa_mpjpe_visible_mm'] = float(np.mean(pampjpe_error[vis_joint_mask]) * 1000)
        elif vis_joint_ids:
            metrics['pa_mpjpe_visible_mm'] = float(np.mean(pampjpe_error[..., vis_joint_ids]) * 1000)
        if occ_joint_mask is not None:
            if np.any(occ_joint_mask):
                metrics['pa_mpjpe_occluded_mm'] = float(np.mean(pampjpe_error[occ_joint_mask]) * 1000)
        elif occ_joint_ids:
            metrics['pa_mpjpe_occluded_mm'] = float(np.mean(pampjpe_error[..., occ_joint_ids]) * 1000)

        # Acceleration Error
        accel_error = self._compute_accel_error(pred_joints, gt_joints)
        metrics['acceleration_error_ms2'] = float(np.mean(accel_error))

        if repr_dict_rec is not  None and repr_dict_clean is not None:
            # Foot Contact Accuracy
            metrics['contact_label_accuracy'] = float(self._compute_foot_contact_accuracy(repr_dict_rec, repr_dict_clean))

        # Foot skating variants:
        # - skating_ratio_rec_gt_floor: prediction skating on the current
        #   evaluator's GT-floor variant.
        # - skating_ratio_gt_floor: GT motion skating only; use as a data
        #   sanity/reference number, not as a method score.
        # - skating_ratio_rec_legacy: prediction skating under the original
        #   RoHM AMASS-table definition, which also uses the GT floor.
        metrics['skating_ratio_rec_gt_floor'] = float(self._compute_foot_skating(pred_joints, gt_joints))
        metrics['skating_ratio_gt_floor'] = float(self._compute_foot_skating(gt_joints))
        metrics['skating_ratio_rec_legacy'] = float(self._compute_foot_skating_legacy(pred_joints, gt_joints))
        metrics['skating_ratio_gt_legacy'] = float(self._compute_foot_skating_legacy(gt_joints, gt_joints))
        
        # Ground Penetration
        gp_freq, gp_dist = self._compute_ground_penetration(pred_joints, gt_joints)
        metrics['ground_penetration_freq_percent'] = float(gp_freq * 100)
        metrics['ground_penetration_dist_mm'] = float(gp_dist * 1000)

        # Head Error
        if repr_dict_rec is not  None and repr_dict_clean is not None:
            head_pos_err, head_rot_err = self._compute_head_error(pred_joints, gt_joints, repr_dict_rec, repr_dict_clean)
            metrics['head_pos_error_mm'] = float(head_pos_err * 1000)
            metrics['head_rot_error_deg'] = float(head_rot_err)

        return metrics


# Registry-backed metric functions
@register_metric(
    "skating_ratio_rec",
    baselines=["ours", _BASELINE_ROHM, _BASELINE_HEAD2BODY],
    requires_gt=False,
    requires_repr=False,
)
def metric_skating_ratio_rec(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_skating(pred_joints)


@register_metric(
    "gnd_percent",
    baselines=["ours", _BASELINE_HEAD2BODY],
    requires_gt=False,
    requires_repr=False,
)
def metric_gnd_percent(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_gnd(pred_joints) * 100


@register_metric(
    "mpjpe_global_mm",
    baselines=[
        "ours",
        _BASELINE_ROHM,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjpe_global_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    mpjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=False)
    return np.mean(mpjpe_error) * 1000


@register_metric(
    "mpjpe_visible_mm",
    baselines=["ours", _BASELINE_ROHM],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjpe_visible_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    vis_joint_ids: Iterable[int],
    vis_joint_mask: Optional[np.ndarray] = None,
    **kwargs,
):
    mpjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=False)
    if vis_joint_mask is not None:
        mask = np.asarray(vis_joint_mask, dtype=bool)
        return np.mean(mpjpe_error[mask]) * 1000 if np.any(mask) else None
    if not vis_joint_ids:
        return None
    return np.mean(mpjpe_error[..., list(vis_joint_ids)]) * 1000


@register_metric(
    "mpjpe_occluded_mm",
    baselines=["ours", _BASELINE_ROHM],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjpe_occluded_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    occ_joint_ids: Iterable[int],
    occ_joint_mask: Optional[np.ndarray] = None,
    **kwargs,
):
    mpjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=False)
    if occ_joint_mask is not None:
        mask = np.asarray(occ_joint_mask, dtype=bool)
        return np.mean(mpjpe_error[mask]) * 1000 if np.any(mask) else None
    if not occ_joint_ids:
        return None
    return np.mean(mpjpe_error[..., list(occ_joint_ids)]) * 1000


@register_metric(
    "pa_mpjpe_global_mm",
    baselines=["ours", _BASELINE_EGOALLO, _BASELINE_TRANSPOSE],
    requires_gt=True,
    requires_repr=False,
)
def metric_pa_mpjpe_global_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    pampjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=True)
    return np.mean(pampjpe_error) * 1000


@register_metric(
    "pa_mpjpe_visible_mm",
    baselines=["ours"],
    requires_gt=True,
    requires_repr=False,
)
def metric_pa_mpjpe_visible_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    vis_joint_ids: Iterable[int],
    vis_joint_mask: Optional[np.ndarray] = None,
    **kwargs,
):
    pampjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=True)
    if vis_joint_mask is not None:
        mask = np.asarray(vis_joint_mask, dtype=bool)
        return np.mean(pampjpe_error[mask]) * 1000 if np.any(mask) else None
    if not vis_joint_ids:
        return None
    return np.mean(pampjpe_error[..., list(vis_joint_ids)]) * 1000


@register_metric(
    "pa_mpjpe_occluded_mm",
    baselines=["ours"],
    requires_gt=True,
    requires_repr=False,
)
def metric_pa_mpjpe_occluded_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    occ_joint_ids: Iterable[int],
    occ_joint_mask: Optional[np.ndarray] = None,
    **kwargs,
):
    pampjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=True)
    if occ_joint_mask is not None:
        mask = np.asarray(occ_joint_mask, dtype=bool)
        return np.mean(pampjpe_error[mask]) * 1000 if np.any(mask) else None
    if not occ_joint_ids:
        return None
    return np.mean(pampjpe_error[..., list(occ_joint_ids)]) * 1000


@register_metric(
    "acceleration_error_ms2",
    baselines=["ours", _BASELINE_ROHM, _BASELINE_HEAD2BODY],
    requires_gt=True,
    requires_repr=False,
)
def metric_acceleration_error_ms2(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    accel_error = evaluator._compute_accel_error(pred_joints, gt_joints)
    return np.mean(accel_error)


@register_metric(
    "contact_label_accuracy",
    baselines=["ours", _BASELINE_ROHM],
    requires_gt=True,
    requires_repr=True,
)
def metric_contact_label_accuracy(
    evaluator: MotionEvaluator,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    return evaluator._compute_foot_contact_accuracy(repr_dict_rec, repr_dict_clean)


@register_metric(
    "skating_ratio_rec_gt_floor",
    baselines=["ours"],
    requires_gt=True,
    requires_repr=False,
)
def metric_skating_ratio_rec_gt_floor(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_skating(pred_joints, gt_joints)


@register_metric(
    "skating_ratio_gt_floor",
    baselines=["ours"],
    requires_gt=True,
    requires_repr=False,
)
def metric_skating_ratio_gt_floor(
    evaluator: MotionEvaluator,
    gt_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_skating(gt_joints)


@register_metric(
    "skating_ratio_rec_legacy",
    baselines=["ours", _BASELINE_ROHM, _BASELINE_HEAD2BODY],
    requires_gt=True,
    requires_repr=False,
)
def metric_skating_ratio_rec_legacy(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_skating_legacy(pred_joints, gt_joints)


@register_metric(
    "skating_ratio_gt_legacy",
    baselines=["ours"],
    requires_gt=True,
    requires_repr=False,
)
def metric_skating_ratio_gt_legacy(
    evaluator: MotionEvaluator,
    gt_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_skating_legacy(gt_joints, gt_joints)


@register_metric(
    "foot_movement_speed_mm_s",
    baselines=[
        _BASELINE_TRANSPOSE,
        _BASELINE_PIP,
        _BASELINE_TIP,
    ],
    requires_gt=False,
    requires_repr=False,
)
def metric_foot_movement_speed_mm_s(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    **kwargs,
):
    return evaluator._compute_foot_movement_speed(pred_joints)


@register_metric(
    "ground_penetration_freq_percent",
    baselines=["ours", _BASELINE_ROHM],
    requires_gt=True,
    requires_repr=False,
)
def metric_ground_penetration_freq_percent(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    gp_freq, _ = evaluator._compute_ground_penetration(pred_joints, gt_joints)
    return gp_freq * 100


@register_metric(
    "ground_penetration_dist_mm",
    baselines=["ours", _BASELINE_ROHM],
    requires_gt=True,
    requires_repr=False,
)
def metric_ground_penetration_dist_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    _, gp_dist = evaluator._compute_ground_penetration(pred_joints, gt_joints)
    return gp_dist * 1000


@register_metric(
    "head_pos_error_mm",
    baselines=["ours", _BASELINE_HEAD2BODY],
    requires_gt=True,
    requires_repr=True,
)
def metric_head_pos_error_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    head_pos_err, _ = evaluator._compute_head_error(
        pred_joints, gt_joints, repr_dict_rec, repr_dict_clean
    )
    return head_pos_err * 1000


@register_metric(
    "head_rot_error_deg",
    baselines=["ours", _BASELINE_HEAD2BODY],
    requires_gt=True,
    requires_repr=True,
)
def metric_head_rot_error_deg(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    _, head_rot_err = evaluator._compute_head_error(
        pred_joints, gt_joints, repr_dict_rec, repr_dict_clean
    )
    return head_rot_err


@register_metric(
    "mpjpe_cm",
    baselines=[
        _BASELINE_EGOEGO,
        _BASELINE_MOCAP_EVERYONE,
        _BASELINE_HMD_POSER,
        _BASELINE_EGOPOSER,
        _BASELINE_EGOCAST,
        _BASELINE_EGOALLO,
        _BASELINE_HEAD2BODY,
        _BASELINE_ROHM,
        _BASELINE_AVATARPOSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjpe_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    mpjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=False)
    return np.mean(mpjpe_error) * 100


@register_metric(
    "pa_mpjpe_cm",
    baselines=[
        _BASELINE_EGOALLO,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_pa_mpjpe_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    pampjpe_error = evaluator._compute_mpjpe(pred_joints, gt_joints, aligned=True)
    return np.mean(pampjpe_error) * 100


@register_metric(
    "mpjve_ms",
    baselines=[],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjve_ms(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    mpjve_error = evaluator._compute_mpjve(pred_joints, gt_joints)
    return np.mean(mpjve_error)


@register_metric(
    "mpjve_cm_s",
    baselines=[
        _BASELINE_EGOPOSER,
        _BASELINE_EGOALLO,
        _BASELINE_HEAD2BODY,
        _BASELINE_HMD_POSER,
        _BASELINE_AVATARPOSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjve_cm_s(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    mpjve_error = evaluator._compute_mpjve(pred_joints, gt_joints)
    return np.mean(mpjve_error) * 100


@register_metric(
    "mpjve_mm_s",
    baselines=[],
    requires_gt=True,
    requires_repr=False,
)
def metric_mpjve_mm_s(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    mpjve_error = evaluator._compute_mpjve(pred_joints, gt_joints)
    return np.mean(mpjve_error) * 1000


@register_metric(
    "jitter",
    baselines=[
        _BASELINE_HMD_POSER,
        _BASELINE_AGROL,
        _BASELINE_PNP,
        _BASELINE_TIP,
        _BASELINE_PIP,
        _BASELINE_TRANSPOSE,
        _BASELINE_LIP,
    ],
    requires_gt=False,
    requires_repr=False,
)
def metric_jitter(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    **kwargs,
):
    jitter = evaluator._compute_jitter(pred_joints)
    return np.mean(jitter)


@register_metric(
    "mpjre_deg",
    baselines=[
        _BASELINE_AVATARPOSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=True,
)
def metric_mpjre_deg(
    evaluator: MotionEvaluator,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    if "R_world_joints" not in repr_dict_rec or "R_world_joints" not in repr_dict_clean:
        return None
    pred_rot_6d = repr_dict_rec["R_world_joints"].reshape(-1, 22, 6)
    gt_rot_6d = repr_dict_clean["R_world_joints"].reshape(-1, 22, 6)
    pred_rot_mat = rot6d_to_rotmat(pred_rot_6d.reshape(-1, 6)).reshape(-1, 22, 3, 3)
    gt_rot_mat = rot6d_to_rotmat(gt_rot_6d.reshape(-1, 6)).reshape(-1, 22, 3, 3)
    rot_err = rotation_geodesic_distance(
        pred_rot_mat.cpu().numpy(),
        gt_rot_mat.cpu().numpy(),
    )
    return np.mean(rot_err)


@register_metric(
    "root_position_error_cm",
    baselines=[
        _BASELINE_MOCAP_EVERYONE,
        _BASELINE_HMD_POSER,
        _BASELINE_AGROL,
        _BASELINE_TRANSPOSE,
        _BASELINE_PIP,
        _BASELINE_TIP,
        _BASELINE_PNP,
        _BASELINE_UMOTION,
        _BASELINE_MAGSHIELD,
        _BASELINE_GROUPIP,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_root_position_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    root_err = np.linalg.norm(pred_joints[..., 0, :] - gt_joints[..., 0, :], axis=-1)
    return np.mean(root_err) * 100


@register_metric(
    "root_position_error_mm",
    baselines=[],
    requires_gt=True,
    requires_repr=False,
)
def metric_root_position_error_mm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    root_err = np.linalg.norm(pred_joints[..., 0, :] - gt_joints[..., 0, :], axis=-1)
    return np.mean(root_err) * 1000


@register_metric(
    "angle_error_deg",
    baselines=[
        _BASELINE_PIP,
        _BASELINE_TIP,
        _BASELINE_PNP,
        _BASELINE_TRANSPOSE,
        _BASELINE_UMOTION,
        _BASELINE_MAGSHIELD,
        _BASELINE_GROUPIP,
    ],
    requires_gt=True,
    requires_repr=True,
)
def metric_angle_error_deg(
    evaluator: MotionEvaluator,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    if "R_world_joints" not in repr_dict_rec or "R_world_joints" not in repr_dict_clean:
        return None
    pred_rot_6d = repr_dict_rec["R_world_joints"].reshape(-1, 22, 6)
    gt_rot_6d = repr_dict_clean["R_world_joints"].reshape(-1, 22, 6)
    pred_rot_mat = rot6d_to_rotmat(pred_rot_6d.reshape(-1, 6)).reshape(-1, 22, 3, 3)
    gt_rot_mat = rot6d_to_rotmat(gt_rot_6d.reshape(-1, 6)).reshape(-1, 22, 3, 3)
    rot_err = rotation_geodesic_distance(
        pred_rot_mat.cpu().numpy(),
        gt_rot_mat.cpu().numpy(),
    )
    return np.mean(rot_err)


@register_metric(
    "sip_error_deg",
    baselines=[
        _BASELINE_PIP,
        _BASELINE_TIP,
        _BASELINE_PNP,
        _BASELINE_TRANSPOSE,
        _BASELINE_UMOTION,
        _BASELINE_MAGSHIELD,
        _BASELINE_GROUPIP,
    ],
    requires_gt=True,
    requires_repr=True,
)
def metric_sip_error_deg(
    evaluator: MotionEvaluator,
    repr_dict_rec: Dict[str, torch.Tensor],
    repr_dict_clean: Dict[str, torch.Tensor],
    **kwargs,
):
    if "R_world_joints" not in repr_dict_rec or "R_world_joints" not in repr_dict_clean:
        return None

    sip_joint_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    sip_indices = [evaluator.joint_name_to_idx[name] for name in sip_joint_names]

    pred_rot_6d = repr_dict_rec["R_world_joints"].reshape(-1, 22, 6)[:, sip_indices, :]
    gt_rot_6d = repr_dict_clean["R_world_joints"].reshape(-1, 22, 6)[:, sip_indices, :]

    pred_rot_mat = rot6d_to_rotmat(pred_rot_6d.reshape(-1, 6)).reshape(-1, 4, 3, 3)
    gt_rot_mat = rot6d_to_rotmat(gt_rot_6d.reshape(-1, 6)).reshape(-1, 4, 3, 3)

    rot_err = rotation_geodesic_distance(
        pred_rot_mat.cpu().numpy(),
        gt_rot_mat.cpu().numpy(),
    )
    return np.mean(rot_err)


@register_metric(
    "positional_error_cm",
    baselines=[
        _BASELINE_PIP,
        _BASELINE_TIP,
        _BASELINE_PNP,
        _BASELINE_TRANSPOSE,
        _BASELINE_UMOTION,
        _BASELINE_MAGSHIELD,
        _BASELINE_GROUPIP,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_positional_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    pred_aligned = evaluator._root_align(pred_joints)
    gt_aligned = evaluator._root_align(gt_joints)
    mpjpe_error = evaluator._compute_mpjpe(pred_aligned, gt_aligned, aligned=False)
    return np.mean(mpjpe_error) * 100


@register_metric(
    "mesh_error_cm",
    baselines=[
        _BASELINE_PIP,
        _BASELINE_TIP,
        _BASELINE_PNP,
        _BASELINE_TRANSPOSE,
        _BASELINE_UMOTION,
        _BASELINE_MAGSHIELD,
        _BASELINE_GROUPIP,
    ],
    requires_gt=True,
    requires_repr=False,
    requires_verts=True,
)
def metric_mesh_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    pred_verts: np.ndarray,
    gt_verts: np.ndarray,
    **kwargs,
):
    pred_aligned = evaluator._root_align(pred_verts)
    gt_aligned = evaluator._root_align(gt_verts)
    per_vert_error = np.linalg.norm(pred_aligned - gt_aligned, axis=-1)
    return np.mean(per_vert_error) * 100


@register_metric(
    "hand_position_error_cm",
    baselines=[
        _BASELINE_HMD_POSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_hand_position_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    hand_indices = [
        evaluator.joint_name_to_idx["left_wrist"],
        evaluator.joint_name_to_idx["right_wrist"],
    ]
    error = np.linalg.norm(
        pred_joints[..., hand_indices, :] - gt_joints[..., hand_indices, :],
        axis=-1,
    )
    return np.mean(error) * 100


@register_metric(
    "upper_body_position_error_cm",
    baselines=[
        _BASELINE_HMD_POSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_upper_body_position_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    upper_names = [
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "head",
        "left_collar",
        "right_collar",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    upper_indices = [evaluator.joint_name_to_idx[name] for name in upper_names]
    error = np.linalg.norm(
        pred_joints[..., upper_indices, :] - gt_joints[..., upper_indices, :],
        axis=-1,
    )
    return np.mean(error) * 100


@register_metric(
    "lower_body_position_error_cm",
    baselines=[
        _BASELINE_HMD_POSER,
        _BASELINE_AGROL,
    ],
    requires_gt=True,
    requires_repr=False,
)
def metric_lower_body_position_error_cm(
    evaluator: MotionEvaluator,
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    **kwargs,
):
    lower_names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ]
    lower_indices = [evaluator.joint_name_to_idx[name] for name in lower_names]
    error = np.linalg.norm(
        pred_joints[..., lower_indices, :] - gt_joints[..., lower_indices, :],
        axis=-1,
    )
    return np.mean(error) * 100
