"""
Utilities for undoing dataset-specific preprocessing so that SMPL-X outputs can be
recovered in world coordinates using the same conventions as the forward kinematics
model. The inverse pipeline mirrors the canonicalization steps in
`data_loaders/dataloader_mocap.py` (homogenous alignment -> floor/XY offset ->
optional y-axis alignment) but applies them on the SMPL-X representation rather than
on model vertices.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from data_sets.motion_representation import (
    recover_from_repr_smpl,
    rotate_repr_smpl_2,
    transform_repr_smpl,
)


def _clone_to_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert value to tensor on the target device/dtype without modifying the source."""
    if torch.is_tensor(value):
        tensor = value.clone().to(device=device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _expand_to_batch(tensor: torch.Tensor, batch_dims: Tuple[int, ...]) -> torch.Tensor:
    """Broadcast tensor to batch_dims."""
    if tensor.ndim == 0:
        return tensor.view(*([1] * len(batch_dims))).expand(*batch_dims)

    # Remove leading singleton dims if they exceed batch rank
    while tensor.ndim > len(batch_dims) and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.shape == batch_dims:
        return tensor

    view_shape = [1] * (len(batch_dims) - tensor.ndim) + list(tensor.shape)
    try:
        return tensor.view(*view_shape).expand(*batch_dims)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot broadcast tensor of shape {tensor.shape} to batch dims {batch_dims}"
        ) from exc


def undo_mocap_preprocessing_on_repr(
    repr_dict: Dict[str, Any],
    meta: Dict[str, Any],
    device: Optional[torch.device] = None,
    smplx_model: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """
    Undo the MoCap canonicalization applied in `DatasetMocap._build_item` on a SMPL-X
    representation dictionary by applying the inverse transforms with
    `rotate_repr_smpl_2` and `transform_repr_smpl`.

    Args:
        repr_dict: Canonicalized SMPL-X representation (tensors or numpy arrays).
        meta: Batch metadata containing preprocessing transforms.
        device: Optional device override. If None, inferred from `repr_dict` tensors.
        smplx_model: Optional SMPL-X model used to reconstruct `local_positions` when
            they are missing or contain non-finite placeholders.

    Returns:
        A new representation dictionary with world-space SMPL-X parameters.
    """
    base_required_keys = ['smplx_trans', 'smplx_rot_6d', 'smplx_body_pose_6d', 'smplx_betas']
    missing = [k for k in base_required_keys if k not in repr_dict]
    if missing:
        raise KeyError(f"Representation missing required keys: {missing}")

    def _reconstruct_local_positions() -> torch.Tensor:
        smpl_keys = ('smplx_rot_6d', 'smplx_body_pose_6d', 'smplx_trans', 'smplx_betas')
        missing_smpl = [k for k in smpl_keys if k not in repr_dict]
        if missing_smpl:
            raise ValueError(f"Cannot reconstruct local_positions; missing SMPL-X keys: {missing_smpl}")
        if smplx_model is None:
            raise ValueError("local_positions are missing or non-finite and no smplx_model was provided.")

        ref_tensor = next((repr_dict[k] for k in smpl_keys if torch.is_tensor(repr_dict[k])), None)
        if device is not None:
            _device = device
        elif ref_tensor is not None:
            _device = ref_tensor.device
        elif hasattr(smplx_model, "device"):
            _device = smplx_model.device
        elif hasattr(smplx_model, "parameters"):
            try:
                _device = next(smplx_model.parameters()).device
            except StopIteration:
                _device = torch.device("cpu")
        else:
            _device = torch.device("cpu")

        _dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32

        smplx_input = {
            k: _clone_to_tensor(repr_dict[k], device=_device, dtype=_dtype) for k in smpl_keys
        }
        with torch.no_grad():
            joints = recover_from_repr_smpl(
                smplx_input,
                recover_mode='smplx_params',
                smplx_model=smplx_model,
                return_verts=False,
                return_full_joints=False,
            )
        if isinstance(joints, tuple):
            joints = joints[0]
        return joints.reshape(*joints.shape[:-2], -1).to(device=_device, dtype=_dtype)

    local_positions = repr_dict.get('local_positions')
    has_valid_local_positions = (
        local_positions is not None
        and (
            (torch.is_tensor(local_positions) and torch.isfinite(local_positions).all())
            or (isinstance(local_positions, np.ndarray) and np.isfinite(local_positions).all())
        )
    )
    if not has_valid_local_positions:
        repr_dict = dict(repr_dict)  # shallow copy to avoid mutating caller
        repr_dict['local_positions'] = _reconstruct_local_positions()

    required_keys = base_required_keys + ['local_positions']

    # Infer device/dtype
    if device is None:
        for key in ('smplx_trans', 'local_positions', 'smplx_rot_6d', 'smplx_body_pose_6d', 'smplx_betas'):
            if torch.is_tensor(repr_dict.get(key)):
                device = repr_dict[key].device
                break
    if device is None:
        device = torch.device('cpu')

    dtype = None
    for key in ('smplx_trans', 'local_positions', 'smplx_rot_6d', 'smplx_body_pose_6d', 'smplx_betas'):
        if torch.is_tensor(repr_dict.get(key)):
            dtype = repr_dict[key].dtype
            break
    if dtype is None:
        dtype = torch.float32

    repr_world: Dict[str, torch.Tensor] = {}
    for key, value in repr_dict.items():
        if key in required_keys:
            repr_world[key] = _clone_to_tensor(value, device=device, dtype=dtype)
        elif torch.is_tensor(value):
            repr_world[key] = value.clone().to(device=device)
        elif isinstance(value, np.ndarray):
            repr_world[key] = torch.as_tensor(value, device=device)
        else:
            # Preserve non-tensor extras (e.g., metadata cached in repr_dict)
            repr_world[key] = value

    batch_dims = repr_world['smplx_trans'].shape[:-1]
    if len(batch_dims) == 0:
        raise ValueError("Expected `smplx_trans` to include at least one batch dimension.")
    target_frames = batch_dims[-1]
    batch_size = batch_dims[0] if len(batch_dims) > 1 else 1

    # Inverse transforms from metadata
    align_mat = meta.get('homogenous_transform_align', np.eye(4))
    align_rot = _clone_to_tensor(align_mat, device=device, dtype=dtype)
    if align_rot.shape[-2:] == (4, 4):
        align_rot = align_rot[..., :3, :3]
    if align_rot.shape[-2:] != (3, 3):
        raise ValueError(f"homogenous_transform_align has invalid shape {align_rot.shape}")
    inv_align_rot = align_rot.transpose(-2, -1)

    yaxis_mat = meta.get('yaxis_align_rot', np.eye(3))
    yaxis_rot = _clone_to_tensor(yaxis_mat, device=device, dtype=dtype)
    if yaxis_rot.shape[-2:] == (4, 4):
        yaxis_rot = yaxis_rot[..., :3, :3]
    if yaxis_rot.shape[-2:] != (3, 3):
        raise ValueError(f"yaxis_align_rot has invalid shape {yaxis_rot.shape}")
    inv_yaxis_rot = yaxis_rot.transpose(-2, -1)

    xy_offset_raw = meta.get('translation_xy_offset', np.zeros(2))
    xy_offset = _clone_to_tensor(xy_offset_raw, device=device, dtype=dtype)
    if xy_offset.shape[-1] != 2:
        raise ValueError(f"translation_xy_offset must have last dim 2, got {xy_offset.shape}")

    floor_offset_raw = meta.get('floor_height_offset', 0.0)
    floor_offset = _clone_to_tensor(floor_offset_raw, device=device, dtype=dtype)
    if floor_offset.ndim > 0:
        if floor_offset.shape[-1] > target_frames:
            floor_offset = floor_offset[..., :target_frames]
        elif len(batch_dims) > 1 and floor_offset.ndim == 1 and floor_offset.shape[-1] == batch_size:
            floor_offset = floor_offset.unsqueeze(-1)
        if floor_offset.shape[-1] not in (1, target_frames):
            raise ValueError(
                f"floor_height_offset length {floor_offset.shape[-1]} is incompatible with sequence length {target_frames}"
            )
    floor_offset = _expand_to_batch(floor_offset, batch_dims)

    transform_inv = torch.eye(4, device=device, dtype=dtype)
    transform_inv = transform_inv.view(*(1,) * len(batch_dims), 4, 4).expand(*batch_dims, 4, 4).clone()
    transform_inv[..., 0, 3] = xy_offset[..., 0]
    transform_inv[..., 1, 3] = xy_offset[..., 1]
    transform_inv[..., 2, 3] = floor_offset

    # Undo preprocessing in reverse order: y-axis align -> floor/XY offset -> homogenous align
    repr_world = rotate_repr_smpl_2(repr_world, rotation_matrix=inv_yaxis_rot)
    repr_world = transform_repr_smpl(repr_world, transform=transform_inv)
    repr_world = rotate_repr_smpl_2(repr_world, rotation_matrix=inv_align_rot)

    return repr_world


def recover_world_from_repr(
    repr_dict: Dict[str, Any],
    meta: Dict[str, Any],
    smplx_model: Any,
    return_full_joints: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover SMPL-X joints and vertices in world coordinates by undoing dataset
    preprocessing on the representation and forwarding through the provided FK model.
    """
    world_repr = undo_mocap_preprocessing_on_repr(repr_dict, meta, smplx_model=smplx_model)
    joints, verts = recover_from_repr_smpl(
        world_repr,
        recover_mode='smplx_params',
        smplx_model=smplx_model,
        return_verts=True,
        return_full_joints=return_full_joints,
    )
    return joints, verts
