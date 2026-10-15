from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from data_sets.motion_representation import populate_repr_dict, recover_from_repr_smpl


def split_repr_tensor(
    full_repr: torch.Tensor,
    repr_list: Sequence[str],
    repr_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    return {name: full_repr[..., repr_indices[name][0] : repr_indices[name][1]] for name in repr_list}


def maybe_unnormalize_full_repr(
    full_repr: torch.Tensor,
    ds,
    sample_norms: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    return ds._unnormalize_full_tensor(full_repr, ds.device, sample_norms=sample_norms)


def populate_smplx_repr_dict(
    repr_dict: Dict[str, torch.Tensor],
    smplx_model,
    output_global_rotations: bool,
) -> Dict[str, torch.Tensor]:
    recover_global = not output_global_rotations
    return populate_repr_dict(repr_dict, smplx_model, recover_global=recover_global)


def decode_smplx_from_full_repr(
    *,
    full_repr: torch.Tensor,
    repr_list: Sequence[str],
    repr_indices: Dict[str, Tuple[int, int]],
    smplx_model,
    output_global_rotations: bool,
    cano_jts_pos: bool,
    return_verts: bool = False,
    return_full_joints: bool = False,
    override_repr_dict: Optional[Dict[str, torch.Tensor]] = None,
    post_override_repr_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    model_device = next(smplx_model.parameters()).device
    full_repr = full_repr.to(device=model_device, dtype=torch.float32)
    repr_dict = split_repr_tensor(full_repr, repr_list, repr_indices)
    if override_repr_dict:
        repr_dict.update(override_repr_dict)
    repr_dict = populate_smplx_repr_dict(
        repr_dict=repr_dict,
        smplx_model=smplx_model,
        output_global_rotations=output_global_rotations,
    )
    if post_override_repr_dict:
        repr_dict.update(post_override_repr_dict)
    smplx_out = recover_from_repr_smpl(
        repr_dict,
        recover_mode="smplx_params",
        cano_jts_pos=cano_jts_pos,
        smplx_model=smplx_model,
        return_verts=return_verts,
        return_full_joints=return_full_joints,
    )
    if return_verts:
        joints, verts = smplx_out
    else:
        joints = smplx_out
        verts = None
    return repr_dict, joints, verts
