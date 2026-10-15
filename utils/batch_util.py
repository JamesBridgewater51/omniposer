import numpy as np
import torch
import smplx
from tqdm import tqdm
from utils.other_utils import TRANS_ROT_MASK_RATIO
from utils.representation_registry import RepresentationRegistry, ModalitySpec
from utils.feature_accessor import select_modalities, insert_modalities, get_modality_dim

from typing import Optional, List


def _interleave_joint_modalities(modalities, data_tensor, repr_indices, num_joints, device, dtype):
    """
    Build an interleaved conditioning tensor for joint-based modalities.
    modalities: ordered list of ModalitySpec (all joint-based).
    data_tensor: [bs, T, feat_dim] containing all reprs.
    
    Output layout: for each joint j, concatenate all modality features.
    Result shape: [bs, T, num_joints * sum(m.per_joint_dim for m in modalities)]
    """
    import einops
    
    bs, T, _ = data_tensor.shape
    per_joint_dims = [m.per_joint_dim or 0 for m in modalities]
    per_joint_total = sum(per_joint_dims)
    
    # Vectorized approach: stack all modalities then rearrange
    # Step 1: Extract and reshape each modality to [bs, T, num_joints, per_joint_dim]
    modality_tensors = []
    for m in modalities:
        s, e = repr_indices[m.name]
        # Shape: [bs, T, num_joints * per_joint_dim] -> [bs, T, num_joints, per_joint_dim]
        mod_data = data_tensor[:, :, s:e].reshape(bs, T, num_joints, m.per_joint_dim)
        modality_tensors.append(mod_data)
    
    # Step 2: Concatenate along the per_joint_dim axis -> [bs, T, num_joints, per_joint_total]
    stacked = torch.cat(modality_tensors, dim=-1)  # [bs, T, num_joints, per_joint_total]
    
    # Step 3: Rearrange to interleaved layout using einops
    # From [bs, T, num_joints, per_joint_total] to [bs, T, num_joints * per_joint_total]
    cond = einops.rearrange(stacked, 'b t j d -> b t (j d)')
    
    return cond


def _build_cond_from_registry(ds, data_tensor, repr_registry: RepresentationRegistry, task: str, num_joints: int, device, dtype):
    """
    Registry-driven conditioning construction. Mirrors legacy behavior but derives dims/topology from registry.
    """
    if task not in repr_registry.cond_modalities:
        raise ValueError(f"cond_modalities for task '{task}' not specified in registry.")
    cond_names = repr_registry.cond_modalities[task]
    joint_modalities: List[ModalitySpec] = []
    global_modalities: List[ModalitySpec] = []
    for name in cond_names:
        mod = repr_registry.get_modality(name)
        (joint_modalities if mod.is_joint_based else global_modalities).append(mod)

    cond_parts = []
    if joint_modalities:
        joint_cond = _interleave_joint_modalities(joint_modalities, data_tensor, ds._repr_indices, num_joints, device, dtype)
        cond_parts.append(joint_cond)
    for m in global_modalities:
        s, e = ds._repr_indices[m.name]
        cond_parts.append(data_tensor[:, :, s:e])

    if not cond_parts:
        raise ValueError(f"No conditioning modalities provided for task '{task}'.")
    return torch.cat(cond_parts, dim=-1)


def process_batch_cond(batch, ds, mask_scheme, task, input_noise=False, num_joints=22):

    item_dict = batch
    device, dtype = item_dict['motion_repr_clean'].device, torch.float32

    store_sample_norms = True if ds.norm_strategy == 'l2' else False
    if store_sample_norms:
        item_dict['motion_repr_clean'], item_dict['extras/sample_norms'] = ds._normalize_full_tensor(item_dict['motion_repr_clean'], device=device, dtype=dtype, store_sample_norms=store_sample_norms)
        # NOTE: have to flatten the dict to avoid issues when we only copy first-level tensor.
        for k, v in item_dict['extras/sample_norms'].items():
            item_dict['extras/sample_norms/' + k] = v
        if input_noise:
            item_dict['motion_repr_noisy'], *_ = ds._normalize_full_tensor(item_dict['motion_repr_noisy'], device=device, dtype=dtype, store_sample_norms=store_sample_norms)
            for k, v in item_dict['extras/sample_norms'].items():
                item_dict['extras/sample_norms/' + k] = v
    else:
        item_dict['motion_repr_clean'] = ds._normalize_full_tensor(item_dict['motion_repr_clean'], device=device, dtype=dtype, store_sample_norms=store_sample_norms)
        if input_noise and 'motion_repr_noisy' in item_dict:
            item_dict['motion_repr_noisy'] = ds._normalize_full_tensor(item_dict['motion_repr_noisy'], device=device, dtype=dtype, store_sample_norms=store_sample_norms)

    bs, T, _ = item_dict['motion_repr_clean'].shape

    # Use noisy repr if available and input_noise is True, otherwise use clean
    key = 'motion_repr_noisy' if (input_noise and 'motion_repr_noisy' in item_dict) else 'motion_repr_clean'

    registry = ds.repr_registry
    if registry is None:
        raise ValueError("repr_registry must be set on dataset for conditioning.")
    nj = registry.joint_count_for_topology(next(iter(registry.topologies.keys()))) or num_joints
    item_dict['cond'] = _build_cond_from_registry(
        ds=ds,
        data_tensor=item_dict[key],
        repr_registry=registry,
        task=task,
        num_joints=nj,
        device=device,
        dtype=dtype,
    )

    if task == 'traj':
        control_cond_parts = []
        if 'smplx_body_pose_6d' in ds._repr_indices:
                s, e = ds._repr_indices['smplx_body_pose_6d']
                control_cond_parts.append(item_dict['motion_repr_clean'][:, :, s:e])
        if 'smplx_betas' in ds._repr_indices:
                s, e = ds._repr_indices['smplx_betas']
                control_cond_parts.append(item_dict['motion_repr_clean'][:, :, s:e])
        if 'foot_contact' in ds._repr_indices:
                s, e = ds._repr_indices['foot_contact']
                control_cond_parts.append(item_dict['motion_repr_clean'][:, :, s:e])
        if control_cond_parts:
            item_dict['control_cond'] = torch.cat(control_cond_parts, dim=-1)
        else: # Create an empty array of correct shape if no control signals
            item_dict['control_cond'] = torch.empty((bs, T, 0), device=device, dtype=dtype)

    if task == 'both':
        traj_item_dict, pose_item_dict = {}, {}
        for k, v in item_dict.items():
            if isinstance(v, torch.Tensor):
                traj_item_dict[k] = v.clone()
                pose_item_dict[k] = v.clone()
        del item_dict
        if input_noise and 'motion_repr_noisy' in pose_item_dict:
            # Use name-based selection to replace trajectory features
            traj_features_clean = select_modalities(pose_item_dict['motion_repr_clean'], ds.traj_repr_name_list, ds._repr_indices)
            pose_item_dict['motion_repr_noisy'] = insert_modalities(
                pose_item_dict['motion_repr_noisy'],
                traj_features_clean,
                ds.traj_repr_name_list,
                ds._repr_indices
            )
        # Build pose cond using registry (traj modalities now included in cond_modalities['pose'])
        nj = registry.joint_count_for_topology(next(iter(registry.topologies.keys()))) or num_joints
        pose_item_dict['cond'] = _build_cond_from_registry(
            ds=ds,
            data_tensor=pose_item_dict[key],
            repr_registry=registry,
            task='pose',
            num_joints=nj,
            device=device,
            dtype=dtype,
        )
        return traj_item_dict, pose_item_dict

    # For task='pose', cond is already built with traj modalities included via registry
    return item_dict

def _split_data(motion_repr, task:str,  ds, cond_feat_dim: int = None, traj_feat_dim: int = None, predicted_modalities: List[str] = None):
    """
    Splits the full data tensor into the part to be modeled and the conditional part.

    Args:
        motion_repr: Full motion representation tensor
        task: 'traj' or 'pose'
        ds: Dataset object with _repr_indices and repr_registry
        cond_feat_dim: (Deprecated) Conditioning feature dimension - no longer used
        traj_feat_dim: (Deprecated) Trajectory feature dimension - no longer used
        predicted_modalities: List of modality names to extract as the modeled part

    Returns:
        (gt_part, modeled_part) tuple where:
        - gt_part: Ground truth part (only for 'traj' task, None for 'pose')
        - modeled_part: The part to be modeled by the network
    """
    if predicted_modalities is None:
        raise ValueError("predicted_modalities must be provided explicitly.")

    # Extract modeled part using name-based selection
    concat_dim = -1 if motion_repr.dim() == 3 else 1
    parts = []
    for name in predicted_modalities:
        s, e = ds._repr_indices[name]
        if motion_repr.dim() == 3:
            parts.append(motion_repr[:, :, s:e])
        else:
            parts.append(motion_repr[:, s:e, ...])
    modeled_part = torch.cat(parts, dim=concat_dim)

    if task == 'traj':
        # For trajectory task, compute gt_part as everything not in traj or cond modalities
        # Get all modality names that are NOT trajectory or conditioning modalities
        traj_modalities = set(ds.traj_repr_name_list)
        cond_modalities = set(ds.cond_repr_name_list_traj)
        excluded_modalities = traj_modalities | cond_modalities

        # Get remaining modalities (pose-related features)
        gt_modality_names = [name for name in ds._repr_list if name not in excluded_modalities]

        if gt_modality_names:
            gt_parts = []
            for name in gt_modality_names:
                s, e = ds._repr_indices[name]
                if motion_repr.dim() == 3:
                    gt_parts.append(motion_repr[:, :, s:e])
                else:
                    gt_parts.append(motion_repr[:, s:e, ...])
            gt_part = torch.cat(gt_parts, dim=concat_dim)
        else:
            # No ground truth part
            gt_part = None

        return gt_part, modeled_part
    else: # pose
        # For pose task, no separate gt_part
        gt_part = None
        return gt_part, modeled_part

def _combine_data(modeled_part, task:str, traj_feats: Optional[torch.Tensor], ds, predicted_modalities: List[str]):
    """
    Combines modeled part with ground truth part to return mixed motion_repr tensor.
    Requires registry-driven predicted_modalities and dataset metadata.

    All tensors use 3D format [bs, T, dim].
    """
    if predicted_modalities is None:
        raise ValueError("predicted_modalities must be provided.")
    if ds is None:
        raise ValueError("dataset metadata (ds) required for reconstruction.")

    if task == 'traj':
        # For traj task, return modeled_part in original format (no conversion)
        return modeled_part

    # modeled_part is [bs, T, dim]
    if modeled_part.dim() != 3:
        raise ValueError(f"_combine_data expects 3D modeled_part [bs, T, dim], got {modeled_part.shape}")

    if traj_feats is None:
        raise ValueError("traj_feats is required for pose task reconstruction.")
    if traj_feats.dim() != 3:
        raise ValueError(f"_combine_data expects 3D traj_feats [bs, T, dim], got {traj_feats.shape}")

    bs, clip_len, _ = modeled_part.shape
    device, dtype = modeled_part.device, modeled_part.dtype
    full = torch.full([bs, clip_len, ds.body_feat_dim], torch.inf, device=device, dtype=dtype)

    # Insert trajectory features according to traj_repr_name_list
    cur_traj = 0
    for name in ds.traj_repr_name_list:
        s, e = ds._repr_indices[name]
        width = e - s
        full[:, :, s:e] = traj_feats[:, :, cur_traj:cur_traj+width]
        cur_traj += width

    # Insert predicted modalities
    cur = 0
    for name in predicted_modalities:
        s, e = ds._repr_indices[name]
        width = e - s
        full[:, :, s:e] = modeled_part[:, :, cur:cur+width]
        cur += width

    return full

def _apply_masking(
    batch,
    ds,
    task: str,
    mask_scheme: str = None,
    visible_ids_override: list = None,
    joint_visibility_mask: Optional[torch.Tensor] = None,
    pattern_name: Optional[str] = None,
    mask_policy: str = "whole_batch",
):
    """
    Apply masking to conditioning features and populate canonical visibility metadata.
    """
    registry = ds.repr_registry
    if registry is None:
        raise ValueError("repr_registry required for masking.")

    bs, T, cond_dim = batch['cond'].shape
    device = batch['cond'].device

    scheme_spec = None
    mask_reg = ds.mask_scheme_registry
    if mask_scheme:
        if mask_reg is None or not mask_reg.has(mask_scheme):
            raise ValueError(f"Mask scheme '{mask_scheme}' not found in mask_scheme_registry.")
        scheme_spec = mask_reg.get(mask_scheme)
        if hasattr(mask_reg, "validate_scheme"):
            mask_reg.validate_scheme(mask_scheme, repr_registry=registry, task=task)

    cond_names = registry.cond_modalities.get(task, [])
    joint_modalities = []
    global_modalities = []
    for name in cond_names:
        mod = registry.get_modality(name)
        (joint_modalities if mod.is_joint_based else global_modalities).append(mod)

    per_joint_total = sum(m.per_joint_dim or 0 for m in joint_modalities)
    num_joints = (
        registry.joint_count_for_topology(joint_modalities[0].topology)
        if joint_modalities
        else registry.joint_count_for_topology(next(iter(registry.topologies.keys())))
    )
    expected_joint_dim = (num_joints or 0) * per_joint_total
    expected_global_dim = sum(m.dim for m in global_modalities)
    assert cond_dim == expected_joint_dim + expected_global_dim, f"cond_dim={cond_dim} != joint_dim={expected_joint_dim} + global_dim={expected_global_dim}"

    cond_mask = torch.ones(bs, T, cond_dim, device=device, dtype=torch.bool)
    cond_joint_mask = torch.ones(bs, T, num_joints, 1, device=device, dtype=torch.bool)
    cond_joint_modality_names = [m.name for m in joint_modalities]
    cond_joint_modality_mask = torch.ones(bs, T, num_joints, len(joint_modalities), device=device, dtype=torch.bool)
    cond_global_modality_names = [m.name for m in global_modalities]
    cond_global_mask = torch.ones(bs, T, len(global_modalities), device=device, dtype=torch.bool)

    def _visible_from_ids(ids):
        vis = torch.zeros(bs, T, num_joints, device=device, dtype=torch.bool)
        if ids is not None and len(ids) > 0:
            vis[:, :, [int(i) for i in ids]] = True
        return vis

    def _visible_from_ids_by_t(ids_by_t):
        if len(ids_by_t) != T:
            raise ValueError(f'visible_ids_by_t length {len(ids_by_t)} != T={T}')
        vis_t = torch.zeros(T, num_joints, device=device, dtype=torch.bool)
        for t_idx, ids in enumerate(ids_by_t):
            if ids is not None and len(ids) > 0:
                vis_t[t_idx, [int(i) for i in ids]] = True
        return vis_t.unsqueeze(0).expand(bs, -1, -1)

    if joint_modalities and per_joint_total > 0:
        if joint_visibility_mask is not None:
            if joint_visibility_mask.dim() == 4 and joint_visibility_mask.shape[-1] == 1:
                joint_visibility_mask = joint_visibility_mask.squeeze(-1)
            if joint_visibility_mask.shape != (bs, T, num_joints):
                raise ValueError(f"joint_visibility_mask shape {joint_visibility_mask.shape} != expected ({bs}, {T}, {num_joints})")
            default_vis = joint_visibility_mask.to(device=device, dtype=torch.bool)
        elif scheme_spec and scheme_spec.visible_ids_by_t is not None:
            if not scheme_spec.allow_time_variation:
                raise ValueError(f"Scheme '{scheme_spec.name}' does not allow time variation.")
            default_vis = _visible_from_ids_by_t(scheme_spec.visible_ids_by_t)
        elif visible_ids_override is not None:
            default_vis = _visible_from_ids(visible_ids_override)
        elif scheme_spec and scheme_spec.visible_ids:
            default_vis = _visible_from_ids(scheme_spec.visible_ids)
        else:
            default_vis = torch.ones(bs, T, num_joints, device=device, dtype=torch.bool)

        joint_block_mask = torch.ones(bs, T, num_joints, per_joint_total, device=device, dtype=torch.bool)
        within = 0
        local_positions_vis = None
        for modality_idx, m in enumerate(joint_modalities):
            width = m.per_joint_dim or 0
            modality_vis = default_vis.clone()
            rule = scheme_spec.modality_spec(m.name) if scheme_spec else None
            if rule and (rule.visible_ids_by_t is not None or rule.visible_ids is not None):
                if rule.visible_ids_by_t is not None:
                    if not scheme_spec.allow_time_variation:
                        raise ValueError(f"Scheme '{scheme_spec.name}' does not allow time variation.")
                    modality_vis &= _visible_from_ids_by_t(rule.visible_ids_by_t)
                else:
                    modality_vis &= _visible_from_ids(rule.visible_ids)
            cond_joint_modality_mask[..., modality_idx] = modality_vis
            joint_block_mask[..., within:within + width] = modality_vis.unsqueeze(-1).expand(bs, T, num_joints, width)
            if m.name == "local_positions":
                local_positions_vis = modality_vis
            within += width

        cond_joint_mask[..., 0] = local_positions_vis if local_positions_vis is not None else default_vis
        joint_flat = joint_block_mask.reshape(bs, T, -1)
        cond_mask[:, :, :expected_joint_dim] = joint_flat
        batch['cond'][:, :, :expected_joint_dim] = torch.where(
            joint_flat,
            batch['cond'][:, :, :expected_joint_dim],
            torch.zeros_like(batch['cond'][:, :, :expected_joint_dim]),
        )

    def _build_global_mask(rule, width):
        mask_local = torch.ones(bs, T, width, device=device, dtype=torch.bool)
        if rule is None:
            return mask_local
        if rule.visible_by_t is not None and scheme_spec and not scheme_spec.allow_time_variation:
            raise ValueError(f"Scheme '{scheme_spec.name}' does not allow time variation.")
        if rule.visible is False:
            mask_local[:] = False
        if rule.masked_dims:
            mask_local[:, :, [int(i) for i in rule.masked_dims]] = False
        if rule.visible_by_t is not None:
            if len(rule.visible_by_t) != T:
                raise ValueError(f'visible_by_t length {len(rule.visible_by_t)} != T={T}')
            vis_t = torch.tensor(rule.visible_by_t, device=device, dtype=torch.bool).view(1, T, 1)
            mask_local &= vis_t.expand(bs, -1, width)
        return mask_local

    cur = expected_joint_dim
    for modality_idx, m in enumerate(global_modalities):
        rule = scheme_spec.modality_spec(m.name) if scheme_spec else None
        mask_local = _build_global_mask(rule, m.dim)
        cond_mask[:, :, cur:cur + m.dim] &= mask_local
        cond_global_mask[..., modality_idx] = mask_local.all(dim=-1)
        mask_value = float(rule.mask_value) if rule is not None else 0.0
        batch['cond'][:, :, cur:cur + m.dim] = torch.where(
            mask_local,
            batch['cond'][:, :, cur:cur + m.dim],
            torch.full_like(batch['cond'][:, :, cur:cur + m.dim], mask_value),
        )
        cur += m.dim

    if pattern_name is None:
        pattern_name = mask_scheme or ("visible_ids_override" if visible_ids_override is not None else "full_visibility")

    batch['cond_mask'] = cond_mask
    batch['cond_joint_mask'] = cond_joint_mask
    batch['cond_joint_modality_mask'] = cond_joint_modality_mask
    batch['cond_joint_modality_names'] = cond_joint_modality_names
    batch['cond_global_mask'] = cond_global_mask
    batch['cond_global_modality_names'] = cond_global_modality_names
    batch['cond_pattern_names'] = [str(pattern_name)] * bs
    batch['cond_mask_policy'] = mask_policy

def _apply_egoexo_like_masking(
    batch,
    ds,
    task: str,
    mask_scheme: str = None,
    mask_joint_ids_np=None,
    pattern_name: Optional[str] = None,
    mask_policy: str = "whole_batch",
    visibility_seed: Optional[int] = None,
):
    """
    Apply egoexo-like masking to smplx joints using probabilistic visibility statistics.

    This function generates a joint visibility mask (either probabilistically using
    SMPLJointVisProb_EgoexoLike, or from occluded_joint_ids) and delegates the actual
    masking to _apply_masking.

    Args:
        batch: Batch dictionary containing 'cond' tensor
        ds: Dataset object with repr_registry and mask_scheme_registry
        task: Task type ('pose', 'traj', or 'both')
        occluded_joint_ids: Optional array of joint IDs that should be MASKED (invisible).
            If None, uses probabilistic masking based on SMPLJointVisProb_EgoexoLike.
    """
    bs, T, *_ = batch['cond'].shape
    device = batch['cond'].device

    # Validate mask scheme early (clearer user feedback)
    if mask_scheme:
        mask_reg = ds.mask_scheme_registry
        if mask_reg is None or not mask_reg.has(mask_scheme):
            raise ValueError(f"Mask scheme '{mask_scheme}' not found in mask_scheme_registry.")

    # Get num_joints from registry
    registry = ds.repr_registry
    if registry is None:
        raise ValueError("ds.repr_registry must be set for masking.")
    num_joints = registry.joint_count_for_topology(next(iter(registry.topologies.keys())))

    # Determine joint visibility mask
    if mask_joint_ids_np is not None:
        # Deterministic masking: mask_joint_ids_np specifies joints to MASK (set to False)
        joint_visibility_mask = torch.ones(bs, T, num_joints, device=device, dtype=torch.bool)
        if len(mask_joint_ids_np) > 0:
            ids = [i for i in mask_joint_ids_np if i < num_joints]
            if ids:
                joint_visibility_mask[:, :, ids] = False
    elif 'cond_joint_mask' in batch:
        # Reuse existing mask if present, currently used in egoexo4d dataset inference time.
        joint_visibility_mask = batch['cond_joint_mask']
        if joint_visibility_mask.dim() == 4:
            joint_visibility_mask = joint_visibility_mask.squeeze(-1)
    else:
        # Probabilistic masking using EgoExo visibility statistics
        from utils.other_utils import SMPLJointID, SMPLJointVisProb_EgoexoLike

        # Set visibility probabilities for each joint (legacy egoexo statistics)
        SMPLJointVisProb_EgoexoLike[SMPLJointID.PELVIS] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_HIP] = 0.9119
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_HIP] = 0.9022
        SMPLJointVisProb_EgoexoLike[SMPLJointID.SPINE1] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_KNEE] = 0.8082
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_KNEE] = 0.7973
        SMPLJointVisProb_EgoexoLike[SMPLJointID.SPINE2] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_ANKLE] = 0.7733
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_ANKLE] = 0.7638
        SMPLJointVisProb_EgoexoLike[SMPLJointID.SPINE3] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_FOOT] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_FOOT] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.NECK] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_COLLAR] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_COLLAR] = 0.0
        SMPLJointVisProb_EgoexoLike[SMPLJointID.HEAD] = 0.8830
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_SHOULDER] = 0.9632
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_SHOULDER] = 0.9782
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_ELBOW] = 0.9388
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_ELBOW] = 0.9705
        SMPLJointVisProb_EgoexoLike[SMPLJointID.LEFT_WRIST] = 0.8797
        SMPLJointVisProb_EgoexoLike[SMPLJointID.RIGHT_WRIST] = 0.9164

        probs = torch.from_numpy(SMPLJointVisProb_EgoexoLike)[None, None, :].to(device)
        generator = None
        if visibility_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(visibility_seed))
        joint_visibility_mask = torch.rand((bs, T, num_joints), device=device, generator=generator) < probs.expand(bs, T, -1)

    # Delegate to _apply_masking with the generated visibility mask
    _apply_masking(
        batch=batch,
        ds=ds,
        task=task,
        mask_scheme=mask_scheme,
        joint_visibility_mask=joint_visibility_mask,
        pattern_name=pattern_name or mask_scheme or "egoexo_like",
        mask_policy=mask_policy,
    )
