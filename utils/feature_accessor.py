"""
Utility functions for accessing motion representation features by name.
"""
import torch
from typing import List, Dict, Tuple


def select_modalities(
    motion_repr: torch.Tensor,
    modality_names: List[str],
    repr_indices: Dict[str, Tuple[int, int]],
    dim: int = None
) -> torch.Tensor:
    """
    Select and concatenate specific modalities from motion representation tensor.

    Args:
        motion_repr: Motion representation tensor of shape [B, T, D] or [B, D, 1, T]
        modality_names: List of modality names to extract (in order)
        repr_indices: Dictionary mapping modality names to (start, end) indices
        dim: Dimension along which to slice features. If None, auto-detect based on tensor shape.
             For 3D tensors [B, T, D], use dim=-1 or None
             For 4D tensors [B, D, 1, T], use dim=1

    Returns:
        Concatenated tensor containing only the requested modalities
    """
    if not modality_names:
        raise ValueError("modality_names cannot be empty")

    # Auto-detect dimension if not specified
    if dim is None:
        if motion_repr.dim() == 3:
            dim = -1
        elif motion_repr.dim() == 4:
            dim = 1
        else:
            raise ValueError(f"Cannot auto-detect dim for tensor with shape {motion_repr.shape}")

    # Extract slices for each modality
    parts = []
    for name in modality_names:
        if name not in repr_indices:
            raise ValueError(f"Modality '{name}' not found in repr_indices. Available: {list(repr_indices.keys())}")
        s, e = repr_indices[name]

        if dim == -1 or dim == motion_repr.ndim - 1:
            # Last dimension slicing
            parts.append(motion_repr[..., s:e])
        elif dim == 1:
            # Second dimension slicing (for 4D tensors [B, D, 1, T])
            parts.append(motion_repr[:, s:e, ...])
        elif dim == 0:
            # First dimension slicing
            parts.append(motion_repr[s:e, ...])
        else:
            raise ValueError(f"Unsupported dim={dim} for tensor with shape {motion_repr.shape}")

    return torch.cat(parts, dim=dim)


def insert_modalities(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    modality_names: List[str],
    repr_indices: Dict[str, Tuple[int, int]],
    dim: int = None
) -> torch.Tensor:
    """
    Insert modalities from source tensor into target tensor at specified positions.

    Args:
        target_tensor: Target tensor to insert into, shape [B, T, D] or [B, D, 1, T]
        source_tensor: Source tensor containing modalities to insert (same format as target)
        modality_names: List of modality names to insert (in order)
        repr_indices: Dictionary mapping modality names to (start, end) indices in target
        dim: Dimension along which to insert features. If None, auto-detect based on tensor shape.
             For 3D tensors [B, T, D], use dim=-1 or None
             For 4D tensors [B, D, 1, T], use dim=1

    Returns:
        Modified target tensor with source modalities inserted
    """
    if not modality_names:
        raise ValueError("modality_names cannot be empty")

    result = target_tensor.clone()

    # Auto-detect dimension if not specified
    if dim is None:
        if result.dim() == 3:
            dim = -1
        elif result.dim() == 4:
            dim = 1
        else:
            raise ValueError(f"Cannot auto-detect dim for tensor with shape {result.shape}")

    # Get indices for all modalities
    indices = get_modality_indices(repr_indices, modality_names)

    # Calculate expected total dimension
    total_dim = sum(e - s for s, e in indices)

    # Verify source tensor has correct size along dim
    source_dim_size = source_tensor.shape[dim]
    if source_dim_size != total_dim:
        raise ValueError(
            f"source_tensor has size {source_dim_size} along dim {dim}, "
            f"but expected {total_dim} for modalities {modality_names}"
        )

    # Insert each modality's values
    current_offset = 0
    for s, e in indices:
        width = e - s

        if dim == -1 or dim == result.ndim - 1:
            # Last dimension insertion
            result[..., s:e] = source_tensor[..., current_offset:current_offset + width]
        elif dim == 1:
            # Second dimension insertion (for 4D tensors)
            result[:, s:e, ...] = source_tensor[:, current_offset:current_offset + width, ...]
        elif dim == 0:
            # First dimension insertion
            result[s:e, ...] = source_tensor[current_offset:current_offset + width, ...]
        else:
            raise ValueError(f"Unsupported dim={dim} for tensor with shape {result.shape}")

        current_offset += width

    return result


def get_modality_indices(
    repr_indices: Dict[str, Tuple[int, int]],
    modality_names: List[str]
) -> List[Tuple[int, int]]:
    """
    Get list of (start, end) index tuples for given modality names.

    Args:
        repr_indices: Dictionary mapping modality names to (start, end) index tuples
        modality_names: List of modality names to look up

    Returns:
        List of (start, end) tuples in the same order as modality_names

    Raises:
        KeyError: If any modality name is not found in repr_indices

    Example:
        >>> repr_indices = {'smplx_rot_6d': (0, 6), 'smplx_trans': (6, 9), 'local_positions': (9, 75)}
        >>> get_modality_indices(repr_indices, ['smplx_rot_6d', 'smplx_trans'])
        [(0, 6), (6, 9)]
    """
    indices = []
    for name in modality_names:
        if name not in repr_indices:
            raise KeyError(f"Modality '{name}' not found in repr_indices. Available: {list(repr_indices.keys())}")
        indices.append(repr_indices[name])
    return indices


def get_modality_dim(modality_names: List[str], repr_indices: Dict[str, Tuple[int, int]]) -> int:
    """
    Calculate total dimension of specified modalities.

    Args:
        modality_names: List of modality names
        repr_indices: Dictionary mapping modality names to (start, end) indices

    Returns:
        Total dimension of all specified modalities
    """
    total_dim = 0
    for name in modality_names:
        if name not in repr_indices:
            raise ValueError(f"Modality '{name}' not found in repr_indices")
        s, e = repr_indices[name]
        total_dim += (e - s)
    return total_dim


def get_traj_features_from_batch(batch: dict, ds, task: str) -> torch.Tensor:
    """
    Extract trajectory features from batch using registry-based approach.

    For pose task, trajectory features are in global section of cond tensor.
    For traj task, we extract from motion_repr_clean.

    Args:
        batch: Batch dictionary containing motion data
        ds: Dataset object with repr_indices and traj_repr_name_list
        task: Task type ('pose' or 'traj')

    Returns:
        Trajectory features tensor
    """
    if task == 'pose':
        # For pose task, traj features are in global section of cond (after joint-based)
        # Use get_cond_modality_indices to find correct positions
        cond_indices = get_cond_modality_indices(ds, 'pose')
        traj_parts = []
        for name in ds.traj_repr_name_list:
            if name in cond_indices:
                s, e = cond_indices[name]
                traj_parts.append(batch['cond'][..., s:e])
        return torch.cat(traj_parts, dim=-1) if traj_parts else batch['cond'][..., :0]
    elif task == 'traj':
        # For traj task, extract from motion_repr_clean using registry
        return select_modalities(
            batch['motion_repr_clean'],
            ds.traj_repr_name_list,
            ds._repr_indices
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_cond_modality_indices(ds, task: str) -> Dict[str, Tuple[int, int]]:
    """
    Compute indices for modalities within the cond tensor for a given task.
    
    The cond tensor layout is: [joint-based interleaved] | [global modalities]
    
    Args:
        ds: Dataset object with repr_registry
        task: Task type ('traj', 'pose', or 'both')
        
    Returns:
        Dictionary mapping modality names to (start, end) indices within cond tensor
    """
    registry = ds.repr_registry
    cond_names = registry.cond_modalities.get(task, [])
    
    # Separate joint-based and global modalities
    joint_modalities = []
    global_modalities = []
    for name in cond_names:
        mod = registry.get_modality(name)
        (joint_modalities if mod.is_joint_based else global_modalities).append(mod)
    
    # Get num_joints
    num_joints = (
        registry.joint_count_for_topology(joint_modalities[0].topology)
        if joint_modalities
        else registry.joint_count_for_topology(next(iter(registry.topologies.keys())))
    )
    
    # Compute joint-based dimension
    per_joint_total = sum(m.per_joint_dim or 0 for m in joint_modalities)
    joint_dim = (num_joints or 0) * per_joint_total
    
    # Build indices dict
    cond_indices = {}
    
    # Joint-based modalities are interleaved, so we can't give simple start/end
    # For now, we only provide indices for global modalities
    # Joint-based modalities would need special handling
    
    # Global modalities start after joint-based section
    cur = joint_dim
    for m in global_modalities:
        cond_indices[m.name] = (cur, cur + m.dim)
        cur += m.dim
    
    return cond_indices


def insert_modalities_into_cond(
    cond_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    modality_names: List[str],
    ds,
    task: str,
) -> torch.Tensor:
    """
    Insert modalities from source tensor into cond tensor at correct positions.
    
    This handles the cond tensor layout which is different from motion_repr:
    - Cond layout: [joint-based interleaved] | [global modalities]
    - Source tensor should contain the modalities in order specified by modality_names
    
    Args:
        cond_tensor: Target cond tensor [B, T, cond_dim]
        source_tensor: Source tensor containing modalities to insert
        modality_names: List of modality names to insert (must be global modalities)
        ds: Dataset object with repr_registry
        task: Task type ('traj', 'pose', or 'both')
        
    Returns:
        Modified cond tensor with source modalities inserted
    """
    result = cond_tensor.clone()
    cond_indices = get_cond_modality_indices(ds, task)
    
    # Verify all modalities are in cond_indices (i.e., global modalities)
    for name in modality_names:
        if name not in cond_indices:
            raise ValueError(f"Modality '{name}' not found in cond indices for task '{task}'. "
                           f"Only global modalities can be inserted. Available: {list(cond_indices.keys())}")
    
    # Insert each modality
    current_offset = 0
    for name in modality_names:
        s, e = cond_indices[name]
        width = e - s
        result[..., s:e] = source_tensor[..., current_offset:current_offset + width]
        current_offset += width
    
    return result
