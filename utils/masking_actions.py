from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from utils.batch_util import _apply_masking, _apply_egoexo_like_masking


def build_mask_action_pool(masking_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    actions = masking_cfg.get("actions")
    if not actions:
        raise ValueError("masking.actions must be defined and non-empty.")
    pool: Dict[str, Dict[str, Any]] = {}
    for action in actions:
        if "name" not in action or "type" not in action:
            raise ValueError(f"Each masking action must have 'name' and 'type': {action}")
        name = action["name"]
        if name in pool:
            raise ValueError(f"Duplicate masking action name: {name}")
        pool[name] = dict(action)
    return pool


def validate_batch_policy(batch_policy: str) -> None:
    if batch_policy not in {"whole_batch", "per_sample"}:
        raise ValueError("masking.batch_policy must be 'whole_batch' or 'per_sample'.")


def validate_masking_config(masking_cfg: Dict[str, Any], action_pool: Dict[str, Dict[str, Any]], mask_scheme_registry=None) -> None:
    validate_batch_policy(masking_cfg.get("batch_policy", "whole_batch"))
    for action_name, action in action_pool.items():
        action_type = action.get("type")
        if action_type not in {"registry", "random_joint_visibility", "custom"}:
            raise ValueError(f"Unsupported mask action type for '{action_name}': {action_type}")
        scheme = action.get("scheme")
        if action_type == "registry" and not scheme:
            raise ValueError(f"registry action '{action_name}' missing scheme.")
        if scheme and mask_scheme_registry is not None and not mask_scheme_registry.has(scheme):
            raise ValueError(f"Masking action '{action_name}' references unknown scheme '{scheme}'.")

    mixture = masking_cfg.get("mixture")
    if not mixture:
        raise ValueError("masking.mixture must be defined and non-empty.")
    for entry in mixture:
        weights = entry.get("weights")
        if not weights:
            raise ValueError(f"masking.mixture entry missing weights: {entry}")
        total = 0.0
        for name, weight in weights.items():
            if name not in action_pool:
                raise ValueError(f"masking.mixture references unknown action '{name}'.")
            value = float(weight)
            if value < 0:
                raise ValueError(f"masking.mixture weight for '{name}' must be non-negative, got {value}.")
            total += value
        if total <= 0:
            raise ValueError(f"masking.mixture weights must have positive total, got: {weights}")


def resolve_random_visible_ids(action: Dict[str, Any], total_joints: int, rng: Optional[torch.Generator] = None) -> List[int]:
    if "num_visible" in action:
        num_visible = int(action["num_visible"])
    elif "max_mask" in action:
        max_mask = int(action["max_mask"])
        if max_mask > 0:
            if rng is None:
                num_visible = total_joints - torch.randint(1, max_mask + 1, (1,)).item()
            else:
                num_visible = total_joints - torch.randint(1, max_mask + 1, (1,), generator=rng).item()
        else:
            num_visible = total_joints
    else:
        raise ValueError(f"random_joint_visibility action requires num_visible or max_mask: {action}")
    num_visible = max(0, min(total_joints, num_visible))
    if rng is None:
        return torch.randperm(total_joints)[:num_visible].tolist()
    return torch.randperm(total_joints, generator=rng)[:num_visible].tolist()


def materialize_mask_action(
    action_name: str,
    action_pool: Dict[str, Dict[str, Any]],
    total_joints: int,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    if action_name not in action_pool:
        raise ValueError(f"Unknown masking action '{action_name}'.")
    action = dict(action_pool[action_name])
    action["name"] = action_name
    if action.get("type") == "random_joint_visibility":
        action["visible_ids"] = resolve_random_visible_ids(action, total_joints, rng=rng)
    if action.get("type") == "custom" and action.get("fn") == "egoexo_like" and rng is not None:
        action["visibility_seed"] = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
    return action


def sample_action_names_from_weights(
    weights: Dict[str, float],
    count: int,
    rng: Optional[torch.Generator] = None,
) -> List[str]:
    if count <= 0:
        return []
    names = list(weights.keys())
    values = torch.tensor([float(weights[name]) for name in names], dtype=torch.float32)
    if torch.any(values < 0):
        raise ValueError(f"masking.mixture weights must be non-negative, got: {weights}")
    total = float(values.sum().item())
    if total <= 0:
        raise ValueError(f"masking.mixture weights must have positive total, got: {weights}")
    sampled = torch.multinomial(values / total, count, replacement=True, generator=rng)
    return [names[int(i)] for i in sampled.tolist()]


def materialize_per_sample_action_plan(
    action_names: Sequence[str],
    action_pool: Dict[str, Dict[str, Any]],
    total_joints: int,
    batch_size: int,
    rng: Optional[torch.Generator] = None,
) -> List[Dict[str, Any]]:
    if len(action_names) != batch_size:
        raise ValueError(f"per-sample action assignment length {len(action_names)} != batch size {batch_size}.")
    return [materialize_mask_action(name, action_pool, total_joints, rng=rng) for name in action_names]


def visible_ids_from_action(
    action: Dict[str, Any],
    mask_scheme_registry,
    total_joints: int,
) -> Optional[List[int]]:
    visible_ids = action.get("visible_ids")
    if visible_ids is not None:
        return list(visible_ids)
    scheme = action.get("scheme")
    if scheme and mask_scheme_registry is not None and mask_scheme_registry.has(scheme):
        return list(mask_scheme_registry.visible_ids(scheme))
    if action.get("type") == "random_joint_visibility":
        return list(range(total_joints))
    return None


def _resolve_action_scheme(action: Dict[str, Any]) -> Optional[str]:
    return action.get("scheme") if action.get("type") in {"registry", "custom"} else None


def apply_mask_action(batch, ds, task: str, action: Dict[str, Any], total_joints: int) -> None:
    action_type = action.get("type")
    if action_type == "registry":
        scheme = action.get("scheme")
        if not scheme:
            raise ValueError(f"registry action missing scheme: {action}")
        _apply_masking(
            batch=batch,
            ds=ds,
            task=task,
            mask_scheme=scheme,
            visible_ids_override=None,
            pattern_name=action.get("name", scheme),
            mask_policy="whole_batch",
        )
        return
    if action_type == "random_joint_visibility":
        visible_ids = action.get("visible_ids")
        if visible_ids is None:
            visible_ids = resolve_random_visible_ids(action, total_joints)
        _apply_masking(
            batch=batch,
            ds=ds,
            task=task,
            mask_scheme=None,
            visible_ids_override=visible_ids,
            pattern_name=action.get("name", "random_joint_visibility"),
            mask_policy="whole_batch",
        )
        return
    if action_type == "custom":
        fn = action.get("fn")
        if fn == "egoexo_like":
            mask_joint_ids_np = None
            visible_ids = action.get("visible_ids")
            if visible_ids is not None:
                mask_joint_ids_np = np.array([i for i in range(total_joints) if i not in visible_ids], dtype=int)
            else:
                scheme = action.get("scheme")
                if scheme and ds.mask_scheme_registry is not None and ds.mask_scheme_registry.has(scheme):
                    ids_to_keep = set(ds.mask_scheme_registry.visible_ids(scheme))
                    mask_joint_ids_np = np.array(list(set(range(total_joints)) - ids_to_keep), dtype=int)
            # NOTE: training samples visibility from the EgoExo4D statistics defined in `_apply_egoxo_like_masking`; evaluation uses batch['cond_joint_mask']; neither path is overridden by `mask_joint_ids_np`.
            _apply_egoexo_like_masking(
                batch=batch,
                ds=ds,
                task=task,
                mask_scheme=scheme,
                mask_joint_ids_np=None,
                pattern_name=action.get("name", fn),
                mask_policy="whole_batch",
                visibility_seed=action.get("visibility_seed"),
            )
            return
        raise ValueError(f"Unsupported custom mask action fn: {fn}")
    raise ValueError(f"Unsupported mask action type: {action_type}")


def apply_per_sample_mask_actions(batch, ds, task: str, action_plan: Sequence[Dict[str, Any]], total_joints: int) -> None:
    bs = batch["cond"].shape[0]
    if len(action_plan) != bs:
        raise ValueError(f"per-sample action plan length {len(action_plan)} != batch size {bs}.")

    cond_chunks = []
    cond_mask_chunks = []
    joint_mask_chunks = []
    joint_modality_mask_chunks = []
    global_mask_chunks = []
    pattern_names = []
    joint_modality_names = None
    global_modality_names = None

    for sample_idx, action in enumerate(action_plan):
        sample_batch = {
            key: (value[sample_idx:sample_idx + 1].clone() if isinstance(value, torch.Tensor) and value.shape[:1] == (bs,) else value)
            for key, value in batch.items()
        }
        apply_mask_action(sample_batch, ds, task, action, total_joints)
        cond_chunks.append(sample_batch["cond"])
        cond_mask_chunks.append(sample_batch["cond_mask"])
        joint_mask_chunks.append(sample_batch["cond_joint_mask"])
        if "cond_joint_modality_mask" in sample_batch:
            joint_modality_mask_chunks.append(sample_batch["cond_joint_modality_mask"])
            joint_modality_names = sample_batch.get("cond_joint_modality_names")
        if "cond_global_mask" in sample_batch:
            global_mask_chunks.append(sample_batch["cond_global_mask"])
            global_modality_names = sample_batch.get("cond_global_modality_names")
        names = sample_batch.get("cond_pattern_names") or [action.get("name", action.get("scheme", action.get("type", "unknown")))]
        pattern_names.append(names[0])

    batch["cond"] = torch.cat(cond_chunks, dim=0)
    batch["cond_mask"] = torch.cat(cond_mask_chunks, dim=0)
    batch["cond_joint_mask"] = torch.cat(joint_mask_chunks, dim=0)
    if joint_modality_mask_chunks:
        batch["cond_joint_modality_mask"] = torch.cat(joint_modality_mask_chunks, dim=0)
        batch["cond_joint_modality_names"] = joint_modality_names
    if global_mask_chunks:
        batch["cond_global_mask"] = torch.cat(global_mask_chunks, dim=0)
        batch["cond_global_modality_names"] = global_modality_names
    batch["cond_pattern_names"] = pattern_names
    batch["cond_mask_policy"] = "per_sample"


def apply_mask_action_plan(batch, ds, task: str, action_or_plan, total_joints: int, batch_policy: str = "whole_batch") -> None:
    validate_batch_policy(batch_policy)
    if batch_policy == "per_sample":
        apply_per_sample_mask_actions(batch, ds, task, action_or_plan, total_joints)
    else:
        apply_mask_action(batch, ds, task, action_or_plan, total_joints)
