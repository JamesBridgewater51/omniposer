"""Action-blind metadata-routed mixture-of-experts primitives.

The routing surface in this module is intentionally restricted to visibility and
representation metadata. Mask action names/ids and pattern bookkeeping may exist
elsewhere in training/evaluation, but they must never influence router logits,
route state, losses, or diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_MOE_KEYS = {"metadata_moe", "local_moe"}
FORBIDDEN_KEY_FRAGMENTS = (
    "mask_action",
    "action_id",
    "action_ids",
    "action_name",
    "action_names",
    "cond_pattern",
    "_per_action",
    "per_action",
    "action_embedding",
    "action_embeddings",
)
ALLOWED_METADATA_TENSOR_KEYS = (
    "cond_visibility_mean",
    "joint_visibility",
    "joint_modality_visibility",
    "global_visibility",
    "group_visibility",
    "hidden_state_summary",
    "route_state_summary",
    "visibility_signal",
)
_ALLOWED_LOSS_SUFFIXES = {
    "load_balance": "load_balance_loss_weight",
    "router_z": "router_z_loss_weight",
    "temporal_consistency": "temporal_consistency_loss_weight",
    "importance": "importance_loss_weight",
    "entropy": "entropy_loss_weight",
    "topk_diversity": "topk_diversity_loss_weight",
}


@dataclass(frozen=True)
class RouteState:
    previous_router_probs: torch.Tensor
    previous_topk_indices: torch.Tensor
    visibility_signal: Optional[torch.Tensor] = None
    initialized: bool = True
    segment_size: Optional[int] = None
    source_steps: Optional[int] = None
    target_steps: Optional[int] = None
    synchronization_scale: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        state = {
            "previous_router_probs": self.previous_router_probs.detach(),
            "previous_topk_indices": self.previous_topk_indices.detach(),
            "initialized": self.initialized,
            "segment_size": self.segment_size,
            "source_steps": self.source_steps,
            "target_steps": self.target_steps,
            "synchronization_scale": self.synchronization_scale,
        }
        if self.visibility_signal is not None:
            state["visibility_signal"] = self.visibility_signal.detach()
        return state


def _to_plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _contains_forbidden_fragment(key: str) -> bool:
    lowered = key.lower()
    if lowered in LEGACY_MOE_KEYS:
        return True
    return any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS)


def assert_action_blind_mapping(mapping: Any, *, context: str = "MoE mapping") -> None:
    """Reject action-aware routing/loss/diagnostic surfaces.

    This validator is recursive over mapping keys. Values are intentionally not
    interpreted except for nested dictionaries/lists so ordinary modality names
    remain valid metadata values.
    """
    plain = _to_plain(mapping)

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                key_str = str(key)
                child_path = f"{path}.{key_str}" if path else key_str
                if _contains_forbidden_fragment(key_str):
                    raise ValueError(f"{context} contains forbidden action-aware key '{child_path}'.")
                visit(value, child_path)
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                visit(value, f"{path}[{idx}]")

    visit(plain, "")


def validate_no_action_keys(mapping: Any, *, context: str = "MoE mapping") -> None:
    assert_action_blind_mapping(mapping, context=context)


def is_moe_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    cfg = _to_plain(config) or {}
    return bool(cfg.get("enabled", False))


def validate_moe_config(
    config: Optional[Mapping[str, Any]],
    *,
    component: str = "moe",
    require_grouping: bool = False,
    require_stage_names: bool = False,
    allowed_routing_granularities: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Validate a single MoE config surface.

    Disabled configs are allowed to omit expensive routing details, but enabled
    configs must provide the explicit semantic surface required by the spec.
    """
    cfg = _to_plain(config) or {"enabled": False}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{component} MoE config must be a mapping.")
    cfg = dict(cfg)
    assert_action_blind_mapping(cfg, context=f"{component} MoE config")

    if not bool(cfg.get("enabled", False)):
        return cfg

    required = [
        "num_experts",
        "top_k",
        "num_shared_experts",
        "segment_size",
        "routing_granularity",
        "router_hidden_dim",
        "expert_hidden_dim",
    ]
    if require_grouping:
        required.append("grouping_name")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"{component} MoE config missing required keys: {missing}")

    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["top_k"])
    num_shared = int(cfg["num_shared_experts"])
    segment_size = int(cfg["segment_size"])
    router_hidden_dim = int(cfg["router_hidden_dim"])
    expert_hidden_dim = int(cfg["expert_hidden_dim"])
    if num_experts <= 0:
        raise ValueError(f"{component} MoE num_experts must be positive.")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"{component} MoE top_k must be in [1, num_experts].")
    if num_shared < 0 or num_shared > num_experts:
        raise ValueError(f"{component} MoE num_shared_experts must be in [0, num_experts].")
    if segment_size <= 0:
        raise ValueError(f"{component} MoE segment_size must be positive.")
    if router_hidden_dim <= 0 or expert_hidden_dim <= 0:
        raise ValueError(f"{component} MoE hidden dimensions must be positive.")

    granularity = str(cfg["routing_granularity"])
    allowed = set(allowed_routing_granularities or {"frame", "segment", "group", "token"})
    if granularity not in allowed:
        raise ValueError(f"{component} MoE routing_granularity '{granularity}' not in {sorted(allowed)}.")

    if require_grouping and not str(cfg.get("grouping_name", "")).strip():
        raise ValueError(f"{component} MoE requires a non-empty grouping_name.")

    if require_stage_names:
        stages = cfg.get("stage_names", None)
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or len(stages) == 0:
            raise ValueError(f"{component} MoE requires a non-empty stage_names list.")

    return cfg


def reject_legacy_moe_surfaces(config: Optional[Mapping[str, Any]], *, context: str) -> None:
    cfg = _to_plain(config) or {}
    if not isinstance(cfg, Mapping):
        return
    for key in cfg.keys():
        if str(key) in LEGACY_MOE_KEYS:
            raise ValueError(f"{context} uses legacy MoE surface '{key}'. Use network_config.moe instead.")
    assert_action_blind_mapping(cfg.get("moe", {}), context=f"{context}.moe")


def _as_float_tensor(value: Any, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if value is None or not torch.is_tensor(value):
        return None
    out = value.detach() if not value.requires_grad else value
    if out.dtype == torch.bool:
        out = out.float()
    else:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def _extract_batch_size_and_steps(batch: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[torch.device]]:
    for key in ("cond_mask", "cond_joint_mask", "cond_joint_modality_mask", "cond_global_mask", "cond"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.shape[0]), int(value.shape[1]), value.device
    return None, None, None


def _normalize_group_items(routing_groups: Any) -> List[Tuple[str, List[int]]]:
    if routing_groups is None:
        return []
    if hasattr(routing_groups, "groups"):
        routing_groups = getattr(routing_groups, "groups")
    if isinstance(routing_groups, Mapping):
        groups = routing_groups.get("groups", routing_groups)
        if hasattr(groups, "groups"):
            groups = getattr(groups, "groups")
        if groups and all(hasattr(value, "groups") for value in groups.values()):
            if len(groups) != 1:
                raise ValueError("Pass a single routing-group mapping, not the full routing_groups registry.")
            groups = getattr(next(iter(groups.values())), "groups")
        return [(str(name), [int(i) for i in indices]) for name, indices in groups.items()]
    return [(f"group_{idx}", [int(i) for i in indices]) for idx, indices in enumerate(routing_groups)]


def build_routing_metadata_from_batch(
    batch: Mapping[str, Any],
    *,
    hidden_state_summary: Optional[torch.Tensor] = None,
    route_state_summary: Optional[torch.Tensor] = None,
    routing_groups: Any = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Build action-blind routing metadata from the allowed batch fields only."""
    bs, steps, inferred_device = _extract_batch_size_and_steps(batch)
    device = device or inferred_device
    metadata: Dict[str, Any] = {}

    cond_mask = _as_float_tensor(batch.get("cond_mask"), device=device, dtype=dtype)
    if cond_mask is not None:
        metadata["cond_visibility_mean"] = cond_mask.mean(dim=-1, keepdim=True)

    joint_mask = _as_float_tensor(batch.get("cond_joint_mask"), device=device, dtype=dtype)
    if joint_mask is not None:
        if joint_mask.dim() == 4:
            joint_visibility = joint_mask.mean(dim=-1)
        elif joint_mask.dim() == 3:
            joint_visibility = joint_mask
        else:
            raise ValueError(f"cond_joint_mask must have shape [B,T,J] or [B,T,J,M], got {tuple(joint_mask.shape)}")
        metadata["joint_visibility"] = joint_visibility

        group_items = _normalize_group_items(routing_groups)
        if group_items:
            group_values = []
            for _, indices in group_items:
                if not indices:
                    raise ValueError("Routing groups cannot be empty.")
                group_values.append(joint_visibility.index_select(2, torch.as_tensor(indices, device=joint_visibility.device)).mean(dim=2))
            metadata["group_visibility"] = torch.stack(group_values, dim=-1)
            metadata["routing_group_names"] = tuple(name for name, _ in group_items)

    joint_modality_mask = _as_float_tensor(batch.get("cond_joint_modality_mask"), device=device, dtype=dtype)
    if joint_modality_mask is not None:
        if joint_modality_mask.dim() != 4:
            raise ValueError(f"cond_joint_modality_mask must have shape [B,T,J,M], got {tuple(joint_modality_mask.shape)}")
        metadata["joint_modality_visibility"] = joint_modality_mask.reshape(joint_modality_mask.shape[0], joint_modality_mask.shape[1], -1)
        names = batch.get("cond_joint_modality_names")
        if names is not None:
            metadata["joint_modality_names"] = tuple(str(name) for name in names)

    global_mask = _as_float_tensor(batch.get("cond_global_mask"), device=device, dtype=dtype)
    if global_mask is not None:
        metadata["global_visibility"] = global_mask.reshape(global_mask.shape[0], global_mask.shape[1], -1)
        names = batch.get("cond_global_modality_names")
        if names is not None:
            metadata["global_modality_names"] = tuple(str(name) for name in names)

    if hidden_state_summary is not None:
        metadata["hidden_state_summary"] = hidden_state_summary
    if route_state_summary is not None:
        metadata["route_state_summary"] = route_state_summary

    if bs is not None:
        metadata["batch_size"] = bs
    if steps is not None:
        metadata["source_steps"] = steps

    assert_action_blind_mapping(metadata, context="routing metadata")
    return metadata


def _resize_time_features(features: torch.Tensor, target_steps: int) -> torch.Tensor:
    if features.shape[1] == target_steps:
        return features
    resized = F.interpolate(features.transpose(1, 2), size=target_steps, mode="nearest")
    return resized.transpose(1, 2)


def routing_metadata_to_frame_features(
    routing_metadata: Optional[Any],
    *,
    target_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert allowed routing metadata to compact per-frame features [B,T,F]."""
    if torch.is_tensor(routing_metadata):
        tensor = routing_metadata.to(device=device, dtype=dtype)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1).expand(-1, target_steps, -1)
        elif tensor.dim() > 3:
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        if tensor.dim() != 3:
            raise ValueError(f"Routing metadata tensor must be [B,F] or [B,T,F], got {tuple(tensor.shape)}")
        return _resize_time_features(tensor, target_steps)

    metadata = _to_plain(routing_metadata) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("routing_metadata must be a mapping, tensor, or None.")
    assert_action_blind_mapping(metadata, context="routing metadata")

    tensors: List[torch.Tensor] = []
    inferred_bs = batch_size or metadata.get("batch_size")
    for key in ALLOWED_METADATA_TENSOR_KEYS:
        value = metadata.get(key)
        if not torch.is_tensor(value):
            continue
        tensor = value.to(device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            if inferred_bs is not None and tensor.shape[0] == int(inferred_bs):
                tensor = tensor.unsqueeze(1).expand(-1, target_steps, -1)
            else:
                tensor = tensor.reshape(1, tensor.shape[0], -1)
        elif tensor.dim() >= 3:
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        if tensor.shape[-1] == 0:
            continue
        tensors.append(_resize_time_features(tensor, target_steps))
        inferred_bs = tensors[-1].shape[0]

    if not tensors:
        if inferred_bs is None:
            inferred_bs = batch_size or 1
        return torch.zeros(int(inferred_bs), target_steps, 1, device=device, dtype=dtype)

    return torch.cat(tensors, dim=-1)


def make_route_state(
    probs: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    visibility_signal: Optional[torch.Tensor] = None,
    segment_size: Optional[int] = None,
    source_steps: Optional[int] = None,
    target_steps: Optional[int] = None,
    synchronization_scale: float = 0.0,
) -> Dict[str, Any]:
    state = RouteState(
        previous_router_probs=probs.detach(),
        previous_topk_indices=topk_indices.detach(),
        visibility_signal=visibility_signal.detach() if visibility_signal is not None else None,
        segment_size=segment_size,
        source_steps=source_steps,
        target_steps=target_steps,
        synchronization_scale=float(synchronization_scale),
    ).as_dict()
    assert_action_blind_mapping(state, context="route state")
    return state


def _previous_probs_from_state(route_state: Optional[Mapping[str, Any]], expected_shape: torch.Size) -> Optional[torch.Tensor]:
    if not isinstance(route_state, Mapping):
        return None
    prev = route_state.get("previous_router_probs")
    if not torch.is_tensor(prev) or tuple(prev.shape) != tuple(expected_shape):
        return None
    return prev


def visibility_bucket_statistics(
    visibility: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    *,
    prefix: str = "visibility",
) -> Dict[str, torch.Tensor]:
    """Return detached bucket counts/means, including zero counts for empty buckets."""
    if _contains_forbidden_fragment(prefix):
        raise ValueError(f"Forbidden diagnostic prefix '{prefix}'.")
    visibility = visibility.detach().float().reshape(-1)
    if values is None:
        values = visibility
    values = values.detach().float().reshape(-1)
    if values.numel() != visibility.numel():
        values = values.mean().expand_as(visibility)
    buckets = {
        "none": visibility <= 0.0,
        "low": (visibility > 0.0) & (visibility < 0.5),
        "mid": (visibility >= 0.5) & (visibility < 1.0),
        "full": visibility >= 1.0,
    }
    stats: Dict[str, torch.Tensor] = {}
    for name, mask in buckets.items():
        count = mask.sum().to(dtype=torch.float32)
        stats[f"{prefix}_{name}_count"] = count
        if bool(mask.any()):
            stats[f"{prefix}_{name}_mean"] = values[mask].mean()
        else:
            stats[f"{prefix}_{name}_mean"] = values.new_zeros(())
    assert_action_blind_mapping(stats, context="MoE diagnostics")
    return stats


class _ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, zero_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetadataRouter(nn.Module):
    """MLP router over action-blind metadata."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        router_hidden_dim: int,
        metadata_dim: Optional[int] = None,
        continuity_bias_weight: float = 0.0,
        global_load_bias_weight: float = 0.0,
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts].")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.continuity_bias_weight = float(continuity_bias_weight)
        self.global_load_bias_weight = float(global_load_bias_weight)
        self.router_noise_std = float(router_noise_std)
        first = nn.Linear(metadata_dim, router_hidden_dim) if metadata_dim is not None else nn.LazyLinear(router_hidden_dim)
        self.net = nn.Sequential(first, nn.GELU(), nn.Linear(router_hidden_dim, self.num_experts))
        self.register_buffer("global_load_bias", torch.zeros(self.num_experts), persistent=False)
        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_route_state: Optional[Dict[str, Any]] = None

    def forward(
        self,
        metadata: torch.Tensor,
        *,
        route_state: Optional[Mapping[str, Any]] = None,
        visibility_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if metadata.dim() < 2:
            raise ValueError(f"Router metadata must be at least 2D, got {tuple(metadata.shape)}")
        flat_metadata = metadata.reshape(-1, metadata.shape[-1])
        logits = self.net(flat_metadata).reshape(*metadata.shape[:-1], self.num_experts)
        if self.global_load_bias_weight:
            logits = logits + self.global_load_bias.to(device=logits.device, dtype=logits.dtype) * self.global_load_bias_weight

        prev_probs = _previous_probs_from_state(route_state, logits.shape)
        if prev_probs is not None and self.continuity_bias_weight:
            logits = logits + prev_probs.to(device=logits.device, dtype=logits.dtype) * self.continuity_bias_weight
        if self.training and self.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        probs = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_probs = probs.reshape(-1, self.num_experts)
        flat_top1 = topk_indices[..., 0].reshape(-1)
        assignment = F.one_hot(flat_top1, num_classes=self.num_experts).to(dtype=flat_probs.dtype)
        mean_probs = flat_probs.mean(dim=0)
        mean_assignment = assignment.mean(dim=0)
        load_balance = (mean_probs * mean_assignment).sum() * self.num_experts
        router_z = torch.logsumexp(logits, dim=-1).square().mean()
        if prev_probs is not None:
            temporal_consistency = F.mse_loss(probs, prev_probs.to(device=probs.device, dtype=probs.dtype))
        else:
            temporal_consistency = probs.new_zeros(())
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

        aux_losses = {
            "load_balance": load_balance,
            "router_z": router_z,
            "temporal_consistency": temporal_consistency,
        }
        diagnostics: Dict[str, torch.Tensor] = {
            "router_entropy": entropy.detach(),
            "router_load_min": mean_assignment.min().detach(),
            "router_load_max": mean_assignment.max().detach(),
            "router_prob_min": mean_probs.min().detach(),
            "router_prob_max": mean_probs.max().detach(),
        }
        for idx in range(self.num_experts):
            diagnostics[f"expert_{idx}_prob_mean"] = mean_probs[idx].detach()
            diagnostics[f"expert_{idx}_top1_rate"] = mean_assignment[idx].detach()
        if visibility_signal is not None:
            diagnostics.update(visibility_bucket_statistics(visibility_signal, entropy.expand_as(visibility_signal.float()), prefix="visibility"))

        state = make_route_state(
            probs,
            topk_indices,
            visibility_signal=visibility_signal,
            synchronization_scale=0.0,
        )
        self.last_aux_losses = aux_losses
        self.last_diagnostics = diagnostics
        self.last_route_state = state
        return {
            "logits": logits,
            "probs": probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "aux_losses": aux_losses,
            "diagnostics": diagnostics,
            "route_state": state,
        }


def _dispatch_experts(
    x: torch.Tensor,
    experts: nn.ModuleList,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_experts: Optional[nn.ModuleList] = None,
) -> torch.Tensor:
    token_shape = x.shape[:-1]
    dim = x.shape[-1]
    flat_x = x.reshape(-1, dim)
    expert_outputs = torch.stack([expert(flat_x).reshape(*token_shape, dim) for expert in experts], dim=-2)
    gather_idx = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, dim)
    selected = torch.gather(expert_outputs, dim=-2, index=gather_idx)
    routed = (selected * topk_weights.unsqueeze(-1)).sum(dim=-2)
    if shared_experts:
        shared = torch.stack([expert(flat_x).reshape(*token_shape, dim) for expert in shared_experts], dim=0).mean(dim=0)
        routed = routed + shared
    return routed


class MetadataRoutedTemporalMoEBlock(nn.Module):
    """Residual temporal MoE block for [B,C,T] tensors."""

    def __init__(self, channels: int, config: Mapping[str, Any], *, component: str = "traj_moe"):
        super().__init__()
        cfg = validate_moe_config(
            config,
            component=component,
            allowed_routing_granularities={"frame", "segment"},
        )
        if not bool(cfg.get("enabled", False)):
            raise ValueError("MetadataRoutedTemporalMoEBlock requires enabled config.")
        self.config = cfg
        self.channels = int(channels)
        self.segment_size = int(cfg["segment_size"])
        self.routing_granularity = str(cfg["routing_granularity"])
        self.residual_scale = float(cfg.get("residual_scale", 1.0))
        self.router = MetadataRouter(
            num_experts=int(cfg["num_experts"]),
            top_k=int(cfg["top_k"]),
            router_hidden_dim=int(cfg["router_hidden_dim"]),
            continuity_bias_weight=float(cfg.get("continuity_bias_weight", 0.0)),
            global_load_bias_weight=float(cfg.get("global_load_bias_weight", 0.0)),
            router_noise_std=float(cfg.get("router_noise_std", 0.0)),
        )
        self.experts = nn.ModuleList([_ExpertMLP(self.channels, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_experts"]))])
        self.shared_experts = nn.ModuleList([_ExpertMLP(self.channels, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_shared_experts"]))])
        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_route_state: Optional[Dict[str, Any]] = None

    def _segment_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        bsz, steps, dim = x.shape
        pad = (self.segment_size - steps % self.segment_size) % self.segment_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        pooled = x.reshape(bsz, -1, self.segment_size, dim).mean(dim=2)
        return pooled, pad

    def _segment_expand(self, x: torch.Tensor, *, original_steps: int) -> torch.Tensor:
        expanded = x.unsqueeze(2).expand(-1, -1, self.segment_size, -1).reshape(x.shape[0], -1, x.shape[-1])
        return expanded[:, :original_steps]

    def forward(
        self,
        x: torch.Tensor,
        *,
        routing_metadata: Optional[Any] = None,
        route_state: Optional[Mapping[str, Any]] = None,
        source_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Temporal MoE expects [B,C,T], got {tuple(x.shape)}")
        bsz, channels, steps = x.shape
        if channels != self.channels:
            raise ValueError(f"Temporal MoE expected {self.channels} channels, got {channels}.")
        x_tokens = x.transpose(1, 2)
        frame_metadata = routing_metadata_to_frame_features(
            routing_metadata,
            target_steps=steps,
            device=x.device,
            dtype=x.dtype,
            batch_size=bsz,
        )
        visibility = frame_metadata[..., :1].detach()

        if self.routing_granularity == "segment":
            expert_tokens, _ = self._segment_pool(x_tokens)
            router_metadata, _ = self._segment_pool(frame_metadata)
            visibility_tokens, _ = self._segment_pool(visibility)
            router_out = self.router(router_metadata, route_state=route_state, visibility_signal=visibility_tokens)
            delta_tokens = _dispatch_experts(
                expert_tokens,
                self.experts,
                router_out["topk_indices"],
                router_out["topk_weights"],
                self.shared_experts,
            )
            delta = self._segment_expand(delta_tokens, original_steps=steps)
            target_steps = expert_tokens.shape[1]
        else:
            router_out = self.router(frame_metadata, route_state=route_state, visibility_signal=visibility)
            delta = _dispatch_experts(
                x_tokens,
                self.experts,
                router_out["topk_indices"],
                router_out["topk_weights"],
                self.shared_experts,
            )
            target_steps = steps

        self.last_aux_losses = router_out["aux_losses"]
        self.last_diagnostics = router_out["diagnostics"]
        self.last_route_state = dict(router_out["route_state"])
        self.last_route_state["segment_size"] = self.segment_size
        self.last_route_state["source_steps"] = source_steps if source_steps is not None else steps
        self.last_route_state["target_steps"] = target_steps
        assert_action_blind_mapping(self.last_diagnostics, context="temporal MoE diagnostics")
        return (x_tokens + self.residual_scale * delta).transpose(1, 2)


class MetadataRoutedGroupedMoEBlock(nn.Module):
    """Grouped body-part MoE for joint tokens [B,T,J,C]."""

    def __init__(self, token_dim: int, routing_groups: Any, config: Mapping[str, Any], *, component: str = "pose_moe"):
        super().__init__()
        cfg = validate_moe_config(
            config,
            component=component,
            require_grouping=True,
            allowed_routing_granularities={"group", "token"},
        )
        if not bool(cfg.get("enabled", False)):
            raise ValueError("MetadataRoutedGroupedMoEBlock requires enabled config.")
        group_items = _normalize_group_items(routing_groups)
        if not group_items:
            raise ValueError("Grouped MoE requires non-empty routing_groups.")
        self.config = cfg
        self.token_dim = int(token_dim)
        self.group_names = tuple(name for name, _ in group_items)
        self.group_indices = tuple(tuple(indices) for _, indices in group_items)
        self.num_groups = len(group_items)
        self.residual_scale = float(cfg.get("residual_scale", 1.0))
        self.group_occlusion_embeddings = nn.Parameter(torch.zeros(self.num_groups, self.token_dim))
        self.router = MetadataRouter(
            num_experts=int(cfg["num_experts"]),
            top_k=int(cfg["top_k"]),
            router_hidden_dim=int(cfg["router_hidden_dim"]),
            continuity_bias_weight=float(cfg.get("continuity_bias_weight", 0.0)),
            global_load_bias_weight=float(cfg.get("global_load_bias_weight", 0.0)),
            router_noise_std=float(cfg.get("router_noise_std", 0.0)),
        )
        self.experts = nn.ModuleList([_ExpertMLP(self.token_dim, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_experts"]))])
        self.shared_experts = nn.ModuleList([_ExpertMLP(self.token_dim, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_shared_experts"]))])
        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_route_state: Optional[Dict[str, Any]] = None

    def _group_visibility(self, routing_metadata: Optional[Any], bsz: int, steps: int, joints: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        metadata = routing_metadata if isinstance(routing_metadata, Mapping) else {}
        group_visibility = metadata.get("group_visibility") if isinstance(metadata, Mapping) else None
        if torch.is_tensor(group_visibility):
            group_visibility = group_visibility.to(device=device, dtype=dtype)
            if group_visibility.shape[1] != steps:
                group_visibility = _resize_time_features(group_visibility, steps)
            return group_visibility
        joint_visibility = metadata.get("joint_visibility") if isinstance(metadata, Mapping) else None
        if torch.is_tensor(joint_visibility):
            joint_visibility = joint_visibility.to(device=device, dtype=dtype)
            if joint_visibility.shape[1] != steps:
                joint_visibility = _resize_time_features(joint_visibility, steps)
            values = []
            for indices in self.group_indices:
                idx = torch.as_tensor(indices, device=device)
                values.append(joint_visibility.index_select(2, idx).mean(dim=2))
            return torch.stack(values, dim=-1)
        return torch.ones(bsz, steps, self.num_groups, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        routing_metadata: Optional[Any] = None,
        route_state: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Grouped MoE expects [B,T,J,C], got {tuple(x.shape)}")
        bsz, steps, joints, dim = x.shape
        if dim != self.token_dim:
            raise ValueError(f"Grouped MoE expected token dim {self.token_dim}, got {dim}.")

        group_tokens = []
        for group_idx, indices in enumerate(self.group_indices):
            idx = torch.as_tensor(indices, device=x.device)
            pooled = x.index_select(2, idx).mean(dim=2)
            group_tokens.append(pooled)
        group_tokens_tensor = torch.stack(group_tokens, dim=2)
        group_visibility = self._group_visibility(routing_metadata, bsz, steps, joints, x.device, x.dtype).clamp(0.0, 1.0)
        occluded = group_visibility <= 0.0
        occlusion_emb = self.group_occlusion_embeddings.view(1, 1, self.num_groups, dim).to(device=x.device, dtype=x.dtype)
        group_tokens_tensor = torch.where(occluded.unsqueeze(-1), occlusion_emb, group_tokens_tensor)

        frame_metadata = routing_metadata_to_frame_features(
            routing_metadata,
            target_steps=steps,
            device=x.device,
            dtype=x.dtype,
            batch_size=bsz,
        )
        group_metadata = frame_metadata.unsqueeze(2).expand(-1, -1, self.num_groups, -1)
        group_metadata = torch.cat([group_metadata, group_visibility.unsqueeze(-1)], dim=-1)
        router_out = self.router(group_metadata, route_state=route_state, visibility_signal=group_visibility.unsqueeze(-1))
        group_delta = _dispatch_experts(
            group_tokens_tensor,
            self.experts,
            router_out["topk_indices"],
            router_out["topk_weights"],
            self.shared_experts,
        )

        joint_delta = torch.zeros_like(x)
        counts = torch.zeros(joints, device=x.device, dtype=x.dtype)
        for group_idx, indices in enumerate(self.group_indices):
            idx = torch.as_tensor(indices, device=x.device)
            joint_delta.index_add_(2, idx, group_delta[:, :, group_idx].unsqueeze(2).expand(-1, -1, idx.numel(), -1))
            counts.index_add_(0, idx, torch.ones(idx.numel(), device=x.device, dtype=x.dtype))
        joint_delta = joint_delta / counts.clamp_min(1.0).view(1, 1, joints, 1)

        self.last_aux_losses = router_out["aux_losses"]
        self.last_diagnostics = dict(router_out["diagnostics"])
        for group_idx, name in enumerate(self.group_names):
            self.last_diagnostics[f"group_{group_idx}_visibility_mean"] = group_visibility[..., group_idx].mean().detach()
        self.last_route_state = router_out["route_state"]
        assert_action_blind_mapping(self.last_diagnostics, context="grouped MoE diagnostics")
        return x + self.residual_scale * joint_delta


class MetadataRoutedFFN(nn.Module):
    """Routed transformer FFN replacement for token tensors [B,S,C]."""

    def __init__(self, d_model: int, config: Mapping[str, Any], *, component: str = "pose_ffn_moe"):
        super().__init__()
        cfg = validate_moe_config(
            config,
            component=component,
            allowed_routing_granularities={"token", "frame", "segment"},
        )
        if not bool(cfg.get("enabled", False)):
            raise ValueError("MetadataRoutedFFN requires enabled config.")
        self.config = cfg
        self.d_model = int(d_model)
        self.router = MetadataRouter(
            num_experts=int(cfg["num_experts"]),
            top_k=int(cfg["top_k"]),
            router_hidden_dim=int(cfg["router_hidden_dim"]),
            continuity_bias_weight=float(cfg.get("continuity_bias_weight", 0.0)),
            global_load_bias_weight=float(cfg.get("global_load_bias_weight", 0.0)),
            router_noise_std=float(cfg.get("router_noise_std", 0.0)),
        )
        self.experts = nn.ModuleList([_ExpertMLP(self.d_model, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_experts"]))])
        self.shared_experts = nn.ModuleList([_ExpertMLP(self.d_model, int(cfg["expert_hidden_dim"])) for _ in range(int(cfg["num_shared_experts"]))])
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_route_state: Optional[Dict[str, Any]] = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        routing_metadata: Optional[Any] = None,
        route_state: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Routed FFN expects [B,S,C], got {tuple(x.shape)}")
        bsz, steps, _ = x.shape
        metadata = routing_metadata_to_frame_features(
            routing_metadata,
            target_steps=steps,
            device=x.device,
            dtype=x.dtype,
            batch_size=bsz,
        )
        visibility = metadata[..., :1].detach()
        router_out = self.router(metadata, route_state=route_state, visibility_signal=visibility)
        out = _dispatch_experts(x, self.experts, router_out["topk_indices"], router_out["topk_weights"], self.shared_experts)
        self.last_aux_losses = router_out["aux_losses"]
        self.last_diagnostics = router_out["diagnostics"]
        self.last_route_state = router_out["route_state"]
        assert_action_blind_mapping(self.last_diagnostics, context="routed FFN diagnostics")
        return self.dropout(out)


def collect_weighted_moe_loss_dict(
    aux_losses: Mapping[str, torch.Tensor],
    config: Optional[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-stage/layer aux losses into allowed public loss names."""
    cfg = _to_plain(config) or {}
    if not bool(cfg.get("enabled", False)):
        return {}
    assert_action_blind_mapping(aux_losses, context=f"{prefix} MoE losses")
    result: Dict[str, torch.Tensor] = {}
    for suffix, weight_key in _ALLOWED_LOSS_SUFFIXES.items():
        matching = [value for key, value in aux_losses.items() if key.endswith(suffix) and torch.is_tensor(value)]
        if not matching:
            continue
        total = matching[0].new_zeros(())
        for value in matching:
            total = total + value
        weight = float(cfg.get(weight_key, 0.0))
        public_key = f"loss_{prefix}_{suffix}"
        if _contains_forbidden_fragment(public_key):
            raise ValueError(f"Forbidden MoE loss key '{public_key}'.")
        result[public_key] = total * weight
    return result


def moe_aux_loss_in_main_loss(config: Optional[Mapping[str, Any]]) -> bool:
    cfg = _to_plain(config) or {}
    return bool(cfg.get("enabled", False)) and bool(cfg.get("moe_aux_loss_in_main_loss", False))


def sum_loss_dict(loss_dict: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
    total: Optional[torch.Tensor] = None
    for key, value in loss_dict.items():
        if _contains_forbidden_fragment(key):
            raise ValueError(f"Forbidden MoE loss key '{key}'.")
        if not torch.is_tensor(value):
            continue
        total = value if total is None else total + value
    return total
