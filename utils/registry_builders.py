from typing import Any, Dict, List

from utils.mask_registry import MaskSchemeRegistry, MaskSchemeSpec, ModalityMaskSpec
from utils.representation_registry import (
    ModalitySpec,
    RepresentationRegistry,
    RoutingGroupSpec,
    TopologySpec,
)


def build_representation_registry(cfg: Dict[str, Any]) -> RepresentationRegistry:
    """
    Construct a RepresentationRegistry from a plain (typically OmegaConf) dict.
    All fields must be provided explicitly; no defaults are injected.
    """
    if cfg is None:
        raise ValueError("representation_registry config must be provided.")

    if "topologies" not in cfg or "modalities" not in cfg:
        raise ValueError("representation_registry config requires 'topologies' and 'modalities' keys.")

    topologies: Dict[str, TopologySpec] = {}
    for name, topo_cfg in cfg["topologies"].items():
        topologies[name] = TopologySpec(
            name=name,
            joint_names=topo_cfg.get("joint_names"),
            joint_count=topo_cfg.get("joint_count"),
        )

    modalities: List[ModalitySpec] = []
    for m in cfg["modalities"]:
        modalities.append(
            ModalitySpec(
                name=m["name"],
                dim=m["dim"],
                is_joint_based=m["is_joint_based"],
                per_joint_dim=m.get("per_joint_dim"),
                topology=m.get("topology"),
                semantic=m.get("semantic"),
            )
        )

    traj_modalities = cfg.get("traj_modalities")
    cond_modalities = cfg.get("cond_modalities")
    predicted_modalities = cfg.get("predicted_modalities")

    if not traj_modalities or not cond_modalities or not predicted_modalities:
        raise ValueError("traj_modalities, cond_modalities, and predicted_modalities must be specified in representation_registry config.")

    routing_groups: Dict[str, RoutingGroupSpec] = {}
    for name, group_cfg in (cfg.get("routing_groups", {}) or {}).items():
        groups_cfg = group_cfg.get("groups", {}) or {}
        routing_groups[name] = RoutingGroupSpec(
            name=name,
            topology=group_cfg["topology"],
            groups={group_name: [int(idx) for idx in indices] for group_name, indices in groups_cfg.items()},
        )

    registry = RepresentationRegistry(
        modalities=modalities,
        topologies=topologies,
        traj_modalities=traj_modalities,
        cond_modalities=cond_modalities,
        predicted_modalities=predicted_modalities,
        routing_groups=routing_groups,
    )
    registry.validate_all_routing_groups()
    return registry


def build_mask_scheme_registry(cfg: Dict[str, Any]) -> MaskSchemeRegistry:
    """Construct a MaskSchemeRegistry from plain dict config."""
    if cfg is None:
        raise ValueError("mask_registry config must be provided.")
    schemes_cfg = cfg.get("schemes", [])
    if not schemes_cfg:
        raise ValueError("mask_registry config must include a non-empty 'schemes' list.")

    default_repr = cfg.get("representation_config")
    reg = MaskSchemeRegistry()
    for scheme in schemes_cfg:
        # Parse per-modality overrides
        modalities_cfg = scheme.get("modalities", {}) or {}
        modalities = {}
        for mod_name, mod_spec in modalities_cfg.items():
            modalities[mod_name] = ModalityMaskSpec(
                visible=mod_spec.get("visible"),
                visible_by_t=mod_spec.get("visible_by_t"),
                masked_dims=mod_spec.get("masked_dims"),
                visible_ids=mod_spec.get("visible_ids"),
                visible_ids_by_t=mod_spec.get("visible_ids_by_t"),
                mask_value=mod_spec.get("mask_value", 0.0),
                description=mod_spec.get("description"),
            )

        reg.add(
            MaskSchemeSpec(
                name=scheme["name"],
                topology=scheme["topology"],
                visible_ids=scheme.get("visible_ids", []),
                visible_ids_by_t=scheme.get("visible_ids_by_t"),
                representation_config=scheme.get("representation_config", default_repr),
                modalities=modalities,
                allow_time_variation=scheme.get("allow_time_variation", False),
                description=scheme.get("description"),
            )
        )
    return reg
