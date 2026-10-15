from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set


@dataclass(frozen=True)
class ModalityMaskSpec:
    """
    Per-modality masking override for conditioning tensors.

    For joint-based modalities:
      - visible_ids: which joints are visible (masked = complement)
      - visible_ids_by_t: optional per-timestep visible joints (len == T)

    For global modalities:
      - visible: mask/unmask the whole modality (True=visible, False=masked)
      - visible_by_t: optional per-timestep visibility (len == T)
      - masked_dims: optional indices within the modality to mask

    Common:
      - mask_value: value written into masked features (default 0.0)
    """
    # For global modalities
    visible: Optional[bool] = None
    visible_by_t: Optional[Sequence[bool]] = None
    masked_dims: Optional[Sequence[int]] = None
    # For joint-based modalities
    visible_ids: Optional[Sequence[int]] = None
    visible_ids_by_t: Optional[Sequence[Sequence[int]]] = None
    # Common
    mask_value: float = 0.0
    description: Optional[str] = None


@dataclass(frozen=True)
class MaskSchemeSpec:
    """
    Describes a masking scheme for a specific topology.
    - name: scheme identifier
    - topology: topology name this scheme applies to
    - representation_config: name of representation registry config to use (e.g., "xyz_only", "xyz_rot")
    - visible_ids: default visible joint ids (masked joints are complement)
    - visible_ids_by_t: optional default visible joints per timestep
    - modalities: per-modality overrides (joint-based or global)
    - allow_time_variation: whether scheme can change per timestep
    - description: optional human-readable notes
    """
    name: str
    topology: str
    visible_ids: Sequence[int] = field(default_factory=list)
    visible_ids_by_t: Optional[Sequence[Sequence[int]]] = None
    modalities: Mapping[str, ModalityMaskSpec] = field(default_factory=dict)
    allow_time_variation: bool = False
    representation_config: Optional[str] = None
    description: Optional[str] = None

    def modality_spec(self, modality_name: str) -> Optional[ModalityMaskSpec]:
        """Get the mask spec for a specific modality, or None if not overridden."""
        return self.modalities.get(modality_name)


@dataclass
class MaskSchemeRegistry:
    """
    Registry of mask schemes keyed by (topology, name).
    """
    schemes: Dict[str, MaskSchemeSpec] = field(default_factory=dict)

    def get(self, name: str) -> MaskSchemeSpec:
        if name not in self.schemes:
            raise KeyError(f"Mask scheme '{name}' not found.")
        return self.schemes[name]

    def add(self, scheme: MaskSchemeSpec) -> None:
        if scheme.name in self.schemes:
            raise ValueError(f"Duplicate mask scheme name: {scheme.name}")
        self.schemes[scheme.name] = scheme

    def has(self, name: str) -> bool:
        return name in self.schemes

    def visible_ids(self, name: str) -> Set[int]:
        return set(self.get(name).visible_ids)

    def validate_scheme(self, name: str, repr_registry=None, task: Optional[str] = None) -> None:
        scheme = self.get(name)
        if repr_registry is None:
            return

        topology_count = repr_registry.joint_count_for_topology(scheme.topology)
        if topology_count is None:
            raise ValueError(f"Mask scheme '{name}' topology '{scheme.topology}' has no joint_count.")

        def _validate_ids(ids, label: str) -> None:
            if ids is None:
                return
            bad_ids = [int(i) for i in ids if int(i) < 0 or int(i) >= topology_count]
            if bad_ids:
                raise ValueError(
                    f"Mask scheme '{name}' {label} has out-of-range joints {bad_ids}; "
                    f"expected [0, {topology_count - 1}]."
                )

        _validate_ids(scheme.visible_ids, "visible_ids")
        if scheme.visible_ids_by_t is not None:
            for t_idx, ids in enumerate(scheme.visible_ids_by_t):
                _validate_ids(ids, f"visible_ids_by_t[{t_idx}]")

        cond_names = set(repr_registry.cond_modalities.get(task, [])) if task else None
        for modality_name, modality_rule in scheme.modalities.items():
            try:
                modality = repr_registry.get_modality(modality_name)
            except KeyError as exc:
                raise ValueError(
                    f"Mask scheme '{name}' references modality '{modality_name}' missing from representation registry."
                ) from exc
            if cond_names is not None and modality_name not in cond_names:
                continue
            if modality.is_joint_based:
                if modality.topology != scheme.topology:
                    raise ValueError(
                        f"Mask scheme '{name}' modality '{modality_name}' topology '{modality.topology}' "
                        f"does not match scheme topology '{scheme.topology}'."
                    )
                _validate_ids(modality_rule.visible_ids, f"modalities.{modality_name}.visible_ids")
                if modality_rule.visible_ids_by_t is not None:
                    for t_idx, ids in enumerate(modality_rule.visible_ids_by_t):
                        _validate_ids(ids, f"modalities.{modality_name}.visible_ids_by_t[{t_idx}]")
            elif modality_rule.visible_ids is not None or modality_rule.visible_ids_by_t is not None:
                raise ValueError(
                    f"Mask scheme '{name}' uses joint visible_ids for non-joint modality '{modality_name}'."
                )
            if modality_rule.masked_dims is not None:
                bad_dims = [int(i) for i in modality_rule.masked_dims if int(i) < 0 or int(i) >= modality.dim]
                if bad_dims:
                    raise ValueError(
                        f"Mask scheme '{name}' modality '{modality_name}' has out-of-range masked_dims {bad_dims}; "
                        f"expected [0, {modality.dim - 1}]."
                    )

    def validate_all(self, repr_registry=None, task: Optional[str] = None) -> None:
        for name in self.schemes:
            self.validate_scheme(name, repr_registry=repr_registry, task=task)
