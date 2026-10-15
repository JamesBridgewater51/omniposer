from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class TopologySpec:
    """Defines a joint topology (e.g., SMPL/SMPLX) used by modalities."""
    name: str
    joint_names: Optional[List[str]] = None
    joint_count: Optional[int] = None


@dataclass(frozen=True)
class ModalitySpec:
    """
    Describes a single representation stream.
    - name: unique identifier (e.g., 'smplx_rot_6d')
    - dim: flattened dimensionality
    - is_joint_based: whether the stream is per-joint and needs per-joint masking/interleaving
    - per_joint_dim: dimensionality per joint if joint-based
    - topology: topology name this modality follows
    - semantic: high-level role (e.g., 'trajectory', 'pose', 'shape', 'contact', 'sensor')
    """
    name: str
    dim: int
    is_joint_based: bool
    per_joint_dim: Optional[int] = None
    topology: Optional[str] = None
    semantic: Optional[str] = None


@dataclass(frozen=True)
class RoutingGroupSpec:
    """Named non-overlapping joint groups for metadata-routed body-part MoE."""
    name: str
    topology: str
    groups: Dict[str, List[int]]


@dataclass
class RepresentationRegistry:
    """
    Registry of modalities and topologies for a dataset/run.
    No defaults or fallbacks are provided: callers must populate all fields explicitly.
    """
    modalities: List[ModalitySpec] = field(default_factory=list)
    topologies: Dict[str, TopologySpec] = field(default_factory=dict)
    traj_modalities: List[str] = field(default_factory=list)  # ordered names for trajectory slice
    cond_modalities: Dict[str, List[str]] = field(default_factory=dict)  # keyed by task ('traj','pose','both')
    predicted_modalities: Dict[str, List[str]] = field(default_factory=dict)  # keyed by task ('traj','pose')
    routing_groups: Dict[str, RoutingGroupSpec] = field(default_factory=dict)

    def get_modality(self, name: str) -> ModalitySpec:
        for m in self.modalities:
            if m.name == name:
                return m
        raise KeyError(f"Modality '{name}' not found in registry.")

    def joint_count_for_topology(self, topology_name: Optional[str]) -> Optional[int]:
        if topology_name is None:
            return None
        topo = self.topologies[topology_name]
        return topo.joint_count

    def validate_routing_group(self, name: str, *, expected_topology: Optional[str] = None) -> RoutingGroupSpec:
        if name not in self.routing_groups:
            raise KeyError(f"Routing group '{name}' not found in registry.")
        spec = self.routing_groups[name]
        if expected_topology is not None and spec.topology != expected_topology:
            raise ValueError(
                f"Routing group '{name}' uses topology '{spec.topology}', expected '{expected_topology}'."
            )
        if spec.topology not in self.topologies:
            raise ValueError(f"Routing group '{name}' references unknown topology '{spec.topology}'.")
        joint_count = self.topologies[spec.topology].joint_count
        if joint_count is None:
            raise ValueError(f"Topology '{spec.topology}' must define joint_count for routing group '{name}'.")
        if not spec.groups:
            raise ValueError(f"Routing group '{name}' must contain at least one group.")
        seen = set()
        for group_name, indices in spec.groups.items():
            if not indices:
                raise ValueError(f"Routing group '{name}.{group_name}' is empty.")
            for idx in indices:
                if idx < 0 or idx >= joint_count:
                    raise ValueError(
                        f"Routing group '{name}.{group_name}' contains joint index {idx} outside [0, {joint_count})."
                    )
                if idx in seen:
                    raise ValueError(f"Routing group '{name}' contains overlapping joint index {idx}.")
                seen.add(idx)
        return spec

    def get_routing_group(self, name: str, *, expected_topology: Optional[str] = None) -> RoutingGroupSpec:
        return self.validate_routing_group(name, expected_topology=expected_topology)

    def get_routing_group_mapping(self, name: str, *, expected_topology: Optional[str] = None) -> Dict[str, List[int]]:
        spec = self.validate_routing_group(name, expected_topology=expected_topology)
        return {group_name: list(indices) for group_name, indices in spec.groups.items()}

    def validate_all_routing_groups(self) -> None:
        for name in self.routing_groups:
            self.validate_routing_group(name)
