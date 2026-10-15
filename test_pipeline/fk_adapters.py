"""
Adapters for mapping model-specific body pose outputs to target kinematic
layouts expected by FK implementations.

Designed to be extensible and decoupled from concrete FK models.
"""
from typing import List
import torch


class BodyPoseAdapter:
    """Base class for adapting body pose tensors before FK."""

    def adapt(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            body_pose_aa: (..., J, 3) axis-angle local rotations.
        Returns:
            Tensor with potentially different joint dimension.
        """
        raise NotImplementedError


class IdentityAdapter(BodyPoseAdapter):
    """Returns input unchanged."""

    def adapt(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        return body_pose_aa


class Smpl21ToSmpl23Adapter(BodyPoseAdapter):
    """Maps 21-joint SMPL body pose to 23-joint layout expected by articulate."""

    def __init__(self, fill_strategy: str = "identity"):
        self.fill_strategy = fill_strategy
        # Default mapping assumes source joints align with SMPL 0-21 (up to wrists).
        self.source_to_target: List[int] = list(range(21))  # 0..20 map directly
        # Target has two extra joints at indices 22 and 23 (hands/palms)
        self.missing_indices = [21, 22]

    def adapt(self, body_pose_aa: torch.Tensor) -> torch.Tensor:
        # Expect (..., 22, 3) or already (..., 24, 3)
        if body_pose_aa.shape[-2] == 23:
            return body_pose_aa
        if body_pose_aa.shape[-2] != 21:
            raise ValueError(
                f"Smpl22ToSmpl24Adapter expects 22 joints, got {body_pose_aa.shape[-2]}"
            )

        *prefix, _, _ = body_pose_aa.shape
        device = body_pose_aa.device
        dtype = body_pose_aa.dtype

        out = torch.zeros(*prefix, 23, 3, device=device, dtype=dtype)
        # Map known joints directly
        out[..., self.source_to_target, :] = body_pose_aa

        # Fill missing joints
        if self.fill_strategy == "identity":
            # leave zeros (identity in axis-angle)
            pass
        elif self.fill_strategy == "copy_wrists":
            left_wrist, right_wrist = 19, 20
            out[..., 21, :] = body_pose_aa[..., left_wrist, :]
            out[..., 22, :] = body_pose_aa[..., right_wrist, :]
        else:
            raise ValueError(f"Unknown fill_strategy: {self.fill_strategy}")

        return out


def build_body_pose_adapter(fk_type: str, fill_strategy: str = "identity") -> BodyPoseAdapter:
    fk_type = fk_type.lower()
    if fk_type == "articulate":
        return Smpl21ToSmpl23Adapter(fill_strategy=fill_strategy)
    return IdentityAdapter()
