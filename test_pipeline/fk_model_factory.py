"""
Factory and lightweight wrappers for selecting a forward-kinematics (FK)
implementation at test time.

Supports the default smplx body model as well as custom implementations such as
third_party/RobustCap/articulate.ParametricModel.
"""
from dataclasses import dataclass
from typing import Any, Optional
import os
import numpy as np
import torch

from utils.konia_transform import angle_axis_to_rotation_matrix
from third_party.RobustCap.articulate.model import ParametricModel
from .fk_adapters import build_body_pose_adapter, BodyPoseAdapter


@dataclass
class FKResult:
    """Simple container that mimics the smplx output fields we consume."""
    joints: torch.Tensor
    vertices: Optional[torch.Tensor] = None


class ArticulateFKModel:
    """FK wrapper around articulate.ParametricModel with a smplx-like API."""

    def __init__(
        self,
        model_path: str,
        use_pose_blendshape: bool,
        device: torch.device,
        pose_adapter: Optional[BodyPoseAdapter] = None,
    ):
        self.model = ParametricModel(
            official_model_file=model_path,
            use_pose_blendshape=use_pose_blendshape,
            device=device,
        )
        # Keep faces aligned with smplx usages in the pipeline
        self.faces = np.asarray(self.model.face)
        self.device = device
        self.pose_adapter = pose_adapter or build_body_pose_adapter("articulate")

    def __call__(self, *args, **kwargs) -> FKResult:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> FKResult:
        # Match smplx signature: return_verts defaults to True
        return_verts = kwargs.pop("return_verts", True)

        global_orient = kwargs.get("global_orient")
        body_pose = kwargs.get("body_pose")
        transl = kwargs.get("transl", None)
        betas = kwargs.get("betas", None)

        if global_orient is None or body_pose is None:
            raise ValueError("ArticulateFKModel expects global_orient and body_pose in axis-angle format")

        batch_size = global_orient.shape[0]
        device = global_orient.device

        body_pose_aa = body_pose.view(batch_size, -1, 3)
        body_pose_aa = self.pose_adapter.adapt(body_pose_aa)

        full_pose_aa = torch.cat([global_orient.view(batch_size, 1, 3), body_pose_aa], dim=1)
        rot_mats = angle_axis_to_rotation_matrix(full_pose_aa.reshape(-1, 3)).view(batch_size, -1, 3, 3)

        _, joints, verts = self.model.forward_kinematics(
            pose=rot_mats,
            shape=betas,
            tran=transl,
            calc_mesh=return_verts,
        )

        return FKResult(joints=joints, vertices=verts if return_verts else None)


ARTICULATE_DATASETS = {"aist", "totalcapture", "3dpw"}
DEFAULT_ARTICULATE_MODEL = os.path.join("data", "body_models", "smpl_model", "models", "SMPL_male.pkl")


def resolve_fk_model_type(dataset_name: str, configured_type: str) -> str:
    """Resolve fk model type using dataset-specific defaults when set to auto."""
    fk_type = (configured_type or "auto").lower()
    if fk_type == "auto":
        return "articulate" if dataset_name.lower() in ARTICULATE_DATASETS else "smplx"
    return fk_type


def build_fk_model(
    dataset_name: str,
    test_config: Any,
    base_smplx_model: Any,
    device: torch.device,
):
    """
    Build the FK model wrapper based on test-time configuration.

    Args:
        dataset_name: current dataset name
        test_config: omegaconf DictConfig or dataclass with fk_model_* fields
        base_smplx_model: smplx instance constructed with training config (for smplx path)
        device: target torch.device
    """
    fk_type = resolve_fk_model_type(dataset_name, getattr(test_config, "fk_model_type", "auto"))

    if fk_type == "smplx":
        return base_smplx_model, fk_type

    model_path = getattr(test_config, "fk_model_path", None) or DEFAULT_ARTICULATE_MODEL
    use_pose_blendshape = bool(getattr(test_config, "fk_model_use_pose_blendshape", True))
    pose_adapter = build_body_pose_adapter(
        fk_type,
        fill_strategy=getattr(test_config, "fk_missing_joint_fill_strategy", "identity"),
    )

    articulate_model = ArticulateFKModel(
        model_path=model_path,
        use_pose_blendshape=use_pose_blendshape,
        device=device,
        pose_adapter=pose_adapter,
    )
    return articulate_model, fk_type
