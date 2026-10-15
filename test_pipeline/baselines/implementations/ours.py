"""
"Ours" baseline implementation - the reference implementation for our method.

This module extracts the current inference logic from BatchProcessor into
the hook-based architecture. It serves as the reference implementation
that other baselines can compare against.

The implementation preserves all existing logic exactly, just reorganized
into the hook structure for consistency with other baselines.
"""
import torch
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..registry import BaselineRegistry
from ..spec import BaselineSpec
from ..context import InferenceContext
from ..hooks.base import PreprocessingHook, InferenceHook, PostprocessingHook, MetricsHook


def _coerce_masking_spec(masking_spec_or_action: Any) -> Tuple[Any, Optional[List[int]]]:
    """Accept both the current masking spec tuple and the legacy bare action."""
    if (
        isinstance(masking_spec_or_action, tuple)
        and len(masking_spec_or_action) == 2
    ):
        return masking_spec_or_action
    return masking_spec_or_action, None


class OursPreprocessingHook(PreprocessingHook):
    """
    Preprocessing hook for our method.
    
    Extracts logic from BatchProcessor.process_batch() and related methods
    for preparing conditioning tensors and applying masking.
    """
    
    def prepare_conditioning(
        self,
        batch: Dict[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare trajectory and pose conditioning batches.
        
        Uses process_batch_cond() from utils.batch_util to create
        conditioning tensors for both TrajNet and PoseNet.
        """
        from utils.batch_util import process_batch_cond
        
        traj_batch, pose_batch = process_batch_cond(
            batch=batch,
            ds=ctx.dataset,
            mask_scheme=None,
            task='both',
            input_noise=ctx.input_noise,
            num_joints=ctx.joints_num,
        )
        
        return traj_batch, pose_batch
    
    def apply_masking(
        self,
        traj_batch: Dict[str, torch.Tensor],
        pose_batch: Dict[str, torch.Tensor],
        masking_spec: Tuple[Any, Optional[List[int]]],
        ctx: InferenceContext,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply test-time masking to conditioning batches.
        
        Supports both fixed mask schemes from registry and EgoExo-style
        probabilistic masking.
        """
        from utils.masking_actions import apply_mask_action_plan

        mask_action, visible_ids_override = _coerce_masking_spec(masking_spec)
        if visible_ids_override is not None:
            mask_action = dict(mask_action or {})
            mask_action["visible_ids"] = list(visible_ids_override)
        batch_policy = getattr(ctx.test_config, "test_batch_policy", "whole_batch")
        apply_mask_action_plan(traj_batch, ctx.dataset, "traj", mask_action, ctx.joints_num, batch_policy)
        apply_mask_action_plan(pose_batch, ctx.dataset, "pose", mask_action, ctx.joints_num, batch_policy)
        
        return traj_batch, pose_batch


class OursInferenceHook(InferenceHook):
    """
    Inference hook for our method.
    
    Extracts logic from BatchProcessor._run_trajnet(), _run_posenet(),
    and trajectory injection into the hook interface.
    """
    
    def run_trajnet(
        self,
        traj_batch: Dict[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        """
        Run TrajNet inference.
        
        If eval_on_posenet_only is set, returns ground truth trajectory
        features instead of running the model.
        """
        from utils.feature_accessor import select_modalities, get_modality_dim
        
        if ctx.test_config.eval_on_posenet_only:
            return select_modalities(
                traj_batch['motion_repr_clean'],
                ctx.traj_repr_name_list,
                ctx.repr_indices
            )
        
        bs, clip_len = traj_batch['cond'].shape[:2]
        traj_feat_dim = get_modality_dim(ctx.traj_repr_name_list, ctx.repr_indices)
        shape = [bs, clip_len, traj_feat_dim]
        
        return ctx.diffusion_trajnet.p_sample_loop(
            model=ctx.model_trajnet,
            batch=traj_batch,
            shape=shape,
            num_inference_steps=ctx.test_config.num_inference_steps,
            progress=ctx.test_config.dist_rank == 0,
        )
    
    def inject_trajectory(
        self,
        pose_batch: Dict[str, torch.Tensor],
        traj_output: torch.Tensor,
        ctx: InferenceContext,
    ) -> Dict[str, torch.Tensor]:
        """
        Inject trajectory output into pose conditioning.
        
        Updates the pose conditioning tensor with predicted trajectory
        features (smplx_trans, smplx_rot_6d) in the global section.
        """
        from utils.feature_accessor import insert_modalities_into_cond
        
        pose_batch['cond'] = insert_modalities_into_cond(
            pose_batch['cond'],
            traj_output,
            ctx.traj_repr_name_list,
            ctx.dataset,
            task='pose'
        )
        
        return pose_batch
    
    def run_posenet(
        self,
        pose_batch: Dict[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        """
        Run PoseNet inference.
        
        If eval_on_trajnet_only is set, returns ground truth motion
        representation instead of running the model.
        """
        if ctx.test_config.eval_on_trajnet_only:
            return pose_batch['motion_repr_clean']
        
        from utils.feature_accessor import get_modality_dim

        bs, clip_len = pose_batch['cond'].shape[:2]
        pose_predicted = ctx.repr_registry.predicted_modalities.get('pose', [])
        if not pose_predicted:
            raise ValueError("predicted_modalities['pose'] is empty; cannot infer PoseNet sampling shape.")
        pose_feat_dim = get_modality_dim(pose_predicted, ctx.repr_indices)
        shape = [bs, clip_len, pose_feat_dim]
        
        result = ctx.diffusion_posenet.p_sample_loop(
            model=ctx.model_posenet,
            batch=pose_batch,
            shape=shape,
            num_inference_steps=ctx.test_config.num_inference_steps,
            progress=ctx.test_config.dist_rank == 0,
        )
        
        if isinstance(result, tuple):
            result = result[0]
        
        return result


class OursPostprocessingHook(PostprocessingHook):
    """
    Postprocessing hook for our method.
    
    Extracts logic from BatchProcessor._reconstruct_results() and
    related methods for output reconstruction.
    """
    
    def reconstruct_repr_dict(
        self,
        pose_output: torch.Tensor,
        ctx: InferenceContext,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct representation dictionary from model output.
        
        Splits the output tensor into individual representation components
        and populates derived representations (e.g., global rotations).
        """
        from data_sets.motion_representation import populate_repr_dict
        
        # Build repr_dict from output
        repr_dict = {}
        for repr_name in ctx.repr_list:
            s, e = ctx.repr_indices[repr_name]
            repr_dict[repr_name] = pose_output[..., s:e].to(ctx.device)
        
        # Populate derived representations
        pose_predicted = ctx.repr_registry.predicted_modalities.get('pose', [])
        include_global_joint_rotations = (
            'R_world_joints' in pose_predicted or 
            'R_root_joints' in pose_predicted
        )
        
        if include_global_joint_rotations:
            repr_dict = populate_repr_dict(repr_dict, ctx.smplx_model, recover_global=False)
        else:
            repr_dict = populate_repr_dict(repr_dict, ctx.smplx_model, recover_global=True)
        
        return repr_dict
    
    def recover_world_frame(
        self,
        repr_dict: Dict[str, torch.Tensor],
        original_batch: Dict[str, Any],
        ctx: InferenceContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover world-frame joints and vertices.
        
        Runs forward kinematics using the FK model to obtain 3D joint
        positions and mesh vertices.
        """
        from data_sets.motion_representation import recover_from_repr_smpl

        seq_len = next(iter(repr_dict.values())).shape[1]
        chunk_size = int(getattr(ctx.test_config, "postprocess_fk_chunk_size", 512))
        chunk_size = max(1, chunk_size)
        if seq_len <= chunk_size:
            joints, verts = recover_from_repr_smpl(
                repr_dict,
                recover_mode='smplx_params',
                # NOTE: `cano_jts_pos` argument had better follow train config's value.
                cano_jts_pos=False,
                smplx_model=ctx.fk_model,
                return_verts=True,
                return_full_joints=False,
            )
            return joints, verts

        joint_chunks = []
        vert_chunks = []
        for start_t in range(0, seq_len, chunk_size):
            end_t = min(start_t + chunk_size, seq_len)
            repr_chunk = {
                name: value[:, start_t:end_t, ...] if torch.is_tensor(value) and value.dim() >= 3 else value
                for name, value in repr_dict.items()
            }
            joints, verts = recover_from_repr_smpl(
                repr_chunk,
                recover_mode='smplx_params',
                cano_jts_pos=False,
                smplx_model=ctx.fk_model,
                return_verts=True,
                return_full_joints=False,
            )
            joint_chunks.append(joints)
            vert_chunks.append(verts)
        return torch.cat(joint_chunks, dim=1), torch.cat(vert_chunks, dim=1)


class OursMetricsHook(MetricsHook):
    """
    Metrics hook for our method.
    
    Computes standard motion reconstruction metrics including MPJPE,
    PA-MPJPE, acceleration error, and physics-based metrics.
    """
    
    def get_metric_names(self) -> List[str]:
        """Return list of metrics computed by this hook."""
        from utils.evaluation import get_metric_names_for_baseline
        return get_metric_names_for_baseline("ours")
    
    def compute_metrics(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        repr_dict_rec: Dict[str, torch.Tensor],
        repr_dict_clean: Dict[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> Dict[str, float]:
        """
        Compute all metrics for our method.
        
        Uses MotionEvaluator from utils.evaluation to compute standard
        metrics with visible/occluded joint separation.
        """
        from utils.evaluation import compute_metrics_for_baseline

        vis_ids = ctx.get_visible_joint_ids()
        occ_ids = ctx.get_occluded_joint_ids()

        return compute_metrics_for_baseline(
            baseline_name="ours",
            pred_joints=pred_joints,
            gt_joints=gt_joints,
            vis_joint_ids=vis_ids,
            occ_joint_ids=occ_ids,
            repr_dict_rec=repr_dict_rec,
            repr_dict_clean=repr_dict_clean,
            fps=30,
        )


# Register the baseline
@BaselineRegistry.register('ours')
def create_ours_spec() -> BaselineSpec:
    """Create the specification for our baseline method."""
    return BaselineSpec(
        name='ours',
        
        # Compatible with all mask schemes we support
        compatible_mask_schemes=[
            'rohm_upper_body_xyz',
            'egoego_head_xyz_rot',
            'imu_6_root_leaf_xyz_rot',
            'imu_5_leaf_xyz_rot',
            'imu_head_hands_root_xyz_rot',
            'imu_xsens_full_xyz_rot',
            'egoexo_prior',
            'random_joints_max15',
        ],
        default_mask_scheme='rohm_upper_body_xyz',
        
        # Representation requirements - supports both
        required_representation_config='xyz_only',  # Default, can work with xyz_rot too
        requires_rotation_input=False,
        
        # Inference characteristics
        uses_sliding_window=True,
        supports_variable_length=True,
        
        # Hook implementations
        preprocessing_hook_cls=OursPreprocessingHook,
        inference_hook_cls=OursInferenceHook,
        postprocessing_hook_cls=OursPostprocessingHook,
        metrics_hook_cls=OursMetricsHook,
        
        # Metrics to compute
        metrics=[
            'mpjpe_global_mm',
            'mpjpe_visible_mm',
            'mpjpe_occluded_mm',
            'pa_mpjpe_global_mm',
            'acceleration_error_ms2',
            'skating_ratio_rec',
            'ground_penetration_freq_percent',
        ],
    )
