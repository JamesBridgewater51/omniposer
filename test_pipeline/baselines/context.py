"""
Inference context - shared state container passed through all hooks.

The InferenceContext provides a single point of access to all resources needed
during inference, avoiding the need to pass many individual parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from ..inference.normalization import NormalizationStats
    from utils.representation_registry import RepresentationRegistry
    from utils.mask_registry import MaskSchemeRegistry

from utils.masking_actions import build_mask_action_pool


@dataclass
class InferenceContext:
    """
    Shared context passed through all hooks during inference.
    
    This dataclass aggregates all resources needed for inference into a single
    container, providing clean access to configs, models, and runtime state.
    
    The context is created once per TestRunner and reused across all batches.
    Per-batch state (current_batch, current_masking_spec) is updated during processing.
    
    Attributes:
        train_config: Training configuration loaded from logdir/config.yaml.
            Contains model architectures, representation registry, mask registry, etc.
            
        test_config: Test configuration from Hydra.
            Contains test_mask_action, num_inference_steps, visualization settings, etc.
            
        dataset: Test dataset instance.
            Provides access to norm_params, _repr_indices, _repr_list, etc.
            
        repr_registry: Representation registry built from train_config.
            Defines modalities, topologies, and predicted features.
            
        mask_scheme_registry: Mask scheme registry built from train_config.
            Defines visible joint IDs for each mask scheme.
            
        model_trajnet: TrajNet model instance (loaded from checkpoint).
        model_posenet: PoseNet model instance (loaded from checkpoint).
        diffusion_trajnet: TrajNet diffusion/training model wrapper.
        diffusion_posenet: PoseNet diffusion/training model wrapper.
        smplx_model: SMPL-X body model from PoseNet.
        fk_model: Forward kinematics model (smplx or articulate).
        
        device: Target torch device for inference.
        clip_len: Training clip length from train_config.
        window_size: Sliding window size (clip_len - 1).
        
        norm_stats: Normalization statistics for motion_repr and cond tensors.
        
        current_batch: Current batch being processed (set per-batch).
        current_masking_spec: Current masking specification tuple for the batch/window.
        sampling_progress_entries: Optional sampling traces collected by hooks.
    """
    # Configs
    train_config: DictConfig
    test_config: DictConfig
    
    # Dataset and registries
    dataset: Any
    repr_registry: 'RepresentationRegistry'
    mask_scheme_registry: 'MaskSchemeRegistry'
    masking_action_pool: Dict[str, Dict[str, Any]]
    
    # Models
    model_trajnet: torch.nn.Module
    model_posenet: torch.nn.Module
    diffusion_trajnet: Any
    diffusion_posenet: Any
    smplx_model: Any
    fk_model: Any
    
    # Runtime parameters
    device: torch.device
    clip_len: int
    window_size: int
    
    # Normalization stats
    norm_stats: 'NormalizationStats'
    
    # Per-batch state (mutable)
    current_batch: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    current_masking_spec: Optional[tuple] = field(default=None, repr=False)
    # Backward-compatible alias for older hooks that looked at only the action.
    current_mask_action: Optional[Dict[str, Any]] = field(default=None, repr=False)
    sampling_progress_entries: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    
    # Derived properties
    @property
    def traj_repr_name_list(self):
        """Trajectory representation names from dataset."""
        return self.dataset.traj_repr_name_list
    
    @property
    def repr_indices(self):
        """Representation indices from dataset."""
        return self.dataset._repr_indices
    
    @property
    def repr_list(self):
        """Representation list from dataset."""
        return self.dataset._repr_list
    
    @property
    def joints_num(self):
        """Number of joints from dataset."""
        return self.dataset.joints_num
    
    @property
    def norm_strategy(self):
        """Normalization strategy from dataset."""
        return self.dataset.norm_strategy
    
    @property
    def input_noise(self) -> bool:
        """Whether input noise is enabled in training config."""
        return self.train_config.representation.input_noise
    
    @property
    def include_global_joint_rotations(self) -> bool:
        """Whether global joint rotations are in the representation."""
        return 'R_world_joints' in self.repr_list
    
    @property
    def repr_abs_only(self) -> bool:
        """Whether representation uses absolute values only (no velocities)."""
        return 'local_vel' not in self.repr_list
    
    def resolve_mask_action(self) -> Dict[str, Any]:
        """Resolve the configured test mask action to a concrete action dict or per-sample plan."""
        from utils.masking_actions import materialize_mask_action, materialize_per_sample_action_plan

        if not self.masking_action_pool:
            raise ValueError("masking.action_pool is empty; check train_config.masking.actions.")
        batch_policy = getattr(self.test_config, "test_batch_policy", "whole_batch")
        if batch_policy == "per_sample":
            action_names = getattr(self.test_config, "test_mask_actions", None)
            if not action_names:
                action_name = getattr(self.test_config, "test_mask_action", None)
                if not action_name:
                    raise ValueError("test_mask_actions or test_mask_action must be specified for per_sample test masking.")
                action_names = [action_name]
            batch_size = 1
            if self.current_batch is not None and "motion_repr_clean" in self.current_batch:
                batch_size = self.current_batch["motion_repr_clean"].shape[0]
            assignments = [action_names[i % len(action_names)] for i in range(batch_size)]
            return materialize_per_sample_action_plan(assignments, self.masking_action_pool, self.joints_num, batch_size)
        action_name = getattr(self.test_config, "test_mask_action", None)
        if not action_name:
            raise ValueError("test_mask_action must be specified for test masking.")
        return materialize_mask_action(action_name, self.masking_action_pool, self.joints_num)

    def get_visible_joint_ids(self) -> list:
        """Get visible joint IDs for current mask action."""
        from utils.masking_actions import visible_ids_from_action

        action = self.resolve_mask_action()
        if isinstance(action, list):
            return list(range(self.joints_num))
        visible_ids = visible_ids_from_action(action, self.mask_scheme_registry, self.joints_num)
        if visible_ids is None:
            return list(range(self.joints_num))
        return list(visible_ids)
    
    def get_occluded_joint_ids(self) -> list:
        """Get occluded joint IDs for current mask action."""
        vis_ids = set(self.get_visible_joint_ids())
        return [i for i in range(self.joints_num) if i not in vis_ids]


def create_inference_context(
    train_config: DictConfig,
    test_config: DictConfig,
    dataset: Any,
    repr_registry: 'RepresentationRegistry',
    mask_scheme_registry: 'MaskSchemeRegistry',
    models: tuple,
    diffusions: tuple,
    smplx_model: Any,
    fk_model: Any,
    device: torch.device,
    clip_len: int,
    norm_stats: 'NormalizationStats',
) -> InferenceContext:
    """
    Factory function to create an InferenceContext.
    
    This function provides a convenient way to create an InferenceContext
    from TestRunner components.
    
    Args:
        train_config: Training configuration from logdir
        test_config: Test configuration from Hydra
        dataset: Test dataset instance
        repr_registry: Representation registry
        mask_scheme_registry: Mask scheme registry
        models: Tuple of (model_posenet, model_trajnet, model_trajnet_control)
        diffusions: Tuple of (diffusion_posenet, diffusion_trajnet, diffusion_trajnet_control)
        smplx_model: SMPL-X body model
        fk_model: Forward kinematics model
        device: Target device
        clip_len: Training clip length
        norm_stats: Normalization statistics
        
    Returns:
        Configured InferenceContext instance
    """
    model_posenet, model_trajnet, _ = models
    diffusion_posenet, diffusion_trajnet, _ = diffusions
    
    masking_action_pool = build_mask_action_pool(
        OmegaConf.to_container(train_config.masking, resolve=True)
    )

    return InferenceContext(
        train_config=train_config,
        test_config=test_config,
        dataset=dataset,
        repr_registry=repr_registry,
        mask_scheme_registry=mask_scheme_registry,
        masking_action_pool=masking_action_pool,
        model_trajnet=model_trajnet,
        model_posenet=model_posenet,
        diffusion_trajnet=diffusion_trajnet,
        diffusion_posenet=diffusion_posenet,
        smplx_model=smplx_model,
        fk_model=fk_model,
        device=device,
        clip_len=clip_len,
        window_size=clip_len - 1,
        norm_stats=norm_stats,
    )
