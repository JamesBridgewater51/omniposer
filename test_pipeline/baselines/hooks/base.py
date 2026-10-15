"""
Abstract base classes for baseline hooks.

All baselines must implement these four hook types. There are no default implementations -
each baseline (including "ours") must provide concrete implementations for all hooks.

Hook Execution Order:
1. PreprocessingHook.prepare_conditioning() - Transform raw batch to model inputs
2. PreprocessingHook.apply_masking() - Apply test-time masking
3. InferenceHook.run_trajnet() - Run trajectory network
4. InferenceHook.inject_trajectory() - Inject trajectory into pose conditioning
5. InferenceHook.run_posenet() - Run pose network
6. PostprocessingHook.reconstruct_repr_dict() - Build representation dictionary
7. PostprocessingHook.recover_world_frame() - Recover world-frame joints/vertices
8. MetricsHook.compute_metrics() - Compute evaluation metrics (after all batches)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from ..context import InferenceContext


class PreprocessingHook(ABC):
    """
    Abstract base class for preprocessing hooks.
    
    Preprocessing hooks transform raw batches from the dataloader into
    the format expected by the inference pipeline. This includes:
    - Preparing conditioning tensors for TrajNet and PoseNet
    - Applying test-time masking based on the mask scheme
    
    Each baseline may have different conditioning requirements (e.g., head-only
    for egocentric methods, IMU sensors for PIP, etc.).
    """
    
    @abstractmethod
    def prepare_conditioning(
        self,
        batch: Dict[str, torch.Tensor],
        ctx: 'InferenceContext',
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare trajectory and pose conditioning batches from raw batch.
        
        This method transforms the raw batch from the dataloader into two
        conditioning dictionaries: one for TrajNet and one for PoseNet.
        
        Args:
            batch: Raw batch from dataloader containing:
                - 'motion_repr_clean': Clean motion representation [bs, T, feat_dim]
                - 'motion_repr_noisy': Noisy motion representation (if input_noise enabled)
                - 'meta': Batch metadata dictionary
                
            ctx: InferenceContext with access to dataset, configs, and models
            
        Returns:
            Tuple of (traj_batch, pose_batch) where each contains:
                - 'motion_repr_clean': Motion representation tensor
                - 'cond': Conditioning tensor for the network
                - 'cond_mask': Temporal mask for conditioning
                - 'cond_joint_mask': Per-joint visibility mask
                - Additional baseline-specific keys as needed
        """
        ...
    
    def apply_masking(
        self,
        traj_batch: Dict[str, torch.Tensor],
        pose_batch: Dict[str, torch.Tensor],
        masking_spec: Tuple[Any, Optional[List[int]]],
        ctx: 'InferenceContext',
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply test-time masking to conditioning batches.
        
        This method applies the specified masking action to both trajectory
        and pose conditioning batches. The action determines which joints
        are visible (conditioning) vs occluded (to be predicted).
        
        Args:
            traj_batch: Trajectory conditioning batch from prepare_conditioning()
            pose_batch: Pose conditioning batch from prepare_conditioning()
            masking_spec: Tuple of (masking_action, visible_ids_override)
                - masking_action: Masking action dict or per-sample action plan
                - visible_ids_override: Optional list of visible joint IDs
            ctx: InferenceContext
            
        Returns:
            Tuple of (masked_traj_batch, masked_pose_batch) with masking applied
        """
        ...


class InferenceHook(ABC):
    """
    Abstract base class for inference hooks.
    
    Inference hooks handle the actual model execution, including:
    - Running TrajNet to predict root trajectory
    - Injecting trajectory predictions into PoseNet conditioning
    - Running PoseNet to predict full body pose
    
    Different baselines may have different inference pipelines (e.g., iterative
    refinement, physics-based optimization, different diffusion strategies).
    """
    
    @abstractmethod
    def run_trajnet(
        self,
        traj_batch: Dict[str, torch.Tensor],
        ctx: 'InferenceContext',
    ) -> torch.Tensor:
        """
        Run trajectory network inference.
        
        Args:
            traj_batch: Masked trajectory conditioning batch
            ctx: InferenceContext with model and diffusion references
            
        Returns:
            Trajectory output tensor [bs, T, traj_feat_dim]
            Contains predicted smplx_trans and smplx_rot_6d features
        """
        ...
    
    @abstractmethod
    def inject_trajectory(
        self,
        pose_batch: Dict[str, torch.Tensor],
        traj_output: torch.Tensor,
        ctx: 'InferenceContext',
    ) -> Dict[str, torch.Tensor]:
        """
        Inject trajectory output into pose conditioning.
        
        This method updates the pose conditioning batch with the predicted
        trajectory features, preparing it for PoseNet inference.
        
        Args:
            pose_batch: Masked pose conditioning batch
            traj_output: Trajectory output from run_trajnet()
            ctx: InferenceContext
            
        Returns:
            Updated pose_batch with trajectory features injected into 'cond'
        """
        ...
    
    @abstractmethod
    def run_posenet(
        self,
        pose_batch: Dict[str, torch.Tensor],
        ctx: 'InferenceContext',
    ) -> torch.Tensor:
        """
        Run pose network inference.
        
        Args:
            pose_batch: Pose conditioning batch with trajectory injected
            ctx: InferenceContext with model and diffusion references
            
        Returns:
            Pose output tensor [bs, T, pose_feat_dim]
            Contains full motion representation features
        """
        ...


class PostprocessingHook(ABC):
    """
    Abstract base class for postprocessing hooks.
    
    Postprocessing hooks transform model outputs into the final representation
    format and recover world-frame coordinates. This includes:
    - Reconstructing the representation dictionary from output tensor
    - Populating derived representations (e.g., global rotations from body pose)
    - Recovering world-frame joints and vertices via forward kinematics
    
    Different baselines may have different output formats or require
    baseline-specific postprocessing (e.g., physics refinement for PIP).
    """
    
    @abstractmethod
    def reconstruct_repr_dict(
        self,
        pose_output: torch.Tensor,
        ctx: 'InferenceContext',
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct representation dictionary from model output.
        
        This method splits the output tensor into individual representation
        components and populates any derived representations.
        
        Args:
            pose_output: Pose output tensor from run_posenet() [bs, T, feat_dim]
            ctx: InferenceContext with dataset repr_indices
            
        Returns:
            Dictionary mapping representation names to tensors:
                - 'smplx_trans': Root translation [bs, T, 3]
                - 'smplx_rot_6d': Root rotation [bs, T, 6]
                - 'smplx_body_pose_6d': Body pose [bs, T, 21*6]
                - 'local_positions': Joint positions [bs, T, 22*3]
                - etc.
        """
        ...
    
    @abstractmethod
    def recover_world_frame(
        self,
        repr_dict: Dict[str, torch.Tensor],
        original_batch: Dict[str, Any],
        ctx: 'InferenceContext',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover world-frame joints and vertices from representation.
        
        This method runs forward kinematics to obtain 3D joint positions
        and mesh vertices in world coordinates.
        
        Args:
            repr_dict: Representation dictionary from reconstruct_repr_dict()
            original_batch: Original batch with metadata for coordinate recovery
            ctx: InferenceContext with fk_model reference
            
        Returns:
            Tuple of (joints, vertices):
                - joints: Joint positions [bs, T, num_joints, 3]
                - vertices: Mesh vertices [bs, T, num_verts, 3]
        """
        ...


class MetricsHook(ABC):
    """
    Abstract base class for metrics hooks.
    
    Metrics hooks compute evaluation metrics specific to each baseline.
    Standard metrics (MPJPE, PA-MPJPE, acceleration error) are common,
    but baselines may add their own metrics (e.g., SIP error for IMU,
    head orientation error for egocentric methods).
    """
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """
        Return list of metric names this hook computes.
        
        Returns:
            List of metric name strings (e.g., ['mpjpe_global_mm', 'pa_mpjpe_global_mm'])
        """
        ...
    
    @abstractmethod
    def compute_metrics(
        self,
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        repr_dict_rec: Dict[str, torch.Tensor],
        repr_dict_clean: Dict[str, torch.Tensor],
        ctx: 'InferenceContext',
    ) -> Dict[str, float]:
        """
        Compute all metrics for this baseline.
        
        This method is called once after all batches have been processed,
        with aggregated predictions and ground truth.
        
        Args:
            pred_joints: Predicted joint positions [N, T, num_joints, 3] or [bs, T, num_joints, 3]
            gt_joints: Ground truth joint positions (same shape as pred_joints)
            repr_dict_rec: Aggregated reconstructed representation dictionary
            repr_dict_clean: Aggregated ground truth representation dictionary
            ctx: InferenceContext with mask scheme info for vis/occ joint separation
            
        Returns:
            Dictionary mapping metric names to float values
        """
        ...
