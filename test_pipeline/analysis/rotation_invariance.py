"""
Rotation invariance testing for motion reconstruction models.
"""
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from utils import dist_util
from utils.masking_actions import (
    build_mask_action_pool,
    materialize_mask_action,
    visible_ids_from_action,
)
from data_sets.motion_representation import populate_repr_dict, rotate_repr_smpl, recover_from_repr_smpl


class RotationInvarianceTester:
    """
    Tests model robustness to input rotations around the Z-axis.
    
    Applies various rotation angles to input data and measures
    how consistently the model reconstructs the motion.
    """
    
    def __init__(
        self,
        batch_processor,
        dataset,
        smplx_model,
        train_config: DictConfig,
        test_config: DictConfig,
    ):
        self.batch_processor = batch_processor
        self.dataset = dataset
        self.smplx_model = smplx_model
        self.train_config = train_config
        self.test_config = test_config
        self.device = dist_util.dev()
    
    def run_sweep(
        self,
        dataloader,
        angles_deg: List[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run rotation invariance sweep across multiple angles.
        
        Args:
            dataloader: Test dataloader
            angles_deg: List of rotation angles in degrees
            
        Returns:
            Dictionary mapping angle strings to metrics dictionaries
        """
        if angles_deg is None:
            angles_deg = [0, 90, 180, 270]
        
        angles_rad = [angle * np.pi / 180.0 for angle in angles_deg]
        per_angle_metrics = {}
        
        for angle_deg, angle_rad in zip(angles_deg, angles_rad):
            print(f"\n[Rotation Invariance] Evaluating angle {angle_deg} deg")
            
            if angle_deg == 0:
                # Baseline: no rotation
                results = self._process_baseline(dataloader)
            else:
                # Apply rotation and process
                results = self._process_with_rotation(dataloader, angle_rad)
            
            # Compute metrics
            metrics = self._compute_metrics(results)
            per_angle_metrics[str(angle_deg)] = metrics
            
            print(f"[Rotation Invariance] Angle {angle_deg} deg metrics:")
            for k in sorted(metrics.keys()):
                print(f"  {k}: {metrics[k]:.4f}")
        
        return per_angle_metrics
    
    def _process_baseline(self, dataloader) -> Dict[str, Any]:
        """Process without rotation (baseline)."""
        repr_dict_rec = {}
        repr_dict_clean = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Baseline")):
            seq_name = self._get_seq_name(batch, batch_idx)
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            result = self.batch_processor.process_batch(batch)
            
            if result is not None:
                repr_dict_rec[seq_name] = result.repr_dict_rec
                if result.repr_dict_clean is not None:
                    repr_dict_clean[seq_name] = result.repr_dict_clean
        
        return {
            'repr_dict_rec': repr_dict_rec,
            'repr_dict_clean': repr_dict_clean if repr_dict_clean else None,
        }
    
    def _process_with_rotation(self, dataloader, angle_rad: float) -> Dict[str, Any]:
        """Process with rotation applied to inputs."""
        repr_dict_rec = {}
        repr_dict_clean = {}
        
        # Build rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rotation {np.degrees(angle_rad):.0f}°")):
            seq_name = self._get_seq_name(batch, batch_idx)
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # Apply rotation to batch
            batch = self._rotate_batch(batch, rotation_matrix)
            
            result = self.batch_processor.process_batch(batch)
            
            if result is not None:
                repr_dict_rec[seq_name] = result.repr_dict_rec
                if result.repr_dict_clean is not None:
                    repr_dict_clean[seq_name] = result.repr_dict_clean
        
        return {
            'repr_dict_rec': repr_dict_rec,
            'repr_dict_clean': repr_dict_clean if repr_dict_clean else None,
        }
    
    def _rotate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        rotation_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply rotation to batch motion representations."""
        # Convert motion_repr to repr_dict
        motion_repr_clean = batch['motion_repr_clean']
        repr_dict_clean = self._motion_repr_to_dict(motion_repr_clean)
        
        # Derive flag from dataset
        include_global_joint_rotations = 'R_world_joints' in self.dataset._repr_list
        
        # Populate derived representations
        if include_global_joint_rotations:
            repr_dict_clean = populate_repr_dict(repr_dict_clean, self.smplx_model, recover_global=False)
        else:
            repr_dict_clean = populate_repr_dict(repr_dict_clean, self.smplx_model, recover_global=True)
        
        # Apply rotation
        repr_dict_clean = rotate_repr_smpl(repr_dict_clean, rotation_matrix)
        
        # Convert back to motion_repr
        rotated_motion_repr = self._dict_to_motion_repr(repr_dict_clean)
        batch['motion_repr_clean'] = rotated_motion_repr
        
        # Also rotate noisy input if present
        if self.train_config.representation.input_noise and 'motion_repr_noisy' in batch:
            motion_repr_noisy = batch['motion_repr_noisy']
            repr_dict_noisy = self._motion_repr_to_dict(motion_repr_noisy)
            
            if include_global_joint_rotations:
                repr_dict_noisy = populate_repr_dict(repr_dict_noisy, self.smplx_model, recover_global=False)
            else:
                repr_dict_noisy = populate_repr_dict(repr_dict_noisy, self.smplx_model, recover_global=True)
            
            repr_dict_noisy = rotate_repr_smpl(repr_dict_noisy, rotation_matrix)
            batch['motion_repr_noisy'] = self._dict_to_motion_repr(repr_dict_noisy)
        
        return batch
    
    def _motion_repr_to_dict(self, motion_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert motion representation tensor to dictionary."""
        repr_dict = {}
        for repr_name in self.dataset._repr_list:
            s, e = self.dataset._repr_indices[repr_name]
            repr_dict[repr_name] = motion_repr[..., s:e]
        return repr_dict
    
    def _dict_to_motion_repr(self, repr_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert representation dictionary back to tensor."""
        return torch.cat([repr_dict[key] for key in self.dataset._repr_list], dim=-1)
    
    def _compute_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute evaluation metrics from results."""
        from utils.evaluation import MotionEvaluator
        
        repr_dict_rec = results['repr_dict_rec']
        repr_dict_clean = results.get('repr_dict_clean')
        
        # Aggregate joint positions
        joint_pos_rec_list = []
        joint_pos_clean_list = []
        
        for seq_name in repr_dict_rec.keys():
            jts_rec, _ = recover_from_repr_smpl(
                repr_dict_rec[seq_name],
                recover_mode='smplx_params',
                cano_jts_pos=True,
                smplx_model=self.smplx_model,
                return_verts=True,
                return_full_joints=False,
            )
            joint_pos_rec_list.append(jts_rec.numpy(force=True))
            
            if repr_dict_clean is not None and seq_name in repr_dict_clean:
                jts_clean, _ = recover_from_repr_smpl(
                    repr_dict_clean[seq_name],
                    recover_mode='smplx_params',
                    cano_jts_pos=True,
                    smplx_model=self.smplx_model,
                    return_verts=True,
                    return_full_joints=False,
                )
                joint_pos_clean_list.append(jts_clean.numpy(force=True))
        
        # Concatenate
        cat_axis = 1 if self.test_config.slice_strategy == 'full_sequence' else 0
        joint_pos_rec = np.concatenate(joint_pos_rec_list, axis=cat_axis)
        
        evaluator = MotionEvaluator(fps=30)
        metrics = evaluator.compute_self_metrics(pred_joints=joint_pos_rec)
        
        if joint_pos_clean_list:
            joint_pos_clean = np.concatenate(joint_pos_clean_list, axis=cat_axis)
            
            # Get vis/occ joint IDs from masking action
            action_pool = build_mask_action_pool(
                OmegaConf.to_container(self.train_config.masking, resolve=True)
            )
            action = materialize_mask_action(
                self.test_config.test_mask_action,
                action_pool,
                self.dataset.joints_num,
            )
            vis_ids = visible_ids_from_action(action, self.dataset.mask_scheme_registry, self.dataset.joints_num)
            if vis_ids is None:
                vis_ids = list(range(self.dataset.joints_num))
            occ_ids = list(set(range(self.dataset.joints_num)) - set(vis_ids))
            
            comparison_metrics = evaluator.compute_comparison_metrics(
                pred_joints=joint_pos_rec,
                gt_joints=joint_pos_clean,
                repr_dict_rec=None,
                repr_dict_clean=None,
                vis_joint_ids=vis_ids,
                occ_joint_ids=occ_ids,
            )
            metrics.update(comparison_metrics)
        
        return metrics
    
    def _get_seq_name(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Extract sequence name from batch."""
        if self.test_config.slice_strategy == 'full_sequence' and self.test_config.dataset_name.lower() == 'amass':
            return batch['meta']['name'][0]
        return batch_idx
    
    def save_results(self, per_angle_metrics: Dict[str, Dict[str, float]]):
        """Save rotation invariance results to JSON."""
        output_path = os.path.join(self.test_config.save_root, 'rotation_invariance_metrics.json')
        with open(output_path, 'w') as f:
            json.dump(per_angle_metrics, f, indent=4)
        print(f"Saved rotation invariance metrics to: {output_path}")
