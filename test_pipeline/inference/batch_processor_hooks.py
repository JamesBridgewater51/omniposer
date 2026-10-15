"""
Hook-based batch processor that delegates to baseline hooks.

This module provides BatchProcessorWithHooks which uses the baseline hook system
while preserving all the complex sliding window and normalization logic.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

from utils.feature_accessor import select_modalities, insert_modalities, get_modality_dim, insert_modalities_into_cond, get_cond_modality_indices

from .sliding_window import SlidingWindowProcessor, WindowConfig
from .result import InferenceResult

if TYPE_CHECKING:
    from ..baselines.context import InferenceContext
    from ..baselines.spec import BaselineSpec
    from ..baselines.hooks.base import PreprocessingHook, InferenceHook, PostprocessingHook


class BatchProcessorWithHooks:
    """
    Batch processor that delegates to baseline hooks.
    
    This class preserves all the complex sliding window and normalization logic
    from the original BatchProcessor, but delegates the core operations to hooks:
    - PreprocessingHook: prepare_conditioning(), apply_masking()
    - InferenceHook: run_trajnet(), inject_trajectory(), run_posenet()
    - PostprocessingHook: reconstruct_repr_dict()
    
    The sliding window orchestration, normalization management, and root offset
    handling remain in this class as they are infrastructure concerns shared
    across all baselines.
    """
    _ROOT_OFFSET_MODALITIES = {
        "local_positions",
        "smplx_trans",
    }
    
    def __init__(
        self,
        ctx: 'InferenceContext',
        baseline_spec: 'BaselineSpec',
    ):
        self.ctx = ctx
        self.spec = baseline_spec
        
        # Instantiate hooks
        self.preprocessing: 'PreprocessingHook' = baseline_spec.preprocessing_hook_cls()
        self.inference: 'InferenceHook' = baseline_spec.inference_hook_cls()
        self.postprocessing: 'PostprocessingHook' = baseline_spec.postprocessing_hook_cls()
        
        # Initialize sliding window processor
        self.window_processor = SlidingWindowProcessor(
            WindowConfig(
                window_size=ctx.window_size,
                overlap_size=ctx.test_config.overlap_size,
            ),
            device=ctx.device,
        )
    
    def get_masking_spec(self) -> tuple:
        """Get masking specification based on configured masking action."""
        mask_action = self.ctx.resolve_mask_action()
        return (mask_action, None)
    
    @torch.no_grad()
    def process_batch(self, batch: Dict[str, torch.Tensor]) -> InferenceResult:
        """Process a single batch through the hook-based pipeline."""
        # Move batch to device
        batch = {k: v.to(self.ctx.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Update context with current batch
        self.ctx.current_batch = batch
        self.ctx.current_masking_spec = None
        self.ctx.current_mask_action = None
        self.ctx.sampling_progress_entries = []
        
        # Phase 1: Preprocessing via hook
        traj_batch, pose_batch = self.preprocessing.prepare_conditioning(batch, self.ctx)
        
        bs, seq_len, _ = pose_batch['motion_repr_clean'].shape
        
        # Check if sliding window is needed
        if seq_len <= self.ctx.window_size:
            return self._process_single_window(traj_batch, pose_batch, batch)
        else:
            return self._process_with_sliding_window(traj_batch, pose_batch, batch)
    
    def _process_single_window(
        self,
        traj_batch: Dict[str, torch.Tensor],
        pose_batch: Dict[str, torch.Tensor],
        original_batch: Dict[str, torch.Tensor],
    ) -> InferenceResult:
        """Process batch without sliding window (short sequences)."""
        sample_norms = self._get_sample_norms(pose_batch)

        masking_spec = self.get_masking_spec()
        self.ctx.current_masking_spec = masking_spec
        self.ctx.current_mask_action = masking_spec[0]
        
        # Apply masking via hook
        traj_batch, pose_batch = self.preprocessing.apply_masking(
            traj_batch, pose_batch, masking_spec, self.ctx
        )

        sample_iter = max(1, int(getattr(self.ctx.test_config, 'sample_iter', 1)))
        pose_cond_base = pose_batch['cond'].clone()
        traj_outputs = []
        pose_outputs = []
        for _ in range(sample_iter):
            # Run TrajNet via hook
            traj_output = self.inference.run_trajnet(traj_batch, self.ctx)

            # Inject trajectory into a fresh pose conditioning tensor for each sample.
            pose_batch_sample = dict(pose_batch)
            pose_batch_sample['cond'] = pose_cond_base.clone()
            pose_batch_sample = self.inference.inject_trajectory(pose_batch_sample, traj_output, self.ctx)

            # Run PoseNet via hook
            pose_output = self.inference.run_posenet(pose_batch_sample, self.ctx)
            traj_outputs.append(traj_output)
            pose_outputs.append(pose_output)

        traj_output = self._aggregate_sample_outputs(traj_outputs)
        pose_output = self._aggregate_sample_outputs(pose_outputs)
        pose_batch['cond'] = self.inference.inject_trajectory(
            dict(pose_batch, cond=pose_cond_base.clone()),
            traj_output,
            self.ctx,
        )['cond']

        # Unnormalize output to recover correct SMPL-X parameters
        pose_output = self.ctx.dataset._unnormalize_full_tensor(
            pose_output, self.ctx.device, sample_norms=sample_norms
        )
        if self.ctx.test_config.dataset_name.lower() in ['ego_exo']:
            pose_output = self._apply_egoexo_offset_recovery(pose_output, original_batch)
        pose_output = self._apply_prediction_postprocess(
            pose_output,
            pose_batch=pose_batch,
            sample_norms=sample_norms,
            cond_is_normalized=True,
        )

        # Reconstruct results via hook
        repr_dict_rec = self.postprocessing.reconstruct_repr_dict(pose_output, self.ctx)

        result = InferenceResult(
            repr_dict_rec=repr_dict_rec,
            masking_spec=masking_spec,
            sampling_progress_entries=list(self.ctx.sampling_progress_entries) or None,
            visible_joint_mask=pose_batch.get('cond_joint_mask', torch.empty(0, device=self.ctx.device))[..., 0].detach().cpu() if 'cond_joint_mask' in pose_batch else None,
            cond_joint_modality_mask=pose_batch.get('cond_joint_modality_mask').detach().cpu() if 'cond_joint_modality_mask' in pose_batch else None,
            cond_joint_modality_names=pose_batch.get('cond_joint_modality_names'),
            cond_pattern_names=pose_batch.get('cond_pattern_names'),
            cond_mask_policy=pose_batch.get('cond_mask_policy'),
        )
        
        # Add ground truth if available
        if self.ctx.test_config.with_gt:
            motion_repr_clean = self.ctx.dataset._unnormalize_full_tensor(
                pose_batch['motion_repr_clean'], self.ctx.device, sample_norms=sample_norms
            )
            result.repr_dict_clean = self._build_repr_dict_from_batch(motion_repr_clean)

        return result
    
    def _process_with_sliding_window(
        self,
        traj_batch: Dict[str, torch.Tensor],
        pose_batch: Dict[str, torch.Tensor],
        original_batch: Dict[str, torch.Tensor],
    ) -> InferenceResult:
        """Process batch using sliding window approach."""
        bs, seq_len, body_feat_dim = pose_batch['motion_repr_clean'].shape
        sample_iter = getattr(self.ctx.test_config, 'sample_iter', 1)
        window_align_to_yaxis = self._resolve_window_align_to_yaxis_flag()
        
        # Get sample norms for l2 normalization
        sample_norms = self._get_sample_norms(pose_batch)
        
        # Unnormalize motion_repr for window processing
        pose_batch['motion_repr_clean'] = self.ctx.dataset._unnormalize_full_tensor(
            pose_batch['motion_repr_clean'], self.ctx.device, sample_norms=sample_norms
        )
        traj_batch['motion_repr_clean'] = self.ctx.dataset._unnormalize_full_tensor(
            traj_batch['motion_repr_clean'], self.ctx.device, sample_norms=sample_norms
        )
        
        # Unnormalize motion_repr_noisy if present
        if self.ctx.input_noise and 'motion_repr_noisy' in pose_batch:
            pose_batch['motion_repr_noisy'] = self.ctx.dataset._unnormalize_full_tensor(
                pose_batch['motion_repr_noisy'], self.ctx.device, sample_norms=sample_norms
            )
            traj_batch['motion_repr_noisy'] = self.ctx.dataset._unnormalize_full_tensor(
                traj_batch['motion_repr_noisy'], self.ctx.device, sample_norms=sample_norms
            )
        
        # Unnormalize cond tensors using unified dataset method (supports all norm strategies)
        traj_batch['cond'] = self.ctx.dataset._unnormalize_cond_tensor(
            traj_batch['cond'], task='traj', device=self.ctx.device
        )
        pose_batch['cond'] = self.ctx.dataset._unnormalize_cond_tensor(
            pose_batch['cond'], task='pose', device=self.ctx.device
        )
        
        # Compute padding and windows
        num_windows = self.window_processor.compute_num_windows(seq_len)
        padded_seq_len = self.window_processor.compute_padded_length(seq_len)
        
        # Initialize output accumulators
        traj_feat_dim = get_modality_dim(self.ctx.traj_repr_name_list, self.ctx.repr_indices)
        val_output_traj = torch.zeros([bs, padded_seq_len, traj_feat_dim], device=self.ctx.device)
        val_output_pose = torch.zeros([bs, padded_seq_len, body_feat_dim], device=self.ctx.device)
        overlap_weights = torch.zeros([bs, padded_seq_len], device=self.ctx.device)
        visible_joint_mask_accum = torch.zeros([bs, padded_seq_len, self.ctx.joints_num], device=self.ctx.device)
        joint_modality_mask_accum = None
        cond_joint_modality_names = None
        cond_pattern_names = None
        cond_mask_policy = None


        beta_range = self.ctx.repr_indices.get('smplx_betas')
        foot_contact_range = self.ctx.repr_indices.get('foot_contact')
        if beta_range is None:
            raise KeyError(
                "Sliding-window inference requires modality 'smplx_betas' in repr_indices for shape fusion."
            )
        if foot_contact_range is None:
            raise KeyError(
                "Sliding-window inference requires modality 'foot_contact' in repr_indices for shape validation."
            )

        beta_contact_threshold = float(getattr(self.ctx.test_config, 'beta_fusion_contact_threshold', 0.5))
        beta_min_contact_ratio = float(getattr(self.ctx.test_config, 'beta_fusion_min_contact_ratio', 0.2))
        beta_geom_eps = float(getattr(self.ctx.test_config, 'beta_fusion_geom_eps', 1e-6))
        if not torch.isfinite(torch.tensor(beta_contact_threshold)):
            raise ValueError(f"beta_fusion_contact_threshold must be finite, got {beta_contact_threshold}.")
        if not (0.0 <= beta_min_contact_ratio <= 1.0):
            raise ValueError(
                f"beta_fusion_min_contact_ratio must be in [0, 1], got {beta_min_contact_ratio}."
            )
        if beta_geom_eps <= 0.0:
            raise ValueError(f"beta_fusion_geom_eps must be > 0, got {beta_geom_eps}.")

        window_beta_candidates: List[torch.Tensor] = []
        window_beta_valid_masks: List[torch.Tensor] = []
        window_beta_contact_ratios: List[torch.Tensor] = []
        
        # Get canonical overlap weights
        canonical_weights = self.window_processor.get_overlap_weights(seq_len)[:self.ctx.window_size, None]
        joint_weights = canonical_weights.view(1, self.ctx.window_size, 1)
        joint_modality_weights = canonical_weights.view(1, self.ctx.window_size, 1, 1)
        
        # Determine keys to copy for each window
        keys_to_slice = [
            'motion_repr_clean', 'motion_repr_noisy', 'cond_mask', 'cond_joint_mask',
            'cond_joint_modality_mask', 'cond_joint_modality_names', 'cond_global_mask',
            'cond',
        ]
        if self.ctx.test_config.dataset_name.lower() == 'ego_exo':
            keys_to_slice.extend(['joints_world_kin_smpl22', 'joints_world_kin_smpl22_norm', 
                                  'joints_world_kin_coco17', 'visible_mask_smpl22', 'visible_mask_coco17'])
        
        # Split into windows
        windows_traj = self.window_processor.split_into_windows(
            traj_batch, keys_to_slice, self.ctx.repr_indices, cond_type='traj'
        )
        windows_pose = self.window_processor.split_into_windows(
            pose_batch, keys_to_slice, self.ctx.repr_indices, cond_type='pose'
        )
        
        # Get masking spec
        masking_spec = self.get_masking_spec()
        self.ctx.current_masking_spec = masking_spec
        self.ctx.current_mask_action = masking_spec[0]
        
        # Process each window
        for win_idx, (win_traj, win_pose) in enumerate(zip(windows_traj, windows_pose)):
            start_t = win_idx * self.window_processor.config.stride
            end_t = start_t + self.ctx.window_size
            
            # Compute root init xy and rotation
            win_root_init_xy = self._compute_window_root_init_xy(
                win_traj,
                masking_spec=masking_spec,
            )
            win_root_init_rot = self._compute_window_root_init_rot(
                win_traj,
                align_to_yaxis=window_align_to_yaxis,
                masking_spec=masking_spec,
            )
            
            # Apply root offset and rotation to cond tensors
            self._apply_root_offset_to_window(win_traj, win_pose, win_root_init_xy)
            if win_root_init_rot is not None:
                self._apply_root_rotate_to_cond(win_traj['cond'], 'traj', win_root_init_rot, mode='forward')
                self._apply_root_rotate_to_cond(win_pose['cond'], 'pose', win_root_init_rot, mode='forward')
            
            # Sync traj features from motion_repr into win_pose['cond']
            traj_features_from_repr = select_modalities(
                win_pose['motion_repr_clean'], 
                self.ctx.traj_repr_name_list, 
                self.ctx.repr_indices
            )
            win_pose['cond'] = insert_modalities_into_cond(
                win_pose['cond'], traj_features_from_repr, 
                self.ctx.traj_repr_name_list, self.ctx.dataset, task='pose'
            )
            
            # Apply root offset and rotation to motion_repr
            self._apply_root_offset_to_motion_repr(win_traj, win_pose, win_root_init_xy)
            if win_root_init_rot is not None:
                self._apply_root_rotate_to_full_repr(win_traj['motion_repr_clean'], win_root_init_rot, mode='forward')
                self._apply_root_rotate_to_full_repr(win_pose['motion_repr_clean'], win_root_init_rot, mode='forward')
                if self.ctx.input_noise and 'motion_repr_noisy' in win_traj:
                    self._apply_root_rotate_to_full_repr(win_traj['motion_repr_noisy'], win_root_init_rot, mode='forward')
                    self._apply_root_rotate_to_full_repr(win_pose['motion_repr_noisy'], win_root_init_rot, mode='forward')
            
            # Normalize cond using unified dataset method (supports all norm strategies)
            win_traj['cond'] = self.ctx.dataset._normalize_cond_tensor(
                win_traj['cond'], task='traj', device=self.ctx.device
            )
            win_pose['cond'] = self.ctx.dataset._normalize_cond_tensor(
                win_pose['cond'], task='pose', device=self.ctx.device
            )
            
            # Normalize motion_repr for windows (model expects normalized input regardless of with_gt)
            win_traj['motion_repr_clean'] = self.ctx.dataset._normalize_full_tensor(
                win_traj['motion_repr_clean'], device=self.ctx.device
            )
            win_pose['motion_repr_clean'] = self.ctx.dataset._normalize_full_tensor(
                win_pose['motion_repr_clean'], device=self.ctx.device
            )
            if self.ctx.input_noise and 'motion_repr_noisy' in win_traj:
                win_traj['motion_repr_noisy'] = self.ctx.dataset._normalize_full_tensor(
                    win_traj['motion_repr_noisy'], device=self.ctx.device
                )
                win_pose['motion_repr_noisy'] = self.ctx.dataset._normalize_full_tensor(
                    win_pose['motion_repr_noisy'], device=self.ctx.device
                )
            
            # Apply masking via hook
            win_traj, win_pose = self.preprocessing.apply_masking(
                win_traj, win_pose, masking_spec, self.ctx
            )
            
            win_pose_cond_base = win_pose['cond'].clone()
            traj_samples = []
            pose_samples = []
            for _ in range(sample_iter):
                # Run TrajNet via hook
                traj_out = self.inference.run_trajnet(win_traj, self.ctx)

                # Inject trajectory via hook (if not eval_on_posenet_only)
                win_pose_sample = dict(win_pose)
                win_pose_sample['cond'] = win_pose_cond_base.clone()
                if not self.ctx.test_config.eval_on_posenet_only:
                    win_pose_sample = self.inference.inject_trajectory(win_pose_sample, traj_out, self.ctx)

                # Run PoseNet via hook
                pose_out = self.inference.run_posenet(win_pose_sample, self.ctx)
                traj_samples.append(traj_out)
                pose_samples.append(pose_out)

            traj_out = self._aggregate_sample_outputs(traj_samples)
            pose_out = self._aggregate_sample_outputs(pose_samples)
            if not self.ctx.test_config.eval_on_posenet_only:
                win_pose['cond'] = self.inference.inject_trajectory(
                    dict(win_pose, cond=win_pose_cond_base.clone()),
                    traj_out,
                    self.ctx,
                )['cond']

            # Unnormalize output and restore root offset
            traj_mean = select_modalities(
                self.ctx.norm_stats.motion_repr_mean.unsqueeze(0).unsqueeze(0), 
                self.ctx.traj_repr_name_list, self.ctx.repr_indices
            ).squeeze(0).squeeze(0)
            traj_std = select_modalities(
                self.ctx.norm_stats.motion_repr_std.unsqueeze(0).unsqueeze(0), 
                self.ctx.traj_repr_name_list, self.ctx.repr_indices
            ).squeeze(0).squeeze(0)
            
            traj_out = traj_out * traj_std + traj_mean
            pose_out = pose_out * self.ctx.norm_stats.motion_repr_std + self.ctx.norm_stats.motion_repr_mean
            pose_out_for_beta = pose_out
            
            # Restore root rotation and offset
            if self.ctx.repr_abs_only:
                if win_root_init_rot is not None:
                    # PoseNet output is partially materialized (some modalities can be placeholders).
                    # To stay geometrically equivalent to rotate_repr_smpl_2, materialize local_positions first.
                    self._materialize_local_positions_from_smplx(
                        pose_out,
                        tensor_name="pose_out before inverse root rotation",
                    )
                    self._apply_root_rotate_to_traj_with_reference_full_repr(
                        traj_out,
                        pose_out,
                        win_root_init_rot,
                        mode='inverse',
                    )
                    self._apply_root_rotate_to_full_repr(pose_out, win_root_init_rot, mode='inverse')

                self._apply_root_offset_to_predicted_repr(
                    traj_out,
                    task='traj',
                    win_root_init_xy=win_root_init_xy,
                    mode='add',
                )
                self._apply_root_offset_to_full_repr(
                    pose_out,
                    win_root_init_xy=win_root_init_xy,
                    mode='add',
                )
                # Re-normalize for accumulation
                traj_out = (traj_out - traj_mean) / traj_std
                pose_out = (pose_out - self.ctx.norm_stats.motion_repr_mean) / self.ctx.norm_stats.motion_repr_std

            if '_window_meta' not in win_pose:
                raise KeyError("Missing '_window_meta' in sliding window payload for beta fusion.")
            pad_len = int(win_pose['_window_meta'].get('pad_len', 0))
            valid_len = self.ctx.window_size - pad_len
            if valid_len <= 0 or valid_len > self.ctx.window_size:
                raise ValueError(
                    f"Invalid effective window length {valid_len} (pad_len={pad_len}, "
                    f"window_size={self.ctx.window_size})."
                )
            win_beta, win_valid_mask, win_contact_ratio = self._collect_window_beta_candidate(
                pose_out=pose_out_for_beta,
                valid_len=valid_len,
                beta_range=beta_range,
                foot_contact_range=foot_contact_range,
                contact_threshold=beta_contact_threshold,
                min_contact_ratio=beta_min_contact_ratio,
            )
            window_beta_candidates.append(win_beta)
            window_beta_valid_masks.append(win_valid_mask)
            window_beta_contact_ratios.append(win_contact_ratio)
            
            # Accumulate with weights
            val_output_traj[:, start_t:end_t, :] += traj_out * canonical_weights
            val_output_pose[:, start_t:end_t, :] += pose_out * canonical_weights
            overlap_weights[:, start_t:end_t] += canonical_weights[..., 0]
            if 'cond_joint_mask' in win_pose:
                visible_joint_mask_accum[:, start_t:end_t, :] += win_pose['cond_joint_mask'][..., 0].float() * joint_weights
            if 'cond_joint_modality_mask' in win_pose:
                if joint_modality_mask_accum is None:
                    joint_modality_mask_accum = torch.zeros(
                        [bs, padded_seq_len, win_pose['cond_joint_modality_mask'].shape[2], win_pose['cond_joint_modality_mask'].shape[3]],
                        device=self.ctx.device,
                    )
                joint_modality_mask_accum[:, start_t:end_t, :, :] += win_pose['cond_joint_modality_mask'].float() * joint_modality_weights
            cond_joint_modality_names = win_pose.get('cond_joint_modality_names', cond_joint_modality_names)
            cond_pattern_names = win_pose.get('cond_pattern_names', cond_pattern_names)
            cond_mask_policy = win_pose.get('cond_mask_policy', cond_mask_policy)

        fused_window_betas = self._fuse_window_betas(
            window_beta_candidates=window_beta_candidates,
            window_beta_valid_masks=window_beta_valid_masks,
            window_beta_contact_ratios=window_beta_contact_ratios,
            geom_eps=beta_geom_eps,
        )
        
        # Normalize by weights
        val_output_traj = val_output_traj / overlap_weights[..., None].clamp(min=1e-8)
        val_output_pose = val_output_pose / overlap_weights[..., None].clamp(min=1e-8)
        
        # Remove padding
        if padded_seq_len > seq_len:
            val_output_traj = val_output_traj[:, :seq_len, :]
            val_output_pose = val_output_pose[:, :seq_len, :]
            visible_joint_mask_accum = visible_joint_mask_accum[:, :seq_len, :]
            if joint_modality_mask_accum is not None:
                joint_modality_mask_accum = joint_modality_mask_accum[:, :seq_len, :, :]
        visible_joint_mask = visible_joint_mask_accum > 0
        joint_modality_mask = joint_modality_mask_accum > 0 if joint_modality_mask_accum is not None else None
        
        # Final unnormalization
        traj_mean = select_modalities(
            self.ctx.norm_stats.motion_repr_mean.unsqueeze(0).unsqueeze(0), 
            self.ctx.traj_repr_name_list, self.ctx.repr_indices
        ).squeeze(0).squeeze(0)
        traj_std = select_modalities(
            self.ctx.norm_stats.motion_repr_std.unsqueeze(0).unsqueeze(0), 
            self.ctx.traj_repr_name_list, self.ctx.repr_indices
        ).squeeze(0).squeeze(0)
        val_output_traj = val_output_traj * traj_std + traj_mean
        val_output_pose = self.ctx.dataset._unnormalize_full_tensor(
            val_output_pose, self.ctx.device, sample_norms=sample_norms
        )
        
        # Apply EgoExo offset recovery
        if self.ctx.test_config.dataset_name.lower() in ['ego_exo']:
            val_output_pose = self._apply_egoexo_offset_recovery(val_output_pose, original_batch)
        
        # Handle eval_on_trajnet_only / eval_on_posenet_only
        if self.ctx.test_config.with_gt:
            if self.ctx.test_config.eval_on_trajnet_only:
                all_modalities = self.ctx.repr_list
                non_traj_modalities = [m for m in all_modalities if m not in self.ctx.traj_repr_name_list]
                non_traj_features = select_modalities(
                    pose_batch['motion_repr_clean'], non_traj_modalities, self.ctx.repr_indices
                )
                val_output_pose = insert_modalities(
                    val_output_pose, non_traj_features, non_traj_modalities, self.ctx.repr_indices
                )
            if self.ctx.test_config.eval_on_posenet_only:
                traj_features_gt = select_modalities(
                    traj_batch['motion_repr_clean'], self.ctx.traj_repr_name_list, self.ctx.repr_indices
                )
                val_output_traj = traj_features_gt.clone()
                val_output_pose = insert_modalities(
                    val_output_pose, traj_features_gt, self.ctx.traj_repr_name_list, self.ctx.repr_indices
                )

        if not self.ctx.test_config.eval_on_trajnet_only:
            s_beta, e_beta = beta_range
            val_output_pose[..., s_beta:e_beta] = fused_window_betas.unsqueeze(1).expand(
                -1, val_output_pose.shape[1], -1
            )
        val_output_pose = self._apply_prediction_postprocess(
            val_output_pose,
            pose_batch=pose_batch,
            visible_joint_mask=visible_joint_mask,
            joint_modality_mask=joint_modality_mask,
            joint_modality_names=cond_joint_modality_names,
            sample_norms=sample_norms,
            cond_is_normalized=False,
        )
        
        # Reconstruct results via hook
        repr_dict_rec = self.postprocessing.reconstruct_repr_dict(val_output_pose, self.ctx)

        result = InferenceResult(
            repr_dict_rec=repr_dict_rec,
            masking_spec=masking_spec,
            sampling_progress_entries=list(self.ctx.sampling_progress_entries) or None,
            visible_joint_mask=visible_joint_mask.detach().cpu(),
            cond_joint_modality_mask=joint_modality_mask.detach().cpu() if joint_modality_mask is not None else None,
            cond_joint_modality_names=cond_joint_modality_names,
            cond_pattern_names=cond_pattern_names,
            cond_mask_policy=cond_mask_policy,
        )
        
        # Add ground truth if available
        if self.ctx.test_config.with_gt:
            result.repr_dict_clean = self._build_repr_dict_from_batch(
                pose_batch['motion_repr_clean']
            )
        
        return result
    
    # Helper methods (preserved from original BatchProcessor)

    def _aggregate_sample_outputs(self, samples: List[torch.Tensor]) -> torch.Tensor:
        if len(samples) == 0:
            raise RuntimeError("Cannot aggregate an empty sample list.")
        if len(samples) == 1:
            return samples[0]

        mode = getattr(self.ctx.test_config, 'sample_aggregation', 'mean')
        if mode == 'last':
            return samples[-1]

        stacked = torch.stack(samples, dim=0)
        if mode == 'mean':
            return stacked.mean(dim=0)
        if mode == 'median':
            return stacked.median(dim=0).values
        raise ValueError(f"Unknown sample_aggregation '{mode}'. Expected one of: last, mean, median.")

    def _apply_prediction_postprocess(
        self,
        motion_repr: torch.Tensor,
        pose_batch: Optional[Dict[str, torch.Tensor]] = None,
        visible_joint_mask: Optional[torch.Tensor] = None,
        joint_modality_mask: Optional[torch.Tensor] = None,
        joint_modality_names: Optional[List[str]] = None,
        sample_norms: Optional[Dict[str, torch.Tensor]] = None,
        cond_is_normalized: bool = True,
    ) -> torch.Tensor:
        """Apply optional inference-time priors to predicted motion representation.

        The postprocess uses only model predictions, static configuration, and
        optionally visible conditioning observations. It deliberately does not read
        ground-truth output tensors or test labels.
        """
        cfg = self.ctx.test_config
        needs_postprocess = any([
            bool(getattr(cfg, "postprocess_rot6d_normalize", False)),
            bool(getattr(cfg, "postprocess_betas_sequence_average", False)),
            bool(getattr(cfg, "postprocess_temporal_smoothing", False)),
            bool(getattr(cfg, "postprocess_contact_binarize", False)),
            bool(getattr(cfg, "postprocess_contact_median", False)),
            bool(getattr(cfg, "postprocess_contact_kinematic", False)),
            bool(getattr(cfg, "postprocess_foot_lock", False)),
            bool(getattr(cfg, "postprocess_observed_joint_anchor", False)),
            bool(getattr(cfg, "postprocess_floor_lift", False)),
        ])
        if not needs_postprocess:
            return motion_repr

        output = motion_repr.clone()

        if bool(getattr(cfg, "postprocess_rot6d_normalize", False)):
            output = self._normalize_rot6d_modalities(output)

        if bool(getattr(cfg, "postprocess_betas_sequence_average", False)):
            output = self._average_sequence_betas(output)

        if bool(getattr(cfg, "postprocess_temporal_smoothing", False)):
            output = self._apply_temporal_smoothing(output)

        if bool(getattr(cfg, "postprocess_contact_binarize", False)):
            output = self._apply_contact_binarize(output)

        if bool(getattr(cfg, "postprocess_contact_median", False)):
            output = self._apply_contact_median(output)

        pred_joints = None
        if bool(getattr(cfg, "postprocess_observed_joint_anchor", False)):
            pred_joints = self._recover_predicted_joints(output)
            if pred_joints is not None:
                output, pred_joints = self._apply_observed_joint_anchor(
                    output,
                    pred_joints,
                    pose_batch=pose_batch,
                    visible_joint_mask=visible_joint_mask,
                    joint_modality_mask=joint_modality_mask,
                    joint_modality_names=joint_modality_names,
                    sample_norms=sample_norms,
                    cond_is_normalized=cond_is_normalized,
                )

        if bool(getattr(cfg, "postprocess_floor_lift", False)):
            if pred_joints is None:
                pred_joints = self._recover_predicted_joints(output)
            if pred_joints is not None:
                output = self._apply_floor_lift(output, pred_joints)
                pred_joints = None

        if bool(getattr(cfg, "postprocess_contact_kinematic", False)):
            if pred_joints is None:
                pred_joints = self._recover_predicted_joints(output)
            if pred_joints is not None:
                output = self._apply_kinematic_contact(output, pred_joints)

        if bool(getattr(cfg, "postprocess_foot_lock", False)):
            if pred_joints is None:
                pred_joints = self._recover_predicted_joints(output)
            if pred_joints is not None:
                output, pred_joints = self._apply_foot_lock(output, pred_joints)

        output = self._invalidate_derived_local_positions(output)
        return output

    def _apply_temporal_smoothing(self, motion_repr: torch.Tensor) -> torch.Tensor:
        cfg = self.ctx.test_config
        kernel_size = int(getattr(self.ctx.test_config, "postprocess_temporal_smoothing_kernel", 5))
        strength = float(getattr(self.ctx.test_config, "postprocess_temporal_smoothing_strength", 0.5))
        modalities = getattr(self.ctx.test_config, "postprocess_temporal_smoothing_modalities", None)
        if modalities is None:
            modalities = ["smplx_trans", "smplx_rot_6d", "smplx_body_pose_6d"]
        if isinstance(modalities, str):
            modalities = [m.strip() for m in modalities.split(",") if m.strip()]

        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        strength = max(0.0, min(1.0, strength))
        if kernel_size <= 1 or strength <= 0.0:
            return motion_repr

        output = motion_repr.clone()
        for name in modalities:
            if name not in self.ctx.repr_indices:
                continue
            start, end = self.ctx.repr_indices[name]
            if end <= start:
                continue
            segment = output[..., start:end]
            smoothed = self._temporal_smooth_tensor(segment, kernel_size=kernel_size)
            segment = segment * (1.0 - strength) + smoothed * strength
            if name in {"smplx_rot_6d", "smplx_body_pose_6d", "R_world_joints", "R_root_joints"}:
                segment = self._normalize_rot6d_feature(segment)
            output[..., start:end] = segment
        return output

    def _normalize_rot6d_modalities(self, motion_repr: torch.Tensor) -> torch.Tensor:
        output = motion_repr.clone()
        for name in ("smplx_rot_6d", "smplx_body_pose_6d", "R_world_joints", "R_root_joints"):
            if name not in self.ctx.repr_indices:
                continue
            start, end = self.ctx.repr_indices[name]
            if end <= start or start >= output.shape[-1] or end > output.shape[-1]:
                continue
            output[..., start:end] = self._normalize_rot6d_feature(output[..., start:end])
        return output

    def _average_sequence_betas(self, motion_repr: torch.Tensor) -> torch.Tensor:
        if "smplx_betas" not in self.ctx.repr_indices or motion_repr.shape[1] <= 1:
            return motion_repr
        start, end = self.ctx.repr_indices["smplx_betas"]
        output = motion_repr.clone()
        beta_mean = output[..., start:end].mean(dim=1, keepdim=True)
        output[..., start:end] = beta_mean.expand(-1, output.shape[1], -1)
        return output

    def _apply_contact_median(self, motion_repr: torch.Tensor) -> torch.Tensor:
        if "foot_contact" not in self.ctx.repr_indices:
            return motion_repr

        kernel_size = int(getattr(self.ctx.test_config, "postprocess_contact_median_kernel", 3))
        threshold = float(getattr(self.ctx.test_config, "postprocess_contact_threshold", 0.5))
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size <= 1 or motion_repr.shape[1] <= 2:
            return motion_repr

        start, end = self.ctx.repr_indices["foot_contact"]
        output = motion_repr.clone()
        contact = (output[..., start:end] > threshold).to(dtype=output.dtype)
        votes = self._temporal_smooth_tensor(contact, kernel_size=kernel_size)
        output[..., start:end] = (votes > 0.5).to(dtype=output.dtype)
        return output

    def _apply_contact_binarize(self, motion_repr: torch.Tensor) -> torch.Tensor:
        if "foot_contact" not in self.ctx.repr_indices:
            return motion_repr

        threshold = float(getattr(self.ctx.test_config, "postprocess_contact_threshold", 0.5))
        start, end = self.ctx.repr_indices["foot_contact"]
        output = motion_repr.clone()
        output[..., start:end] = (output[..., start:end] > threshold).to(dtype=output.dtype)
        return output

    def _apply_kinematic_contact(
        self,
        motion_repr: torch.Tensor,
        pred_joints: torch.Tensor,
    ) -> torch.Tensor:
        """Overwrite or fuse foot-contact logits from prediction-only kinematics.

        This mirrors the dataset contact-label rule: contact is active when a
        foot joint is low enough and its frame-to-frame squared displacement is
        below a small threshold. The floor can be estimated from the prediction
        itself to avoid using GT floor height at test time.
        """
        if "foot_contact" not in self.ctx.repr_indices:
            return motion_repr
        if pred_joints.shape[1] <= 1:
            return motion_repr

        cfg = self.ctx.test_config
        joint_ids = getattr(cfg, "postprocess_contact_kinematic_joint_ids", None)
        if joint_ids is None:
            joint_ids = [7, 10, 8, 11]
        joint_ids = [int(i) for i in joint_ids if 0 <= int(i) < pred_joints.shape[2]]
        if not joint_ids:
            return motion_repr

        s_contact, e_contact = self.ctx.repr_indices["foot_contact"]
        contact_dim = max(0, e_contact - s_contact)
        if contact_dim <= 0:
            return motion_repr
        joint_ids = joint_ids[:contact_dim]
        if not joint_ids:
            return motion_repr

        height_thresholds = getattr(cfg, "postprocess_contact_kinematic_height_thresholds", None)
        if height_thresholds is None:
            height_thresholds = [0.18, 0.15, 0.18, 0.15]
        if isinstance(height_thresholds, (int, float)):
            height_thresholds = [float(height_thresholds)]
        height_thresholds = [float(v) for v in height_thresholds]
        if len(height_thresholds) < len(joint_ids):
            height_thresholds.extend([height_thresholds[-1] if height_thresholds else 0.15] * (
                len(joint_ids) - len(height_thresholds)
            ))
        height_thresholds = height_thresholds[:len(joint_ids)]
        height_thresholds_t = torch.as_tensor(
            height_thresholds,
            device=pred_joints.device,
            dtype=pred_joints.dtype,
        ).view(1, 1, -1)

        foot = pred_joints[:, :, joint_ids, :]
        foot_delta = foot[:, 1:] - foot[:, :-1]
        sq_disp = foot_delta.pow(2).sum(dim=-1)

        foot_height = foot[:, :-1, :, 2]
        floor_mode = str(getattr(cfg, "postprocess_contact_kinematic_floor_mode", "predicted_min"))
        if floor_mode == "predicted_min":
            floor_z = foot[..., 2].amin(dim=(1, 2), keepdim=True)
            foot_height = foot_height - floor_z
        elif floor_mode == "target_z":
            target_z = float(getattr(cfg, "postprocess_floor_lift_target_z", 0.0))
            foot_height = foot_height - target_z
        elif floor_mode != "absolute":
            raise ValueError(
                "postprocess_contact_kinematic_floor_mode must be one of "
                "{'predicted_min', 'absolute', 'target_z'}"
            )

        vel_thresh = max(0.0, float(getattr(cfg, "postprocess_contact_kinematic_vel_thresh", 5e-5)))
        contact = (sq_disp < vel_thresh) & (foot_height < height_thresholds_t)
        contact = contact.to(dtype=motion_repr.dtype)

        last_frame = str(getattr(cfg, "postprocess_contact_kinematic_last_frame", "repeat"))
        if last_frame == "repeat":
            contact = torch.cat([contact, contact[:, -1:, :]], dim=1)
        elif last_frame == "zero":
            pad = torch.zeros(
                contact.shape[0],
                1,
                contact.shape[2],
                device=contact.device,
                dtype=contact.dtype,
            )
            contact = torch.cat([contact, pad], dim=1)
        else:
            raise ValueError("postprocess_contact_kinematic_last_frame must be one of {'repeat', 'zero'}")

        output = motion_repr.clone()
        pred_contact = output[..., s_contact:s_contact + len(joint_ids)]
        mode = str(getattr(cfg, "postprocess_contact_kinematic_mode", "replace"))
        if mode == "replace":
            fused = contact
        else:
            pred_binary = (pred_contact > float(getattr(cfg, "postprocess_contact_threshold", 0.5))).to(
                dtype=motion_repr.dtype
            )
            if mode == "or":
                fused = torch.maximum(pred_binary, contact)
            elif mode == "and":
                fused = pred_binary * contact
            elif mode == "blend":
                weight = max(0.0, min(1.0, float(getattr(
                    cfg, "postprocess_contact_kinematic_blend_weight", 1.0
                ))))
                fused = pred_contact * (1.0 - weight) + contact * weight
            else:
                raise ValueError(
                    "postprocess_contact_kinematic_mode must be one of {'replace', 'or', 'and', 'blend'}"
                )
        output[..., s_contact:s_contact + len(joint_ids)] = fused
        return output

    def _apply_foot_lock(
        self,
        motion_repr: torch.Tensor,
        pred_joints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "smplx_trans" not in self.ctx.repr_indices or "foot_contact" not in self.ctx.repr_indices:
            return motion_repr, pred_joints
        if pred_joints.shape[1] <= 1:
            return motion_repr, pred_joints

        cfg = self.ctx.test_config
        joint_ids = getattr(cfg, "postprocess_foot_lock_joint_ids", None)
        if joint_ids is None:
            joint_ids = [7, 10, 8, 11]
        joint_ids = [int(i) for i in joint_ids if 0 <= int(i) < pred_joints.shape[2]]
        if not joint_ids:
            return motion_repr, pred_joints

        s_contact, e_contact = self.ctx.repr_indices["foot_contact"]
        contact = motion_repr[..., s_contact:e_contact]
        if contact.shape[-1] < len(joint_ids):
            joint_ids = joint_ids[:contact.shape[-1]]
        if len(joint_ids) == 0:
            return motion_repr, pred_joints
        contact = contact[..., :len(joint_ids)] > float(
            getattr(cfg, "postprocess_foot_lock_contact_threshold", 0.5)
        )
        if contact.shape[1] != pred_joints.shape[1]:
            return motion_repr, pred_joints
        if not torch.any(contact[:, 1:] & contact[:, :-1]):
            return motion_repr, pred_joints

        strength = max(0.0, min(1.0, float(getattr(cfg, "postprocess_foot_lock_strength", 1.0))))
        max_step = float(getattr(cfg, "postprocess_foot_lock_max_shift_per_frame", 0.03))
        max_total = float(getattr(cfg, "postprocess_foot_lock_max_total_shift", 0.20))
        no_contact_decay = max(0.0, min(1.0, float(getattr(cfg, "postprocess_foot_lock_no_contact_decay", 1.0))))
        if strength <= 0.0:
            return motion_repr, pred_joints

        bs, seq_len = pred_joints.shape[:2]
        foot_xy = pred_joints[:, :, joint_ids, :2]
        corrections = torch.zeros(bs, seq_len, 2, device=motion_repr.device, dtype=motion_repr.dtype)

        for t in range(1, seq_len):
            prev_corr = corrections[:, t - 1]
            pair_contact = contact[:, t] & contact[:, t - 1]
            weights = pair_contact.to(dtype=motion_repr.dtype)
            count = weights.sum(dim=-1, keepdim=True)
            target_corr = prev_corr * no_contact_decay
            active = count.squeeze(-1) > 0
            if torch.any(active):
                desired = foot_xy[:, t - 1] + prev_corr[:, None, :] - foot_xy[:, t]
                desired = (desired * weights[..., None]).sum(dim=1) / count.clamp_min(1.0)
                target_corr = torch.where(active[:, None], desired, target_corr)

            delta = (target_corr - prev_corr) * strength
            if max_step > 0.0:
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                delta_scale = (max_step / delta_norm.clamp_min(1e-8)).clamp(max=1.0)
                delta = delta * delta_scale
            corr = prev_corr + delta
            if max_total > 0.0:
                corr_norm = torch.linalg.vector_norm(corr, dim=-1, keepdim=True)
                corr_scale = (max_total / corr_norm.clamp_min(1e-8)).clamp(max=1.0)
                corr = corr * corr_scale
            corrections[:, t] = corr

        output = motion_repr.clone()
        s_trans, _ = self.ctx.repr_indices["smplx_trans"]
        output[..., s_trans:s_trans + 2] = output[..., s_trans:s_trans + 2] + corrections
        pred_joints = pred_joints.clone()
        pred_joints[..., :2] = pred_joints[..., :2] + corrections[:, :, None, :]
        return output, pred_joints

    def _apply_observed_joint_anchor(
        self,
        motion_repr: torch.Tensor,
        pred_joints: torch.Tensor,
        pose_batch: Optional[Dict[str, torch.Tensor]],
        visible_joint_mask: Optional[torch.Tensor],
        joint_modality_mask: Optional[torch.Tensor],
        joint_modality_names: Optional[List[str]],
        sample_norms: Optional[Dict[str, torch.Tensor]],
        cond_is_normalized: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "smplx_trans" not in self.ctx.repr_indices or pose_batch is None or "cond" not in pose_batch:
            return motion_repr, pred_joints

        observed = self._extract_observed_local_positions(
            pose_batch=pose_batch,
            visible_joint_mask=visible_joint_mask,
            joint_modality_mask=joint_modality_mask,
            joint_modality_names=joint_modality_names,
            sample_norms=sample_norms,
            cond_is_normalized=cond_is_normalized,
        )
        if observed is None:
            return motion_repr, pred_joints

        obs_pos, obs_mask = observed
        if obs_pos.shape[:3] != pred_joints.shape[:3]:
            return motion_repr, pred_joints

        cfg = self.ctx.test_config
        joint_ids = getattr(cfg, "postprocess_observed_joint_anchor_joint_ids", None)
        if joint_ids is not None:
            joint_ids = [int(i) for i in joint_ids if 0 <= int(i) < obs_mask.shape[2]]
            if not joint_ids:
                return motion_repr, pred_joints
            joint_ids_t = torch.as_tensor(joint_ids, device=obs_mask.device, dtype=torch.long)
            keep = torch.zeros_like(obs_mask)
            keep.index_fill_(dim=2, index=joint_ids_t, value=True)
            obs_mask = obs_mask & keep

        dims = getattr(cfg, "postprocess_observed_joint_anchor_dims", None)
        if dims is None:
            dims = [0, 1, 2]
        dims = [int(d) for d in dims if int(d) in (0, 1, 2)]
        if not dims:
            return motion_repr, pred_joints

        min_joints = max(1, int(getattr(cfg, "postprocess_observed_joint_anchor_min_joints", 1)))
        strength = max(0.0, min(1.0, float(getattr(cfg, "postprocess_observed_joint_anchor_strength", 1.0))))
        max_shift = float(getattr(cfg, "postprocess_observed_joint_anchor_max_shift", 0.25))

        weights = obs_mask.to(dtype=motion_repr.dtype).unsqueeze(-1)
        counts = weights.sum(dim=2, keepdim=False).clamp_min(1.0)
        delta = ((obs_pos - pred_joints) * weights).sum(dim=2) / counts
        enough = (weights.squeeze(-1).sum(dim=2) >= min_joints).unsqueeze(-1)
        delta = torch.where(enough, delta, torch.zeros_like(delta))

        dim_mask = torch.zeros_like(delta)
        dim_mask[..., dims] = 1.0
        delta = delta * dim_mask * strength

        if bool(getattr(cfg, "postprocess_observed_joint_anchor_smooth_delta", False)):
            kernel_size = int(getattr(cfg, "postprocess_observed_joint_anchor_smooth_kernel", 3))
            smooth_strength = max(
                0.0,
                min(1.0, float(getattr(cfg, "postprocess_observed_joint_anchor_smooth_strength", 0.5))),
            )
            if kernel_size > 1 and smooth_strength > 0.0:
                smoothed_delta = self._temporal_smooth_tensor(delta, kernel_size=kernel_size)
                delta = delta * (1.0 - smooth_strength) + smoothed_delta * smooth_strength

        if max_shift > 0.0:
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            scale = (max_shift / norm.clamp_min(1e-8)).clamp(max=1.0)
            delta = delta * scale

        s_trans, _ = self.ctx.repr_indices["smplx_trans"]
        output = motion_repr.clone()
        output[..., s_trans:s_trans + 3] = output[..., s_trans:s_trans + 3] + delta
        pred_joints = pred_joints + delta.unsqueeze(-2)
        return output, pred_joints

    def _apply_floor_lift(self, motion_repr: torch.Tensor, pred_joints: torch.Tensor) -> torch.Tensor:
        if "smplx_trans" not in self.ctx.repr_indices:
            return motion_repr

        cfg = self.ctx.test_config
        joint_ids = getattr(cfg, "postprocess_floor_lift_joint_ids", None)
        if joint_ids is None:
            joint_ids = [10, 11]
        joint_ids = [int(i) for i in joint_ids if 0 <= int(i) < pred_joints.shape[2]]
        if not joint_ids:
            return motion_repr
        target_z = float(getattr(cfg, "postprocess_floor_lift_target_z", 0.0))
        tolerance = max(0.0, float(getattr(cfg, "postprocess_floor_lift_tolerance", 0.01)))
        max_shift = float(getattr(cfg, "postprocess_floor_lift_max_shift", 0.10))

        foot_z = pred_joints[:, :, joint_ids, 2]
        mode = str(getattr(cfg, "postprocess_floor_lift_mode", "sequence"))
        if mode == "sequence":
            min_foot_z = foot_z.amin(dim=(1, 2), keepdim=True)
        elif mode == "frame":
            min_foot_z = foot_z.amin(dim=2, keepdim=True)
        else:
            raise ValueError("postprocess_floor_lift_mode must be one of {'sequence', 'frame'}")
        shift = (target_z - min_foot_z - tolerance).clamp_min(0.0)
        if max_shift > 0.0:
            shift = shift.clamp(max=max_shift)
        if mode == "frame":
            kernel_size = int(getattr(cfg, "postprocess_floor_lift_smooth_kernel", 1))
            strength = max(0.0, min(1.0, float(getattr(cfg, "postprocess_floor_lift_smooth_strength", 0.0))))
            if kernel_size > 1 and strength > 0.0:
                if kernel_size % 2 == 0:
                    kernel_size += 1
                smoothed_shift = self._temporal_smooth_tensor(shift, kernel_size=kernel_size)
                shift = shift * (1.0 - strength) + smoothed_shift * strength
        if not torch.any(shift > 0):
            return motion_repr

        output = motion_repr.clone()
        s_trans, _ = self.ctx.repr_indices["smplx_trans"]
        output[..., s_trans + 2:s_trans + 3] = output[..., s_trans + 2:s_trans + 3] + shift
        return output

    def _extract_observed_local_positions(
        self,
        pose_batch: Dict[str, torch.Tensor],
        visible_joint_mask: Optional[torch.Tensor],
        joint_modality_mask: Optional[torch.Tensor],
        joint_modality_names: Optional[List[str]],
        sample_norms: Optional[Dict[str, torch.Tensor]],
        cond_is_normalized: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if "cond" not in pose_batch:
            return None

        cond = pose_batch["cond"]
        if cond_is_normalized:
            cond = self.ctx.dataset._unnormalize_cond_tensor(
                cond,
                task="pose",
                device=self.ctx.device,
                sample_norms=sample_norms,
            )

        num_joints, per_joint_total, lp_offset, lp_dim = self._get_joint_cond_layout(task="pose")
        if lp_dim < 3:
            return None
        joint_dim = num_joints * per_joint_total
        observed_pos = cond[..., :joint_dim].reshape(
            cond.shape[0], cond.shape[1], num_joints, per_joint_total
        )[..., lp_offset:lp_offset + 3]

        if joint_modality_mask is None:
            joint_modality_mask = pose_batch.get("cond_joint_modality_mask")
        if joint_modality_names is None:
            joint_modality_names = pose_batch.get("cond_joint_modality_names")

        obs_mask = None
        if joint_modality_mask is not None and joint_modality_names is not None and "local_positions" in joint_modality_names:
            local_idx = joint_modality_names.index("local_positions")
            obs_mask = joint_modality_mask[..., local_idx]
        elif visible_joint_mask is not None:
            obs_mask = visible_joint_mask
        elif "cond_joint_mask" in pose_batch:
            obs_mask = pose_batch["cond_joint_mask"]
            if obs_mask.dim() == 4:
                obs_mask = obs_mask[..., 0]

        if obs_mask is None:
            return None
        obs_mask = obs_mask.to(device=observed_pos.device, dtype=torch.bool)
        return observed_pos, obs_mask

    def _recover_predicted_joints(self, motion_repr: torch.Tensor) -> Optional[torch.Tensor]:
        from data_sets.motion_representation import recover_from_repr_smpl

        required = ("smplx_rot_6d", "smplx_body_pose_6d", "smplx_trans", "smplx_betas")
        missing = [name for name in required if name not in self.ctx.repr_indices]
        if missing:
            if not getattr(self, "_warned_missing_fk_postprocess_modalities", False):
                print(
                    "[WARN] Skipping FK-dependent prediction postprocess because "
                    f"the active representation is missing modalities: {missing}"
                )
                self._warned_missing_fk_postprocess_modalities = True
            return None

        chunk_size = int(getattr(self.ctx.test_config, "postprocess_fk_chunk_size", 512))
        chunk_size = max(1, chunk_size)
        chunks = []
        for start_t in range(0, motion_repr.shape[1], chunk_size):
            end_t = min(start_t + chunk_size, motion_repr.shape[1])
            repr_chunk = motion_repr[:, start_t:end_t, :]
            data_dict = {}
            for name in required:
                s, e = self.ctx.repr_indices[name]
                data_dict[name] = repr_chunk[..., s:e]
            joints = recover_from_repr_smpl(
                data_dict,
                recover_mode="smplx_params",
                cano_jts_pos=False,
                smplx_model=self.ctx.fk_model,
                return_verts=False,
                return_full_joints=False,
            )
            chunks.append(joints)
        return torch.cat(chunks, dim=1)

    def _invalidate_derived_local_positions(self, motion_repr: torch.Tensor) -> torch.Tensor:
        if "local_positions" not in self.ctx.repr_indices:
            return motion_repr
        cfg = self.ctx.test_config
        smpl_param_postprocess_enabled = any([
            bool(getattr(cfg, "postprocess_rot6d_normalize", False)),
            bool(getattr(cfg, "postprocess_betas_sequence_average", False)),
            bool(getattr(cfg, "postprocess_temporal_smoothing", False)),
            bool(getattr(cfg, "postprocess_foot_lock", False)),
            bool(getattr(cfg, "postprocess_observed_joint_anchor", False)),
            bool(getattr(cfg, "postprocess_floor_lift", False)),
        ])
        if not smpl_param_postprocess_enabled:
            return motion_repr
        start, end = self.ctx.repr_indices["local_positions"]
        output = motion_repr.clone()
        output[..., start:end] = torch.inf
        return output

    @staticmethod
    def _temporal_smooth_tensor(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if tensor.dim() < 3 or tensor.shape[1] <= 2:
            return tensor
        kernel_size = min(kernel_size, tensor.shape[1] if tensor.shape[1] % 2 == 1 else tensor.shape[1] - 1)
        if kernel_size <= 1:
            return tensor
        pad = kernel_size // 2
        x = tensor.transpose(1, 2)
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)
        return x.transpose(1, 2)

    @staticmethod
    def _normalize_rot6d_feature(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] % 6 != 0:
            return tensor
        original_shape = tensor.shape
        rot6d = tensor.reshape(*tensor.shape[:-1], -1, 6)
        # The repository stores 6D rotations as the first two matrix columns
        # flattened from shape [3, 2], i.e. [c0x, c1x, c0y, c1y, c0z, c1z].
        a1 = F.normalize(rot6d[..., [0, 2, 4]], dim=-1, eps=1e-6)
        a2_raw = rot6d[..., [1, 3, 5]]
        a2 = a2_raw - (a1 * a2_raw).sum(dim=-1, keepdim=True) * a1
        a2 = F.normalize(a2, dim=-1, eps=1e-6)
        normalized = torch.stack((a1, a2), dim=-1).reshape(*rot6d.shape[:-1], 6)
        return normalized.reshape(original_shape)
    
    def _get_sample_norms(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Extract sample norms for l2 normalization strategy."""
        if self.ctx.norm_strategy == 'l2':
            return {
                k[len("extras/sample_norms/"):]: v
                for k, v in batch.items()
                if k.startswith('extras/sample_norms')
            }
        return None

    def _collect_window_beta_candidate(
        self,
        pose_out: torch.Tensor,
        valid_len: int,
        beta_range: Tuple[int, int],
        foot_contact_range: Tuple[int, int],
        contact_threshold: float,
        min_contact_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract one beta candidate per window and mark whether the window is valid.

        A window is valid only when foot-ground contact appears frequently enough,
        preventing stair/airborne segments from polluting global shape estimation.
        """
        if valid_len <= 0 or valid_len > pose_out.shape[1]:
            raise ValueError(
                f"valid_len must be in (0, {pose_out.shape[1]}], got {valid_len}."
            )

        s_beta, e_beta = beta_range
        s_fc, e_fc = foot_contact_range
        if e_beta <= s_beta:
            raise ValueError(f"Invalid smplx_betas range: [{s_beta}, {e_beta}).")
        if e_fc <= s_fc:
            raise ValueError(f"Invalid foot_contact range: [{s_fc}, {e_fc}).")

        window_pose = pose_out[:, :valid_len, :]
        window_betas = window_pose[..., s_beta:e_beta]
        window_contact = window_pose[..., s_fc:e_fc]

        if not torch.isfinite(window_betas).all():
            nan_count = int(torch.isnan(window_betas).sum().item())
            inf_count = int(torch.isinf(window_betas).sum().item())
            raise ValueError(
                "Window betas contain non-finite values; "
                f"nan={nan_count}, inf={inf_count}."
            )
        if not torch.isfinite(window_contact).all():
            nan_count = int(torch.isnan(window_contact).sum().item())
            inf_count = int(torch.isinf(window_contact).sum().item())
            raise ValueError(
                "Window foot_contact contains non-finite values; "
                f"nan={nan_count}, inf={inf_count}."
            )

        beta_candidate = window_betas.mean(dim=1)
        frame_has_contact = (window_contact > contact_threshold).any(dim=-1)
        contact_ratio = frame_has_contact.float().mean(dim=1)
        valid_mask = contact_ratio >= min_contact_ratio
        return beta_candidate, valid_mask, contact_ratio

    def _fuse_window_betas(
        self,
        window_beta_candidates: List[torch.Tensor],
        window_beta_valid_masks: List[torch.Tensor],
        window_beta_contact_ratios: List[torch.Tensor],
        geom_eps: float,
    ) -> torch.Tensor:
        """
        Fuse per-window betas into one sequence-level shape using signed geometric mean.
        """
        if len(window_beta_candidates) == 0:
            raise RuntimeError("No beta candidates were collected from sliding windows.")
        if len(window_beta_candidates) != len(window_beta_valid_masks):
            raise RuntimeError("Mismatch between number of beta candidates and validity masks.")
        if len(window_beta_candidates) != len(window_beta_contact_ratios):
            raise RuntimeError("Mismatch between number of beta candidates and contact ratios.")

        beta_stack = torch.stack(window_beta_candidates, dim=0)  # [W, B, D_beta]
        valid_stack = torch.stack(window_beta_valid_masks, dim=0)  # [W, B]
        ratio_stack = torch.stack(window_beta_contact_ratios, dim=0)  # [W, B]

        if beta_stack.dim() != 3:
            raise RuntimeError(
                f"Expected beta_stack to be rank-3 [W, B, D], got shape {tuple(beta_stack.shape)}."
            )
        if valid_stack.shape != beta_stack.shape[:2]:
            raise RuntimeError(
                f"valid_stack shape {tuple(valid_stack.shape)} is incompatible with "
                f"beta_stack shape {tuple(beta_stack.shape)}."
            )
        if ratio_stack.shape != beta_stack.shape[:2]:
            raise RuntimeError(
                f"ratio_stack shape {tuple(ratio_stack.shape)} is incompatible with "
                f"beta_stack shape {tuple(beta_stack.shape)}."
            )

        _, bs, beta_dim = beta_stack.shape
        fused = torch.zeros((bs, beta_dim), device=beta_stack.device, dtype=beta_stack.dtype)

        for b_idx in range(bs):
            valid_idx = valid_stack[:, b_idx]
            valid_count = int(valid_idx.sum().item())
            if valid_count == 0:
                ratios = ratio_stack[:, b_idx].detach().cpu().tolist()
                raise RuntimeError(
                    "No valid beta windows for sample index "
                    f"{b_idx}. contact_ratios_per_window={ratios}. "
                    "Adjust masking strategy or beta_fusion_min_contact_ratio."
                )

            betas_for_sample = beta_stack[valid_idx, b_idx, :]
            beta_sign = torch.sign(betas_for_sample.mean(dim=0))
            beta_magnitude = torch.exp(
                torch.log(betas_for_sample.abs().clamp_min(geom_eps)).mean(dim=0)
            )
            fused[b_idx] = beta_sign * beta_magnitude

        if not torch.isfinite(fused).all():
            nan_count = int(torch.isnan(fused).sum().item())
            inf_count = int(torch.isinf(fused).sum().item())
            raise ValueError(
                "Fused betas contain non-finite values; "
                f"nan={nan_count}, inf={inf_count}."
            )
        return fused
    
    def _compute_window_root_init_xy(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Compute an XY origin for a sliding window without reading hidden joints."""
        bs = win_traj['cond'].shape[0]
        window_size = win_traj['cond'].shape[1]

        num_joints, per_joint_total, lp_offset, lp_dim = self._get_joint_cond_layout(task='traj')
        joint_dim = num_joints * per_joint_total
        joint_part = win_traj['cond'][:, :, :joint_dim].reshape(bs, window_size, num_joints, per_joint_total)
        local_pos = joint_part[..., lp_offset:lp_offset + lp_dim]

        if not getattr(self.ctx.test_config, 'use_root_init_xy_in_sliding_window_preprocessing', True):
            policy = "visible_mean"
        else:
            policy = str(getattr(self.ctx.test_config, 'root_init_xy_policy', 'visible_safe'))

        root_xy = local_pos[:, 0:1, 0:1, :2].clone()
        if policy == "legacy_root":
            return root_xy
        if policy == "zero":
            return torch.zeros_like(root_xy)
        if policy not in {"visible_safe", "visible_mean"}:
            raise ValueError(
                "root_init_xy_policy must be one of "
                "{'visible_safe', 'visible_mean', 'legacy_root', 'zero'}, "
                f"got {policy!r}."
            )

        joint_mask = self._window_joint_visibility_after_masking(win_traj, masking_spec)
        visible_xy = self._first_frame_visible_xy_mean(local_pos, joint_mask)
        if policy == "visible_mean":
            return visible_xy

        if joint_mask is None:
            return root_xy
        if joint_mask.dim() == 4:
            root_visible = joint_mask[:, 0:1, 0:1, 0]
        else:
            root_visible = joint_mask[:, 0:1, 0:1]
        return torch.where(root_visible.unsqueeze(-1), root_xy, visible_xy)

    def _window_joint_visibility_after_masking(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple],
    ) -> Optional[torch.Tensor]:
        """Return cond_joint_mask for the current test mask without mutating the window."""
        if masking_spec is None:
            return win_traj.get('cond_joint_mask')

        from utils.masking_actions import apply_mask_action_plan

        mask_action, _ = masking_spec
        dry_batch: Dict[str, Any] = {'cond': win_traj['cond'].clone()}
        if 'cond_joint_mask' in win_traj and win_traj['cond_joint_mask'] is not None:
            dry_batch['cond_joint_mask'] = win_traj['cond_joint_mask'].clone()

        apply_mask_action_plan(
            dry_batch,
            self.ctx.dataset,
            'traj',
            mask_action,
            self.ctx.joints_num,
            getattr(self.ctx.test_config, "test_batch_policy", "whole_batch"),
        )
        return dry_batch.get('cond_joint_mask')

    @staticmethod
    def _first_frame_visible_xy_mean(
        local_pos: torch.Tensor,
        joint_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average first-frame visible joint XY, returning zero if no joint is visible."""
        root_xy = local_pos[:, 0:1, 0:1, :2].clone()
        if joint_mask is None:
            return root_xy
        if joint_mask.dim() == 4:
            joint_vis = joint_mask[:, 0:1, :, 0]
        else:
            joint_vis = joint_mask[:, 0:1, :]
        first_frame_joints = local_pos[:, 0:1, :, :2]
        vis = joint_vis.unsqueeze(-1).expand_as(first_frame_joints)
        visible_joints = torch.where(vis, first_frame_joints, torch.zeros_like(first_frame_joints))
        visibility_count = joint_vis.sum(dim=2, keepdim=True).clamp(min=1).unsqueeze(-1)
        return visible_joints.sum(dim=2, keepdim=True) / visibility_count
    
    def _apply_root_offset_to_window(
        self,
        win_traj: Dict[str, torch.Tensor],
        win_pose: Dict[str, torch.Tensor],
        win_root_init_xy: torch.Tensor,
    ):
        """
        Apply root XY offset to window cond tensors.
        win_traj['cond'] and win_pose['cond'] are laid out as [joint-based modalities | global modalities].
        All modalities that carry absolute position information are shifted. Currently covered:
        - joint-based: local_positions (XY)
        - global: smplx_trans (XY)
        """
        self._apply_root_offset_to_cond(win_traj['cond'], task='traj', win_root_init_xy=win_root_init_xy, mode='subtract')
        self._apply_root_offset_to_cond(win_pose['cond'], task='pose', win_root_init_xy=win_root_init_xy, mode='subtract')

        
    def _apply_root_offset_to_motion_repr(
        self,
        win_traj: Dict[str, torch.Tensor],
        win_pose: Dict[str, torch.Tensor],
        win_root_init_xy: torch.Tensor,
    ):
        """Apply root XY offset to motion_repr tensors."""
        if self.ctx.test_config.with_gt:
            self._apply_root_offset_to_full_repr(win_traj['motion_repr_clean'], win_root_init_xy, mode='subtract')
            self._apply_root_offset_to_full_repr(win_pose['motion_repr_clean'], win_root_init_xy, mode='subtract')
            
            if self.ctx.input_noise:
                if 'motion_repr_noisy' in win_traj:
                    self._apply_root_offset_to_full_repr(win_traj['motion_repr_noisy'], win_root_init_xy, mode='subtract')
                if 'motion_repr_noisy' in win_pose:
                    self._apply_root_offset_to_full_repr(win_pose['motion_repr_noisy'], win_root_init_xy, mode='subtract')
    
    def _apply_egoexo_offset_recovery(
        self,
        motion_repr_rec: torch.Tensor,
        original_batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply EgoExo-specific offset recovery."""
        s_trans, _ = self.ctx.repr_indices['smplx_trans']
        
        norm_params = original_batch['meta']['normalization_params']
        translation_xy_offset = norm_params.get('translation_xy_offset')
        floor_height_offset = norm_params.get('floor_height_offset')
        
        if translation_xy_offset is not None:
            if isinstance(translation_xy_offset, torch.Tensor):
                translation_xy_offset = translation_xy_offset[0].cpu().numpy()
            motion_repr_rec[:, :, s_trans] += translation_xy_offset[0]
            motion_repr_rec[:, :, s_trans+1] += translation_xy_offset[1]
        
        if floor_height_offset is not None:
            if isinstance(floor_height_offset, torch.Tensor):
                floor_height_offset = floor_height_offset.item()
            motion_repr_rec[:, :, s_trans+2] += floor_height_offset
    
        return motion_repr_rec

    def _get_joint_cond_layout(self, task: str) -> Tuple[int, int, int, int]:
        """
        Get joint-based layout info for the cond tensor.

        Returns:
            (num_joints, per_joint_total, local_pos_offset, local_pos_dim)
        """
        registry = self.ctx.repr_registry
        cond_names = registry.cond_modalities.get(task)
        joint_modalities = []
        for name in cond_names:
            mod = registry.get_modality(name)
            if mod.is_joint_based:
                joint_modalities.append(mod)

        if not joint_modalities:
            raise ValueError(f"No joint-based modalities found for task '{task}'")

        num_joints = registry.joint_count_for_topology(joint_modalities[0].topology)
        per_joint_total = sum(mod.per_joint_dim or 0 for mod in joint_modalities)

        local_pos_offset = 0
        local_pos_dim = None
        cur = 0
        for mod in joint_modalities:
            if mod.name == 'local_positions':
                local_pos_offset = cur
                local_pos_dim = mod.per_joint_dim or 0
                break
            cur += mod.per_joint_dim or 0

        if local_pos_dim is None:
            raise ValueError(f"'local_positions' not found in joint modalities for task '{task}'")

        return num_joints, per_joint_total, local_pos_offset, local_pos_dim

    def _apply_root_offset_to_cond(
        self,
        cond_tensor: torch.Tensor,
        task: str,
        win_root_init_xy: torch.Tensor,
        mode: str,
    ) -> None:
        """Apply root XY offset to cond tensor in-place."""
        if mode not in ("subtract", "add"):
            raise ValueError(f"Unsupported mode: {mode}")
        sign = -1.0 if mode == "subtract" else 1.0
        bs, window_size = cond_tensor.shape[:2]

        cond_names = self.ctx.repr_registry.cond_modalities.get(task, [])
        if "local_positions" in cond_names and "local_positions" in self._ROOT_OFFSET_MODALITIES:
            num_joints, per_joint_total, lp_offset, _ = self._get_joint_cond_layout(task=task)
            joint_dim = num_joints * per_joint_total
            joint_part = cond_tensor[:, :, :joint_dim].reshape(bs, window_size, num_joints, per_joint_total)
            joint_part[..., lp_offset:lp_offset + 2] = (
                joint_part[..., lp_offset:lp_offset + 2] + sign * win_root_init_xy
            )
            cond_tensor[:, :, :joint_dim] = joint_part.reshape(bs, window_size, -1).clone()

        cond_indices = get_cond_modality_indices(self.ctx.dataset, task=task)
        for name in self._ROOT_OFFSET_MODALITIES:
            mod_range = cond_indices.get(name)
            if not mod_range:
                continue
            mod = self.ctx.repr_registry.get_modality(name)
            if mod.is_joint_based:
                continue
            s_mod, _ = mod_range
            cond_tensor[:, :, s_mod:s_mod + 2] = (
                cond_tensor[:, :, s_mod:s_mod + 2] + sign * win_root_init_xy[:, 0, :, :]
            )

    def _apply_root_offset_to_full_repr(
        self,
        full_repr: torch.Tensor,
        win_root_init_xy: torch.Tensor, # [B, 1, 1, 2]
        mode: str,
    ) -> None:
        """Apply root XY offset to full motion_repr tensor in-place."""
        if mode not in ("subtract", "add"):
            raise ValueError(f"Unsupported mode: {mode}")
        sign = -1.0 if mode == "subtract" else 1.0
        root_offset = win_root_init_xy.squeeze(1) # [B, 1, 2]
        registry = self.ctx.repr_registry
        for name in self._ROOT_OFFSET_MODALITIES:
            if name not in self.ctx.repr_indices:
                continue
            mod = registry.get_modality(name)
            s_mod, e_mod = self.ctx.repr_indices[name]
            if mod.is_joint_based:
                num_joints = self.ctx.joints_num
                per_joint_dim = mod.per_joint_dim or 0
                if per_joint_dim < 2:
                    continue
                local_pos = full_repr[:, :, s_mod:e_mod].reshape(
                    full_repr.shape[0], full_repr.shape[1], num_joints, per_joint_dim
                )
                local_pos[..., :2] = local_pos[..., :2] + sign * root_offset[:, :, None, :]
                full_repr[:, :, s_mod:e_mod] = local_pos.reshape(full_repr.shape[0], full_repr.shape[1], -1)
            else:
                full_repr[:, :, s_mod:s_mod + 2] = full_repr[:, :, s_mod:s_mod + 2] + sign * root_offset

    def _apply_root_offset_to_global_modalities(
        self,
        full_repr: torch.Tensor,
        win_root_init_xy: torch.Tensor,
        mode: str,
    ) -> None:
        """
        Apply root XY offset only on global trajectory modalities.

        This method is for partially materialized full representations where
        local pose modalities can legally contain placeholders (e.g., inf).
        """
        if mode not in ("subtract", "add"):
            raise ValueError(f"Unsupported mode: {mode}")
        if "smplx_trans" not in self.ctx.repr_indices:
            raise KeyError("Expected modality 'smplx_trans' in repr_indices for root-offset restoration.")

        sign = -1.0 if mode == "subtract" else 1.0
        root_offset = win_root_init_xy.squeeze(1)  # [B, 1, 2]
        s_tr, _ = self.ctx.repr_indices["smplx_trans"]
        full_repr[:, :, s_tr:s_tr + 2] = full_repr[:, :, s_tr:s_tr + 2] + sign * root_offset

    def _assert_modalities_finite(
        self,
        tensor: torch.Tensor,
        modality_names: Tuple[str, ...],
        tensor_name: str,
    ) -> None:
        """Fail loudly if required modalities contain NaN/Inf."""
        for name in modality_names:
            if name not in self.ctx.repr_indices:
                raise KeyError(f"Missing required modality '{name}' in repr_indices.")
            s, e = self.ctx.repr_indices[name]
            view = tensor[..., s:e]
            if not torch.isfinite(view).all():
                nan_count = int(torch.isnan(view).sum().item())
                inf_count = int(torch.isinf(view).sum().item())
                raise ValueError(
                    f"{tensor_name}.{name} contains non-finite values: "
                    f"nan={nan_count}, inf={inf_count}."
                )

    def _apply_root_offset_to_predicted_repr(
        self,
        predicted_repr: torch.Tensor,
        task: str,
        win_root_init_xy: torch.Tensor,
        mode: str,
    ) -> None:
        """Apply root XY offset to predicted-modalities layout in-place."""
        if mode not in ("subtract", "add"):
            raise ValueError(f"Unsupported mode: {mode}")
        sign = -1.0 if mode == "subtract" else 1.0
        predicted = self.ctx.repr_registry.predicted_modalities.get(task, [])
        if not predicted:
            return

        bs, T, _ = predicted_repr.shape
        cur = 0
        for name in predicted:
            s, e = self.ctx.repr_indices[name]
            width = e - s
            if name not in self._ROOT_OFFSET_MODALITIES:
                cur += width
                continue
            mod = self.ctx.repr_registry.get_modality(name)
            if mod.is_joint_based:
                per_joint_dim = mod.per_joint_dim or 0
                if per_joint_dim < 2:
                    cur += width
                    continue
                num_joints = self.ctx.joints_num
                local_pos = predicted_repr[:, :, cur:cur + width].reshape(bs, T, num_joints, per_joint_dim)
                local_pos[..., :2] = local_pos[..., :2] + sign * win_root_init_xy
                predicted_repr[:, :, cur:cur + width] = local_pos.reshape(bs, T, -1)
            else:
                predicted_repr[:, :, cur:cur + 2] = (
                    predicted_repr[:, :, cur:cur + 2] + sign * win_root_init_xy[:, 0, :, :]
                )
            cur += width
    
    def _build_repr_dict_from_batch(self, motion_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build representation dictionary from motion representation tensor."""
        from data_sets.motion_representation import populate_repr_dict
        
        repr_dict = {}
        for repr_name in self.ctx.repr_list:
            s, e = self.ctx.repr_indices[repr_name]
            repr_dict[repr_name] = motion_repr[..., s:e].to(self.ctx.device)
        
        pose_predicted = self.ctx.repr_registry.predicted_modalities.get('pose', [])
        include_global_joint_rotations = 'R_world_joints' in pose_predicted or 'R_root_joints' in pose_predicted
        
        if include_global_joint_rotations:
            repr_dict = populate_repr_dict(repr_dict, self.ctx.smplx_model, recover_global=False)
        else:
            repr_dict = populate_repr_dict(repr_dict, self.ctx.smplx_model, recover_global=True)
        
        return repr_dict

    @staticmethod
    def _read_required_cfg_value(cfg_obj: Any, key: str, owner: str) -> Any:
        """
        Read a required key from DictConfig/dict/object and fail loudly if absent.

        This keeps preprocessing semantics explicit and prevents silent fallback
        when train/test configuration contracts diverge.
        """
        _missing = object()

        if isinstance(cfg_obj, dict):
            if key not in cfg_obj:
                raise KeyError(f"Missing required key '{key}' in {owner}.")
            return cfg_obj[key]

        if hasattr(cfg_obj, "get"):
            value = cfg_obj.get(key, _missing)
            if value is not _missing:
                return value

        if hasattr(cfg_obj, key):
            return getattr(cfg_obj, key)

        raise KeyError(f"Missing required key '{key}' in {owner}.")

    def _resolve_window_align_to_yaxis_flag(self) -> bool:
        """
        Resolve the alignment contract from train_config and validate test dataset consistency.

        Sliding-window canonicalization must follow the representation contract used
        during training, otherwise full_sequence inference diverges from random clip
        inference semantics.
        """
        train_datasets = self._read_required_cfg_value(
            self.ctx.train_config,
            "train_datasets",
            owner="train_config",
        )
        if len(train_datasets) == 0:
            raise ValueError("train_config.train_datasets must contain at least one dataset config.")

        train_dataset_entry = train_datasets[0]
        train_dataset_cfg = self._read_required_cfg_value(
            train_dataset_entry,
            "dataset",
            owner="train_config.train_datasets[0]",
        )
        train_align_to_yaxis = bool(
            self._read_required_cfg_value(
                train_dataset_cfg,
                "align_to_yaxis",
                owner="train_config.train_datasets[0].dataset",
            )
        )

        if not hasattr(self.ctx.dataset, "align_to_yaxis"):
            raise AttributeError("Inference dataset is missing required attribute 'align_to_yaxis'.")

        test_align_to_yaxis = bool(self.ctx.dataset.align_to_yaxis)
        if test_align_to_yaxis != train_align_to_yaxis:
            raise RuntimeError(
                "align_to_yaxis mismatch between train/test configs: "
                f"train={train_align_to_yaxis}, test={test_align_to_yaxis}. "
                "Fix the dataset config mismatch instead of masking it in inference."
            )

        return train_align_to_yaxis

    def _compute_window_root_init_rot(
        self,
        win_traj: Dict[str, torch.Tensor],
        align_to_yaxis: bool,
        masking_spec: Optional[tuple] = None,
    ) -> Optional[torch.Tensor]:
        if not align_to_yaxis:
            return None

        policy = str(getattr(self.ctx.test_config, "root_init_rot_policy", "disabled"))
        if policy == "disabled":
            return None

        s, e = self.ctx.repr_indices.get('local_positions', (None, None))
        if s is None:
            return None

        bs = win_traj['motion_repr_clean'].shape[0]
        if policy == "legacy_gt":
            if not getattr(self, "_warned_legacy_gt_root_init_rot", False):
                print(
                    "[WARN] root_init_rot_policy=legacy_gt reads motion_repr_clean "
                    "for per-window yaw; use only for reproducing legacy runs."
                )
                self._warned_legacy_gt_root_init_rot = True
            local_pos0 = win_traj['motion_repr_clean'][:, 0, s:e].view(bs, -1, 3).detach().cpu()
            joint_visible = torch.ones(
                bs,
                local_pos0.shape[1],
                dtype=torch.bool,
                device=local_pos0.device,
            )
        elif policy == "visible_safe":
            local_pos0, joint_visible = self._first_frame_visible_local_positions(
                win_traj,
                masking_spec=masking_spec,
            )
            if local_pos0 is None or joint_visible is None:
                return self._rotation_to_yaxis_from_visible_rotations(
                    win_traj,
                    masking_spec=masking_spec,
                )
            local_pos0 = local_pos0.detach().cpu()
            joint_visible = joint_visible.detach().cpu().bool()
            rot_mat = self._rotation_to_yaxis_from_visible_pairs(
                local_pos0,
                joint_visible,
                device=win_traj['motion_repr_clean'].device,
            )
            if rot_mat is not None:
                return rot_mat
            return self._rotation_to_yaxis_from_visible_rotations(
                win_traj,
                masking_spec=masking_spec,
            )
        elif policy == "visible_rotation":
            return self._rotation_to_yaxis_from_visible_rotations(
                win_traj,
                masking_spec=masking_spec,
            )
        else:
            raise ValueError(
                "root_init_rot_policy must be one of "
                "{'disabled', 'visible_safe', 'visible_rotation', 'legacy_gt'}, "
                f"got {policy!r}."
            )

    def _first_frame_visible_local_positions(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_joints, per_joint_total, lp_offset, lp_dim = self._get_joint_cond_layout(task='traj')
        if lp_dim < 3:
            return None, None
        bs, window_size = win_traj['cond'].shape[:2]
        joint_dim = num_joints * per_joint_total
        joint_part = win_traj['cond'][:, :, :joint_dim].reshape(bs, window_size, num_joints, per_joint_total)
        local_pos0 = joint_part[:, 0, :, lp_offset:lp_offset + 3]

        joint_mask = self._window_joint_visibility_after_masking(win_traj, masking_spec)
        if joint_mask is None:
            joint_visible = torch.ones(
                bs,
                num_joints,
                dtype=torch.bool,
                device=local_pos0.device,
            )
        elif joint_mask.dim() == 4:
            joint_visible = joint_mask[:, 0, :, 0].bool()
        else:
            joint_visible = joint_mask[:, 0, :].bool()
        return local_pos0, joint_visible

    def _first_frame_visible_joint_rotations(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple],
        modality_name: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        registry = self.ctx.repr_registry
        cond_names = registry.cond_modalities.get("traj", [])
        joint_modalities = []
        for name in cond_names:
            mod = registry.get_modality(name)
            if mod.is_joint_based:
                joint_modalities.append(mod)

        if not joint_modalities:
            return None, None

        num_joints = registry.joint_count_for_topology(joint_modalities[0].topology)
        per_joint_total = sum(mod.per_joint_dim or 0 for mod in joint_modalities)
        offset = 0
        rot_dim = None
        for mod in joint_modalities:
            dim = mod.per_joint_dim or 0
            if mod.name == modality_name:
                rot_dim = dim
                break
            offset += dim

        if rot_dim != 6:
            return None, None

        bs, window_size = win_traj['cond'].shape[:2]
        joint_dim = num_joints * per_joint_total
        joint_part = win_traj['cond'][:, :, :joint_dim].reshape(bs, window_size, num_joints, per_joint_total)
        rot6d0 = joint_part[:, 0, :, offset:offset + 6]

        joint_visible = self._first_frame_visible_modality_mask(
            win_traj,
            masking_spec=masking_spec,
            modality_name=modality_name,
            num_joints=num_joints,
        )
        return rot6d0, joint_visible

    def _first_frame_visible_modality_mask(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple],
        modality_name: str,
        num_joints: int,
    ) -> Optional[torch.Tensor]:
        joint_mask = self._window_joint_visibility_after_masking(win_traj, masking_spec)
        if joint_mask is not None:
            if joint_mask.dim() == 4:
                base_visible = joint_mask[:, 0, :, 0].bool()
            else:
                base_visible = joint_mask[:, 0, :].bool()
        else:
            base_visible = torch.ones(
                win_traj['cond'].shape[0],
                num_joints,
                dtype=torch.bool,
                device=win_traj['cond'].device,
            )

        modality_mask = win_traj.get('cond_joint_modality_mask')
        if modality_mask is None:
            return base_visible

        modality_names = win_traj.get('cond_joint_modality_names')
        if modality_names is None:
            return base_visible
        if isinstance(modality_names, tuple):
            modality_names = list(modality_names)
        if modality_name not in modality_names:
            return torch.zeros_like(base_visible)

        mod_idx = modality_names.index(modality_name)
        if modality_mask.dim() == 4:
            mod_visible = modality_mask[:, 0, :, mod_idx].bool()
        elif modality_mask.dim() == 5:
            mod_visible = modality_mask[:, 0, :, mod_idx, 0].bool()
        else:
            return base_visible
        return base_visible & mod_visible

    @staticmethod
    def _rotation_to_yaxis_from_visible_pairs(
        local_pos0: torch.Tensor,
        joint_visible: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        from data_sets.motion_representation import face_joint_indx
        from data_sets.common.quaternion import qbetween
        from scipy.spatial.transform import Rotation as R

        bs = local_pos0.shape[0]
        dtype = local_pos0.dtype
        l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
        pair_defs = ((l_hip, r_hip), (sdr_l, sdr_r))
        across_sum = torch.zeros(bs, 3, dtype=dtype, device=local_pos0.device)
        pair_count = torch.zeros(bs, 1, dtype=dtype, device=local_pos0.device)

        for left_idx, right_idx in pair_defs:
            if left_idx >= local_pos0.shape[1] or right_idx >= local_pos0.shape[1]:
                continue
            pair_visible = joint_visible[:, left_idx] & joint_visible[:, right_idx]
            pair_vec = local_pos0[:, right_idx] - local_pos0[:, left_idx]
            pair_norm = torch.linalg.vector_norm(pair_vec, dim=-1)
            pair_valid = pair_visible & torch.isfinite(pair_vec).all(dim=-1) & (pair_norm > 1e-6)
            across_sum = across_sum + torch.where(pair_valid[:, None], pair_vec, torch.zeros_like(pair_vec))
            pair_count = pair_count + pair_valid.to(dtype=dtype).unsqueeze(-1)

        valid = pair_count.squeeze(-1) > 0
        if not torch.any(valid):
            return None

        across = across_sum / pair_count.clamp_min(1.0)
        across_norm = torch.linalg.vector_norm(across, dim=-1, keepdim=True)
        valid = valid & torch.isfinite(across).all(dim=-1) & (across_norm.squeeze(-1) > 1e-6)
        if not torch.any(valid):
            return None
        across = torch.where(valid[:, None], across / across_norm.clamp_min(1e-6), torch.zeros_like(across))

        z_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=local_pos0.device).expand(bs, -1)
        forward = torch.cross(z_axis, across, dim=-1)
        forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
        valid = valid & torch.isfinite(forward).all(dim=-1) & (forward_norm.squeeze(-1) > 1e-6)
        if not torch.any(valid):
            return None
        forward = torch.where(valid[:, None], forward / forward_norm.clamp_min(1e-6), torch.zeros_like(forward))

        target = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=local_pos0.device).expand(bs, -1)
        root_rot_quat = torch.zeros(bs, 4, dtype=dtype, device=local_pos0.device)
        root_rot_quat[:, 0] = 1.0
        root_rot_quat[valid] = qbetween(forward[valid], target[valid])
        valid = valid & torch.isfinite(root_rot_quat).all(dim=-1)
        if not torch.any(valid):
            return None

        rot_mat = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(bs, 1, 1)
        valid_quat = root_rot_quat[valid].numpy()
        valid_rot = torch.from_numpy(R.from_quat(valid_quat, scalar_first=True).as_matrix()).float()
        rot_mat[valid.cpu()] = valid_rot
        return rot_mat.to(device)

    def _rotation_to_yaxis_from_visible_rotations(
        self,
        win_traj: Dict[str, torch.Tensor],
        masking_spec: Optional[tuple],
    ) -> Optional[torch.Tensor]:
        modality_name = str(getattr(
            self.ctx.test_config,
            "root_init_rot_visible_rotation_modality",
            "R_world_joints",
        ))
        rot6d0, joint_visible = self._first_frame_visible_joint_rotations(
            win_traj,
            masking_spec=masking_spec,
            modality_name=modality_name,
        )
        if rot6d0 is None or joint_visible is None:
            return None

        joint_ids = getattr(self.ctx.test_config, "root_init_rot_visible_rotation_joint_ids", None)
        if joint_ids is None:
            preferred = [15, 0]
            joint_ids = preferred + [i for i in range(rot6d0.shape[1]) if i not in preferred]
        joint_ids = [int(i) for i in joint_ids if 0 <= int(i) < rot6d0.shape[1]]
        if not joint_ids:
            return None

        axis_name = str(getattr(self.ctx.test_config, "root_init_rot_visible_rotation_axis", "y"))
        axis_sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_key = axis_name[1:] if axis_name.startswith("-") else axis_name
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis_key]

        from data_sets.common.quaternion import qbetween, rot6d_to_rotmat
        from scipy.spatial.transform import Rotation as R

        bs = rot6d0.shape[0]
        dtype = rot6d0.dtype
        rot_device = rot6d0.device
        rot_mats = rot6d_to_rotmat(rot6d0.reshape(-1, 6)).reshape(bs, rot6d0.shape[1], 3, 3)

        forward = torch.zeros(bs, 3, dtype=dtype, device=rot_device)
        valid = torch.zeros(bs, dtype=torch.bool, device=rot_device)
        for joint_idx in joint_ids:
            candidate = rot_mats[:, joint_idx, :, axis_idx] * axis_sign
            candidate = candidate.clone()
            candidate[:, 2] = 0.0
            candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
            candidate_valid = (
                joint_visible[:, joint_idx]
                & torch.isfinite(candidate).all(dim=-1)
                & (candidate_norm > 1e-6)
                & (~valid)
            )
            if torch.any(candidate_valid):
                forward[candidate_valid] = candidate[candidate_valid] / candidate_norm[candidate_valid, None]
                valid[candidate_valid] = True

        if not torch.any(valid):
            return None

        target = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device_cpu).expand(bs, -1)
        root_rot_quat = torch.zeros(bs, 4, dtype=dtype, device=device_cpu)
        root_rot_quat[:, 0] = 1.0
        root_rot_quat[valid] = qbetween(forward[valid], target[valid])
        valid = valid & torch.isfinite(root_rot_quat).all(dim=-1)
        if not torch.any(valid):
            return None

        rot_mat = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(bs, 1, 1)
        valid_quat = root_rot_quat[valid].detach().cpu().numpy()
        valid_rot = torch.from_numpy(R.from_quat(valid_quat, scalar_first=True).as_matrix()).float()
        rot_mat[valid.detach().cpu()] = valid_rot
        return rot_mat.to(win_traj['motion_repr_clean'].device)

    def _get_predicted_modality_ranges(self, task: str) -> Dict[str, Tuple[int, int]]:
        """Build per-modality index ranges inside predicted-layout tensors."""
        predicted = self.ctx.repr_registry.predicted_modalities.get(task, [])
        ranges: Dict[str, Tuple[int, int]] = {}
        cur = 0
        for name in predicted:
            if name not in self.ctx.repr_indices:
                raise KeyError(f"Modality '{name}' from predicted_modalities[{task}] is missing in repr_indices.")
            s_full, e_full = self.ctx.repr_indices[name]
            width = e_full - s_full
            ranges[name] = (cur, cur + width)
            cur += width
        return ranges

    def _materialize_local_positions_from_smplx(
        self,
        full_repr: torch.Tensor,
        tensor_name: str,
    ) -> None:
        """
        Rebuild local_positions from SMPL-X modalities in-place when placeholders exist.

        This is required before any rotate_repr_smpl_2-equivalent transform that couples
        local_positions with smplx_trans.
        """
        if "local_positions" not in self.ctx.repr_indices:
            raise KeyError(f"{tensor_name}: missing required modality 'local_positions'.")

        s_lp, e_lp = self.ctx.repr_indices["local_positions"]
        local_pos = full_repr[..., s_lp:e_lp]
        if torch.isfinite(local_pos).all():
            return

        base_required = ("smplx_rot_6d", "smplx_trans", "smplx_betas")
        for name in base_required:
            if name not in self.ctx.repr_indices:
                raise KeyError(
                    f"{tensor_name}: cannot rebuild local_positions; missing required modality '{name}'."
                )
            s, e = self.ctx.repr_indices[name]
            if not torch.isfinite(full_repr[..., s:e]).all():
                nan_count = int(torch.isnan(full_repr[..., s:e]).sum().item())
                inf_count = int(torch.isinf(full_repr[..., s:e]).sum().item())
                raise ValueError(
                    f"{tensor_name}: modality '{name}' contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count}); cannot rebuild local_positions."
                )

        smpl_input = {
            "smplx_rot_6d": full_repr[..., self.ctx.repr_indices["smplx_rot_6d"][0]:self.ctx.repr_indices["smplx_rot_6d"][1]],
            "smplx_trans": full_repr[..., self.ctx.repr_indices["smplx_trans"][0]:self.ctx.repr_indices["smplx_trans"][1]],
            "smplx_betas": full_repr[..., self.ctx.repr_indices["smplx_betas"][0]:self.ctx.repr_indices["smplx_betas"][1]],
        }

        has_body_pose = "smplx_body_pose_6d" in self.ctx.repr_indices
        has_world_rot = "R_world_joints" in self.ctx.repr_indices
        if has_body_pose:
            s_bp, e_bp = self.ctx.repr_indices["smplx_body_pose_6d"]
            body_pose = full_repr[..., s_bp:e_bp]
            if not torch.isfinite(body_pose).all():
                nan_count = int(torch.isnan(body_pose).sum().item())
                inf_count = int(torch.isinf(body_pose).sum().item())
                raise ValueError(
                    f"{tensor_name}: smplx_body_pose_6d is non-finite (nan={nan_count}, inf={inf_count}); "
                    "cannot rebuild local_positions."
                )
            smpl_input["smplx_body_pose_6d"] = body_pose
        elif has_world_rot:
            s_rw, e_rw = self.ctx.repr_indices["R_world_joints"]
            world_rot = full_repr[..., s_rw:e_rw]
            if not torch.isfinite(world_rot).all():
                nan_count = int(torch.isnan(world_rot).sum().item())
                inf_count = int(torch.isinf(world_rot).sum().item())
                raise ValueError(
                    f"{tensor_name}: R_world_joints is non-finite (nan={nan_count}, inf={inf_count}); "
                    "cannot derive smplx_body_pose_6d for local_positions reconstruction."
                )
            smpl_input["R_world_joints"] = world_rot
            from data_sets.motion_representation import populate_repr_dict
            populate_repr_dict(smpl_input, self.ctx.smplx_model, recover_global=False)
        else:
            raise KeyError(
                f"{tensor_name}: cannot rebuild local_positions because neither "
                "'smplx_body_pose_6d' nor 'R_world_joints' is available."
            )

        from data_sets.motion_representation import recover_from_repr_smpl
        with torch.no_grad():
            joints = recover_from_repr_smpl(
                smpl_input,
                recover_mode="smplx_params",
                smplx_model=self.ctx.smplx_model,
                return_verts=False,
                return_full_joints=False,
            )
        if isinstance(joints, tuple):
            joints = joints[0]
        full_repr[..., s_lp:e_lp] = joints.reshape(*full_repr.shape[:2], -1).to(
            device=full_repr.device,
            dtype=full_repr.dtype,
        )
        if not torch.isfinite(full_repr[..., s_lp:e_lp]).all():
            nan_count = int(torch.isnan(full_repr[..., s_lp:e_lp]).sum().item())
            inf_count = int(torch.isinf(full_repr[..., s_lp:e_lp]).sum().item())
            raise ValueError(
                f"{tensor_name}: local_positions reconstruction produced non-finite values "
                f"(nan={nan_count}, inf={inf_count})."
            )

    def _apply_root_rotate_to_traj_with_reference_full_repr(
        self,
        traj_repr: torch.Tensor,
        reference_full_repr: torch.Tensor,
        rot_mat: torch.Tensor,
        mode: str,
    ) -> None:
        """
        Apply rotate_repr_smpl_2-equivalent rotation on traj predicted layout.

        If traj representation contains smplx_trans but not local_positions, local_positions
        are borrowed from reference_full_repr to preserve coupled root translation semantics.
        """
        if mode not in ("forward", "inverse"):
            raise ValueError(f"Unsupported mode: {mode}")
        if rot_mat is None:
            return

        predicted = self.ctx.repr_registry.predicted_modalities.get("traj", [])
        if not predicted:
            return

        applied_rot_mat = rot_mat if mode == "forward" else rot_mat.transpose(-1, -2)
        bs, T = traj_repr.shape[:2]
        rot_mat_T = applied_rot_mat.transpose(-1, -2).unsqueeze(1)
        ranges = self._get_predicted_modality_ranges("traj")

        has_lp = "local_positions" in ranges
        has_trans = "smplx_trans" in ranges
        has_rot6d = "smplx_rot_6d" in ranges

        J = None
        trans = None
        num_joints = self.ctx.joints_num

        if has_lp:
            s_lp, e_lp = ranges["local_positions"]
            J_full = traj_repr[..., s_lp:e_lp].reshape(bs, T, num_joints, -1)
            J = J_full[..., :3]
            if not torch.isfinite(J).all():
                nan_count = int(torch.isnan(J).sum().item())
                inf_count = int(torch.isinf(J).sum().item())
                raise ValueError(
                    "traj_repr.local_positions contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count}); cannot apply root rotation."
                )
        if has_trans:
            s_tr, _ = ranges["smplx_trans"]
            trans = traj_repr[..., s_tr:s_tr + 3]
            if not torch.isfinite(trans).all():
                nan_count = int(torch.isnan(trans).sum().item())
                inf_count = int(torch.isinf(trans).sum().item())
                raise ValueError(
                    "traj_repr.smplx_trans contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count}); cannot apply root rotation."
                )

        if has_trans and not has_lp:
            if "local_positions" not in self.ctx.repr_indices:
                raise KeyError(
                    "traj predicted modalities include smplx_trans but no local_positions, "
                    "and reference_full_repr lacks local_positions."
                )
            s_lp_ref, e_lp_ref = self.ctx.repr_indices["local_positions"]
            ref_local = reference_full_repr[..., s_lp_ref:e_lp_ref]
            if not torch.isfinite(ref_local).all():
                nan_count = int(torch.isnan(ref_local).sum().item())
                inf_count = int(torch.isinf(ref_local).sum().item())
                raise ValueError(
                    "reference_full_repr.local_positions contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count}); cannot rotate traj smplx_trans."
                )
            J = ref_local.reshape(bs, T, num_joints, 3)

        if has_lp and has_trans:
            J_root_ = J[..., 0:1, :] - trans.unsqueeze(-2)
            RJ_root_ = torch.matmul(J_root_, rot_mat_T)
            new_J = torch.matmul(J, rot_mat_T)
            s_lp, e_lp = ranges["local_positions"]
            J_full = traj_repr[..., s_lp:e_lp].reshape(bs, T, num_joints, -1)
            J_full[..., :3] = new_J
            traj_repr[..., s_lp:e_lp] = J_full.reshape(bs, T, -1).clone()
            s_tr, _ = ranges["smplx_trans"]
            new_trans = (
                RJ_root_.squeeze(-2)
                - J_root_.squeeze(-2)
                + torch.matmul(trans.unsqueeze(-2), rot_mat_T).squeeze(-2)
            )
            traj_repr[..., s_tr:s_tr + 3] = new_trans
        elif has_lp:
            s_lp, e_lp = ranges["local_positions"]
            new_J = torch.matmul(J, rot_mat_T)
            J_full = traj_repr[..., s_lp:e_lp].reshape(bs, T, num_joints, -1)
            J_full[..., :3] = new_J
            traj_repr[..., s_lp:e_lp] = J_full.reshape(bs, T, -1).clone()
        elif has_trans:
            s_tr, _ = ranges["smplx_trans"]
            J_root_ = J[..., 0:1, :] - trans.unsqueeze(-2)
            RJ_root_ = torch.matmul(J_root_, rot_mat_T)
            new_trans = (
                RJ_root_.squeeze(-2)
                - J_root_.squeeze(-2)
                + torch.matmul(trans.unsqueeze(-2), rot_mat_T).squeeze(-2)
            )
            traj_repr[..., s_tr:s_tr + 3] = new_trans

        if has_rot6d:
            s_rot, e_rot = ranges["smplx_rot_6d"]
            rot6d = traj_repr[..., s_rot:e_rot].reshape(-1, 6)
            from utils.konia_transform import rotate_6d_repr
            rot_mat_expanded = applied_rot_mat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 3, 3)
            new_rot6d = rotate_6d_repr(rot6d, rot_mat_expanded).reshape(bs, T, -1)
            traj_repr[..., s_rot:e_rot] = new_rot6d

    def _apply_root_rotate_to_cond(self, cond_tensor: torch.Tensor, task: str, rot_mat: torch.Tensor, mode: str) -> None:
        if mode not in ("forward", "inverse"):
            raise ValueError(f"Unsupported mode: {mode}")
        if rot_mat is None:
            return

        applied_rot_mat = rot_mat if mode == "forward" else rot_mat.transpose(-1, -2)
        bs, T = cond_tensor.shape[:2]
        rot_mat_T = applied_rot_mat.transpose(-1, -2).unsqueeze(1)

        cond_names = self.ctx.repr_registry.cond_modalities.get(task, [])
        cond_indices = get_cond_modality_indices(self.ctx.dataset, task=task)
        has_lp = "local_positions" in cond_names
        has_trans = "smplx_trans" in cond_names

        if has_trans and not has_lp:
            raise ValueError(
                f"cond[{task}] contains smplx_trans but not local_positions. "
                "rotate_repr_smpl_2-equivalent transform requires both modalities."
            )

        J = None
        if has_lp:
            num_joints, per_joint_total, lp_offset, _ = self._get_joint_cond_layout(task=task)
            joint_dim = num_joints * per_joint_total
            joint_part = cond_tensor[:, :, :joint_dim].reshape(bs, T, num_joints, per_joint_total)
            J = joint_part[..., lp_offset:lp_offset + 3]
            if not torch.isfinite(J).all():
                nan_count = int(torch.isnan(J).sum().item())
                inf_count = int(torch.isinf(J).sum().item())
                raise ValueError(
                    f"cond[{task}].local_positions contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count})."
                )

        if has_trans:
            if "smplx_trans" not in cond_indices:
                raise KeyError(f"cond[{task}] is missing global index for modality 'smplx_trans'.")
            s_tr, _ = cond_indices["smplx_trans"]
            trans = cond_tensor[:, :, s_tr:s_tr + 3]
            if not torch.isfinite(trans).all():
                nan_count = int(torch.isnan(trans).sum().item())
                inf_count = int(torch.isinf(trans).sum().item())
                raise ValueError(
                    f"cond[{task}].smplx_trans contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count})."
                )

            J_root_ = J[..., 0:1, :] - trans.unsqueeze(-2)
            RJ_root_ = torch.matmul(J_root_, rot_mat_T)
            new_trans = (
                RJ_root_.squeeze(-2)
                - J_root_.squeeze(-2)
                + torch.matmul(trans.unsqueeze(-2), rot_mat_T).squeeze(-2)
            )
            cond_tensor[:, :, s_tr:s_tr + 3] = new_trans

        if has_lp:
            new_J = torch.matmul(J, rot_mat_T)
            joint_part[..., lp_offset:lp_offset + 3] = new_J
            cond_tensor[:, :, :joint_dim] = joint_part.reshape(bs, T, -1).clone()

        if "smplx_rot_6d" in cond_names:
            if "smplx_rot_6d" not in cond_indices:
                raise KeyError(f"cond[{task}] is missing global index for modality 'smplx_rot_6d'.")
            s_rot, e_rot = cond_indices["smplx_rot_6d"]
            rot6d = cond_tensor[..., s_rot:e_rot].reshape(-1, 6)
            from utils.konia_transform import rotate_6d_repr
            rot_mat_expanded = applied_rot_mat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 3, 3)
            new_rot6d = rotate_6d_repr(rot6d, rot_mat_expanded).reshape(bs, T, -1)
            cond_tensor[..., s_rot:e_rot] = new_rot6d

    def _apply_root_rotate_to_global_modalities(
        self,
        full_repr: torch.Tensor,
        rot_mat: torch.Tensor,
        mode: str,
    ) -> None:
        """
        Backward-compatible wrapper.

        Rotation semantics are strictly delegated to _apply_root_rotate_to_full_repr,
        which enforces rotate_repr_smpl_2 coupling constraints.
        """
        self._apply_root_rotate_to_full_repr(full_repr, rot_mat, mode)

    def _apply_root_rotate_to_full_repr(self, full_repr: torch.Tensor, rot_mat: torch.Tensor, mode: str) -> None:
        if mode not in ("forward", "inverse"):
            raise ValueError(f"Unsupported mode: {mode}")
        if rot_mat is None:
            return

        applied_rot_mat = rot_mat if mode == "forward" else rot_mat.transpose(-1, -2)
        bs, T = full_repr.shape[:2]
        rot_mat_T = applied_rot_mat.transpose(-1, -2).unsqueeze(1)

        has_lp = "local_positions" in self.ctx.repr_indices
        has_trans = "smplx_trans" in self.ctx.repr_indices
        has_rot6d = "smplx_rot_6d" in self.ctx.repr_indices

        if has_trans and not has_lp:
            raise ValueError(
                "full_repr contains smplx_trans but not local_positions. "
                "rotate_repr_smpl_2-equivalent transform requires both modalities."
            )

        if has_lp:
            s_lp, e_lp = self.ctx.repr_indices["local_positions"]
            J = full_repr[..., s_lp:e_lp].reshape(bs, T, -1, 3)
            if not torch.isfinite(J).all():
                nan_count = int(torch.isnan(J).sum().item())
                inf_count = int(torch.isinf(J).sum().item())
                raise ValueError(
                    "full_repr.local_positions contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count}); cannot apply coupled root rotation."
                )

            if has_trans:
                s_tr, _ = self.ctx.repr_indices["smplx_trans"]
                trans = full_repr[..., s_tr:s_tr + 3]
                if not torch.isfinite(trans).all():
                    nan_count = int(torch.isnan(trans).sum().item())
                    inf_count = int(torch.isinf(trans).sum().item())
                    raise ValueError(
                        "full_repr.smplx_trans contains non-finite values "
                        f"(nan={nan_count}, inf={inf_count}); cannot apply coupled root rotation."
                    )
                J_root_ = J[..., 0:1, :] - trans.unsqueeze(-2)
                RJ_root_ = torch.matmul(J_root_, rot_mat_T)
                new_trans = (
                    RJ_root_.squeeze(-2)
                    - J_root_.squeeze(-2)
                    + torch.matmul(trans.unsqueeze(-2), rot_mat_T).squeeze(-2)
                )
                full_repr[..., s_tr:s_tr + 3] = new_trans

            new_J = torch.matmul(J, rot_mat_T)
            full_repr[..., s_lp:e_lp] = new_J.reshape(bs, T, -1)

        if has_rot6d:
            s_rot, e_rot = self.ctx.repr_indices["smplx_rot_6d"]
            rot6d = full_repr[..., s_rot:e_rot].reshape(-1, 6)
            from utils.konia_transform import rotate_6d_repr
            rot_mat_expanded = applied_rot_mat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 3, 3)
            new_rot6d = rotate_6d_repr(rot6d, rot_mat_expanded).reshape(bs, T, -1)
            full_repr[..., s_rot:e_rot] = new_rot6d

    def _apply_root_rotate_to_predicted_repr(self, predicted_repr: torch.Tensor, task: str, rot_mat: torch.Tensor, mode: str) -> None:
        if mode not in ("forward", "inverse"):
            raise ValueError(f"Unsupported mode: {mode}")
        if rot_mat is None:
            return

        predicted = self.ctx.repr_registry.predicted_modalities.get(task, [])
        if not predicted:
            return

        applied_rot_mat = rot_mat if mode == "forward" else rot_mat.transpose(-1, -2)
        bs, T = predicted_repr.shape[:2]
        rot_mat_T = applied_rot_mat.transpose(-1, -2).unsqueeze(1)
        ranges = self._get_predicted_modality_ranges(task)

        has_lp = "local_positions" in ranges
        has_trans = "smplx_trans" in ranges
        has_rot6d = "smplx_rot_6d" in ranges

        if has_trans and not has_lp:
            raise ValueError(
                f"predicted_repr[{task}] contains smplx_trans but not local_positions. "
                "rotate_repr_smpl_2-equivalent transform requires both modalities."
            )

        if has_lp:
            s_lp, e_lp = ranges["local_positions"]
            num_joints = self.ctx.joints_num
            J_full = predicted_repr[..., s_lp:e_lp].reshape(bs, T, num_joints, -1)
            J = J_full[..., :3]
            if not torch.isfinite(J).all():
                nan_count = int(torch.isnan(J).sum().item())
                inf_count = int(torch.isinf(J).sum().item())
                raise ValueError(
                    f"predicted_repr[{task}].local_positions contains non-finite values "
                    f"(nan={nan_count}, inf={inf_count})."
                )

            if has_trans:
                s_tr, _ = ranges["smplx_trans"]
                trans = predicted_repr[..., s_tr:s_tr + 3]
                if not torch.isfinite(trans).all():
                    nan_count = int(torch.isnan(trans).sum().item())
                    inf_count = int(torch.isinf(trans).sum().item())
                    raise ValueError(
                        f"predicted_repr[{task}].smplx_trans contains non-finite values "
                        f"(nan={nan_count}, inf={inf_count})."
                    )
                J_root_ = J[..., 0:1, :] - trans.unsqueeze(-2)
                RJ_root_ = torch.matmul(J_root_, rot_mat_T)
                new_trans = (
                    RJ_root_.squeeze(-2)
                    - J_root_.squeeze(-2)
                    + torch.matmul(trans.unsqueeze(-2), rot_mat_T).squeeze(-2)
                )
                predicted_repr[..., s_tr:s_tr + 3] = new_trans

            new_J = torch.matmul(J, rot_mat_T)
            J_full[..., :3] = new_J
            predicted_repr[..., s_lp:e_lp] = J_full.reshape(bs, T, -1).clone()

        if has_rot6d:
            s_rot, e_rot = ranges["smplx_rot_6d"]
            rot6d = predicted_repr[..., s_rot:e_rot].reshape(-1, 6)
            from utils.konia_transform import rotate_6d_repr
            rot_mat_expanded = applied_rot_mat.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, 3, 3)
            new_rot6d = rotate_6d_repr(rot6d, rot_mat_expanded).reshape(bs, T, -1)
            predicted_repr[..., s_rot:e_rot] = new_rot6d
