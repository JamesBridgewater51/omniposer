# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

from collections.abc import Mapping

from utils.other_utils import *
import smplx
from utils.repr_adapter import decode_smplx_from_full_repr
from model.heads import *
from model.heads import FrequencyAdapter, MaskedConditioningEncoder
from utils.evaluation import MotionEvaluator, procrustes_align
from typing import Literal, Optional, Dict, Any

# Use centralized construct_class_by_name following the IMM/dnnlib pattern
from utils.construct import construct_class_by_name
from utils.feature_accessor import select_modalities, get_modality_dim
from model.metadata_moe import build_routing_metadata_from_batch, reject_legacy_moe_surfaces


class PoseNet(nn.Module):
    """
    PoseNet: A flexible pose estimation/generation network.

    Note: `include_global_joint_rotations` and `repr_abs_only` are now derived
    from the dataset's representation registry, not passed as parameters.
    """

    def __init__(self, dataset,
                 body_model_path='',
                 device=None,
                 finetune_posenet=True,
                 abstract_model_type:Literal['generative', 'discriminative', 'descriptive']='generative',
                 ############ Config-based architecture construction
                 network_config: Optional[Dict[str, Any]] = None,
                 ############ Masking configuration
                 cano_jts_pos: bool = False,
                 use_masked_conditioner: bool = True,
                 ############ Architecture hyperparameters (for masked conditioner)
                 latent_dim: int = 256,
                 ############ Loss weights
                 weight_loss_rec_repr_full_body=0.0,
                 weight_loss_repr_foot_contact_mse=0.0,
                 weight_loss_joint_pos_global=0.0,
                 weight_loss_joint_vel_global=0.0,
                 weight_loss_joint_smooth=0.0,
                 weight_loss_foot_skating=0.0,
                 weight_loss_visible_joint_pos_global=0.0,
                 start_skating_loss_epoch=0,
                 ):
        super().__init__()
        # Store both train and test datasets; test_ds can be set later via set_test_dataset()
        self._train_ds = dataset
        self._test_ds = None  # Will be set via set_test_dataset()
        self.task = 'pose'  # PoseNet always operates on pose task

        # Derive representation flags from dataset's representation registry
        # These flags are used to determine:
        # 1. include_global_joint_rotations: Whether the model's OUTPUT uses R_world_joints vs. smplx_body_pose_6d
        # 2. repr_abs_only: Whether velocity representations (smplx_trans_vel, smplx_rot_vel) are present
        cond_modalities = self._train_ds.repr_registry.cond_modalities.get(self.task, [])
        predicted_modalities = self._train_ds.repr_registry.predicted_modalities.get(self.task, [])
        repr_list = self._train_ds._repr_list

        # Check if joint rotations are in CONDITIONING (used to configure MaskedConditioningEncoder)
        self.has_rotation_in_cond = 'R_world_joints' in cond_modalities or 'R_root_joints' in cond_modalities

        # Check if joint rotations are in PREDICTIONS (used to determine output representation)
        # For now, we infer this from whether smplx_body_pose_6d is NOT in predicted_modalities
        # because if we're predicting R_world_joints, we wouldn't also predict smplx_body_pose_6d
        self.include_global_joint_rotations = 'smplx_body_pose_6d' not in predicted_modalities

        # Check if velocity representations are present
        self.repr_abs_only = 'local_vel' not in repr_list

        self.abstract_model_type = abstract_model_type
        self.finetune_posenet = finetune_posenet
        self.cano_jts_pos = cano_jts_pos
        self.latent_dim = latent_dim
        self.use_masked_conditioner = use_masked_conditioner

        # Use dataset-computed dimensions
        self.model_feat_dim = self._train_ds.pose_feat_dim
        self.cond_feat_dim = self._train_ds.cond_feat_dim_pose

        # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
        self.foot_joint_index_list = [7, 10, 8, 11]
        self.foot_skating_vel_thres = 0.1
        self.fps = 30
        self.evaluator = MotionEvaluator(fps=30)

        self.mse_loss = nn.MSELoss(reduction='none').to(device)
        self.device = device

        self.weight_loss_rec_repr_full_body = weight_loss_rec_repr_full_body
        self.weight_loss_repr_foot_contact_mse = weight_loss_repr_foot_contact_mse
        self.weight_loss_joint_pos_global = weight_loss_joint_pos_global
        self.weight_loss_joint_vel_global = weight_loss_joint_vel_global
        self.weight_loss_joint_smooth = weight_loss_joint_smooth
        self.weight_loss_foot_skating = weight_loss_foot_skating
        self.weight_loss_visible_joint_pos_global = weight_loss_visible_joint_pos_global
        self.start_skating_loss_epoch = start_skating_loss_epoch

        self.smplx_model = smplx.create(model_path=body_model_path, model_type="smplx",
                                 gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)
        
        reject_legacy_moe_surfaces(network_config, context="PoseNet network_config")
        self.network_kwargs = network_config
        self.last_pose_moe_aux_losses = {}
        self.last_pose_moe_diagnostics = {}
        self.last_pose_route_state = None
        self.last_pose_ffn_moe_aux_losses = {}
        self.last_pose_ffn_moe_diagnostics = {}
        self.last_pose_ffn_route_state = {}

        assert network_config is not None
        self._init_from_config(network_config)

    def set_test_dataset(self, test_ds):
        """Set the test dataset for evaluation context."""
        self._test_ds = test_ds

    def _get_dataset_for_context(self, is_eval: bool = False):
        """
        Get the appropriate dataset based on context.
        
        Args:
            is_eval: If True, use test dataset for unnormalization; otherwise use train dataset.
        
        Returns:
            The appropriate dataset. For attributes that must be identical between train/test
            (e.g., _repr_indices, _repr_list), either dataset can be used.
        """
        if is_eval and self._test_ds is not None:
            return self._test_ds
        return self._train_ds

    def _init_from_config(self, network_kwargs: Dict[str, Any]):
        """
        Initialize architecture from config dict using construct_class_by_name pattern.

        This follows the IMM/dnnlib pattern where network_kwargs contains:
        - class_name: Full path to the architecture class
        - Other parameters passed to the architecture constructor
        """
        # Inject derived representation flags into architecture config
        # This allows architectures to receive these values without them being in YAML configs
        updated_kwargs = dict(network_kwargs)
        updated_kwargs['include_global_joint_rotations'] = self.include_global_joint_rotations
        updated_kwargs['repr_abs_only'] = self.repr_abs_only
        
        # Inject dynamic cond dimensions from registry
        # New cond structure: [joint-based interleaved] | [global including traj]
        # cond_feat_dim = joint-based dimension
        # traj_feat_dim = global dimension (including traj modalities)
        cond_modalities = self._train_ds.repr_registry.cond_modalities[self.task]
        joint_dim = 0
        global_dim = 0
        for mod_name in cond_modalities:
            mod = self._train_ds.repr_registry.get_modality(mod_name)
            if mod.is_joint_based:
                joint_dim += mod.dim
            else:
                global_dim += mod.dim
        updated_kwargs['cond_feat_dim'] = joint_dim
        updated_kwargs['traj_feat_dim'] = global_dim
        per_joint_dim = 0
        joint_modality_names = []
        global_modality_names = []
        for mod_name in cond_modalities:
            mod = self._train_ds.repr_registry.get_modality(mod_name)
            if mod.is_joint_based:
                per_joint_dim += mod.per_joint_dim
                joint_modality_names.append(mod_name)
            else:
                global_modality_names.append(mod_name)
        grouping_name = None
        moe_cfg = updated_kwargs.get('moe') or {}
        if isinstance(moe_cfg, Mapping):
            grouping_name = (moe_cfg.get('grouped') or {}).get('grouping_name')
        routing_groups = None
        if grouping_name:
            routing_groups = self._train_ds.repr_registry.get_routing_group_mapping(grouping_name, expected_topology='smplx_22')
        updated_kwargs['routing_groups'] = routing_groups
        updated_kwargs['per_joint_cond_dim'] = per_joint_dim
        updated_kwargs['joint_modality_names'] = joint_modality_names
        updated_kwargs['global_modality_names'] = global_modality_names

        # Construct the architecture using the class_name pattern
        self.architecture = construct_class_by_name(**updated_kwargs)
        
        # Determine if we need to handle finetune heads separately
        # For GenericDiT and discriminative architectures without built-in heads
        self._has_external_finetune_heads = False
        
        if self.finetune_posenet and not hasattr(self.architecture, 'smplx_betas_head'):
            # Architecture doesn't have built-in finetune heads, we add them here
            self._has_external_finetune_heads = True
            self._init_finetune_heads()
        
        # Initialize masked conditioner (handled externally, not inside architecture)
        # This follows the pattern from TrajNet._init_from_config
        self.num_joints = 22
        if self.use_masked_conditioner:
            # Calculate encoder dimensions from registry (registry-driven interface)
            # cond_modalities['pose'] now explicitly includes traj modalities as global
            cond_modalities = self._train_ds.repr_registry.cond_modalities[self.task]
            per_joint_dim = 0
            global_dim = 0
            for mod_name in cond_modalities:
                mod = self._train_ds.repr_registry.get_modality(mod_name)
                if mod.is_joint_based:
                    per_joint_dim += mod.per_joint_dim
                else:
                    global_dim += mod.dim

            # No prefix - traj modalities are now part of global_dim via cond_modalities['pose']
            prefix_dim = 0

            self.masked_conditioner = MaskedConditioningEncoder(
                cond_dim=self.cond_feat_dim,  # No prefix to add
                per_joint_dim=per_joint_dim,
                num_joints=self.num_joints,
                global_dim=global_dim,
                prefix_dim=prefix_dim,
                hidden_dim=self.latent_dim // 2,
                num_blocks=2,
                use_time=self.abstract_model_type == 'generative',
                time_dim=self.latent_dim,
                dropout=0.1
            )
    
    def _init_finetune_heads(self):
        """Initialize finetune heads externally for architectures that don't have them."""
        if self.include_global_joint_rotations:
            self.R_world_joints_head = nn.Linear(REPR_DIM_DICT['R_world_joints'], REPR_DIM_DICT['R_world_joints'])
            self.R_root_joints_head = nn.Linear(REPR_DIM_DICT['R_root_joints'], REPR_DIM_DICT['R_root_joints'])
        else:
            self.smplx_body_pose_6d_head = nn.Linear(REPR_DIM_DICT['smplx_body_pose_6d'], REPR_DIM_DICT['smplx_body_pose_6d'])
        self.smplx_betas_head = nn.Linear(REPR_DIM_DICT['smplx_betas'], REPR_DIM_DICT['smplx_betas'])
        self.smplx_foot_contact_head = nn.Linear(REPR_DIM_DICT['foot_contact'], REPR_DIM_DICT['foot_contact'])
        
        # Identity-like init
        def init_identity_linear(layer: nn.Linear):
            with torch.no_grad():
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
                diag = min(layer.in_features, layer.out_features)
                for i in range(diag):
                    layer.weight[i, i] = 1.0

        if self.include_global_joint_rotations:
            init_identity_linear(self.R_world_joints_head)
            init_identity_linear(self.R_root_joints_head)
        else:
            init_identity_linear(self.smplx_body_pose_6d_head)
        init_identity_linear(self.smplx_betas_head)
        init_identity_linear(self.smplx_foot_contact_head)

    def _build_routing_metadata(self, batch):
        return build_routing_metadata_from_batch(
            batch,
            routing_groups=self._train_ds.repr_registry.get_routing_group_mapping('body_part_6', expected_topology='smplx_22'),
            device=self.device,
        )

    def reset_route_state(self) -> None:
        self.last_pose_route_state = None
        self.last_pose_ffn_route_state = {}
        if hasattr(self, 'architecture') and hasattr(self.architecture, 'reset_route_state'):
            self.architecture.reset_route_state()

    def get_moe_loss_dict(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'architecture') or not hasattr(self.architecture, 'get_moe_loss_dict'):
            return {}
        return self.architecture.get_moe_loss_dict()

    def moe_aux_loss_in_main_loss(self) -> bool:
        return bool(hasattr(self, 'architecture') and hasattr(self.architecture, 'moe_aux_loss_in_main_loss') and self.architecture.moe_aux_loss_in_main_loss())

    def _forward_from_config(self, batch, t, d=None, **kwargs):
        """
        Forward pass using config-based architecture.

        This follows the pattern from TrajNet._forward_config:
        - Prepares main input based on abstract_model_type
        - Handles masked conditioner externally (not inside architecture)
        - Calls architecture.compute_time_embedding if available for masked conditioner
        - Dispatches to the constructed architecture
        - Applies external finetune heads if needed

        All tensors use 3D format [bs, T, dim].
        """
        # Prepare main input based on model type
        # For PoseNet: cond is [bs, T, cond_dim], x_t is [bs, T, model_feat_dim]
        cond = batch['cond']  # [bs, T, cond_dim]
        cond_mask = batch['cond_mask']  # [bs, T, cond_dim] feature-level mask

        # For discriminative mode, x_t should be the same as cond (not from batch)
        if self.abstract_model_type == 'discriminative':
            x_t = cond
        else:
            x_t = batch.get('x_t', None)

        time_emb = None
        if self.abstract_model_type == 'generative' and hasattr(self.architecture, 'compute_time_embedding'):
            time_emb = self.architecture.compute_time_embedding(t, d)

        # Apply masked conditioner if enabled (following TrajNet pattern)
        # The masked conditioner expects [bs, T, cond_dim] format
        if self.use_masked_conditioner:
            cond = self.masked_conditioner(cond, cond_mask, time_emb)

        architecture_kwargs = dict(kwargs)
        architecture_kwargs['routing_metadata'] = self._build_routing_metadata(batch)
        architecture_kwargs['route_state'] = kwargs.get('route_state', self.last_pose_route_state)
        architecture_kwargs['ffn_route_state'] = kwargs.get('ffn_route_state', self.last_pose_ffn_route_state)
        if time_emb is not None:
            architecture_kwargs['time_emb'] = time_emb
        output = self.architecture(
            x_t=x_t,
            t=t,
            d=d,
            cond=cond,
            **architecture_kwargs
        )

        self.last_pose_moe_aux_losses = getattr(self.architecture, 'last_pose_moe_aux_losses', {})
        self.last_pose_moe_diagnostics = getattr(self.architecture, 'last_pose_moe_diagnostics', {})
        self.last_pose_route_state = getattr(self.architecture, 'last_pose_route_state', None)
        self.last_pose_ffn_moe_aux_losses = getattr(self.architecture, 'last_pose_ffn_moe_aux_losses', {})
        self.last_pose_ffn_moe_diagnostics = getattr(self.architecture, 'last_pose_ffn_moe_diagnostics', {})
        self.last_pose_ffn_route_state = getattr(self.architecture, 'last_pose_ffn_route_state', {})

        # Apply external finetune heads if needed
        # output is [bs, T, model_feat_dim]
        if self._has_external_finetune_heads and self.finetune_posenet:
            smplx_betas_pred = self.smplx_betas_head(
                output[:, :, -REPR_DIM_DICT['smplx_betas']-REPR_DIM_DICT['foot_contact']:-REPR_DIM_DICT['foot_contact']]
            )
            smplx_foot_contact_pred = self.smplx_foot_contact_head(output[:, :, -REPR_DIM_DICT['foot_contact']:])

            if self.include_global_joint_rotations:
                R_world_joints_pred = self.R_world_joints_head(output[:, :, :REPR_DIM_DICT['R_world_joints']])
                R_root_joints_pred = self.R_root_joints_head(
                    output[:, :, REPR_DIM_DICT['R_world_joints']:REPR_DIM_DICT['R_world_joints']+REPR_DIM_DICT['R_root_joints']]
                )
                output = torch.cat([R_world_joints_pred, R_root_joints_pred, smplx_betas_pred, smplx_foot_contact_pred], dim=-1)
            else:
                smplx_body_pose_6d_pred = self.smplx_body_pose_6d_head(output[:, :, :REPR_DIM_DICT['smplx_body_pose_6d']])
                output = torch.cat([smplx_body_pose_6d_pred, smplx_betas_pred, smplx_foot_contact_pred], dim=-1)

        return output

    def forward(self, batch, t, d=None, **kwargs):
        return self._forward_from_config(batch, t, d, **kwargs)

    def compute_losses_with_smpl(self, batch, model_output, smplx_model=None, epoch=0, compute_eval_metrics=False, **kwargs):
        loss_dict = {}

        # Select dataset based on context: use test_ds for eval, train_ds for training
        ds = self._get_dataset_for_context(is_eval=compute_eval_metrics)

        # model_output is [bs, T, full_feat_dim], batch['motion_repr_clean'] is [bs, T, full_feat_dim]
        full_repr_rec_normalized = model_output
        full_repr_clean_normalized = batch['motion_repr_clean']

        if ds.norm_strategy == 'l2':
            sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in batch.items() if k.startswith('extras/sample_norms')}
        else:
            sample_norms = None

        full_repr_clean_unnormalized = ds._unnormalize_full_tensor(full_repr_clean_normalized, self.device, sample_norms=sample_norms)
        full_repr_rec_unnormalized   = ds._unnormalize_full_tensor(full_repr_rec_normalized, self.device, sample_norms=sample_norms)

        # MSE loss on only the normalized pose slices used by the objective.
        pose_components_loss = []
        if self.include_global_joint_rotations:
            s_grj, e_grj = ds._repr_indices['R_world_joints']
            loss_grj = (full_repr_rec_normalized[:, :, s_grj:e_grj] - full_repr_clean_normalized[:, :, s_grj:e_grj]).square()
            pose_components_loss.append(loss_grj)
            loss_dict['R_world_joints'] = loss_grj.mean()
        else:
            s_bp, e_bp = ds._repr_indices['smplx_body_pose_6d']
            loss_bp = (full_repr_rec_normalized[:, :, s_bp:e_bp] - full_repr_clean_normalized[:, :, s_bp:e_bp]).square()
            pose_components_loss.append(loss_bp)
            loss_dict['smplx_body_pose_6d'] = loss_bp.mean()
        s, e = ds._repr_indices['smplx_betas']
        loss_betas = (full_repr_rec_normalized[:, :, s:e] - full_repr_clean_normalized[:, :, s:e]).square()
        pose_components_loss.append(loss_betas)
        loss_dict['smplx_betas'] = loss_betas.mean()

        loss_dict['loss_rec_repr_full_body'] = torch.cat(pose_components_loss, dim=-1).mean()

        # Loss for 'foot_contact'
        s_fc, e_fc = ds._repr_indices['foot_contact']
        loss_dict['loss_repr_foot_contact_mse'] = (
            model_output[:, :, s_fc:e_fc] - batch['motion_repr_clean'][:, :, s_fc:e_fc]
        ).square().mean()

        dbg = False
        with torch.no_grad():
            repr_dict_clean_unnorm, joint_pos_clean, verts_clean = decode_smplx_from_full_repr(
                full_repr=full_repr_clean_unnormalized,
                repr_list=ds._repr_list,
                repr_indices=ds._repr_indices,
                smplx_model=smplx_model,
                output_global_rotations=self.include_global_joint_rotations,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=dbg,
            )
        repr_dict_rec_unnorm, joint_pos_rec_from_smpl, verts_rec = decode_smplx_from_full_repr(
            full_repr=full_repr_rec_unnormalized,
            repr_list=ds._repr_list,
            repr_indices=ds._repr_indices,
            smplx_model=smplx_model,
            output_global_rotations=self.include_global_joint_rotations,
            cano_jts_pos=self.cano_jts_pos,
            return_verts=dbg,
        )

        if dbg:
            from data_sets.motion_representation import SMPLH_KINTREE
            from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence

            cfg = SmplxVisConfig(
                window_title="PoseNet Debug",
                num_repeats=1,
                fixed_cam=False,
                camera_follow=True,
                show_mesh=True,
                show_skeleton=True,
            )
            for idx in range(joint_pos_clean.shape[0]):
                visualize_smplx_sequence(
                    joints=joint_pos_clean,
                    vertices=verts_clean,
                    faces=smplx_model.faces,
                    parent_indices=SMPLH_KINTREE,
                    batch_idx=idx,
                    cfg=cfg,
                )
                
        loss_dict['loss_joint_pos_global'] = self.mse_loss(joint_pos_rec_from_smpl, joint_pos_clean).mean()

        # Visible joint loss (using cond_joint_mask [bs, T, 22, 1])
        joint_mask = batch['cond_joint_mask']
        assert joint_mask is not None
        pos_mask = joint_mask[..., 0]
        cond_mask_expanded = pos_mask.unsqueeze(-1)  # Shape: [bs, T, 22, 1]
        # Ensure joint_pos_rec_from_smpl and joint_pos_clean are [bs, T, 22, 3]
        loss_dict['loss_visible_joint_pos_global'] = (
            (joint_pos_rec_from_smpl - joint_pos_clean).square() * cond_mask_expanded
        ).sum() / (cond_mask_expanded.sum() * 3 + 1e-6)

        joint_vel_clean = joint_pos_clean[:, 1:] - joint_pos_clean[:, 0:-1]
        joint_vel_rec_from_smpl = joint_pos_rec_from_smpl[:, 1:] - joint_pos_rec_from_smpl[:, 0:-1]
        
        loss_dict['loss_joint_vel_global'] = self.mse_loss(joint_vel_rec_from_smpl, joint_vel_clean).mean()

        joint_acc_rec_from_smpl = joint_vel_rec_from_smpl[:, 1:] - joint_vel_rec_from_smpl[:, 0:-1]
        loss_dict['loss_joint_smooth'] = torch.mean(joint_acc_rec_from_smpl ** 2)


        contact_lbl_gt_unnorm = repr_dict_clean_unnorm['foot_contact'] # [bs, T, 4]

        foot_vel_smpl = torch.norm((joint_pos_rec_from_smpl[:, 1:, self.foot_joint_index_list] - joint_pos_rec_from_smpl[:, :-1, self.foot_joint_index_list]) * self.fps, dim=-1)
        mask_skate_smpl = (foot_vel_smpl > self.foot_skating_vel_thres) & (contact_lbl_gt_unnorm[:, :-1] > 0.5)
        if mask_skate_smpl.sum() > 0:
            loss_dict['loss_foot_skating'] = (foot_vel_smpl * mask_skate_smpl).sum() / mask_skate_smpl.sum()
        else:
            loss_dict['loss_foot_skating'] = torch.tensor(0.0, device=self.device)

        current_weight_foot_skating = self.weight_loss_foot_skating if epoch >= self.start_skating_loss_epoch else 0.0
        
        loss_dict["loss"] = (
            self.weight_loss_rec_repr_full_body * loss_dict['loss_rec_repr_full_body'] +
            self.weight_loss_repr_foot_contact_mse * loss_dict['loss_repr_foot_contact_mse'] +
            self.weight_loss_joint_pos_global * loss_dict['loss_joint_pos_global'] +
            self.weight_loss_visible_joint_pos_global * loss_dict['loss_visible_joint_pos_global'] +
            self.weight_loss_joint_vel_global * loss_dict['loss_joint_vel_global'] +
            self.weight_loss_joint_smooth * loss_dict['loss_joint_smooth'] +
            current_weight_foot_skating * loss_dict['loss_foot_skating']
        )


        if compute_eval_metrics:
            with torch.no_grad():
                # NOTE: we don't directly call MotionEvaluator._compute_comparison_metrics since it assumes a constant batch['cond_mask'], However, here our batch['cond_mask'] is not constant.
                pred_joints_np = joint_pos_rec_from_smpl.cpu().numpy()
                gt_joints_np = joint_pos_clean.cpu().numpy()

                # Shape: (B, T, 22, 2) if include_global_joint_rotations set to True, otherwise (B, T, 22, 1) -> (B, T, 22)
                # NOTE: this currently uses the last dim of cond_joint_mask, which ignores R_world_joints masking schedules, so it is not a true reflection.
                joint_mask = batch['cond_joint_mask']
                vis_mask_np = joint_mask[..., :1].squeeze(-1).cpu().numpy()

                # 1. MPJPE Calculation
                mpjpe_error = np.linalg.norm(pred_joints_np - gt_joints_np, axis=-1)
                loss_dict['eval_mpjpe_global_mm'] = torch.tensor(np.mean(mpjpe_error) * 1000)
                if np.any(vis_mask_np):
                    loss_dict['eval_mpjpe_visible_mm'] = torch.tensor(np.mean(mpjpe_error[vis_mask_np]) * 1000)
                if np.any(~vis_mask_np):
                    loss_dict['eval_mpjpe_occluded_mm'] = torch.tensor(np.mean(mpjpe_error[~vis_mask_np]) * 1000)

                # 2. PA-MPJPE Calculation
                pred_joints_aligned_np = procrustes_align(gt_joints_np, pred_joints_np)
                pampjpe_error = np.linalg.norm(pred_joints_aligned_np - gt_joints_np, axis=-1)
                loss_dict['eval_pa_mpjpe_global_mm'] = torch.tensor(np.mean(pampjpe_error) * 1000)
                if np.any(vis_mask_np):
                    loss_dict['eval_pa_mpjpe_visible_mm'] = torch.tensor(np.mean(pampjpe_error[vis_mask_np]) * 1000)
                if np.any(~vis_mask_np):
                    loss_dict['eval_pa_mpjpe_occluded_mm'] = torch.tensor(np.mean(pampjpe_error[~vis_mask_np]) * 1000)

                # 3. Other global metrics
                accel_error = self.evaluator._compute_accel_error(pred_joints_np, gt_joints_np)
                loss_dict['eval_acceleration_error_ms2'] = torch.tensor(np.mean(accel_error))
                
                contact_acc = self.evaluator._compute_foot_contact_accuracy(repr_dict_rec_unnorm, repr_dict_clean_unnorm)
                loss_dict['eval_contact_label_accuracy'] = torch.tensor(contact_acc)

        store_pred = kwargs.get("store_pred", False)
        if store_pred:
            batch['extras/pred'] = model_output
        return loss_dict

    def guide_skating_with_smpl(self, batch, model_output, inputs, smplx_model=None, compute_grad='x_t'):
        # model_output: [bs, T, full_feat_dim]
        with torch.enable_grad():

            ########## obtain global joint coordinate
            full_repr_rec_normalized = model_output  # [bs, T, body_feat_dim]
            full_repr_clean_normalized = batch['motion_repr_clean']  # [bs, T, body_feat_dim]

            if self._train_ds.norm_strategy == 'l2':
                sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in batch.items() if k.startswith('extras/sample_norms')}
            else:
                sample_norms = None

            full_repr_rec_unnormalized = self._train_ds._unnormalize_full_tensor(full_repr_rec_normalized, device=self.device, sample_norms=sample_norms)
            full_repr_clean_unnormalized = self._train_ds._unnormalize_full_tensor(full_repr_clean_normalized, device=self.device, sample_norms=sample_norms)

            repr_dict_clean_unnorm, joint_pos_clean, _ = decode_smplx_from_full_repr(
                full_repr=full_repr_clean_unnormalized,
                repr_list=self._train_ds._repr_list,
                repr_indices=self._train_ds._repr_indices,
                smplx_model=smplx_model,
                output_global_rotations=self.include_global_joint_rotations,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=False,
            )
            repr_dict_rec_unnorm, joint_pos_rec_from_smpl, _ = decode_smplx_from_full_repr(
                full_repr=full_repr_rec_unnormalized,
                repr_list=self._train_ds._repr_list,
                repr_indices=self._train_ds._repr_indices,
                smplx_model=smplx_model,
                output_global_rotations=self.include_global_joint_rotations,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=False,
            )

            joint_vel_clean = joint_pos_clean[:, 1:] - joint_pos_clean[:, 0:-1]
            joint_vel_rec_from_smpl = joint_pos_rec_from_smpl[:, 1:] - joint_pos_rec_from_smpl[:, 0:-1]
            
            loss_joint_vel_global_from_smpl = self.mse_loss(joint_vel_rec_from_smpl, joint_vel_clean).mean()

            joint_acc_rec_from_smpl = joint_vel_rec_from_smpl[:, 1:] - joint_vel_rec_from_smpl[:, 0:-1]
            loss_joint_smooth_from_smpl = torch.mean(joint_acc_rec_from_smpl ** 2)

            joint_mask = batch['cond_joint_mask']
            assert joint_mask is not None
            pos_mask = joint_mask[..., 0]
            cond_mask_expanded = pos_mask.unsqueeze(-1)  # Shape: [bs, T, 22, 1]
            # Ensure joint_pos_rec_from_smpl and joint_pos_clean are [bs, T, 22, 3]
            loss_visible_joint_pos_global_from_abs_traj = (self.mse_loss(joint_pos_rec_from_smpl * cond_mask_expanded, joint_pos_clean * cond_mask_expanded).sum()) / (cond_mask_expanded.sum() * 3 + 1e-6)  # Multiply by 3 for xyz

            ############# foot skating loss
            contact_lbl_rec = full_repr_rec_unnormalized[:, :, -4:].detach().clone()  # [bs, clip_len, 4] 1 - in contact, 0 - not in contact
            contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
            contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0

            ### joints from smpl
            foot_joint_rec_vel_from_smpl = (joint_pos_rec_from_smpl[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_smpl[:, 0:-1, self.foot_joint_index_list]) * self.fps
            foot_joint_rec_vel_from_smpl = torch.norm(foot_joint_rec_vel_from_smpl, dim=-1)
            mask_skating_from_smpl = (foot_joint_rec_vel_from_smpl - self.foot_skating_vel_thres).gt(0)
            mask_skating_from_smpl = mask_skating_from_smpl * contact_lbl_rec[:, 0:-1]
            masked_foot_joint_rec_vel_from_smpl = foot_joint_rec_vel_from_smpl * mask_skating_from_smpl
            if mask_skating_from_smpl.sum() != 0:
                loss_foot_skating_from_smpl = masked_foot_joint_rec_vel_from_smpl.sum() / mask_skating_from_smpl.sum()
            else:
                loss_foot_skating_from_smpl = torch.tensor(0.0).to(self.device)

            if mask_skating_from_smpl.sum() != 0:
                loss_foot_skating = loss_foot_skating_from_smpl

                # Compute the guidance gradient via autograd (RoHM-style).
                loss_foot_skating = loss_visible_joint_pos_global_from_abs_traj
                grad_skating = torch.autograd.grad([-loss_foot_skating], [inputs])[0]  # [bs, body_feat_dim, 1, T] same shape as x_t, body_feat_dim=294
                # print('loss_foot_skating_from_smpl: ', loss_foot_skating_from_smpl)
                grad_skating[:, -4:, :, :] = 0  # for contact label
            else:
                assert False, "Shouldn't reach here since we have a skating foot."
                grad_skating = torch.tensor(0.0).to(self.device)
            model_output.detach()

        return grad_skating


    def guide_2d_projection_with_smpl(self, batch, out, denoise_t, compute_grad='x_t'):
        # model_output: [bs, T, pose_feat_dim]
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, T, body_feat_dim]

            ########## obtain global joint coordinate
            full_repr_rec = x_t  # [bs, T, body_feat_dim]
            full_repr_rec = full_repr_rec * torch.from_numpy(self._train_ds.Std).to(self.device) + torch.from_numpy(self._train_ds.Mean).to(self.device)

            # reconstruct joint positions
            traj_feat_dim = sum(
                self._train_ds._repr_indices[name][1] - self._train_ds._repr_indices[name][0]
                for name in self._train_ds.traj_repr_name_list
            )
            _, joint_pos_rec_from_smpl, _ = decode_smplx_from_full_repr(
                full_repr=full_repr_rec,
                repr_list=self._train_ds._repr_list,
                repr_indices=self._train_ds._repr_indices,
                smplx_model=self.smplx_model,
                output_global_rotations=self.include_global_joint_rotations,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=False,
            )

            ######### obtain joint coordinate in scene coord
            batch_size, clip_len = joint_pos_rec_from_smpl.shape[0], joint_pos_rec_from_smpl.shape[1]
            trans_cano2scene = torch.linalg.inv(batch['transf_matrix'])  # [bs, 4, 4]
            R, T = trans_cano2scene[:, 0:3, 0:3], trans_cano2scene[:, 0:3, -1]
            R = R.unsqueeze(1).repeat(1, clip_len, 1, 1).reshape(-1, 3, 3)
            T = T.unsqueeze(1).unsqueeze(1).repeat(1, clip_len, self._train_ds.joints_num, 1).reshape(-1, self._train_ds.joints_num, 3)
            temp = joint_pos_rec_from_smpl.reshape(batch_size*clip_len, -1, 3)
            joints_rec_scene_coord_from_smpl = torch.matmul(R, temp.permute(0, 2, 1)).permute(0, 2, 1) + T
            joints_rec_scene_coord_from_smpl = joints_rec_scene_coord_from_smpl.reshape(batch_size, clip_len, self._train_ds.joints_num, 3)

            ###### trans to camera coord
            joints_rec_scene_coord_from_smpl = joints_rec_scene_coord_from_smpl.reshape(batch_size*clip_len, -1, 3)
            joints_rec_cam_coord = torch.matmul(torch.linalg.inv(self._train_ds.cam_R), (joints_rec_scene_coord_from_smpl - self._train_ds.cam_t).permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size*clip_len, 22, 3]
            ###### project to 2d
            focal_length = batch['focal_length'].unsqueeze(1).repeat(1, clip_len, 1).reshape(-1, 2)  # [batch_size*clip_len, 2]
            camera_center = batch['camera_center'].unsqueeze(1).repeat(1, clip_len, 1).reshape(-1, 2)
            joints_rec_2d = perspective_projection(points=joints_rec_cam_coord,
                                                   focal_length=focal_length,
                                                   camera_center=camera_center)  # [batch_size*clip_len, 22, 2]
            joints_rec_2d = joints_rec_2d.reshape(batch_size, clip_len, -1, 2)  # [batch_size, clip_len, 22, 2]
            loss_joints_2d = self.l1_loss(joints_rec_2d, batch['keypoints_2d'][:, :clip_len, :, 0:2])  # [batch_size, clip_len, 22, 2]
            loss_joints_2d = loss_joints_2d * batch['keypoints_2d'][:, :clip_len, :, [-1]]
            # currently select those joints in simplx main body 22 joints for 2d loss
            # more joints could be added for better alignment with image observations
            loss_joints_2d = loss_joints_2d[:, :, [16, 18, 20, 17, 19, 21, 4, 5, 7, 8]]
            loss_joints_2d = loss_joints_2d.mean()

            # print('loss_joints_2d: ', loss_joints_2d)
            grad_joints_2d = torch.autograd.grad([-loss_joints_2d], [x_t])[0]  # [bs, T, feat_dim] same shape as x_t
            grad_joints_2d[:, :, 0:traj_feat_dim] = 0
            grad_joints_2d[:, :, -4:] = 0
            x_t.detach()

        return grad_joints_2d
