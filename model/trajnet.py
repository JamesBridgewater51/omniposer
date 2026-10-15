import einops
from utils.repr_adapter import decode_smplx_from_full_repr
from data_sets.common.quaternion import rot6d_to_rotmat
from utils.other_utils import *
from utils.konia_transform import angle_axis_to_rotation_matrix
from model.heads import *
from model.heads import FrequencyAdapter
from utils.evaluation import MotionEvaluator, procrustes_align
from utils.construct import construct_class_by_name
from typing import Literal, Optional, Dict, Any
from utils.batch_util import process_batch_cond
from utils.feature_accessor import select_modalities, get_modality_dim
from model.metadata_moe import (
    build_routing_metadata_from_batch,
    reject_legacy_moe_surfaces,
)


class ControlNet(nn.Module):

    def __init__(self,time_dim=32, control_cond_dim=272, traj_feat_dim=4, mid_dim=256,):
        super().__init__()
        self.control_cond_dim = control_cond_dim
        self.traj_feat_dim = traj_feat_dim

        #################################### controlnet ####################################
        # dim of control condition local body feature --> dim of traj feature
        self.control_zero_conv_0 = zero_module(nn.Conv1d(self.control_cond_dim, self.traj_feat_dim, 1, padding=0))

        self.control_enc1 = ResidualTemporalBlock(self.traj_feat_dim, mid_dim // 8, input_t=True, t_embed_dim=time_dim)
        self.control_zero_conv_1 = zero_module(nn.Conv1d(mid_dim // 8, 32, 1, padding=0))
        self.control_downsample1 = Downsample1d(mid_dim // 8 * 2)

        self.control_enc2 = ResidualTemporalBlock(mid_dim // 8 * 2, mid_dim // 4, input_t=True, t_embed_dim=time_dim)
        self.control_zero_conv_2 = zero_module(nn.Conv1d(mid_dim // 8 * 2, mid_dim // 8, 1, padding=0))
        self.control_downsample2 = Downsample1d(mid_dim // 4 * 2)

        self.control_enc3 = ResidualTemporalBlock(mid_dim // 4 * 2, mid_dim // 2, input_t=True, t_embed_dim=time_dim)
        self.control_zero_conv_3 = zero_module(nn.Conv1d(mid_dim // 4 * 2, mid_dim // 4, 1, padding=0))
        self.control_downsample3 = Downsample1d(mid_dim // 2 * 2)

        self.control_enc4 = ResidualTemporalBlock(mid_dim // 2 * 2, mid_dim, input_t=True, t_embed_dim=time_dim)
        self.control_zero_conv_4 = zero_module(nn.Conv1d(mid_dim, mid_dim // 4 * 2, 1, padding=0))
        self.control_downsample4 = Downsample1d(mid_dim * 2)

        ###########
        self.control_mid_block1 = ResidualTemporalBlock(mid_dim * 2, mid_dim, input_t=True, t_embed_dim=time_dim)
        self.control_mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, input_t=True, t_embed_dim=time_dim)
        self.control_zero_conv_mid = zero_module(nn.Conv1d(mid_dim, mid_dim, 1, padding=0))
        ###########

    def forward(self, batch, h_cond, time_embed):
        '''
            batch['control_cond'] : [bs, T, control_feat_dim]
            time_embed: [bs, 32], diffusion timestep embedding
        '''
        control_cond = batch['control_cond']

        ################# control net
        control_cond = einops.rearrange(control_cond, 'b h t -> b t h')  # [bs, 272, T]
        control_cond = self.control_zero_conv_0(control_cond)  # [bs, self.traj_dim, T]

        control_cond = self.control_enc1(control_cond, time_embed)  # [bs, mid_dim/8, T]
        control_cond_zero_conv_1 = self.control_zero_conv_1(control_cond)
        control_cond = self.control_downsample1(torch.cat([control_cond, h_cond[0]], dim=1))  # [bs, mid_dim/8 * 2, T/2]

        control_cond = self.control_enc2(control_cond, time_embed)  # [bs, mid_dim/4, T/2]
        control_cond_zero_conv_2 = self.control_zero_conv_2(control_cond)
        control_cond = self.control_downsample2(torch.cat([control_cond, h_cond[1]], dim=1))  # [bs, mid_dim/4*2, T/4]

        control_cond = self.control_enc3(control_cond, time_embed)  # [bs, mid_dim/2, T/4]
        control_cond_zero_conv_3 = self.control_zero_conv_3(control_cond)
        control_cond = self.control_downsample3(torch.cat([control_cond, h_cond[2]], dim=1))  # [bs, mid_dim/2*2, T/8]

        control_cond = self.control_enc4(control_cond, time_embed)  # [bs, mid_dim, T/8]
        control_cond_zero_conv_4 = self.control_zero_conv_4(control_cond)
        control_cond = self.control_downsample4(torch.cat([control_cond, h_cond[3]], dim=1))  # [bs, mid_dim*2, T/16]

        control_cond = self.control_mid_block1(control_cond, time_embed)  # [bs, mid_dim, T/16]
        control_cond = self.control_mid_block2(control_cond, time_embed)  # [bs, mid_dim, T/16]
        control_cond_zero_conv_mid = self.control_zero_conv_mid(control_cond)

        return control_cond_zero_conv_1, control_cond_zero_conv_2, control_cond_zero_conv_3, \
            control_cond_zero_conv_4, control_cond_zero_conv_mid




class TrajNet(nn.Module):
    """
    TrajNet: Trajectory prediction network for motion capture data.
    
    This class supports both legacy architecture initialization (via traj_arch_type)
    and config-based architecture instantiation (via network_config).
    
    Config-based instantiation example:
        network_config = {
            'class_name': 'model.architectures.UNetTCN',
            'input_dim': 12,
            'cond_dim': 66,
            'output_dim': 12,
            'mid_dim': 256,
            'time_dim': 32,
            'input_t': True,
            'model_type': 'flow_matching',
        }
        trajnet = TrajNet(network_config=network_config, ...)
    """

    def __init__(self,
                 device=None, dataset=None,
                 cano_jts_pos: bool = False,
                 use_masked_conditioner: bool = True,
                 abstract_model_type:Literal['generative', 'discriminative', 'descriptive'] = 'generative',
                 network_config: Optional[Dict[str, Any]] = None,
                 weight_loss_root_rec_repr=0.0,
                 weight_loss_root_pos_global=0.0, weight_loss_root_vel_global=0.0,
                 weight_loss_root_rot_vel_from_abs_traj=0.0,
                 weight_loss_root_smplx_transl_vel=0.0, weight_loss_root_smplx_rot_vel=0.0,
                 weight_loss_root_smooth=0.0,
                 weight_loss_root_rot_cos_smooth_from_abs_traj=0.0,
                 use_freq_transform=False,
                 freq_config=None,
                 ):
        super().__init__()

        self.use_freq_transform = use_freq_transform
        self.freq_config = freq_config if freq_config is not None else {}

        # Store both train and test datasets; test_ds can be set later via set_test_dataset()
        self._train_ds = dataset
        self._test_ds = None  # Will be set via set_test_dataset()
        self.cano_jts_pos = cano_jts_pos
        self.task = 'traj'  # TrajNet always operates on trajectory task

        # Derive representation flags from dataset's representation registry
        # These flags are used to determine:
        # 1. include_global_joint_rotations: Whether the model's OUTPUT uses R_world_joints vs. smplx_body_pose_6d
        # 2. repr_abs_only: Whether velocity representations (smplx_trans_vel, smplx_rot_vel) are present
        cond_modalities = self._train_ds.repr_registry.cond_modalities.get(self.task, [])
        predicted_modalities = self._train_ds.repr_registry.predicted_modalities.get(self.task, [])
        repr_list = self._train_ds._repr_list

        # Check if joint rotations are in CONDITIONING (used to configure MaskedConditioningEncoder)
        self.has_rotation_in_cond = 'R_world_joints' in cond_modalities or 'R_root_joints' in cond_modalities

        # Check if velocity representations are present
        self.repr_abs_only = 'local_vel' not in repr_list

        # For TrajNet, we don't typically predict joint rotations (only root rotation as smplx_rot_6d)
        # So this flag is primarily used for conditioning encoder configuration
        self.include_global_joint_rotations = self.has_rotation_in_cond

        self.abstract_model_type = abstract_model_type

        self.num_joints = 22

        # Use dataset-computed dimensions
        self._traj_feat_dim = get_modality_dim(self._train_ds.traj_repr_name_list, self._train_ds._repr_indices)
        self.cond_feat_dim = self._train_ds.cond_feat_dim_traj
        self.use_masked_conditioner = use_masked_conditioner

        if self.use_freq_transform:
             feat_dim = self._traj_feat_dim if self.abstract_model_type == 'generative' else self.cond_feat_dim
             self.freq_adapter = FrequencyAdapter(feat_dim, **self.freq_config)

        self.weight_loss_root_rec_repr = weight_loss_root_rec_repr
        self.weight_loss_root_pos_global = weight_loss_root_pos_global
        self.weight_loss_root_vel_global = weight_loss_root_vel_global
        self.weight_loss_root_rot_vel_from_abs_traj = weight_loss_root_rot_vel_from_abs_traj
        self.weight_loss_root_smplx_transl_vel = weight_loss_root_smplx_transl_vel
        self.weight_loss_root_smplx_rot_vel = weight_loss_root_smplx_rot_vel
        self.weight_loss_root_smooth = weight_loss_root_smooth
        self.weight_loss_root_rot_cos_smooth_from_abs_traj = weight_loss_root_rot_cos_smooth_from_abs_traj

        self.mse_loss = nn.MSELoss(reduction='none').to(device)
        self.device = device
        self.evaluator = MotionEvaluator(fps=30)

        # Store network config for config-based instantiation
        reject_legacy_moe_surfaces(network_config, context="TrajNet network_config")
        self.network_config = network_config
        self.last_traj_moe_aux_losses = {}
        self.last_traj_moe_diagnostics = {}
        self.last_traj_route_state = {}

        # Initialize architecture based on type
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
    
    def _init_from_config(self, network_config: Dict[str, Any]):
        """
        Initialize architecture from config dictionary using construct_class_by_name.
        
        This allows specifying any architecture class in the config, following the
        IMM/dnnlib pattern.
        
        Args:
            network_config: Dictionary with 'class_name' and constructor kwargs.
                           If dimensions are not specified, they will be inferred from dataset.
            time_dim: Time embedding dimension (used as fallback if not in config)
            mid_dim: Hidden dimension (used as fallback if not in config)
        """
        # For generative models, input is x_t (traj_feat_dim), for discriminative, input is cond
        if self.abstract_model_type == 'generative':
            input_dim = self._traj_feat_dim
        else:
            input_dim = self.cond_feat_dim
        
        cond_dim = self.cond_feat_dim
        output_dim = self._traj_feat_dim
        
        # Construct the architecture
        self.backbone = construct_class_by_name(
            **network_config,
            input_dim=input_dim,
            cond_dim=cond_dim,
            output_dim=output_dim,
        )
        # Initialize masked conditioner if needed (still task-specific)
        if self.use_masked_conditioner:
            # Calculate encoder dimensions from registry (registry-driven interface)
            cond_modalities = self._train_ds.repr_registry.cond_modalities[self.task]
            per_joint_dim = 0
            global_dim = 0
            for mod_name in cond_modalities:
                mod = self._train_ds.repr_registry.get_modality(mod_name)
                if mod.is_joint_based:
                    per_joint_dim += mod.per_joint_dim
                else:
                    global_dim += mod.dim

            # No prefix for traj task
            prefix_dim = 0

            self.masked_conditioner = MaskedConditioningEncoder(
                cond_dim=self.cond_feat_dim + prefix_dim,
                per_joint_dim=per_joint_dim,
                num_joints=self.num_joints,
                global_dim=global_dim,
                prefix_dim=prefix_dim,
                hidden_dim=256,
                num_blocks=2,
                use_time=self.abstract_model_type == 'generative',
                time_dim=32,
                dropout=0.1
            )

    def _build_routing_metadata(self, batch):
        return build_routing_metadata_from_batch(
            batch,
            routing_groups=self._train_ds.repr_registry.get_routing_group_mapping('body_part_6', expected_topology='smplx_22'),
            device=self.device,
        )

    def reset_route_state(self) -> None:
        self.last_traj_route_state = {}
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'reset_route_state'):
            self.backbone.reset_route_state()

    def get_moe_loss_dict(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'backbone') or not hasattr(self.backbone, 'get_moe_loss_dict'):
            return {}
        return self.backbone.get_moe_loss_dict()

    def moe_aux_loss_in_main_loss(self) -> bool:
        return bool(hasattr(self, 'backbone') and hasattr(self.backbone, 'moe_aux_loss_in_main_loss') and self.backbone.moe_aux_loss_in_main_loss())

    def _forward_config(self, batch, t, d=None, **kwargs):
        """
        Forward pass using config-based architecture.
        
        The architecture is expected to have a forward method with signature:
            forward(x, cond, t=None, d=None) -> output
        
        Where:
            x: [bs, T, input_dim] main input (x_t for generative, cond for discriminative)
            cond: [bs, T, cond_dim] conditioning signal
            t: [bs] time step (optional, for generative models)
            d: [bs] step size (optional, for meanflow/shortcut)
        """
        # Prepare main input
        if self.abstract_model_type == 'generative':
            x = batch['x_t']  # [bs, T, traj_dim]
        else:
            x = batch['cond']  # [bs, T, cond_dim]
        
        # Apply frequency transform if enabled
        if self.use_freq_transform:
            # [bs, T, dim] -> [bs, dim, T]
            x = x.permute(0, 2, 1)
            x = self.freq_adapter(x)
            x = x.permute(0, 2, 1)
        
        # Prepare conditioning
        cond = batch['cond']  # [bs, T, cond_dim]
        
        time_emb = None
        if self.abstract_model_type == 'generative' and hasattr(self, 'backbone') and hasattr(self.backbone, 'compute_time_embedding'):
            time_emb = self.backbone.compute_time_embedding(t, d)

        # Apply masked conditioner if enabled
        if self.use_masked_conditioner:
            cond_mask = batch['cond_mask']  # [bs, T, cond_dim] feature-level mask
            cond = self.masked_conditioner(cond, cond_mask, time_emb)

        # Forward through backbone
        routing_metadata = self._build_routing_metadata(batch)
        route_state = kwargs.get('route_state', self.last_traj_route_state)
        backbone_kwargs = {'t': t, 'd': d, 'routing_metadata': routing_metadata, 'route_state': route_state}
        if time_emb is not None:
            backbone_kwargs['time_emb'] = time_emb
        output = self.backbone(x, cond, **backbone_kwargs)
        self.last_traj_moe_aux_losses = getattr(self.backbone, 'last_traj_moe_aux_losses', {})
        self.last_traj_moe_diagnostics = getattr(self.backbone, 'last_traj_moe_diagnostics', {})
        self.last_traj_route_state = getattr(self.backbone, 'last_traj_route_state', {})

        return output

    def forward(self, batch, t, d=None, **kwargs):
        return self._forward_config(batch, t, d, **kwargs)
   

    def compute_losses_with_smpl(self, batch, model_output, smplx_model=None, compute_eval_metrics=False, **kwargs):
        loss_dict = {}

        # Select dataset based on context: use test_ds for eval, train_ds for training
        ds = self._get_dataset_for_context(is_eval=compute_eval_metrics)

        # model_output is the predicted (normalized) trajectory part, with dimension self.model_output_dim
        # batch['motion_repr_clean'] is the full ground truth (normalized) representation
        
        # Concatenate predicted trajectory part (model_output) with GT pose part
        # The pose part starts after the full dataset trajectory feature dimension
        # Get non-trajectory modalities (pose part)
        non_traj_modalities = [m for m in ds._repr_list if m not in ds.traj_repr_name_list]
        gt_pose_part_normalized = select_modalities(batch['motion_repr_clean'], non_traj_modalities, ds._repr_indices)

        # full_repr_rec is normalized
        # model_output must have dimension self._traj_feat_dim for this to be a direct replacement.
        assert model_output.shape[-1] == self._traj_feat_dim

        full_repr_rec_traj_part = model_output
        full_repr_rec_normalized = torch.cat([full_repr_rec_traj_part, gt_pose_part_normalized], dim=-1)

        full_repr_clean_normalized = batch['motion_repr_clean']

        if ds.norm_strategy == 'l2':
            sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in batch.items() if k.startswith('extras/sample_norms')}
        else:
            sample_norms = None
        
        full_repr_clean_unnormalized = ds._unnormalize_full_tensor(full_repr_clean_normalized, self.device, sample_norms=sample_norms)
        full_repr_rec_unnormalized = ds._unnormalize_full_tensor(full_repr_rec_normalized, self.device, sample_norms=sample_norms)

        traj_repr_keys_abs = ['smplx_rot_6d', 'smplx_trans']
        traj_repr_keys_vel = ['smplx_rot_vel', 'smplx_trans_vel']
        current_traj_repr_keys = traj_repr_keys_abs + (traj_repr_keys_vel if not self.repr_abs_only else [])

        traj_losses = []
        for key in current_traj_repr_keys:
            if key in ds._repr_indices:
                s, e = ds._repr_indices[key]
                loss_key = (full_repr_rec_normalized[:, :, s:e] - full_repr_clean_normalized[:, :, s:e]).square()
                traj_losses.append(loss_key)
                loss_dict[f'loss_repr_traj_{key}'] = loss_key.mean()
            else:
                loss_dict[f'loss_repr_traj_{key}'] = torch.tensor(0.0, device=self.device)

        loss_dict['loss_root_rec_repr'] = torch.cat(traj_losses, dim=-1).mean() if traj_losses else full_repr_rec_normalized.new_zeros(())

        with torch.no_grad():
            repr_dict_clean_unnorm, joint_pos_clean, _ = decode_smplx_from_full_repr(
                full_repr=full_repr_clean_unnormalized,
                repr_list=ds._repr_list,
                repr_indices=ds._repr_indices,
                smplx_model=smplx_model,
                output_global_rotations=self.include_global_joint_rotations,
                cano_jts_pos=self.cano_jts_pos,
                return_verts=False,
            )
        post_override = (
            {"smplx_body_pose_6d": repr_dict_clean_unnorm["smplx_body_pose_6d"]}
            if self.include_global_joint_rotations
            else {
                "R_world_joints": repr_dict_clean_unnorm["R_world_joints"],
                "R_root_joints": repr_dict_clean_unnorm["R_root_joints"],
            }
        )
        repr_dict_rec_unnorm, joint_pos_rec_from_smpl, _ = decode_smplx_from_full_repr(
            full_repr=full_repr_rec_unnormalized,
            repr_list=ds._repr_list,
            repr_indices=ds._repr_indices,
            smplx_model=smplx_model,
            output_global_rotations=self.include_global_joint_rotations,
            cano_jts_pos=self.cano_jts_pos,
            return_verts=False,
            post_override_repr_dict=post_override,
        )

        root_pos_clean = joint_pos_clean[:, :, 0]
        root_pos_rec_from_smpl = joint_pos_rec_from_smpl[:, :, 0]

        loss_dict['loss_root_pos_global'] = self.mse_loss(root_pos_rec_from_smpl, root_pos_clean).mean()

        # Visible joint loss (using cond_joint_mask [bs, T, 22, 1])
        joint_mask = batch['cond_joint_mask']
        assert joint_mask is not None
        pos_mask = joint_mask[..., 0]
        cond_mask_expanded = pos_mask.unsqueeze(-1)  # Shape: [bs, T, 22, 1]
        # Ensure joint_pos_rec_from_smpl and joint_pos_clean are [bs, T, 22, 3]
        loss_dict['loss_visible_joint_pos_global'] = (
            (joint_pos_rec_from_smpl - joint_pos_clean).square() * cond_mask_expanded
        ).sum() / (cond_mask_expanded.sum() * 3 + 1e-6)

        root_vel_clean = root_pos_clean[:, 1:] - root_pos_clean[:, 0:-1]
        root_vel_rec_from_smpl = root_pos_rec_from_smpl[:, 1:] - root_pos_rec_from_smpl[:, 0:-1]
        
        loss_dict['loss_root_vel_global'] = self.mse_loss(root_vel_rec_from_smpl, root_vel_clean).mean()

        if not self.repr_abs_only:
            for k_opt in [ 'loss_root_smplx_rot_vel', 'loss_root_smplx_transl_vel']:
                loss_dict[k_opt] = torch.tensor(0.0, device=self.device)

            bs = joint_pos_clean.shape[0]
            global_orient_mat_rec = rot6d_to_rotmat(repr_dict_rec_unnorm['smplx_rot_6d'].reshape(-1, 6))
            global_orient_mat_rec = global_orient_mat_rec.reshape(bs, -1, 3, 3)
            dRdt_rec = global_orient_mat_rec[:, 1:] - global_orient_mat_rec[:, 0:-1]
            smplx_rot_vel_rec = estimate_angular_velocity(global_orient_mat_rec[:, 0:-1], dRdt_rec)
            smplx_rot_vel_clean = repr_dict_clean_unnorm['smplx_rot_vel'][:, 0:smplx_rot_vel_rec.shape[1]] # Align length
            loss_dict['loss_root_smplx_rot_vel'] = self.mse_loss(smplx_rot_vel_rec, smplx_rot_vel_clean).mean()

            smplx_transl_vel_rec = repr_dict_rec_unnorm['smplx_trans'][:, 1:] - repr_dict_rec_unnorm['smplx_trans'][:, 0:-1]
            smplx_transl_vel_clean = repr_dict_clean_unnorm['smplx_trans_vel'][:, 0:smplx_transl_vel_rec.shape[1]] # Align length
            loss_dict['loss_root_smplx_transl_vel'] = self.mse_loss(smplx_transl_vel_rec, smplx_transl_vel_clean).mean()

        root_acc_rec_from_smpl = root_vel_rec_from_smpl[:, 1:] - root_vel_rec_from_smpl[:, 0:-1]
        loss_dict['loss_root_smooth'] = torch.mean(root_acc_rec_from_smpl ** 2)
        
        loss_dict["loss"] = (
            self.weight_loss_root_rec_repr * loss_dict['loss_root_rec_repr'] +
            self.weight_loss_root_pos_global * (
                loss_dict['loss_root_pos_global'] ) +
            self.weight_loss_root_vel_global * (
                loss_dict['loss_root_vel_global'] ) +
            self.weight_loss_root_smooth * (
                loss_dict['loss_root_smooth'] )
        )

        if not self.repr_abs_only:
            loss_dict['loss'] += (
                self.weight_loss_root_smplx_rot_vel * loss_dict['loss_root_smplx_rot_vel'] +
                self.weight_loss_root_smplx_transl_vel * loss_dict['loss_root_smplx_transl_vel']
            )

        if compute_eval_metrics:
            with torch.no_grad():
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
        # model_output: [bs, pose_feat_dim, 1, T]

        dataset_std_torch = torch.from_numpy(self._train_ds.Std).to(self.device).float()
        dataset_mean_torch = torch.from_numpy(self._train_ds.Mean).to(self.device).float()

        # Get non-trajectory modalities (pose part)
        non_traj_modalities = [m for m in self._train_ds._repr_list if m not in self._train_ds.traj_repr_name_list]
        gt_pose_part_normalized = select_modalities(batch['motion_repr_clean'], non_traj_modalities, self._train_ds._repr_indices)

        # full_repr_rec is normalized
        # model_output must have dimension self._traj_feat_dim for this to be a direct replacement.
        assert model_output.shape[-1] == self._traj_feat_dim

        full_repr_rec_traj_part = model_output
        full_repr_rec = torch.cat([full_repr_rec_traj_part, gt_pose_part_normalized], dim=-1)

        loss_rec_traj_repr_all_normalized = self.mse_loss(batch['motion_repr_clean'], full_repr_rec)
        
        full_repr_clean_unnormalized = batch['motion_repr_clean'] * dataset_std_torch + dataset_mean_torch
        full_repr_rec_unnormalized = full_repr_rec * dataset_std_torch + dataset_mean_torch

        loss_components_for_sum = loss_rec_traj_repr_all_normalized.clone()
        
        traj_repr_keys_abs = ['smplx_rot_6d', 'smplx_trans']
        traj_repr_keys_vel = ['smplx_rot_vel', 'smplx_trans_vel']
        current_traj_repr_keys = traj_repr_keys_abs + (traj_repr_keys_vel if not self.repr_abs_only else [])
        
        repr_dict_clean_unnorm, joint_pos_clean, _ = decode_smplx_from_full_repr(
            full_repr=full_repr_clean_unnormalized,
            repr_list=self._train_ds._repr_list,
            repr_indices=self._train_ds._repr_indices,
            smplx_model=smplx_model,
            output_global_rotations=self.include_global_joint_rotations,
            cano_jts_pos=self.cano_jts_pos,
            return_verts=False,
        )
        post_override = (
            {"smplx_body_pose_6d": repr_dict_clean_unnorm["smplx_body_pose_6d"]}
            if self.include_global_joint_rotations
            else {
                "R_world_joints": repr_dict_clean_unnorm["R_world_joints"],
                "R_root_joints": repr_dict_clean_unnorm["R_root_joints"],
            }
        )
        repr_dict_rec_unnorm, joint_pos_rec_from_smpl, _ = decode_smplx_from_full_repr(
            full_repr=full_repr_rec_unnormalized,
            repr_list=self._train_ds._repr_list,
            repr_indices=self._train_ds._repr_indices,
            smplx_model=smplx_model,
            output_global_rotations=self.include_global_joint_rotations,
            cano_jts_pos=self.cano_jts_pos,
            return_verts=False,
            post_override_repr_dict=post_override,
        )

        root_pos_clean = joint_pos_clean[:, :, 0]
        root_pos_rec_from_smpl = joint_pos_rec_from_smpl[:, :, 0]

        # Visible joint loss (using cond_joint_mask [bs, T, 22, 1])
        joint_mask = batch['cond_joint_mask']
        assert joint_mask is not None
        pos_mask = joint_mask[..., 0]
        cond_mask_expanded = pos_mask.unsqueeze(-1)  # Shape: [bs, T, 22, 1]
        # Ensure joint_pos_rec_from_smpl and joint_pos_clean are [bs, T, 22, 3]
        loss_visible_joint_pos_global_from_abs_traj = (self.mse_loss(joint_pos_rec_from_smpl * cond_mask_expanded, joint_pos_clean * cond_mask_expanded).sum()) / (cond_mask_expanded.sum() * 3 + 1e-6)  # Multiply by 3 for xyz

        # Compute the guidance gradient via autograd (RoHM-style).
        loss_foot_skating = loss_rec_traj_repr_all_normalized.mean()
        grad_skating = torch.autograd.grad([-loss_foot_skating], [inputs])[0]  # [bs, body_feat_dim, 1, T] same shape as x_t, body_feat_dim=294
        # print('loss_foot_skating_from_smpl: ', loss_foot_skating_from_smpl)
        model_output.detach()

        return grad_skating
