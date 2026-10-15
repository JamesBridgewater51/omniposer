# Copyright (c) Meta Platforms, Inc. and affiliates.
# Task-specific DiT architecture for pose estimation/generation

import torch
import torch.nn as nn
from collections.abc import Mapping
from typing import Optional, Literal, Dict, Any

from model.heads import (
    CrossAttentionBlock,
    PositionalEncoding,
    TimestepEmbedder,
    InputProcess,
    OutputProcess,
    FrequencyAdapter,
)
from model.metadata_moe import (
    MetadataRoutedGroupedMoEBlock,
    collect_weighted_moe_loss_dict,
    is_moe_enabled,
    moe_aux_loss_in_main_loss,
    validate_moe_config,
)
from utils.other_utils import REPR_DIM_DICT


class SpecificPoseDiT(nn.Module):
    """
    Task-specific DiT architecture for pose estimation/generation.
    
    This module explicitly handles the pose-specific conditioning:
    - Split conditioning into trajectory features and local body pose features
    - Apply masked conditioning encoder for handling occlusions
    - Separate encoders for trajectory and joint conditioning
    - Combine them through a final combiner
    
    This is the "aware" version that knows about:
    - Input structure (traj_feat_dim + pose features)
    - Physical meaning of features (trajectory vs body pose)
    - Task-specific conditioning (masked conditioner, etc.)
    
    Config-based construction via class_name pattern:
        class_name: model.architectures.pose_dit.SpecificPoseDiT
        model_feat_dim: 294
        cond_feat_dim: 66
        traj_feat_dim: 330
        latent_dim: 256
        ...
    """
    
    def __init__(
        self,
        # Feature dimensions
        model_feat_dim: int,
        cond_feat_dim: int,
        traj_feat_dim: int,
        # Architecture hyperparameters
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        nfeats: int = 1,
        # RoPE and FiLM options
        use_rope: bool = True,
        use_film: bool = False,
        # Model type options
        abstract_model_type: Literal['generative', 'discriminative'] = 'generative',
        generative_model_type: Literal['diffusion', 'meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching'] = 'diffusion',
        # Masked conditioning options
        num_joints: int = 22,
        # Representation options
        include_global_joint_rotations: bool = False,
        repr_abs_only: bool = True,
        # Finetune head options
        finetune_posenet: bool = True,
        # Frequency transform options
        use_freq_transform: bool = False,
        freq_config: Optional[Dict[str, Any]] = None,
        # Metadata MoE options
        moe: Optional[Dict[str, Any]] = None,
        routing_groups: Optional[Any] = None,
        per_joint_cond_dim: Optional[int] = None,
        joint_modality_names: Optional[list] = None,
        global_modality_names: Optional[list] = None,
        # Device
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.model_feat_dim = model_feat_dim
        self.cond_feat_dim = cond_feat_dim
        self.traj_feat_dim = traj_feat_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.nfeats = nfeats
        self.use_rope = use_rope
        self.use_film = use_film
        self.abstract_model_type = abstract_model_type
        self.generative_model_type = generative_model_type
        self.num_joints = num_joints
        self.include_global_joint_rotations = include_global_joint_rotations
        self.repr_abs_only = repr_abs_only
        self.finetune_posenet = finetune_posenet
        self.use_freq_transform = use_freq_transform
        self.freq_config = freq_config if freq_config is not None else {}
        self.device = device
        self.moe_config = moe or {}
        self.grouped_moe_config = validate_moe_config(
            self.moe_config.get("grouped", {"enabled": False}) if isinstance(self.moe_config, Mapping) else {"enabled": False},
            component="pose_moe",
            require_grouping=bool(isinstance(self.moe_config, Mapping) and self.moe_config.get("grouped", {}).get("enabled", False)),
            allowed_routing_granularities={"group", "token"},
        )
        self.routed_ffn_moe_config = validate_moe_config(
            self.moe_config.get("routed_ffn", {"enabled": False}) if isinstance(self.moe_config, Mapping) else {"enabled": False},
            component="pose_ffn_moe",
            allowed_routing_granularities={"token", "frame", "segment"},
        )
        self.grouped_moe_enabled = is_moe_enabled(self.grouped_moe_config)
        self.routed_ffn_moe_enabled = is_moe_enabled(self.routed_ffn_moe_config)
        self.routing_groups = routing_groups
        self.per_joint_cond_dim = per_joint_cond_dim
        self.joint_modality_names = joint_modality_names or []
        self.global_modality_names = global_modality_names or []
        self.last_pose_moe_aux_losses = {}
        self.last_pose_moe_diagnostics = {}
        self.last_pose_route_state = None
        self.last_pose_ffn_moe_aux_losses = {}
        self.last_pose_ffn_moe_diagnostics = {}
        self.last_pose_ffn_route_state = {}

        # Frequency adapter (optional)
        if self.use_freq_transform:
            feat_dim = self.model_feat_dim if self.abstract_model_type == 'generative' else (self.traj_feat_dim + self.cond_feat_dim)
            self.freq_adapter = FrequencyAdapter(feat_dim, **self.freq_config)
        
        # Input processing for x_t (generative mode)
        self.pose_encoder = InputProcess(model_feat_dim * nfeats, latent_dim)
        
        # Trajectory condition encoder
        self.traj_cond_encoder = InputProcess(traj_feat_dim, latent_dim)
        
        # Joint condition encoder (linear projection)
        self.joint_cond_encoder = nn.Linear(cond_feat_dim, latent_dim)
        if self.grouped_moe_enabled:
            if self.per_joint_cond_dim is None:
                if cond_feat_dim % num_joints != 0:
                    raise ValueError("per_joint_cond_dim must be provided when cond_feat_dim is not divisible by num_joints.")
                self.per_joint_cond_dim = cond_feat_dim // num_joints
            self.joint_token_encoder = nn.Linear(self.per_joint_cond_dim, latent_dim)
            self.grouped_moe_block = MetadataRoutedGroupedMoEBlock(
                latent_dim,
                routing_groups,
                self.grouped_moe_config,
                component="pose_moe",
            )
        else:
            self.grouped_moe_block = None

        # Note: Masked conditioning is now handled externally by PoseNet
        # This allows for better separation of concerns and consistent API
        # across different architectures
        
        # Final combiner for trajectory and joint conditioning
        self.final_cond_combiner = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder layers
        self.seqTransEncoder = nn.ModuleList([
            CrossAttentionBlock(
                latent_dim,
                num_heads,
                dropout,
                activation,
                use_rope=use_rope,
                use_film=use_film,
                d_film_emb=latent_dim if use_film else None,
                routed_ffn_config=self.routed_ffn_moe_config if self.routed_ffn_moe_enabled else None,
            )
            for _ in range(num_layers)
        ])
        
        # Time embedders (for generative models)
        if abstract_model_type == 'generative':
            if generative_model_type in ['meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching']:
                self.t_embedder = TimestepEmbedder(latent_dim)
                self.d_embedder = TimestepEmbedder(latent_dim)
                if generative_model_type in ['meanflow', 'shortcut']:
                    self.time_d_combiner = nn.Sequential(
                        nn.Linear(latent_dim * 2, latent_dim * 4),
                        nn.Mish(),
                        nn.Linear(latent_dim * 4, latent_dim)
                    )
            else:  # 'diffusion'
                self.embed_timestep = TimestepEmbedder(latent_dim)
        
        # Output processing
        self.output_process = OutputProcess(model_feat_dim, latent_dim, nfeats)
        
        # Finetune heads (optional)
        if finetune_posenet:
            if include_global_joint_rotations:
                self.R_world_joints_head = nn.Linear(REPR_DIM_DICT['R_world_joints'], REPR_DIM_DICT['R_world_joints'])
                self.R_root_joints_head = nn.Linear(REPR_DIM_DICT['R_root_joints'], REPR_DIM_DICT['R_root_joints'])
            else:
                self.smplx_body_pose_6d_head = nn.Linear(REPR_DIM_DICT['smplx_body_pose_6d'], REPR_DIM_DICT['smplx_body_pose_6d'])
            self.smplx_betas_head = nn.Linear(REPR_DIM_DICT['smplx_betas'], REPR_DIM_DICT['smplx_betas'])
            self.smplx_foot_contact_head = nn.Linear(REPR_DIM_DICT['foot_contact'], REPR_DIM_DICT['foot_contact'])
            
            # Identity-like initialization
            self._init_finetune_heads()
    
    def _init_finetune_heads(self):
        """Initialize finetune heads with identity-like transformation."""
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
    
    def reset_route_state(self) -> None:
        self.last_pose_route_state = None
        self.last_pose_ffn_route_state = {}

    def get_moe_loss_dict(self) -> Dict[str, torch.Tensor]:
        losses = {}
        losses.update(collect_weighted_moe_loss_dict(
            self.last_pose_moe_aux_losses,
            self.grouped_moe_config,
            prefix="pose_moe",
        ))
        losses.update(collect_weighted_moe_loss_dict(
            self.last_pose_ffn_moe_aux_losses,
            self.routed_ffn_moe_config,
            prefix="pose_ffn_moe",
        ))
        return losses

    def moe_aux_loss_in_main_loss(self) -> bool:
        return moe_aux_loss_in_main_loss(self.grouped_moe_config) or moe_aux_loss_in_main_loss(self.routed_ffn_moe_config)

    def compute_time_embedding(self, t: torch.Tensor, d: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Compute time embedding based on model type.
        
        This method is called externally by PoseNet to get time embedding for
        the masked conditioner.
        
        Args:
            t: [bs] time step
            d: [bs] step size for meanflow/shortcut models (optional)
            
        Returns:
            time_emb: [bs, latent_dim] time embedding, or None if not generative
        """
        if self.abstract_model_type != 'generative':
            return None
            
        if self.generative_model_type == 'meanflow':
            t_emb = self.t_embedder(t * 1000).squeeze(0)
            d_emb = self.d_embedder(d * 1000).squeeze(0)
            emb = self.time_d_combiner(torch.cat([t_emb, d_emb], dim=-1))
        elif self.generative_model_type == 'shortcut':
            t_emb = self.t_embedder(t * 1000).squeeze(0)
            d_emb = self.d_embedder(d * 1000).squeeze(0)
            emb = self.time_d_combiner(torch.cat([t_emb, d_emb], dim=-1))
        elif self.generative_model_type in ['consistency', 'improved_consistency']:
            emb = self.t_embedder(t).squeeze(0)
        elif self.generative_model_type == 'flow_matching':
            emb = self.t_embedder(t * 1000).squeeze(0)
        else:  # 'diffusion'
            emb = self.embed_timestep(t).squeeze(0)
        
        return emb
    
    def forward(
        self,
        x_t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for SpecificPoseDiT.

        Args:
            x_t: Noisy input [bs, T, model_feat_dim] for generative mode.
            t: Timestep [bs] for generative models.
            d: Step size [bs] for meanflow/shortcut models.
            cond: Full conditioning tensor [bs, T, traj_feat_dim + cond_feat_dim].
                  Note: Masked conditioning should be applied externally before calling this.

        Returns:
            Output tensor [bs, T, model_feat_dim].
        """
        if cond.dim() != 3:
            raise ValueError(f"SpecificPoseDiT expects cond [bs, T, dim], got {cond.shape}")
        routing_metadata = kwargs.get("routing_metadata")
        route_state = kwargs.get("route_state") or self.last_pose_route_state
        ffn_route_state = kwargs.get("ffn_route_state") or self.last_pose_ffn_route_state
        self.last_pose_moe_aux_losses = {}
        self.last_pose_moe_diagnostics = {}
        self.last_pose_ffn_moe_aux_losses = {}
        self.last_pose_ffn_moe_diagnostics = {}

        # Compute time embedding for generative mode if caller did not precompute it.
        # compute_time_embedding returns [bs, latent_dim]
        emb = time_emb if time_emb is not None else self.compute_time_embedding(t, d)

        # For generative mode, we need [1, bs, latent_dim] for sequence concatenation
        if self.abstract_model_type == 'generative' and emb is not None:
            emb = emb.unsqueeze(0)  # [1, bs, latent_dim]

        # Apply frequency transform if enabled (generative mode)
        if self.abstract_model_type == 'generative':
            if x_t.dim() != 3:
                raise ValueError(f"SpecificPoseDiT expects x_t [bs, T, dim], got {x_t.shape}")
            if self.use_freq_transform:
                x_t_in = x_t.permute(0, 2, 1)  # [bs, T, dim] -> [bs, dim, T]
                x_t_out = self.freq_adapter(x_t_in)
                x_t = x_t_out.permute(0, 2, 1)  # [bs, dim, T] -> [bs, T, dim]

            x_proc = self.pose_encoder(x_t)  # [T, bs, latent_dim]
            xseq = torch.cat((emb, x_proc), axis=0)  # [T+1, bs, latent_dim]
            xseq = self.sequence_pos_encoder(xseq)  # [T+1, bs, latent_dim]
            x_query = xseq[1:].permute(1, 0, 2)  # [bs, T, latent_dim]

        # Process conditioning: cond is [bs, T, cond_full_dim]
        cond_full = cond

        # Apply frequency transform if enabled (discriminative mode)
        if self.abstract_model_type == 'discriminative':
            if self.use_freq_transform:
                cond_full = cond_full.permute(0, 2, 1)  # [bs, T, dim] -> [bs, dim, T]
                cond_full = self.freq_adapter(cond_full)
                cond_full = cond_full.permute(0, 2, 1)  # [bs, dim, T] -> [bs, T, dim]

        bs, T, _ = cond_full.shape

        # New cond structure: [joint-based interleaved] | [global including traj]
        # cond_feat_dim = joint-based dim, traj_feat_dim = global dim (including traj modalities)
        # (1) Process Joint/Pose Condition (joint-based features)
        joint_based_cond = cond_full[:, :, :self.cond_feat_dim]

        # (2) Process Trajectory/Global Condition (global features including traj)
        # Note: Masked conditioning is now applied externally by PoseNet before calling this
        global_cond = cond_full[:, :, self.cond_feat_dim:]

        if self.grouped_moe_enabled:
            if joint_based_cond.shape[-1] != self.num_joints * self.per_joint_cond_dim:
                raise ValueError(
                    f"Grouped MoE expected joint conditioning dim {self.num_joints * self.per_joint_cond_dim}, "
                    f"got {joint_based_cond.shape[-1]}."
                )
            joint_tokens = joint_based_cond.reshape(bs, T, self.num_joints, self.per_joint_cond_dim)
            joint_tokens = self.joint_token_encoder(joint_tokens)
            joint_tokens = self.grouped_moe_block(
                joint_tokens,
                routing_metadata=routing_metadata,
                route_state=route_state,
            )
            joint_delta = joint_tokens.mean(dim=2)
            self.last_pose_moe_aux_losses = self.grouped_moe_block.last_aux_losses
            self.last_pose_moe_diagnostics = self.grouped_moe_block.last_diagnostics
            self.last_pose_route_state = self.grouped_moe_block.last_route_state
        joint_latent = self.joint_cond_encoder(joint_based_cond)  # [bs, T, latent_dim]
        if self.grouped_moe_enabled:
            joint_latent = joint_latent + joint_delta

        cond_traj_latent = self.traj_cond_encoder(global_cond)  # [T, bs, latent_dim]
        cond_traj_latent = cond_traj_latent.permute(1, 0, 2)  # [bs, T, latent_dim]

        combined_cond_features = torch.cat([cond_traj_latent, joint_latent], dim=-1)
        cond_kv = self.final_cond_combiner(combined_cond_features)  # [bs, T, latent_dim]

        # Select output initialization based on mode
        if self.abstract_model_type == 'generative':
            output = x_query
        else:
            output = cond_kv

        # Apply transformer layers
        ffn_route_states = {}
        ffn_aux_losses = {}
        ffn_diagnostics = {}
        for layer_idx, layer in enumerate(self.seqTransEncoder):
            film_cond = emb.squeeze(0) if self.use_film and self.abstract_model_type == 'generative' else None
            layer_metadata = routing_metadata
            layer_state = ffn_route_state.get(layer_idx) if isinstance(ffn_route_state, dict) else None
            output = layer(
                output,
                cond_kv,
                film_cond=film_cond,
                routing_metadata=layer_metadata,
                route_state=layer_state,
            )
            if hasattr(layer, 'last_ffn_route_state') and layer.last_ffn_route_state is not None:
                ffn_route_states[layer_idx] = layer.last_ffn_route_state
            for loss_name, value in getattr(layer, 'last_ffn_aux_losses', {}).items():
                ffn_aux_losses[f"layer_{layer_idx}_{loss_name}"] = value
            for diag_name, value in getattr(layer, 'last_ffn_diagnostics', {}).items():
                ffn_diagnostics[f"layer_{layer_idx}_{diag_name}"] = value
        self.last_pose_ffn_route_state = ffn_route_states
        self.last_pose_ffn_moe_aux_losses = ffn_aux_losses
        self.last_pose_ffn_moe_diagnostics = ffn_diagnostics

        # Transpose back to [T, bs, latent_dim] for output processing
        output = output.permute(1, 0, 2)

        output = self.output_process(output)  # [bs, T, model_feat_dim]

        # Apply finetune heads
        if self.finetune_posenet:
            # output is already [bs, T, model_feat_dim]

            # NOTE: the slicing order is assumed to follow: smplx_body_pose_6d / R_world_joints+R_root_joints, smplx_betas, foot_contact
            smplx_betas_pred = self.smplx_betas_head(
                output[:, :, -REPR_DIM_DICT['smplx_betas']-REPR_DIM_DICT['foot_contact']:-REPR_DIM_DICT['foot_contact']]
            )
            smplx_foot_contact_pred = self.smplx_foot_contact_head(
                output[:, :, -REPR_DIM_DICT['foot_contact']:]
            )

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
    
    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

