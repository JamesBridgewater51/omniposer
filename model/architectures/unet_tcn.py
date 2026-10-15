# Copyright (c) Meta Platforms, Inc. and affiliates.
# Task-agnostic UNet TCN architecture for trajectory/sequence modeling.

import torch
import torch.nn as nn
import einops
from typing import Any, Dict, Optional, Literal

from model.heads import (
    ResidualTemporalBlock,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    SinusoidalPosEmb,
)
from model.metadata_moe import (
    MetadataRoutedTemporalMoEBlock,
    collect_weighted_moe_loss_dict,
    is_moe_enabled,
    moe_aux_loss_in_main_loss,
    validate_moe_config,
)


class UNetTCN(nn.Module):
    """
    Task-agnostic UNet TCN architecture for sequence-to-sequence modeling.
    
    Supports both generative (with time embedding) and discriminative modes.
    In generative mode, the model takes noisy input x_t and conditioning cond,
    along with a time embedding.
    In discriminative mode, the model takes only conditioning cond as input.
    
    Architecture:
        - Conditioning encoder: processes the conditioning signal
        - UNet encoder-decoder: processes the main input with skip connections
        - Optional time embedding injection at each layer
    
    Config example:
        network:
            class_name: model.architectures.unet_tcn.UNetTCN
            input_dim: 12  # dimension of main input (x_t or cond)
            cond_dim: 66   # dimension of conditioning signal
            output_dim: 12 # dimension of output
            mid_dim: 256   # hidden dimension
            time_dim: 32   # time embedding dimension
            input_t: true  # whether to use time embedding
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        output_dim: int,
        mid_dim: int = 256,
        time_dim: int = 32,
        input_t: bool = True,
        generative_model_type: Literal['diffusion', 'meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching'] = 'diffusion',
        moe: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.mid_dim = mid_dim
        self.time_dim = time_dim
        self.input_t = input_t
        self.generative_model_type = generative_model_type
        self.moe_config = validate_moe_config(
            moe or {"enabled": False},
            component="traj_moe",
            require_grouping=bool(moe and moe.get("enabled", False)),
            require_stage_names=bool(moe and moe.get("enabled", False)),
            allowed_routing_granularities={"frame", "segment"},
        )
        self.moe_enabled = is_moe_enabled(self.moe_config)
        self.traj_moe_blocks = nn.ModuleDict()
        self.last_traj_moe_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_traj_moe_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_traj_route_state: Dict[str, Dict[str, Any]] = {}

        # Time embedding
        if self.input_t:
            if self.generative_model_type in ['meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching']:
                self.t_embedder = nn.Sequential(
                    SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.Mish(), nn.Linear(time_dim * 4, time_dim),
                )
                self.d_embedder = nn.Sequential(
                    SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.Mish(), nn.Linear(time_dim * 4, time_dim),
                )
                if self.generative_model_type in ['meanflow', 'shortcut']:
                    self.time_d_combiner = nn.Sequential(
                        nn.Linear(time_dim * 2, time_dim * 4), nn.Mish(), nn.Linear(time_dim * 4, time_dim)
                    )
            else:  # 'diffusion'
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.Mish(), nn.Linear(time_dim * 4, time_dim),
                )

        # UNet encoder
        self.diff_enc1 = ResidualTemporalBlock(input_dim, mid_dim // 8, input_t=input_t, t_embed_dim=time_dim)
        self.diff_downsample1 = Downsample1d(mid_dim // 8 * 2)

        self.diff_enc2 = ResidualTemporalBlock(mid_dim // 8 * 2, mid_dim // 4, input_t=input_t, t_embed_dim=time_dim)
        self.diff_downsample2 = Downsample1d(mid_dim // 4 * 2)

        self.diff_enc3 = ResidualTemporalBlock(mid_dim // 4 * 2, mid_dim // 2, input_t=input_t, t_embed_dim=time_dim)
        self.diff_downsample3 = Downsample1d(mid_dim // 2 * 2)

        self.diff_enc4 = ResidualTemporalBlock(mid_dim // 2 * 2, mid_dim, input_t=input_t, t_embed_dim=time_dim)
        self.diff_downsample4 = Downsample1d(mid_dim * 2)

        # Middle blocks
        self.diff_mid_block1 = ResidualTemporalBlock(mid_dim * 2, mid_dim, input_t=input_t, t_embed_dim=time_dim)
        self.diff_mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, input_t=input_t, t_embed_dim=time_dim)

        # UNet decoder
        self.diff_upsample4 = Upsample1d(mid_dim)
        self.diff_dec4 = ResidualTemporalBlock(mid_dim * 2, mid_dim // 2, input_t=input_t, t_embed_dim=time_dim)

        self.diff_upsample3 = Upsample1d(mid_dim // 2)
        self.diff_dec3 = ResidualTemporalBlock(mid_dim // 2 * 2, mid_dim // 4, input_t=input_t, t_embed_dim=time_dim)

        self.diff_upsample2 = Upsample1d(mid_dim // 4)
        self.diff_dec2 = ResidualTemporalBlock(mid_dim // 4 * 2, mid_dim // 8, input_t=input_t, t_embed_dim=time_dim)

        self.diff_upsample1 = Upsample1d(mid_dim // 8)
        self.diff_dec1 = ResidualTemporalBlock(mid_dim // 8 * 2, 32, input_t=input_t, t_embed_dim=time_dim)

        self.diff_final_conv = nn.Sequential(
            Conv1dBlock(32, 32, kernel_size=5),
            nn.Conv1d(32, output_dim, 1),
        )

        # Conditioning encoder
        self.cond_enc1 = ResidualTemporalBlock(cond_dim, mid_dim // 8, input_t=False)
        self.cond_downsample1 = Downsample1d(mid_dim // 8)

        self.cond_enc2 = ResidualTemporalBlock(mid_dim // 8, mid_dim // 4, input_t=False)
        self.cond_downsample2 = Downsample1d(mid_dim // 4)

        self.cond_enc3 = ResidualTemporalBlock(mid_dim // 4, mid_dim // 2, input_t=False)
        self.cond_downsample3 = Downsample1d(mid_dim // 2)

        self.cond_enc4 = ResidualTemporalBlock(mid_dim // 2, mid_dim, input_t=False)
        self.cond_downsample4 = Downsample1d(mid_dim)

        if self.moe_enabled:
            stage_channels = {
                "diff_enc1": mid_dim // 8,
                "diff_enc2": mid_dim // 4,
                "diff_enc3": mid_dim // 2,
                "diff_enc4": mid_dim,
                "diff_mid_block1": mid_dim,
                "diff_mid_block2": mid_dim,
                "diff_dec4": mid_dim // 2,
                "diff_dec3": mid_dim // 4,
                "diff_dec2": mid_dim // 8,
                "diff_dec1": 32,
            }
            for stage_name in self.moe_config["stage_names"]:
                if stage_name not in stage_channels:
                    raise ValueError(f"Unknown TrajNet MoE stage '{stage_name}'.")
                self.traj_moe_blocks[stage_name] = MetadataRoutedTemporalMoEBlock(
                    stage_channels[stage_name],
                    self.moe_config,
                    component=f"traj_moe.{stage_name}",
                )

    def compute_time_embedding(self, t: torch.Tensor, d: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Compute time embedding based on model type."""
        if not self.input_t:
            return None
            
        if self.generative_model_type == 'meanflow':
            assert d is not None, "MeanFlow requires 'r' timestep (passed as 'd')."
            t_emb = self.t_embedder(t * 1000)
            d_emb = self.d_embedder(d * 1000)
            time_emb = self.time_d_combiner(torch.cat([t_emb, d_emb], dim=-1))
        elif self.generative_model_type == 'shortcut':
            assert d is not None, "Shortcut model requires step size 'd'."
            t_emb = self.t_embedder(t * 1000)
            d_emb = self.d_embedder(d * 1000)
            time_emb = self.time_d_combiner(torch.cat([t_emb, d_emb], dim=-1))
        elif self.generative_model_type in ['consistency', 'improved_consistency']:
            time_emb = self.t_embedder(t)
        elif self.generative_model_type == 'flow_matching':
            time_emb = self.t_embedder(t * 1000)
        else:  # 'diffusion'
            time_emb = self.time_mlp(t)
        
        return time_emb

    def encode_condition(self, cond: torch.Tensor) -> list:
        """
        Encode conditioning signal through the conditioning encoder.
        
        Args:
            cond: [bs, cond_dim, T] conditioning signal
            
        Returns:
            List of conditioning features at different scales for skip connections
        """
        h_cond = []

        cond = self.cond_enc1(cond, None)  # [bs, mid_dim/8, T]
        h_cond.append(cond)
        cond = self.cond_downsample1(cond)  # [bs, mid_dim/8, T/2]

        cond = self.cond_enc2(cond, None)  # [bs, mid_dim/4, T/2]
        h_cond.append(cond)
        cond = self.cond_downsample2(cond)  # [bs, mid_dim/4, T/4]

        cond = self.cond_enc3(cond, None)  # [bs, mid_dim/2, T/4]
        h_cond.append(cond)
        cond = self.cond_downsample3(cond)  # [bs, mid_dim/2, T/8]

        cond = self.cond_enc4(cond, None)  # [bs, mid_dim, T/8]
        h_cond.append(cond)

        return h_cond

    def reset_route_state(self) -> None:
        self.last_traj_route_state = {}

    def _apply_moe_stage(
        self,
        stage_name: str,
        output: torch.Tensor,
        routing_metadata: Optional[Any],
        route_state: Optional[Dict[str, Any]],
        source_steps: int,
    ) -> torch.Tensor:
        if stage_name not in self.traj_moe_blocks:
            return output
        block_route_state = route_state.get(stage_name) if isinstance(route_state, dict) else None
        output = self.traj_moe_blocks[stage_name](
            output,
            routing_metadata=routing_metadata,
            route_state=block_route_state,
            source_steps=source_steps,
        )
        return output

    def _collect_moe_outputs(self) -> None:
        aux_losses = {}
        diagnostics = {}
        route_state = {}
        for stage_name, block in self.traj_moe_blocks.items():
            for loss_name, value in block.last_aux_losses.items():
                aux_losses[f"{stage_name}_{loss_name}"] = value
            for diag_name, value in block.last_diagnostics.items():
                diagnostics[f"{stage_name}_{diag_name}"] = value
            if block.last_route_state is not None:
                route_state[stage_name] = block.last_route_state
        self.last_traj_moe_aux_losses = aux_losses
        self.last_traj_moe_diagnostics = diagnostics
        self.last_traj_route_state = route_state

    def get_moe_loss_dict(self) -> Dict[str, torch.Tensor]:
        losses = collect_weighted_moe_loss_dict(
            self.last_traj_moe_aux_losses,
            self.moe_config,
            prefix="traj_moe",
        )
        return losses

    def moe_aux_loss_in_main_loss(self) -> bool:
        return moe_aux_loss_in_main_loss(self.moe_config)

    def forward(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        routing_metadata: Optional[Any] = None,
        route_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet TCN.

        Args:
            x_t: [bs, T, input_dim] main input sequence
            cond: [bs, T, cond_dim] conditioning sequence
            t: [bs] time step for generative models (optional)
            d: [bs] step size for meanflow/shortcut models (optional)

        Returns:
            output: [bs, T, output_dim] output sequence
        """
        if x_t.dim() != 3 or cond.dim() != 3:
            raise ValueError(f"UNetTCN expects 3D tensors [bs, T, dim], got x_t: {x_t.shape}, cond: {cond.shape}")
        source_steps = x_t.shape[1]
        self.last_traj_moe_aux_losses = {}
        self.last_traj_moe_diagnostics = {}

        # Compute time embedding if caller did not precompute it for masked conditioning.
        if time_emb is None:
            time_emb = self.compute_time_embedding(t, d) if t is not None else None

        # Rearrange to channel-first format for conv layers
        # Input is [bs, T, dim], need [bs, dim, T]
        x_t = einops.rearrange(x_t, 'b t h -> b h t')  # [bs, input_dim, T]
        cond = einops.rearrange(cond, 'b t h -> b h t')  # [bs, cond_dim, T]

        # Encode conditioning
        h_cond = self.encode_condition(cond)

        # UNet encoder with skip connections
        h_diff = []

        output = self.diff_enc1(x_t, time_emb)  # [bs, mid_dim/8, T]
        output = self._apply_moe_stage("diff_enc1", output, routing_metadata, route_state, source_steps)
        h_diff.append(output)
        output = self.diff_downsample1(torch.cat([output, h_cond[0]], dim=1))  # [bs, mid_dim/8 * 2, T/2]

        output = self.diff_enc2(output, time_emb)  # [bs, mid_dim/4, T/2]
        output = self._apply_moe_stage("diff_enc2", output, routing_metadata, route_state, source_steps)
        h_diff.append(output)
        output = self.diff_downsample2(torch.cat([output, h_cond[1]], dim=1))  # [bs, mid_dim/4*2, T/4]

        output = self.diff_enc3(output, time_emb)  # [bs, mid_dim/2, T/4]
        output = self._apply_moe_stage("diff_enc3", output, routing_metadata, route_state, source_steps)
        h_diff.append(output)
        output = self.diff_downsample3(torch.cat([output, h_cond[2]], dim=1))  # [bs, mid_dim/2*2, T/8]

        output = self.diff_enc4(output, time_emb)  # [bs, mid_dim, T/8]
        output = self._apply_moe_stage("diff_enc4", output, routing_metadata, route_state, source_steps)
        h_diff.append(output)
        output = self.diff_downsample4(torch.cat([output, h_cond[3]], dim=1))  # [bs, mid_dim*2, T/16]

        # Middle blocks
        output = self.diff_mid_block1(output, time_emb)  # [bs, mid_dim, T/16]
        output = self._apply_moe_stage("diff_mid_block1", output, routing_metadata, route_state, source_steps)
        output = self.diff_mid_block2(output, time_emb)  # [bs, mid_dim, T/16]
        output = self._apply_moe_stage("diff_mid_block2", output, routing_metadata, route_state, source_steps)

        # UNet decoder
        output = self.diff_upsample4(output)  # [bs, mid_dim, T/8]
        output = self.diff_dec4(torch.cat([output, h_diff[-1]], dim=1), time_emb)  # [bs, mid_dim/2, T/8]
        output = self._apply_moe_stage("diff_dec4", output, routing_metadata, route_state, source_steps)

        output = self.diff_upsample3(output)  # [bs, mid_dim/2, T/4]
        output = self.diff_dec3(torch.cat([output, h_diff[-2]], dim=1), time_emb)  # [bs, mid_dim/4, T/4]
        output = self._apply_moe_stage("diff_dec3", output, routing_metadata, route_state, source_steps)

        output = self.diff_upsample2(output)  # [bs, mid_dim/4, T/2]
        output = self.diff_dec2(torch.cat([output, h_diff[-3]], dim=1), time_emb)  # [bs, mid_dim/8, T/2]
        output = self._apply_moe_stage("diff_dec2", output, routing_metadata, route_state, source_steps)

        output = self.diff_upsample1(output)  # [bs, mid_dim/8, T]
        output = self.diff_dec1(torch.cat([output, h_diff[-4]], dim=1), time_emb)  # [bs, 32, T]
        output = self._apply_moe_stage("diff_dec1", output, routing_metadata, route_state, source_steps)

        self._collect_moe_outputs()
        output = self.diff_final_conv(output)  # [bs, output_dim, T]
        # Convert from channel-first [bs, output_dim, T] to [bs, T, output_dim]
        output = einops.rearrange(output, 'b h t -> b t h')  # [bs, T, output_dim]

        return output


class UNetTCNWithControlNet(UNetTCN):
    """
    UNet TCN with ControlNet-style conditioning.
    
    This extends UNetTCN with additional control conditioning that can be
    injected at various stages of the UNet decoder.
    
    Config example:
        network:
            class_name: model.architectures.unet_tcn.UNetTCNWithControlNet
            input_dim: 12
            cond_dim: 66
            output_dim: 12
            control_cond_dim: 272
            mid_dim: 256
            time_dim: 32
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        output_dim: int,
        control_cond_dim: int,
        mid_dim: int = 256,
        time_dim: int = 32,
        input_t: bool = True,
        generative_model_type: Literal['diffusion', 'meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching'] = 'diffusion',
    ):
        super().__init__(
            input_dim=input_dim,
            cond_dim=cond_dim,
            output_dim=output_dim,
            mid_dim=mid_dim,
            time_dim=time_dim,
            input_t=input_t,
            generative_model_type=generative_model_type,
        )
        
        self.control_cond_dim = control_cond_dim

        # ControlNet encoder
        from model.heads import zero_module
        
        self.control_zero_conv_0 = zero_module(nn.Conv1d(control_cond_dim, input_dim, 1, padding=0))

        self.control_enc1 = ResidualTemporalBlock(input_dim, mid_dim // 8, input_t=input_t, t_embed_dim=time_dim)
        self.control_zero_conv_1 = zero_module(nn.Conv1d(mid_dim // 8, 32, 1, padding=0))
        self.control_downsample1 = Downsample1d(mid_dim // 8 * 2)

        self.control_enc2 = ResidualTemporalBlock(mid_dim // 8 * 2, mid_dim // 4, input_t=input_t, t_embed_dim=time_dim)
        self.control_zero_conv_2 = zero_module(nn.Conv1d(mid_dim // 8 * 2, mid_dim // 8, 1, padding=0))
        self.control_downsample2 = Downsample1d(mid_dim // 4 * 2)

        self.control_enc3 = ResidualTemporalBlock(mid_dim // 4 * 2, mid_dim // 2, input_t=input_t, t_embed_dim=time_dim)
        self.control_zero_conv_3 = zero_module(nn.Conv1d(mid_dim // 4 * 2, mid_dim // 4, 1, padding=0))
        self.control_downsample3 = Downsample1d(mid_dim // 2 * 2)

        self.control_enc4 = ResidualTemporalBlock(mid_dim // 2 * 2, mid_dim, input_t=input_t, t_embed_dim=time_dim)
        self.control_zero_conv_4 = zero_module(nn.Conv1d(mid_dim, mid_dim // 4 * 2, 1, padding=0))
        self.control_downsample4 = Downsample1d(mid_dim * 2)

        self.control_mid_block1 = ResidualTemporalBlock(mid_dim * 2, mid_dim, input_t=input_t, t_embed_dim=time_dim)
        self.control_mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, input_t=input_t, t_embed_dim=time_dim)
        self.control_zero_conv_mid = zero_module(nn.Conv1d(mid_dim, mid_dim, 1, padding=0))

    def encode_control(
        self,
        control_cond: torch.Tensor,
        h_cond: list,
        time_emb: Optional[torch.Tensor],
    ) -> tuple:
        """
        Encode control conditioning through the ControlNet encoder.
        
        Args:
            control_cond: [bs, control_cond_dim, T] control conditioning signal
            h_cond: List of conditioning features from the main conditioning encoder
            time_emb: [bs, time_dim] time embedding
            
        Returns:
            Tuple of control features to be added to the UNet decoder
        """
        control_cond = self.control_zero_conv_0(control_cond)  # [bs, input_dim, T]

        control_cond = self.control_enc1(control_cond, time_emb)  # [bs, mid_dim/8, T]
        control_zero_conv_1 = self.control_zero_conv_1(control_cond)
        control_cond = self.control_downsample1(torch.cat([control_cond, h_cond[0]], dim=1))  # [bs, mid_dim/8 * 2, T/2]

        control_cond = self.control_enc2(control_cond, time_emb)  # [bs, mid_dim/4, T/2]
        control_zero_conv_2 = self.control_zero_conv_2(control_cond)
        control_cond = self.control_downsample2(torch.cat([control_cond, h_cond[1]], dim=1))  # [bs, mid_dim/4*2, T/4]

        control_cond = self.control_enc3(control_cond, time_emb)  # [bs, mid_dim/2, T/4]
        control_zero_conv_3 = self.control_zero_conv_3(control_cond)
        control_cond = self.control_downsample3(torch.cat([control_cond, h_cond[2]], dim=1))  # [bs, mid_dim/2*2, T/8]

        control_cond = self.control_enc4(control_cond, time_emb)  # [bs, mid_dim, T/8]
        control_zero_conv_4 = self.control_zero_conv_4(control_cond)
        control_cond = self.control_downsample4(torch.cat([control_cond, h_cond[3]], dim=1))  # [bs, mid_dim*2, T/16]

        control_cond = self.control_mid_block1(control_cond, time_emb)  # [bs, mid_dim, T/16]
        control_cond = self.control_mid_block2(control_cond, time_emb)  # [bs, mid_dim, T/16]
        control_zero_conv_mid = self.control_zero_conv_mid(control_cond)

        return (
            control_zero_conv_1,
            control_zero_conv_2,
            control_zero_conv_3,
            control_zero_conv_4,
            control_zero_conv_mid,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        control_cond: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet TCN with ControlNet.
        
        Args:
            x: [bs, T, input_dim] main input sequence
            cond: [bs, T, cond_dim] conditioning sequence
            control_cond: [bs, T, control_cond_dim] control conditioning (optional)
            t: [bs] time step for generative models (optional)
            d: [bs] step size for meanflow/shortcut models (optional)
            
        Returns:
            output: [bs, T, output_dim] output sequence
        """
        # Compute time embedding if caller did not precompute it for masked conditioning.
        if time_emb is None:
            time_emb = self.compute_time_embedding(t, d) if t is not None else None

        # Rearrange to channel-first format for conv layers
        x = einops.rearrange(x, 'b t h -> b h t')  # [bs, input_dim, T]
        cond = einops.rearrange(cond, 'b t h -> b h t')  # [bs, cond_dim, T]

        # Encode conditioning
        h_cond = self.encode_condition(cond)

        # Encode control conditioning if provided
        if control_cond is not None:
            control_cond = einops.rearrange(control_cond, 'b t h -> b h t')  # [bs, control_cond_dim, T]
            control_features = self.encode_control(control_cond, h_cond, time_emb)
        else:
            control_features = None

        # UNet encoder with skip connections
        h_diff = []

        output = self.diff_enc1(x, time_emb)  # [bs, mid_dim/8, T]
        h_diff.append(output)
        output = self.diff_downsample1(torch.cat([output, h_cond[0]], dim=1))

        output = self.diff_enc2(output, time_emb)  # [bs, mid_dim/4, T/2]
        h_diff.append(output)
        output = self.diff_downsample2(torch.cat([output, h_cond[1]], dim=1))

        output = self.diff_enc3(output, time_emb)  # [bs, mid_dim/2, T/4]
        h_diff.append(output)
        output = self.diff_downsample3(torch.cat([output, h_cond[2]], dim=1))

        output = self.diff_enc4(output, time_emb)  # [bs, mid_dim, T/8]
        h_diff.append(output)
        output = self.diff_downsample4(torch.cat([output, h_cond[3]], dim=1))

        # Middle blocks
        output = self.diff_mid_block1(output, time_emb)
        output = self.diff_mid_block2(output, time_emb)
        
        if control_features is not None:
            output = output + control_features[4]  # control_zero_conv_mid

        # UNet decoder with control conditioning
        output = self.diff_upsample4(output)
        output = self.diff_dec4(torch.cat([output, h_diff[-1]], dim=1), time_emb)
        if control_features is not None:
            output = output + control_features[3]  # control_zero_conv_4

        output = self.diff_upsample3(output)
        output = self.diff_dec3(torch.cat([output, h_diff[-2]], dim=1), time_emb)
        if control_features is not None:
            output = output + control_features[2]  # control_zero_conv_3

        output = self.diff_upsample2(output)
        output = self.diff_dec2(torch.cat([output, h_diff[-3]], dim=1), time_emb)
        if control_features is not None:
            output = output + control_features[1]  # control_zero_conv_2

        output = self.diff_upsample1(output)
        output = self.diff_dec1(torch.cat([output, h_diff[-4]], dim=1), time_emb)
        if control_features is not None:
            output = output + control_features[0]  # control_zero_conv_1

        output = self.diff_final_conv(output)  # [bs, output_dim, T]
        output = einops.rearrange(output, 'b h t -> b t h')  # [bs, T, output_dim]

        return output

