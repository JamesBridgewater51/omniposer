# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
import torch.nn.functional as F
import math
import numpy as np
from typing import Any, Dict, Optional
from utils.other_utils import REPR_DIM_DICT, JOINT_REPR_DIM_DICT
from model.metadata_moe import MetadataRoutedFFN, is_moe_enabled, validate_moe_config

##################################################################
######################### for TrajNet ############################
##################################################################
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class FrequencyAdapter(nn.Module):
    def __init__(self, channels, num_bands=4, method='band_gating'):
        super().__init__()
        self.method = method
        self.channels = channels
        self.num_bands = num_bands
        
        if self.method == 'band_gating':
            # Learnable scaling factors for each band per channel
            # Initialized to 0.0 so it starts as identity (since we add residual)
            # But if we multiply, we should init to 1.0?
            # The logic below is x + irfft(fft(x) * scale). 
            # If scale is 0, result is x + 0 = x. Identity.
            # If scale is learnable, it adds frequency components.
            self.band_gates = nn.Parameter(torch.zeros(num_bands, channels, 1, dtype=torch.float32)) # [Bands, C, 1]
        elif self.method == 'freq_mix':
             # Simple channel mixing in frequency domain
             self.mix_layer = nn.Linear(channels, channels)

    def forward(self, x):
        # x: [bs, channels, seq_len]
        if self.method == 'band_gating':
            # x is real, so rfft
            x_fft = torch.fft.rfft(x, dim=-1) # [bs, C, F]
            F_bin = x_fft.shape[-1]
            
            # Split F into num_bands
            band_width = F_bin // self.num_bands
            
            scales = []
            for i in range(self.num_bands):
                # Last band might be larger due to integer division
                w = band_width if i < self.num_bands - 1 else F_bin - i * band_width
                if w > 0:
                   scales.append(self.band_gates[i].expand(-1, w)) # [C, w]

            if len(scales) > 0:
                scale_tensor = torch.cat(scales, dim=-1).to(x.device) # [C, F]
                scale_tensor = scale_tensor.unsqueeze(0) # [1, C, F]
                
                x_filtered = x_fft * scale_tensor
                x_out = torch.fft.irfft(x_filtered, n=x.shape[-1], dim=-1)
                return x + x_out 
            return x
            
        elif self.method == 'freq_mix':
            # x: [bs, C, T]
            x_fft = torch.fft.rfft(x, dim=-1) # [bs, C, F]
            # Transpose to [bs, F, C] for Linear
            x_fft_t = x_fft.permute(0, 2, 1)
            
            # Process Real and Imag separately or as complex?
            # Linear layer doesn't support complex inputs directly in older pytorch, 
            # but supports it in newer. Let's treat as real concatenation.
            real = x_fft_t.real
            imag = x_fft_t.imag
            
            # Apply same mixing to both or different? Shared is phase-preserving-ish.
            real = self.mix_layer(real)
            imag = self.mix_layer(imag)
            
            x_fft_out = torch.complex(real, imag)
            x_out = torch.fft.irfft(x_fft_out.permute(0, 2, 1), n=x.shape[-1], dim=-1)
            return x + x_out
            
        return x

class ResidualTemporalBlock(nn.Module):
    """
    This class is copied from https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
    """
    def __init__(self, inp_channels=4, out_channels=64, input_t=False, t_embed_dim=32, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.input_t = input_t
        if input_t:
            self.time_mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(t_embed_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if self.input_t:
            out = out + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)



class SinusoidalPosEmb(nn.Module):
    # forward: x: [bs], returns: [bs, dim]
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -exponent)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x):
        inv_freq = self.inv_freq
        if inv_freq.device != x.device:
            inv_freq = inv_freq.to(device=x.device)
        emb = x.float()[:, None] * inv_freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class MaskedConditioningEncoder(nn.Module):
    def __init__(self, 
             # Essential structure info (New Design)
             cond_dim: int,
             per_joint_dim: int = None,
             num_joints: int = 22,
             global_dim: int = 0,
             prefix_dim: int = 0,
             input_t: bool = True,
             
             # Processing
             hidden_dim: int = 256,
             num_blocks: int = 2,
             use_time: bool = True,
             time_dim: int = 128,
             mid_dim: int = 256,
             dropout: float = 0.1,
             
             # Legacy arguments (for backward compatibility)
             **kwargs):
        super().__init__()

        # --- Standard Initialization ---
        if per_joint_dim is None:
            raise ValueError("per_joint_dim must be provided (or inferred from legacy args)")
            
        self.per_joint_dim = per_joint_dim
        self.num_joints = num_joints
        self.global_dim = global_dim
        self.prefix_dim = prefix_dim
        self.use_time = use_time
        
        # Total input dimension for the temporal blocks
        self.total_dim = prefix_dim + num_joints * per_joint_dim + global_dim
        assert self.total_dim == cond_dim
        
        # 1. Learnable embeddings for MASKED features (MAE-style)
        # Joint embeddings: one per joint
        self.joint_embeddings = nn.Parameter(torch.randn(num_joints, per_joint_dim))
        # Prefix embedding: single vector for masked prefix features
        if prefix_dim > 0:
            self.prefix_embedding = nn.Parameter(torch.randn(prefix_dim))
        # Global embedding: single vector for masked global features
        if global_dim > 0:
            self.global_embedding = nn.Parameter(torch.randn(global_dim))
        
        # 2. Input Projection
        self.input_proj = Conv1dBlock(self.total_dim, hidden_dim, kernel_size=3)
        
        # 3. Temporal Blocks
        self.temporal_blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, hidden_dim, input_t=use_time, t_embed_dim=time_dim) 
             for _ in range(num_blocks)]
        )
        
        # 4. Output Projection
        self.output_proj = nn.Conv1d(hidden_dim, self.total_dim, 1)

    def forward(self,
                cond: torch.Tensor,                    # [bs, T, total_dim]
                cond_mask: torch.Tensor,               # [bs, T, total_dim] feature-level mask
                time_emb: Optional[torch.Tensor]=None  # [bs, time_dim]
                ) -> torch.Tensor:
        """
        Apply MAE-style masking to conditioning and process through temporal blocks.
        
        Args:
            cond: Conditioning tensor [bs, T, prefix_dim + num_joints*per_joint_dim + global_dim]
            cond_mask: Feature-level visibility mask [bs, T, total_dim]. True=Visible, False=Masked.
            time_emb: Time embedding [bs, time_dim]
            
        Returns:
            Processed conditioning of same shape as input
        """
        bs, T, _ = cond.shape
        
        if cond_mask is None:
            raise ValueError('cond_mask must be provided for MaskedConditioningEncoder')
        
        # Slice cond_mask into three parts
        joint_start = self.prefix_dim
        joint_end = self.prefix_dim + self.num_joints * self.per_joint_dim
        
        # 1. Process Prefix Features
        if self.prefix_dim > 0:
            prefix_feat = cond[..., :self.prefix_dim]  # [bs, T, prefix_dim]
            prefix_mask = cond_mask[..., :self.prefix_dim] > 0.5  # [bs, T, prefix_dim]
            prefix_emb = self.prefix_embedding.view(1, 1, -1).expand(bs, T, -1)
            prefix_feat_masked = torch.where(prefix_mask, prefix_feat, prefix_emb)
        else:
            prefix_feat_masked = None
            
        # 2. Process Joint Features
        joint_feat_flat = cond[..., joint_start:joint_end]  # [bs, T, num_joints*per_joint_dim]
        joint_mask_flat = cond_mask[..., joint_start:joint_end]  # [bs, T, num_joints*per_joint_dim]
        
        # Reshape to [bs, T, num_joints, per_joint_dim]
        joint_feat = joint_feat_flat.view(bs, T, self.num_joints, self.per_joint_dim)
        joint_mask = (joint_mask_flat.view(bs, T, self.num_joints, self.per_joint_dim) > 0.5)
        
        # Expand embeddings and apply
        joint_emb = self.joint_embeddings.view(1, 1, self.num_joints, self.per_joint_dim).expand(bs, T, -1, -1)
        joint_feat_masked = torch.where(joint_mask, joint_feat, joint_emb)
        joint_feat_masked_flat = joint_feat_masked.view(bs, T, -1)
        
        # 3. Process Global Features
        if self.global_dim > 0:
            global_feat = cond[..., joint_end:]  # [bs, T, global_dim]
            global_mask = cond_mask[..., joint_end:] > 0.5  # [bs, T, global_dim]
            global_emb = self.global_embedding.view(1, 1, -1).expand(bs, T, -1)
            global_feat_masked = torch.where(global_mask, global_feat, global_emb)
        else:
            global_feat_masked = None
            
        # 4. Reconstruct full feature vector
        parts = []
        if prefix_feat_masked is not None:
            parts.append(prefix_feat_masked)
        parts.append(joint_feat_masked_flat)
        if global_feat_masked is not None:
            parts.append(global_feat_masked)
            
        x_in = torch.cat(parts, dim=-1)  # [bs, T, total_dim]
        
        # 5. Temporal Processing
        # Permute to [bs, dim, T] for Conv1d
        x = x_in.permute(0, 2, 1)
        
        x = self.input_proj(x)
        for block in self.temporal_blocks:
            x = block(x, time_emb)
        x = self.output_proj(x)
        
        # Return [bs, T, dim]
        return x.permute(0, 2, 1)

##################################################################
######################### for PoseNet ############################
##################################################################

class CrossAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dropout=0.1, 
                 activation="gelu",
                 use_rope=False,
                 use_film=False,
                 d_film_emb=None,
                 routed_ffn_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        if use_film:
            assert d_film_emb is not None, "d_film_emb must be provided if use_film is True"

        self.nhead = nhead
        self.d_head = d_model // nhead
        self.use_rope = use_rope
        self.use_film = use_film
        self.routed_ffn_config = validate_moe_config(
            routed_ffn_config or {"enabled": False},
            component="pose_ffn_moe",
            allowed_routing_granularities={"token", "frame", "segment"},
        )
        self.routed_ffn_enabled = is_moe_enabled(self.routed_ffn_config)
        self.last_ffn_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_ffn_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_ffn_route_state = None

        # Attention Layers
        self.sa_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.sa_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ca_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.ca_kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.ca_out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Feed-Forward Block
        d_ff = d_model * 4
        self.ff_linear1 = nn.Linear(d_model, d_ff)
        self.ff_activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_linear2 = nn.Linear(d_ff, d_model)
        self.ff_dropout2 = nn.Dropout(dropout)
        if self.routed_ffn_enabled:
            self.routed_ffn = MetadataRoutedFFN(d_model, self.routed_ffn_config, component="pose_ffn_moe")
        else:
            self.routed_ffn = None

        # Normalization and Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_res = nn.Dropout(dropout)

        # --- MODIFICATION: Use the provided RotaryEmbedding class ---
        if self.use_rope:
            # Instantiate the new, more powerful RotaryEmbedding class
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)
        
        if self.use_film:
            self.film_norm = nn.LayerNorm(d_ff, elementwise_affine=False, bias=False)
            self.film_proj = zero_module(nn.Linear(d_film_emb, d_ff * 2))

    def _self_attention(self, x, attn_mask=None):
        q, k, v = self.sa_qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.nhead)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.nhead)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.nhead)
        
        # --- MODIFICATION: Apply RoPE using the provided class method ---
        if self.use_rope:
            # The API is a direct replacement as the default seq_dim=-2 matches our layout
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
            
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.ff_dropout1.p if self.training else 0.0)
        out = rearrange(out, 'b h s d -> b s (h d)')
        return self.sa_out_proj(out)

    def _cross_attention(self, x, cond, key_padding_mask=None):
        q = self.ca_q_proj(x)
        k, v = self.ca_kv_proj(cond).chunk(2, dim=-1)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.nhead)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.nhead)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.nhead)
        
        # --- MODIFICATION: Apply RoPE using the provided class method ---
        if self.use_rope:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.ff_dropout1.p if self.training else 0.0)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.ca_out_proj(out)

    def _feed_forward(self, x, film_cond=None, routing_metadata=None, route_state=None):
        if self.routed_ffn is not None:
            h = self.routed_ffn(x, routing_metadata=routing_metadata, route_state=route_state)
            self.last_ffn_aux_losses = self.routed_ffn.last_aux_losses
            self.last_ffn_diagnostics = self.routed_ffn.last_diagnostics
            self.last_ffn_route_state = self.routed_ffn.last_route_state
            return h
        self.last_ffn_aux_losses = {}
        self.last_ffn_diagnostics = {}
        self.last_ffn_route_state = None
        # film_cond: [bs, d_film_emb]
        h = self.ff_linear1(x)
        h = self.ff_activation(h)
        
        if self.use_film:
            if film_cond is None:
                raise ValueError("film_cond must be provided when use_film is True")
            scale, shift = self.film_proj(film_cond).chunk(2, dim=-1)
            h = self.film_norm(h)
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            
        h = self.ff_dropout1(h)
        h = self.ff_linear2(h)
        h = self.ff_dropout2(h)
        return h

    def forward(self, x, cond, x_mask=None, cond_mask=None, film_cond=None, routing_metadata=None, route_state=None):
        x_res = self._self_attention(self.norm1(x), attn_mask=x_mask)
        x = x + self.dropout_res(x_res)
        
        x_res = self._cross_attention(self.norm2(x), cond, key_padding_mask=cond_mask)
        x = x + self.dropout_res(x_res)
        
        x_res = self._feed_forward(self.norm3(x), film_cond=film_cond, routing_metadata=routing_metadata, route_state=route_state)
        x = x + x_res
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]: 0, 1, 2, ..., 4999
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # torch.arange(0, d_model, 2): [256]: 0, 2, 4, 6, 8, ..., 510  div_term: [256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # [5000, 1, 512]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        time_embed_dim = self.latent_dim * 4
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.latent_dim),
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, self.latent_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(timesteps).unsqueeze(0)


class InputProcess(nn.Module):
    """Project a [bs, T, dim] sequence into latent tokens [T, bs, latent_dim] for transformer backbones."""
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"InputProcess expects [bs, T, dim], got {x.shape}")
        x = x.permute(1, 0, 2)  # [bs, T, dim] -> [T, bs, dim]
        x = self.poseEmbedding(x)  # [T, bs, latent_dim]
        return x


class OutputProcess(nn.Module):
    """Project latent tokens [T, bs, latent_dim] back to a [bs, T, output_feats] sequence."""
    def __init__(self, output_feats, latent_dim, nfeats=1):
        super().__init__()
        self.output_feats = output_feats
        self.latent_dim = latent_dim
        self.nfeats = nfeats  # Kept for backward compatibility but not used in 3D output
        self.poseFinal = nn.Linear(self.latent_dim, self.output_feats)

    def forward(self, output):
        if output.dim() != 3:
            raise ValueError(f"OutputProcess expects [T, bs, latent_dim], got {output.shape}")
        output = self.poseFinal(output)  # [T, bs, output_feats]
        output = output.permute(1, 0, 2)  # [bs, T, output_feats]
        return output
