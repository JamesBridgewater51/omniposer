# Copyright (c) Meta Platforms, Inc. and affiliates.
# Generic (task-agnostic) DiT architecture

import torch
import torch.nn as nn
from typing import Optional, Literal

from model.heads import (
    CrossAttentionBlock,
    PositionalEncoding,
    TimestepEmbedder,
    InputProcess,
    OutputProcess,
)


class GenericDiT(nn.Module):
    """
    Task-agnostic DiT (Diffusion Transformer) architecture.
    
    This module operates in a black-box fashion:
    - Generative mode: maps (x_t, t, cond) -> output through cascaded transformer blocks
    - Discriminative mode: maps cond -> output through transformer blocks (no timestep)
    
    It does NOT have any knowledge of:
    - Input structure (trajectory vs pose features)
    - Physical meaning of features
    - Task-specific conditioning (masked conditioner, traj_cond_encoder, etc.)
    
    Config-based construction via class_name pattern:
        class_name: model.architectures.generic_dit.GenericDiT
        input_dim: 294
        output_dim: 294
        cond_dim: 330
        latent_dim: 256
        ...
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_rope: bool = True,
        use_film: bool = False,
        abstract_model_type: Literal['generative', 'discriminative'] = 'generative',
        generative_model_type: Literal['diffusion', 'meanflow', 'shortcut', 'consistency', 'improved_consistency', 'flow_matching'] = 'diffusion',
        nfeats: int = 1,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_rope = use_rope
        self.use_film = use_film
        self.abstract_model_type = abstract_model_type
        self.generative_model_type = generative_model_type
        self.nfeats = nfeats
        self.device = device
        
        # Input/Output Processing
        self.input_encoder = InputProcess(input_dim * nfeats, latent_dim)
        self.cond_encoder = nn.Linear(cond_dim, latent_dim)
        self.output_process = OutputProcess(output_dim, latent_dim, nfeats)
        
        # Positional Encoding
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        # Transformer Encoder Layers
        self.seqTransEncoder = nn.ModuleList([
            CrossAttentionBlock(
                latent_dim, 
                num_heads, 
                dropout, 
                activation, 
                use_rope=use_rope, 
                use_film=use_film, 
                d_film_emb=latent_dim if use_film else None
            )
            for _ in range(num_layers)
        ])
        
        # Time Embedder (for generative models)
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
    
    def compute_time_embedding(self, t: torch.Tensor, d: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
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
            emb = self.embed_timestep(t * 1000).squeeze(0)
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
        Forward pass for generic DiT.

        Args:
            x_t: Noisy input [bs, T, input_dim] for generative, None for discriminative.
            t: Timestep [bs] for generative models.
            d: Step size [bs] for meanflow/shortcut models.
            cond: Conditioning tensor [bs, T, cond_dim].

        Returns:
            Output tensor [bs, T, output_dim].
        """
        if cond.dim() != 3:
            raise ValueError(f"GenericDiT expects cond [bs, T, dim], got {cond.shape}")

        bs, T, _ = cond.shape
        cond_seq = cond  # [bs, T, cond_dim]

        # Encode conditioning to latent space using linear layer
        cond_kv = self.cond_encoder(cond_seq)  # [bs, T, latent_dim]
        # Note: CrossAttentionBlock expects [b, t, d] format, which cond_kv already is
        
        if self.abstract_model_type == 'generative':
            if x_t.dim() != 3:
                raise ValueError(f"GenericDiT expects x_t [bs, T, dim], got {x_t.shape}")

            emb = time_emb if time_emb is not None else self.compute_time_embedding(t, d)

            # Process main input: x_t [bs, T, input_dim] -> [T, bs, latent_dim]
            x_seq = self.input_encoder(x_t)  # [T, bs, latent_dim]

            # Add time embedding to sequence
            emb_seq = emb.unsqueeze(0).expand(T, -1, -1)  # [T, bs, latent_dim]
            x_seq = x_seq + emb_seq

            # Add positional encoding
            x_seq = self.sequence_pos_encoder(x_seq)  # [T, bs, latent_dim]

            # Convert to [bs, T, latent_dim] for CrossAttentionBlock
            x_seq = x_seq.permute(1, 0, 2)  # [bs, T, latent_dim]

            # Prepare film embedding
            film_emb = emb if self.use_film else None

            # Transformer encoder with cross-attention to conditioning
            for layer in self.seqTransEncoder:
                x_seq = layer(x_seq, cond_kv, film_cond=film_emb)

            # Convert back to [T, bs, latent_dim] for OutputProcess
            x_seq = x_seq.permute(1, 0, 2)  # [T, bs, latent_dim]

            # Output processing
            output = self.output_process(x_seq)  # [bs, T, output_dim]

        else:  # discriminative
            # Process main input (which is cond for discriminative)
            x_seq = self.input_encoder(cond_seq)  # [T, bs, latent_dim]

            # Add positional encoding
            x_seq = self.sequence_pos_encoder(x_seq)  # [T, bs, latent_dim]

            # Convert to [bs, T, latent_dim] for CrossAttentionBlock
            x_seq = x_seq.permute(1, 0, 2)  # [bs, T, latent_dim]

            # Transformer encoder with self-attention (no conditioning needed)
            for layer in self.seqTransEncoder:
                x_seq = layer(x_seq, cond_kv, None)

            # Convert back to [T, bs, latent_dim] for OutputProcess
            x_seq = x_seq.permute(1, 0, 2)  # [T, bs, latent_dim]

            # Output processing
            output = self.output_process(x_seq)  # [bs, T, output_dim]

        return output
    
    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

