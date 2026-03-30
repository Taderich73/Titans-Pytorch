# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory Cross-Attention for Titans (MLX Implementation).

Provides a cross-attention mechanism that reads from NeuralLongTermMemory's
weight matrix rows. This gives the model a second read interface into the
same memory that's already being written to by the surprise-driven update
mechanism.

MLP retrieval (existing): nonlinear function of query — precise key-value lookup.
Cross-attention (this module): linear blend of memory directions — soft discovery
of which associations are relevant to the current context.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


class MemoryCrossAttention(nn.Module):
    """Cross-attention from token representations to NeuralLTM weight rows.

    Q from the residual stream, K/V from the memory MLP's first weight matrix.
    Gated output — gate initialized near-zero so MCA has no effect until
    the gate learns to open.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        dim = config.dim
        self.num_heads = config.mca_num_heads
        self.head_dim = dim // self.num_heads

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

        gate_out = 1 if config.mca_gate_type == "scalar" else dim
        self.Wg = nn.Linear(dim, gate_out, bias=True)
        self.Wg.bias = mx.full(self.Wg.bias.shape, config.mca_gate_bias_init)

        self.norm = nn.RMSNorm(dim)

    def __call__(self, x: mx.array, memory_weights: mx.array) -> mx.array:
        """Cross-attend from x to memory weight rows.

        Args:
            x: Token representations [B, T, dim]
            memory_weights: First weight matrix from NeuralLTM
                [num_rows, dim] where num_rows is dim (linear) or
                memory_hidden_dim (deep)

        Returns:
            Gated output [B, T, dim] — net contribution for residual add
        """
        B, T, dim = x.shape
        num_rows = memory_weights.shape[0]

        Q = self.Wq(self.norm(x))  # [B, T, dim]
        K = self.Wk(memory_weights)  # [num_rows, dim]
        V = self.Wv(memory_weights)  # [num_rows, dim]

        # Reshape for multi-head attention
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, num_rows, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, num_rows, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Broadcast K, V across batch
        K = mx.broadcast_to(K, (B, self.num_heads, num_rows, self.head_dim))
        V = mx.broadcast_to(V, (B, self.num_heads, num_rows, self.head_dim))

        # Scaled dot-product attention (no causal mask — all rows visible)
        scale = self.head_dim ** -0.5
        attn_scores = Q @ K.transpose(0, 1, 3, 2) * scale
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ V  # [B, heads, T, head_dim]

        # Reshape and project
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, dim)
        attn_out = self.Wo(attn_out)

        # Gated output
        gate = mx.sigmoid(self.Wg(x))
        return gate * attn_out
