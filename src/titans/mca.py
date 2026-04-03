# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory Cross-Attention for Titans (PyTorch Implementation).

Cross-attention from token representations to NeuralLongTermMemory's
weight matrix rows. Gated output — gate initialized near-zero so MCA
has no effect until the gate learns to open.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig
from titans.models import RMSNorm


class MemoryCrossAttention(nn.Module):
    """Cross-attention from token representations to NeuralLTM weight rows.

    Q from the residual stream, K/V from the memory MLP's first weight matrix.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        dim = config.dim
        self.dim = dim
        self.num_heads = config.mca_num_heads
        self.head_dim = dim // self.num_heads

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

        gate_out = 1 if config.mca_gate_type == "scalar" else dim
        self.Wg = nn.Linear(dim, gate_out, bias=True)
        nn.init.constant_(self.Wg.bias, config.mca_gate_bias_init)

        self.norm = RMSNorm(dim)

    def forward(
        self, x: torch.Tensor, memory_weights: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend from x to memory weight rows.

        Args:
            x: Token representations [B, T, dim]
            memory_weights: First weight matrix from NeuralLTM
                [num_rows, dim] — detached, no gradient flow to memory

        Returns:
            Gated output [B, T, dim]
        """
        B, T, dim = x.shape
        num_rows = memory_weights.shape[0]

        Q = self.Wq(self.norm(x))  # [B, T, dim]
        K = self.Wk(memory_weights)  # [num_rows, dim]
        V = self.Wv(memory_weights)  # [num_rows, dim]

        # Multi-head reshape
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(1, num_rows, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(1, num_rows, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        K = K.expand(B, -1, -1, -1)
        V = V.expand(B, -1, -1, -1)

        # Scaled dot-product attention (no causal mask — all rows visible)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, scale=self.head_dim**-0.5
        )

        # Reshape and project
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, dim)
        attn_out = self.Wo(attn_out)

        # Gated output
        gate = torch.sigmoid(self.Wg(x))
        return gate * attn_out
