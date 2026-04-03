# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Attention Residuals for Titans (PyTorch Implementation).

Block Attention Residuals (AttnRes paper, arXiv 2603.15031) replace fixed
residual connections with learned softmax attention over prior block
representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from titans.models import RMSNorm


class BlockAttnRes(nn.Module):
    """Block Attention Residuals (AttnRes paper Eq. 2-6).

    Per-layer pseudo-query w_l (Linear(dim, 1)) computes softmax attention
    over completed block representations. Zero-initialized for uniform
    initial weights (matching standard residual behavior at training start).
    """

    def __init__(self, dim: int, logit_clip: float = 30.0) -> None:
        super().__init__()
        self.dim = dim
        self.logit_clip = logit_clip
        self.attn_res_norm = RMSNorm(dim)
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.attn_res_proj.weight)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AttnRes input for this layer.

        Args:
            blocks: Completed block representations [b_0, ..., b_{n-1}],
                each shape (batch, seq, dim).
            partial_block: Current intra-block partial sum (batch, seq, dim),
                or None if at block boundary.

        Returns:
            (h, attn_weights):
                h: Attention-weighted input (batch, seq, dim)
                attn_weights: Distribution over sources (batch, seq, num_sources)
        """
        sources = list(blocks)
        if partial_block is not None:
            sources.append(partial_block)

        if not sources:
            raise ValueError("BlockAttnRes requires at least one source")

        # Single source: skip attention
        if len(sources) == 1:
            v = sources[0]
            ones = torch.ones(*v.shape[:2], 1, device=v.device, dtype=v.dtype)
            return v, ones

        # Stack: (num_sources, batch, seq, dim)
        V = torch.stack(sources, dim=0)

        # Keys: RMSNorm prevents large-magnitude layers from dominating
        K = self.attn_res_norm(V)

        # Pseudo-query logits: (N, B, T, D) -> (N, B, T, 1) -> (N, B, T)
        logits = self.attn_res_proj(K).squeeze(-1)
        logits = torch.clamp(logits, -self.logit_clip, self.logit_clip)

        # Softmax over sources (dim=0)
        attn_weights = torch.softmax(logits, dim=0)  # (N, B, T)

        # Weighted sum
        h = torch.sum(attn_weights.unsqueeze(-1) * V, dim=0)  # (B, T, D)

        # Transpose to (B, T, N)
        attn_weights = attn_weights.permute(1, 2, 0)

        return h, attn_weights


class AttnResMemoryGate:
    """Extracts importance signal from AttnRes attention weights.

    Takes the weight assigned to the most recent source as the memory
    learning rate modulator.
    """

    def __call__(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Extract importance from AttnRes attention distribution.

        Args:
            attn_weights: Shape (batch, seq, num_sources) from BlockAttnRes.

        Returns:
            Scalar importance weight (mean over batch and sequence).
        """
        importance = attn_weights[:, :, -1]  # (B, T)
        return torch.mean(importance)
