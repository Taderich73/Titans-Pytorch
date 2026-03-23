"""
Attention Residuals for TNT (MLX Implementation).

Implements Block Attention Residuals from "Attention Residuals"
(Kimi Team, arXiv 2603.15031).

BlockAttnRes replaces fixed residual connections between layers with
learned softmax attention over prior block representations. Each layer
gets a pseudo-query vector w_l that determines how to weight earlier
block outputs.

AttnResMemoryGate extracts an importance signal from the AttnRes
attention weights to modulate the memory update learning rate.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class BlockAttnRes(nn.Module):
    """Block Attention Residuals (Eq. 2-6 from AttnRes paper).

    Computes attention-weighted input from prior block representations
    and the current intra-block partial sum. Each layer owns one instance.

    The pseudo-query w_l is the weight vector of a Linear(dim, 1) projection.
    Initialized to zero so initial attention weights are uniform across
    sources, matching standard residual behavior at the start of training.

    Args:
        dim: Model dimension
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.attn_res_norm = nn.RMSNorm(dim)
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        # Zero-init pseudo-query for uniform initial weights
        self.attn_res_proj.weight = mx.zeros_like(self.attn_res_proj.weight)

    def __call__(
        self,
        blocks: list[mx.array],
        partial_block: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Compute AttnRes input for this layer.

        Args:
            blocks: Completed block representations [b_0, ..., b_{n-1}],
                each shape (batch, seq, dim).
            partial_block: Current intra-block partial sum (batch, seq, dim),
                or None if first layer in a new block.

        Returns:
            Tuple of (h_l, attn_weights):
                h_l: Attention-weighted input (batch, seq, dim)
                attn_weights: Attention distribution (batch, seq, num_sources)
        """
        # Collect all sources
        sources = list(blocks)
        if partial_block is not None:
            sources.append(partial_block)

        # Single source: skip attention, weight = 1.0
        if len(sources) == 1:
            v = sources[0]
            ones = mx.ones((*v.shape[:2], 1))
            return v, ones

        # Stack sources: (num_sources, batch, seq, dim)
        V = mx.stack(sources, axis=0)

        # Keys: RMSNorm prevents large-magnitude layers from dominating
        K = self.attn_res_norm(V)

        # Pseudo-query logits: w_l^T . k_i for each source
        # attn_res_proj: (dim, 1), K: (N, B, T, D) -> logits: (N, B, T, 1)
        logits = self.attn_res_proj(K)
        logits = logits.squeeze(-1)  # (N, B, T)

        # Clamp logits to prevent exp() overflow in softmax
        logits = mx.clip(logits, -30.0, 30.0)

        # Softmax over sources dimension (axis=0)
        attn_weights = mx.softmax(logits, axis=0)  # (N, B, T)

        # Weighted sum: h = sum alpha_i * V_i
        # attn_weights: (N, B, T), V: (N, B, T, D)
        h = mx.sum(attn_weights[..., None] * V, axis=0)  # (B, T, D)

        # Transpose weights to (B, T, N) for downstream use
        attn_weights = mx.transpose(attn_weights, (1, 2, 0))  # (B, T, N)

        return h, attn_weights


class AttnResMemoryGate:
    """Extracts importance signal from AttnRes attention weights.

    Takes the attention weight assigned to the most recent source
    (the last element in the sources list, typically the intra-block
    partial sum) as the importance signal. Returns a scalar multiplier
    for the memory learning rate.

    Scalar averaging matches the existing pattern in NeuralLongTermMemory
    where theta, alpha, eta are batch-averaged because memory weights
    are shared across the batch.
    """

    def __call__(self, attn_weights: mx.array) -> mx.array:
        """Extract importance from AttnRes attention distribution.

        Args:
            attn_weights: Shape (batch, seq, num_sources) from BlockAttnRes.

        Returns:
            Scalar importance weight (mean over batch and sequence).
        """
        importance = attn_weights[:, :, -1]  # (B, T)
        return mx.mean(importance)
