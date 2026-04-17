# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Q-K Projection for TNT local memory retrieval (PyTorch).

Implements the paper-specified per-position projection (TNT Eq. 7)
via a causal-masked attention decomposition:

    proj_q_t = M_t . q_t
             = (carry + sum_{j<=t, j in chunk} k_j k_j^T) . q_t
             = carry @ q_t + sum_{j<=t, j in chunk} <k_j, q_t> . k_j

The key algebraic step factors the per-position ``D x D`` matrix
``M_t`` as

    M_t . q_t = carry @ q_t + K^T . (mask_t * (K . q_t))

where ``K`` is the chunk's key matrix and ``mask_t`` zeroes future
positions. This is computed for all ``t`` at once with two einsums and
a causal triangular mask -- no per-position ``D x D`` matrix is ever
materialised. Compute is ``O(B * C^2 * D)``, memory ``O(B * C^2)`` for
the masked scores (plus ``O(B * C * D)`` for the outputs), instead of
``O(B * C * D^2)`` for the naive materialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QKProjection(nn.Module):
    """Efficient per-position Q-K projection for TNT local memory.

    Args:
        dim: Embedding dimension ``D``. Keys and queries are expected to be
            L2-normalised along the last dim by the caller.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        carry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project queries causally and return the updated carry.

        For each position ``t`` in the chunk, the projected query is

            proj_q_t = carry @ q_t + sum_{j<=t} <k_j, q_t> * k_j

        which equals ``M_t @ q_t`` where
        ``M_t = carry + sum_{j<=t} k_j k_j^T`` -- without ever
        materialising the per-position ``D x D`` matrix.

        Args:
            queries: ``(B, C, D)`` queries for this chunk.
            keys: ``(B, C, D)`` L2-normalised keys for this chunk.
            carry: ``(D, D)`` accumulated ``sum k k^T`` from earlier chunks.

        Returns:
            Tuple of ``(projected_queries, new_carry)``.
            ``projected_queries`` has shape ``(B, C, D)``.
            ``new_carry`` has shape ``(D, D)`` and equals ``carry`` plus
            the batch-mean of ``sum_{j in chunk} k_j k_j^T``.
        """
        b, c, _ = queries.shape
        # scores[b, t, j] = <k_j, q_t>  -- shape (B, C, C).
        scores = torch.einsum("bjd,btd->btj", keys, queries)
        # Causal mask: zero out contributions from j > t.
        mask = torch.tril(
            torch.ones(c, c, device=queries.device, dtype=queries.dtype),
        )
        scores = scores * mask
        # sum_{j<=t} <k_j, q_t> * k_j, batched over (b, t) -> (B, C, D).
        chunk_contrib = torch.einsum("btj,bjd->btd", scores, keys)
        # carry @ q_t, batched over (b, t).
        carry_contrib = torch.einsum("de,bte->btd", carry, queries)
        projected = carry_contrib + chunk_contrib

        # Per-chunk carry update: sum_{j in chunk} k_j k_j^T, batch-mean reduced.
        new_carry = carry + torch.einsum("bcd,bce->de", keys, keys) / b
        return projected, new_carry

    def update_carry(
        self,
        keys: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy carry-only update kept for the chunk-mean fallback path.

        Used exclusively when the ``tnt_qk_projection == "chunk_mean"``
        config option is selected (to be added in a follow-up task) to
        reproduce pre-fix behaviour for loading old checkpoints.

        Args:
            keys: ``(B, C, D)`` L2-normalised keys.
            carry: ``(D, D)`` current carry.

        Returns:
            Updated carry with shape ``(D, D)``.
        """
        b, c, d = keys.shape
        k_flat = keys.reshape(b * c, d)
        return carry + (k_flat.T @ k_flat) / b
