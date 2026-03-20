# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Q-K Projection for TNT local memory retrieval.

Implements Section 4.1.2 of the TNT paper. Resolves the domain mismatch
between memory compression (key space) and retrieval (query space) by
projecting queries onto the subspace spanned by observed keys.

Math (TNT Eq. 7):
    o_t = f(V, q_t) + f(W_t, M_t · q_t)

    where M_t = Σ_{τ=ξ(t,C_L)}^{t} (k_τ k_τ^T / ||k_τ||²)

For L2-normalized keys (||k_τ|| = 1), this simplifies to:
    M_t = Σ_{τ=ξ(t,C_L)}^{t} k_τ k_τ^T

The projection matrix M_t is a running sum of outer products, reset at
shard boundaries.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class QKProjection(nn.Module):
    """Q-K Projection for TNT local memory retrieval.

    Projects query vectors onto the subspace spanned by observed keys,
    resolving the domain mismatch between compression (key space) and
    retrieval (query space).

    The projection matrix M_t = Σ k_τ k_τ^T is maintained as a running
    sum and reset at shard boundaries.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def compute_projection_matrix(
        self,
        keys: mx.array,
        carry_over: mx.array,
    ) -> mx.array:
        """Compute per-position projection matrices via prefix sum.

        Args:
            keys: L2-normalized key vectors (batch, chunk_len, dim)
            carry_over: Projection matrix carried from previous chunk (dim, dim)

        Returns:
            Per-position projection matrices (batch, chunk_len, dim, dim)
        """
        B, C, D = keys.shape

        # Outer products: k_τ k_τ^T for each position
        # (B, C, D, 1) @ (B, C, 1, D) → (B, C, D, D)
        outer_products = mx.expand_dims(keys, -1) @ mx.expand_dims(keys, -2)

        # Prefix sum along chunk dimension (cumulative sum of outer products)
        cumsum = mx.cumsum(outer_products, axis=1)  # (B, C, D, D)

        # Add carry-over state from previous chunk
        # carry_over: (D, D) → (1, 1, D, D) for broadcasting
        cumsum = cumsum + mx.reshape(carry_over, (1, 1, D, D))

        return cumsum  # (B, C, D, D)

    def project_queries(
        self,
        queries: mx.array,
        projection_matrices: mx.array,
    ) -> mx.array:
        """Project queries onto key subspace.

        Args:
            queries: Query vectors (batch, chunk_len, dim)
            projection_matrices: Per-position projections (batch, chunk_len, dim, dim)

        Returns:
            Projected queries (batch, chunk_len, dim)
        """
        # (B, C, D, D) @ (B, C, D, 1) → (B, C, D, 1) → (B, C, D)
        projected = projection_matrices @ mx.expand_dims(queries, -1)
        return projected.squeeze(-1)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        carry_over: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute Q-K projected queries and update carry-over state.

        Args:
            queries: Query vectors (batch, chunk_len, dim)
            keys: L2-normalized key vectors (batch, chunk_len, dim)
            carry_over: Projection matrix from previous chunk (dim, dim)

        Returns:
            Tuple of (projected_queries, new_carry_over)
        """
        projections = self.compute_projection_matrix(keys, carry_over)
        projected_queries = self.project_queries(queries, projections)

        # New carry-over = final projection matrix in this chunk
        # projections[:, -1] is (B, D, D) — average over batch since
        # weights are shared
        new_carry_over = mx.mean(projections[:, -1], axis=0)  # (D, D)

        return projected_queries, new_carry_over


def update_projection_state(
    carry_over: mx.array,
    chunk_keys: mx.array,
    reset: bool,
) -> tuple[mx.array, mx.array]:
    """Update projection state, handling periodic resets.

    Convenience function for managing projection state across chunks,
    including shard-boundary resets.

    Args:
        carry_over: Current carry-over projection matrix (dim, dim)
        chunk_keys: L2-normalized keys for this chunk (batch, chunk_len, dim)
        reset: Whether this chunk starts a new shard (zeros the carry-over)

    Returns:
        Tuple of (new_carry_over, per_position_projections)
    """
    if reset:
        carry_over = mx.zeros_like(carry_over)

    D = carry_over.shape[0]
    proj = QKProjection(D)
    projections = proj.compute_projection_matrix(chunk_keys, carry_over)

    # New carry-over = final projection matrix in this chunk
    new_carry_over = mx.mean(projections[:, -1], axis=0)  # (D, D)

    return new_carry_over, projections
