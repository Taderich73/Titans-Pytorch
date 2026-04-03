# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Q-K Projection for TNT local memory retrieval (PyTorch).

Resolves domain mismatch between memory compression (key space) and
retrieval (query space) by projecting queries onto the subspace spanned
by observed keys (TNT Eq. 7):

    M_t = Σ k_τ k_τ^T (for L2-normalized keys)
    projected_q = M_t · q_t
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QKProjection(nn.Module):
    """Q-K Projection for TNT local memory retrieval."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def compute_projection_matrix(
        self, keys: torch.Tensor, carry_over: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position projection matrices via prefix sum.

        Args:
            keys: L2-normalized (batch, chunk_len, dim)
            carry_over: (dim, dim)

        Returns:
            (batch, chunk_len, dim, dim)
        """
        outer_products = keys.unsqueeze(-1) @ keys.unsqueeze(-2)
        cumsum = torch.cumsum(outer_products, dim=1)
        cumsum = cumsum + carry_over.reshape(1, 1, self.dim, self.dim)
        return cumsum

    def project_queries(
        self, queries: torch.Tensor, projection_matrices: torch.Tensor,
    ) -> torch.Tensor:
        projected = projection_matrices @ queries.unsqueeze(-1)
        return projected.squeeze(-1)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, carry_over: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projections = self.compute_projection_matrix(keys, carry_over)
        projected_queries = self.project_queries(queries, projections)
        new_carry_over = torch.mean(projections[:, -1], dim=0)
        return projected_queries, new_carry_over

    def update_carry(
        self, keys: torch.Tensor, carry_over: torch.Tensor,
    ) -> torch.Tensor:
        """Efficient carry-over update: O(B*C*D + D*D)."""
        B, C, D = keys.shape
        k_flat = keys.reshape(B * C, D)
        new_carry = carry_over + (k_flat.T @ k_flat) / B
        return new_carry
