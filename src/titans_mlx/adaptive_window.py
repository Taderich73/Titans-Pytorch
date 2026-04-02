# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Adaptive window sizing for sliding window attention.

Per-layer learned soft masking that replaces the hard boolean window
boundary with a differentiable sigmoid falloff. Each layer learns its
own effective window size from the input hidden state.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


class AdaptiveWindowPredictor(nn.Module):
    """Predicts per-position soft window boundaries.

    A lightweight linear projection maps each position's hidden state
    to a scalar "falloff center" — the effective window size for that
    query position. A sigmoid with configurable temperature converts
    query-key distances into soft mask weights.

    Args:
        config: TitansConfig with adaptive window fields set.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.min_window = config.adaptive_window_min
        self.max_window = config.effective_adaptive_window_max
        self.temperature = config.adaptive_window_temperature
        self.window_range = self.max_window - self.min_window

        # Linear projection: dim -> 1 scalar per position
        self.proj = nn.Linear(config.dim, 1, bias=True)

        # Initialize bias to produce mid-range falloff centers
        self.proj.weight = mx.random.normal(self.proj.weight.shape) * config.init_std
        self.proj.bias = mx.zeros((1,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute soft window mask from hidden states.

        Args:
            x: Hidden states (batch, seq_len, dim)

        Returns:
            Tuple of:
            - soft_mask: (batch, 1, seq_len, seq_len) mask weights in [0, 1]
            - falloff_centers: (batch, seq_len, 1) effective window sizes
        """
        batch, seq_len, _ = x.shape

        # Predict falloff center per position: (batch, seq_len, 1)
        raw = self.proj(x)  # (batch, seq_len, 1)
        # Scale to [min_window, max_window]
        falloff_centers = self.min_window + self.window_range * mx.sigmoid(raw)

        # Build distance matrix: distance[i, j] = i - j
        positions = mx.arange(seq_len)
        row_idx = mx.expand_dims(positions, axis=1)  # (seq_len, 1)
        col_idx = mx.expand_dims(positions, axis=0)  # (1, seq_len)
        distance = (row_idx - col_idx).astype(mx.float32)  # (seq_len, seq_len)

        # Expand falloff_centers for broadcasting: (batch, seq_len, 1) -> (batch, seq_len, 1)
        # distance: (seq_len, seq_len) broadcasts with falloff: (batch, seq_len, 1)
        # Result: (batch, seq_len, seq_len)
        soft_mask = mx.sigmoid(
            self.temperature * (falloff_centers - distance)
        )

        # Enforce causality: zero out future positions (where col > row)
        causal = (col_idx <= row_idx).astype(mx.float32)  # (seq_len, seq_len)
        soft_mask = soft_mask * causal

        # Add head dimension: (batch, 1, seq_len, seq_len)
        soft_mask = mx.expand_dims(soft_mask, axis=1)

        return soft_mask, falloff_centers
