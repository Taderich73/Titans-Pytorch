# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Adaptive window sizing for sliding window attention (PyTorch).

Paper alignment: N/A — novel extension. ``AdaptiveWindowPredictor`` is not
specified by any reference paper; it is a project-specific learned
soft-masking scheme documented in ``docs/adaptive_window_sizing.md``.

Per-layer learned soft masking that replaces the hard boolean window
boundary with a differentiable sigmoid falloff.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn

from titans.config import TitansConfig


@lru_cache(maxsize=32)
def _adaptive_window_grid(
    seq_len: int, device_str: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row_idx, col_idx, causal_mask) cached per (seq_len, device)."""
    device = torch.device(device_str)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    row_idx = positions.unsqueeze(1)
    col_idx = positions.unsqueeze(0)
    causal = (col_idx <= row_idx).float()
    return row_idx, col_idx, causal


class AdaptiveWindowPredictor(nn.Module):
    """Predicts per-position soft window boundaries."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.min_window = config.adaptive_window_min
        self.max_window = config.effective_adaptive_window_max
        self.temperature = config.adaptive_window_temperature
        self.window_range = self.max_window - self.min_window

        self.proj = nn.Linear(config.dim, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft window mask from hidden states.

        Args:
            x: Hidden states (batch, seq_len, dim)

        Returns:
            (soft_mask, falloff_centers):
                soft_mask: (batch, 1, seq_len, seq_len)
                falloff_centers: (batch, seq_len, 1)
        """
        batch, seq_len, _ = x.shape

        raw = self.proj(x)  # (batch, seq_len, 1)
        falloff_centers = self.min_window + self.window_range * torch.sigmoid(raw)

        row_idx, col_idx, causal = _adaptive_window_grid(seq_len, str(x.device))
        distance = row_idx - col_idx  # (seq_len, seq_len)

        soft_mask = torch.sigmoid(self.temperature * (falloff_centers - distance))
        soft_mask = soft_mask * causal
        soft_mask = soft_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)

        return soft_mask, falloff_centers


def compute_window_regularization(
    falloff_centers: list[torch.Tensor],
    max_window: int,
) -> torch.Tensor:
    """Efficiency regularization penalizing large windows."""
    if not falloff_centers:
        return torch.tensor(0.0)
    layer_means = [torch.mean(fc / max_window) for fc in falloff_centers]
    return torch.mean(torch.stack(layer_means))
