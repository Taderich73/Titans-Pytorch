# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Persistent Memory Module for Titans (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from titans.config import TitansConfig


class PersistentMemory(nn.Module):
    """Persistent Memory tokens — learnable, data-independent.

    These tokens are prepended to the input sequence and encode
    task-specific knowledge. They remain fixed during inference.
    """

    tokens: nn.Parameter | None

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.num_tokens = config.num_persistent_tokens
        self.dim = config.dim

        if self.num_tokens > 0:
            self.tokens = nn.Parameter(
                torch.randn(self.num_tokens, self.dim) * config.init_std
            )
        else:
            self.tokens = None

    def forward(self, batch_size: int) -> torch.Tensor | None:
        """Get persistent memory tokens expanded for batch.

        Args:
            batch_size: Batch size

        Returns:
            Persistent tokens (batch, num_tokens, dim) or None if num_tokens=0
        """
        if self.tokens is None:
            return None
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
