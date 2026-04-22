# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory update gate machinery.

Shared support for the learn-at-test-time update equations of
:class:`~titans.memory.NeuralLongTermMemory`:

* numeric constants used by the L2 normalization and degenerate-gate branches
  of the parallel update,
* the activation lookup used by :class:`MemoryMLP`,
* :class:`MemoryMLP` itself — the inner weight-storing MLP whose weights are
  mutated by the alpha/eta/theta gated update,
* :func:`apply_huber_clip` — the Huber-δ gate applied to the prediction error
  when ``memory_objective == "huber"``.

The alpha/eta/theta/δ gate *projections* themselves are ``nn.Linear`` layers
owned by :class:`~titans.memory.NeuralLongTermMemory` (they see
data-dependent inputs and need to be registered as submodules of the main
memory module), so they live in ``core.py``; this module holds only the
stateless, reusable helpers they rely on.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig

# Numeric thresholds shared by L2 normalization and the degenerate-gate branch
# of the parallel update. Kept module-level (not config) so they match the
# paper's fixed ε choices and stay constant-foldable under torch.compile.
_L2_NORM_EPS: float = 1e-8
_DEGENERATE_THRESHOLD: float = 1e-6


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable] = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def apply_huber_clip(
    raw_error: torch.Tensor, delta: torch.Tensor | None
) -> torch.Tensor:
    """Apply the Huber-δ gate to a pre-clipped prediction error.

    Outside the δ-ball the error saturates at ``±δ``; inside, the raw
    (already memory_error_clip-clipped) error passes through. When ``delta``
    is ``None`` the input is returned unchanged, letting callers unconditionally
    route through this helper without an extra branch.

    Args:
        raw_error: Prediction error already clipped to ``±memory_error_clip``.
        delta: Per-sample Huber threshold, shape ``(B, 1, 1)`` or broadcastable.

    Returns:
        Tensor with the same shape/dtype as ``raw_error``.
    """
    if delta is None:
        return raw_error
    abs_error = torch.abs(raw_error)
    return torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))


class MemoryMLP(nn.Module):
    """MLP that stores information in its weights."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim
        self.activation = get_activation(config.activation)

        layers: list[nn.Linear] = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        self.layers = nn.ModuleList(layers)
        self._init_weights(config.init_std)

        self._layer_shapes = [tuple(layer.weight.shape) for layer in self.layers]
        self._ref_dtype = self.layers[0].weight.dtype

    def _init_weights(self, std: float) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=std)

    def forward_with_weights(
        self, x: torch.Tensor, weights: list[torch.Tensor]
    ) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]

    def zero_weights_like(self, device: torch.device) -> list[torch.Tensor]:
        """Return zero-initialized tensors matching each layer's weight shape."""
        return [
            torch.zeros(shape, dtype=self._ref_dtype, device=device)
            for shape in self._layer_shapes
        ]

    def get_base_weights(self) -> list[torch.Tensor]:
        """Return live weight parameter references (with autograd graph)."""
        return [layer.weight for layer in self.layers]
