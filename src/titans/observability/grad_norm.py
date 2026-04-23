"""Feature A: pre-clip global gradient norm."""

from __future__ import annotations

import math

from torch import nn


def global_grad_norm(model: nn.Module) -> float:
    """Compute global L2 norm across all parameter gradients.

    Intended to be called between accelerator.backward(loss) and
    accelerator.clip_grad_norm_(...), on a sync-gradient step.

    Args:
        model: Any nn.Module. Parameters without .grad are ignored.

    Returns:
        The global L2 norm as a Python float. Returns 0.0 if no gradients
        are attached.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total_sq += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total_sq)
