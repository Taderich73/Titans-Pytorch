"""Feature D: forward hooks capturing per-chunk gate alpha values."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class GateHookRegistry:
    """Register forward hooks on every block's gate_decay_proj.

    On each forward, the hook stashes the most recent output tensor per block.
    snapshot() applies sigmoid and computes cross-block / cross-batch stats
    suitable for logging.

    Handles both non-TNT (block.memory.gate_decay_proj) and TNT
    (block.memory.global_memory.memory.gate_decay_proj) module paths.
    """

    def __init__(self, model: Any) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._outputs: dict[int, torch.Tensor] = {}
        self._num_blocks: int = 0

        for idx, block in enumerate(getattr(model, "blocks", [])):
            proj = _find_gate_decay_proj(block)
            if proj is None:
                continue
            handle = proj.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
            self._num_blocks = max(self._num_blocks, idx + 1)

    def _make_hook(self, block_idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            # Stash detached, cpu-agnostic copy; keep on same device to avoid
            # pointless transfers. Overwrite prior value for this block.
            self._outputs[block_idx] = output.detach()

        return hook

    def snapshot(self) -> dict[str, Any]:
        """Apply sigmoid to stashed tensors and compute aggregate stats.

        Returns an empty dict when no forward has populated the registry.
        """
        if not self._outputs:
            return {}

        # Per-block alpha: mean over batch/time/feature dims.
        per_block: list[float] = []
        flat_values: list[float] = []
        for idx in sorted(self._outputs.keys()):
            raw = self._outputs[idx].float()
            alpha = torch.sigmoid(raw)
            per_block.append(float(alpha.mean().item()))
            flat_values.extend(alpha.flatten().tolist())

        if not flat_values:
            return {}

        mean = sum(flat_values) / len(flat_values)
        if len(flat_values) >= 2:
            var = sum((v - mean) ** 2 for v in flat_values) / len(flat_values)
            std = math.sqrt(var)
        else:
            std = 0.0

        return {
            "gate/alpha_mean": mean,
            "gate/alpha_std": std,
            "gate/alpha_min": min(flat_values),
            "gate/alpha_max": max(flat_values),
            "gate/alpha_per_block": per_block,
        }

    def clear(self) -> None:
        """Drop all stashed tensors (call after snapshot to free memory)."""
        self._outputs.clear()

    def remove(self) -> None:
        """Deregister every hook. Idempotent."""
        for handle in self._handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self._handles.clear()
        self._outputs.clear()


def _find_gate_decay_proj(block: Any) -> nn.Module | None:
    """Locate the gate_decay_proj on a block, supporting TNT and non-TNT layouts."""
    mem = getattr(block, "memory", None)
    if mem is None:
        return None
    # TNT path.
    global_mem = getattr(mem, "global_memory", None)
    if global_mem is not None:
        inner = getattr(global_mem, "memory", None)
        if inner is not None:
            proj = getattr(inner, "gate_decay_proj", None)
            if isinstance(proj, nn.Module):
                return proj
    # Non-TNT path.
    proj = getattr(mem, "gate_decay_proj", None)
    if isinstance(proj, nn.Module):
        return proj
    return None
