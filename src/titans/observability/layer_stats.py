"""Feature C: per-block memory state and weight norms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayerStats:
    """Container for per-block norms with aggregate accessors.

    state_norms and weight_norms are index-aligned with blocks. A None entry
    in state_norms indicates the block has no state yet (e.g. pre-warmup).
    """

    state_norms: list[float | None]
    weight_norms: list[float | None]

    # ---- aggregates over state_norms (skipping None) ------------------------

    @property
    def state_mean(self) -> float:
        vals = [v for v in self.state_norms if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def state_std(self) -> float:
        vals = [v for v in self.state_norms if v is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return math.sqrt(var)

    @property
    def state_min(self) -> float:
        vals = [v for v in self.state_norms if v is not None]
        return min(vals) if vals else 0.0

    @property
    def state_max(self) -> float:
        vals = [v for v in self.state_norms if v is not None]
        return max(vals) if vals else 0.0

    # ---- aggregates over weight_norms --------------------------------------

    @property
    def weight_mean(self) -> float:
        vals = [v for v in self.weight_norms if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def weight_std(self) -> float:
        vals = [v for v in self.weight_norms if v is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return math.sqrt(var)

    @property
    def weight_min(self) -> float:
        vals = [v for v in self.weight_norms if v is not None]
        return min(vals) if vals else 0.0

    @property
    def weight_max(self) -> float:
        vals = [v for v in self.weight_norms if v is not None]
        return max(vals) if vals else 0.0

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce JSONL-ready payload with namespaced keys."""
        return {
            "layer/state_norm_mean": self.state_mean,
            "layer/state_norm_std": self.state_std,
            "layer/state_norm_min": self.state_min,
            "layer/state_norm_max": self.state_max,
            "layer/state_norm_per_block": list(self.state_norms),
            "layer/weight_norm_mean": self.weight_mean,
            "layer/weight_norm_std": self.weight_std,
            "layer/weight_norm_min": self.weight_min,
            "layer/weight_norm_max": self.weight_max,
            "layer/weight_norm_per_block": list(self.weight_norms),
        }


def collect_layer_stats(
    unwrapped_model: Any,
    memory_states: list[Any] | None,
) -> LayerStats:
    """Walk all blocks and extract per-block state and weight norms.

    Handles TNT (block.memory.global_memory.memory) and non-TNT
    (block.memory.memory) paths. Returns a LayerStats whose lists are aligned
    with unwrapped_model.blocks.

    Args:
        unwrapped_model: The model post-accelerator.unwrap_model().
        memory_states: Per-block memory state objects from the training loop.
            May be None or contain None entries.

    Returns:
        LayerStats.
    """
    blocks = list(getattr(unwrapped_model, "blocks", []))
    num = len(blocks)
    state_norms: list[float | None] = []
    weight_norms: list[float | None] = []

    for i, block in enumerate(blocks):
        state = (
            memory_states[i]
            if memory_states is not None and i < len(memory_states)
            else None
        )
        state_norms.append(_extract_state_norm(state))
        weight_norms.append(_extract_weight_norm(block))

    # Defensive: ensure list lengths match block count even when empty.
    while len(state_norms) < num:
        state_norms.append(None)
    while len(weight_norms) < num:
        weight_norms.append(None)

    return LayerStats(state_norms=state_norms, weight_norms=weight_norms)


def _extract_state_norm(state: Any) -> float | None:
    """Return L2 norm of the state's first weights tensor, or None if absent."""
    if state is None:
        return None
    # TNT path: state.global_state.weights[0]
    g_state = getattr(state, "global_state", None)
    if g_state is not None and getattr(g_state, "weights", None):
        try:
            return float(g_state.weights[0].detach().float().norm().item())
        except (AttributeError, IndexError):
            pass
    # Non-TNT path: state.weights[0]
    weights = getattr(state, "weights", None)
    if weights:
        try:
            return float(weights[0].detach().float().norm().item())
        except (AttributeError, IndexError):
            return None
    return None


def _extract_weight_norm(block: Any) -> float | None:
    """Return L2 norm of block's memory inner first layer weight."""
    mem = getattr(block, "memory", None)
    if mem is None:
        return None

    # TNT path: block.memory.global_memory.memory.layers[0].weight
    inner = getattr(getattr(mem, "global_memory", None), "memory", None)
    # Non-TNT path: block.memory.memory.layers[0].weight
    if inner is None:
        inner = getattr(mem, "memory", None)
    if inner is None:
        return None

    layers = getattr(inner, "layers", None)
    if not layers:
        return None
    try:
        w = layers[0].weight
        if not isinstance(w, torch.Tensor):
            return None
        return float(w.detach().float().norm().item())
    except (AttributeError, IndexError):
        return None
