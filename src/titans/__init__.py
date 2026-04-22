# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Titans: Learning to Memorize at Test Time — PyTorch Implementation.

Public API (stable surface for 0.7.x)
-------------------------------------
The names exposed in :data:`__all__` are the curated, supported top-level
imports. Everything else lives behind a sub-module path (``titans.attention``,
``titans.memory``, ``titans.lora``, …).

Legacy top-level names that were exported in 0.6.x still resolve via
:pep:`562` lazy ``__getattr__``, but each access emits a
:class:`DeprecationWarning` pointing at the canonical sub-module path.
These shims will be removed in 0.8.

Usage::

    import torch
    from titans import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    x = torch.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)

See :doc:`docs/api.md` for the full public-surface reference.
"""

from __future__ import annotations

import warnings
from typing import Any

# --- Stable public API (eager imports) -----------------------------------
# Keep this list tight. Every name here is a backward-compat contract.
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans.memory_dump import load_memory_states, save_memory_states
from titans.models import TitansLMM, TitansMAC, TitansMAG, TitansMAL

__version__ = "0.7.0"

__all__ = [
    # Config
    "TitansConfig",
    # Models
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    "TNTMemoryState",
    # Persistence
    "save_memory_states",
    "load_memory_states",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
]

# --- Deprecated top-level re-exports (removed in 0.8) --------------------
# Each entry maps the old top-level name to the canonical sub-module it
# should be imported from going forward. The shim preserves import
# compatibility for one release while nudging users toward the sub-module.
#
# NOTE: P9 will move ``checkpoint_types``, ``checkpoint_signals``,
# ``novelty_detector`` and ``memory_checkpointer`` under
# ``titans.checkpointing``. When that lands, update the corresponding
# targets here.
_DEPRECATED_EXPORTS: dict[str, str] = {
    # titans.adaptive_window
    "AdaptiveWindowPredictor": "titans.adaptive_window",
    "compute_window_regularization": "titans.adaptive_window",
    # titans.attention
    "RotaryPositionEmbedding": "titans.attention",
    "SegmentedAttention": "titans.attention",
    "SlidingWindowAttention": "titans.attention",
    "log_sdpa_backend": "titans.attention",
    # titans.attn_res
    "BlockAttnRes": "titans.attn_res",
    # titans.mca
    "MemoryCrossAttention": "titans.mca",
    # titans.qk_projection
    "QKProjection": "titans.qk_projection",
    # titans.tnt_memory
    "GlobalMemory": "titans.tnt_memory",
    "HierarchicalMemory": "titans.tnt_memory",
    "LocalMemory": "titans.tnt_memory",
    # titans.persistent
    "PersistentMemory": "titans.persistent",
    # titans.quantize_state
    "QuantizedMemoryState": "titans.quantize_state",
    "QuantizedTensor": "titans.quantize_state",
    "quantize_memory_state": "titans.quantize_state",
    "quantize_tensor": "titans.quantize_state",
    # titans.lora
    "LoRALinear": "titans.lora",
    "count_lora_parameters": "titans.lora",
    "load_adapters": "titans.lora",
    "merge_lora_weights": "titans.lora",
    "save_adapters": "titans.lora",
    "set_lora_enabled": "titans.lora",
    "wrap_lora_layers": "titans.lora",
    # titans.checkpoint_types (P9 will relocate to titans.checkpointing)
    "CheckpointEntry": "titans.checkpoint_types",
    "GateSnapshot": "titans.checkpoint_types",
    "MemoryCheckpointConfig": "titans.checkpoint_types",
    "SignalFrame": "titans.checkpoint_types",
    "TransitionRecord": "titans.checkpoint_types",
    # titans.checkpoint_signals (P9 will relocate to titans.checkpointing)
    "build_signal_frame": "titans.checkpoint_signals",
    "compute_momentum_norms": "titans.checkpoint_signals",
    "compute_momentum_shift": "titans.checkpoint_signals",
    "compute_weight_delta": "titans.checkpoint_signals",
    "compute_weight_norms": "titans.checkpoint_signals",
    # titans.novelty_detector (P9 will relocate to titans.checkpointing)
    "StatisticalNoveltyDetector": "titans.novelty_detector",
    "TriggerDecision": "titans.novelty_detector",
    # titans.memory_checkpointer (P9 will relocate to titans.checkpointing)
    "MemoryCheckpointer": "titans.memory_checkpointer",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader for deprecated top-level re-exports.

    Resolves legacy names that used to live in ``titans.__all__`` by
    importing them from their canonical sub-module, while emitting a
    :class:`DeprecationWarning` that points at the new import path.

    The warning fires on every access (the resolved attribute is NOT
    cached on the module). This is intentional: caching would hide the
    deprecation from every subsequent importer in the same process and
    make per-name assertions fragile. The per-access cost is a dict
    lookup plus an ``importlib`` cache hit — negligible.

    Raises:
        AttributeError: If ``name`` is not a known deprecated export.
    """
    submodule = _DEPRECATED_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(submodule)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {submodule!r} has no attribute {name!r} "
            f"(stale deprecation shim in titans.__init__)"
        ) from exc

    warnings.warn(
        (
            f"`titans.{name}` is deprecated and will be removed in 0.8. "
            f"Import it from `{submodule}` instead "
            f"(e.g. `from {submodule} import {name}`)."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    """Expose both stable and deprecated names for tab-completion."""
    return sorted(set(__all__) | set(_DEPRECATED_EXPORTS) | {"__version__"})
