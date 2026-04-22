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

The auto-checkpointing stack (``MemoryCheckpointer``, detectors, signal
helpers, data types) lives under :mod:`titans.checkpointing` and is
loaded only on demand; importing :mod:`titans` does not drag it in.

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

import importlib
import warnings
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

# --- Schema version (P5) --------------------------------------------------
# Bump whenever ANY persisted state layout changes: MemoryState fields,
# TNTMemoryState fields, HF config shape, saved-dict top-level keys, npz
# key layout emitted by save_memory_states, etc. See MIGRATIONS.md at the
# repo root for the per-version change log and migration protocol.
#
# Version 1 is the first versioned schema. Anything written before P5 —
# .pt / .safetensors / .npz without the ``titans_schema_version`` key —
# is considered "unversioned" and loads via a best-effort path that
# assumes the pre-0.7 layout.
TITANS_SCHEMA_VERSION: int = 1
"""Current checkpoint schema version.

Bump when any persisted state layout changes: ``MemoryState``,
``TNTMemoryState``, HF config shape, saved-dict top-level keys, etc. See
``MIGRATIONS.md`` at the repo root."""

# --- Stable public API (eager imports) -----------------------------------
# Keep this list tight. Every name here is a backward-compat contract.
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans.memory_dump import load_memory_states, save_memory_states
from titans.models import TitansLMM, TitansMAC, TitansMAG, TitansMAL

try:
    __version__ = _pkg_version("titans")
except PackageNotFoundError:  # editable-install / uninstalled checkout fallback
    __version__ = "0.0.0+unknown"

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
    # Schema version
    "TITANS_SCHEMA_VERSION",
]

# --- Deprecated top-level re-exports (removed in 0.8) --------------------
# Each entry maps the old top-level name to the canonical sub-module it
# should be imported from going forward. The shim preserves import
# compatibility for one release while nudging users toward the sub-module.
#
# The auto-checkpointing entries point at the ``titans.checkpointing``
# package (new canonical home as of P9) rather than at the legacy
# top-level shim modules (``titans.novelty_detector`` etc.). That way
# ``from titans import StatisticalNoveltyDetector`` emits exactly one
# DeprecationWarning — from here — instead of also triggering the
# shim-module warning.
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
    # titans.checkpointing — data types (previously titans.checkpoint_types)
    "CheckpointEntry": "titans.checkpointing",
    "GateSnapshot": "titans.checkpointing",
    "MemoryCheckpointConfig": "titans.checkpointing",
    "SignalFrame": "titans.checkpointing",
    "TransitionRecord": "titans.checkpointing",
    # titans.checkpointing — signal helpers (previously titans.checkpoint_signals)
    "build_signal_frame": "titans.checkpointing",
    "compute_momentum_norms": "titans.checkpointing",
    "compute_momentum_shift": "titans.checkpointing",
    "compute_weight_delta": "titans.checkpointing",
    "compute_weight_norms": "titans.checkpointing",
    # titans.checkpointing — detection (previously titans.novelty_detector)
    "StatisticalNoveltyDetector": "titans.checkpointing",
    "TriggerDecision": "titans.checkpointing",
    # titans.checkpointing — orchestrator (previously titans.memory_checkpointer)
    "MemoryCheckpointer": "titans.checkpointing",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader for deprecated top-level re-exports.

    Resolves legacy names that used to live in ``titans.__all__`` by
    importing them from their canonical sub-module, while emitting a
    :class:`DeprecationWarning` that points at the new import path.

    The resolved attribute is cached on the module after the first
    warning. This is the idiomatic PEP 562 pattern (numpy/scipy/pandas)
    and is required so that ``from titans import X`` emits exactly one
    warning: CPython's ``IMPORT_FROM`` probes ``__getattr__`` twice
    (once for the attribute, once for a potential submodule), and
    without caching each probe fires a fresh warning. Tests that need
    to observe repeat accesses should either ``importlib.reload(titans)``
    between cases or run each access in a fresh subprocess.

    Raises:
        AttributeError: If ``name`` is not a known deprecated export.
    """
    submodule = _DEPRECATED_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(submodule)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - guards shim drift
        raise AttributeError(
            f"Deprecation shim entry 'titans.{name}' -> '{submodule}.{name}' "
            "is stale: the submodule no longer exposes that attribute."
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
    # Cache on the module so subsequent accesses do not trigger
    # __getattr__ again (idiomatic PEP 562 pattern used by numpy/scipy).
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose both stable and deprecated names for tab-completion."""
    return sorted(set(__all__) | set(_DEPRECATED_EXPORTS) | {"__version__"})
