# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Auto-checkpointing subsystem.

Utilities for signal-based detection of memory-update transitions and
snapshotting the memory state when novelty spikes. Import what you need
from this subpackage only if you're wiring ``auto_checkpoint=True`` or
running the inference script with ``--auto-checkpoint``; the rest of
titans does not depend on anything here.

Public surface
--------------
- :class:`MemoryCheckpointer` — orchestrator and ring buffer.
- :class:`StatisticalNoveltyDetector` / :class:`TriggerDecision` — detection.
- :class:`MemoryCheckpointConfig` / :class:`TransitionRecord` /
  :class:`CheckpointEntry` / :class:`SignalFrame` / :class:`GateSnapshot` —
  data types.
- :func:`build_signal_frame`, :func:`compute_weight_delta`,
  :func:`compute_momentum_shift`, :func:`compute_weight_norms`,
  :func:`compute_momentum_norms` — signal-compute helpers.
"""

from __future__ import annotations

from .memory_checkpointer import MemoryCheckpointer
from .novelty_detector import StatisticalNoveltyDetector, TriggerDecision
from .signals import (
    build_signal_frame,
    compute_momentum_norms,
    compute_momentum_shift,
    compute_weight_delta,
    compute_weight_norms,
)
from .types import (
    CheckpointEntry,
    GateSnapshot,
    MemoryCheckpointConfig,
    SignalFrame,
    TransitionRecord,
)

__all__ = [
    # Orchestrator
    "MemoryCheckpointer",
    # Detection
    "StatisticalNoveltyDetector",
    "TriggerDecision",
    # Data types
    "CheckpointEntry",
    "GateSnapshot",
    "MemoryCheckpointConfig",
    "SignalFrame",
    "TransitionRecord",
    # Signal-compute helpers
    "build_signal_frame",
    "compute_momentum_norms",
    "compute_momentum_shift",
    "compute_weight_delta",
    "compute_weight_norms",
]
