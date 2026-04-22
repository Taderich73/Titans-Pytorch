# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory subsystem — NeuralLongTermMemory and its state containers.

Public surface of the ``titans.memory`` subpackage. The legacy
``from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState``
import path is preserved exactly as it was when this was a single module.

Layout
------
- :mod:`titans.memory.state` — :class:`MemoryState`, :class:`TNTMemoryState`
  dataclasses.
- :mod:`titans.memory.gates` — :class:`MemoryMLP`, :func:`get_activation`,
  :func:`apply_huber_clip`, and the numeric constants used by the
  L2-norm / degenerate-gate branches of the parallel update.
- :mod:`titans.memory.core` — :class:`NeuralLongTermMemory` — the
  learn-at-test-time memory module.
"""

from __future__ import annotations

from titans.memory.core import NeuralLongTermMemory
from titans.memory.gates import (
    MemoryMLP,
    apply_huber_clip,
    get_activation,
)
from titans.memory.state import MemoryState, TNTMemoryState

__all__ = [
    "MemoryMLP",
    "MemoryState",
    "NeuralLongTermMemory",
    "TNTMemoryState",
    "apply_huber_clip",
    "get_activation",
]
