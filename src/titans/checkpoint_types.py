# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Backward-compatibility shim.

``titans.checkpoint_types`` has moved to :mod:`titans.checkpointing.types`
(P9 — the file was renamed to ``types.py`` inside the ``checkpointing``
package, where the ``checkpoint_`` prefix is redundant).

This shim will be removed in 0.8. Update your imports::

    # Old
    from titans.checkpoint_types import MemoryCheckpointConfig

    # New — prefer the subpackage's public surface
    from titans.checkpointing import MemoryCheckpointConfig

    # Or, for direct submodule access
    from titans.checkpointing.types import MemoryCheckpointConfig
"""

from __future__ import annotations

import warnings

from titans.checkpointing.types import (  # noqa: F401
    CheckpointEntry,
    GateSnapshot,
    MemoryCheckpointConfig,
    SignalFrame,
    TransitionRecord,
)

warnings.warn(
    "`titans.checkpoint_types` has moved to `titans.checkpointing.types` "
    "(or use `from titans.checkpointing import ...` for the public API). "
    "This shim will be removed in 0.8.",
    DeprecationWarning,
    stacklevel=2,
)
