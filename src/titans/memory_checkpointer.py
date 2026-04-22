# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Backward-compatibility shim.

``titans.memory_checkpointer`` has moved to
:mod:`titans.checkpointing.memory_checkpointer` (P9).

This shim will be removed in 0.8. Update your imports::

    # Old
    from titans.memory_checkpointer import MemoryCheckpointer

    # New — prefer the subpackage's public surface
    from titans.checkpointing import MemoryCheckpointer

    # Or, for direct submodule access
    from titans.checkpointing.memory_checkpointer import MemoryCheckpointer
"""

from __future__ import annotations

import warnings

from titans.checkpointing.memory_checkpointer import (  # noqa: F401
    CheckpointerState,
    MemoryCheckpointer,
)

warnings.warn(
    "`titans.memory_checkpointer` has moved to `titans.checkpointing.memory_checkpointer` "
    "(or use `from titans.checkpointing import ...` for the public API). "
    "This shim will be removed in 0.8.",
    DeprecationWarning,
    stacklevel=2,
)
