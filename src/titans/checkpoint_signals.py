# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Backward-compatibility shim.

``titans.checkpoint_signals`` has moved to
:mod:`titans.checkpointing.signals` (P9 — the file was renamed to
``signals.py`` inside the ``checkpointing`` package, where the
``checkpoint_`` prefix is redundant).

This shim will be removed in 0.8. Update your imports::

    # Old
    from titans.checkpoint_signals import build_signal_frame

    # New — prefer the subpackage's public surface
    from titans.checkpointing import build_signal_frame

    # Or, for direct submodule access
    from titans.checkpointing.signals import build_signal_frame
"""

from __future__ import annotations

import warnings

from titans.checkpointing.signals import (  # noqa: F401
    build_signal_frame,
    compute_momentum_norms,
    compute_momentum_shift,
    compute_weight_delta,
    compute_weight_norms,
)

warnings.warn(
    "`titans.checkpoint_signals` has moved to `titans.checkpointing.signals` "
    "(or use `from titans.checkpointing import ...` for the public API). "
    "This shim will be removed in 0.8.",
    DeprecationWarning,
    stacklevel=2,
)
