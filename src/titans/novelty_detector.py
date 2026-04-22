# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Backward-compatibility shim.

``titans.novelty_detector`` has moved to
:mod:`titans.checkpointing.novelty_detector` (P9).

This shim will be removed in 0.8. Update your imports::

    # Old
    from titans.novelty_detector import StatisticalNoveltyDetector

    # New — prefer the subpackage's public surface
    from titans.checkpointing import StatisticalNoveltyDetector

    # Or, for direct submodule access
    from titans.checkpointing.novelty_detector import StatisticalNoveltyDetector
"""

from __future__ import annotations

import warnings

from titans.checkpointing.novelty_detector import (  # noqa: F401
    NoveltyDetector,
    StatisticalNoveltyDetector,
    TriggerDecision,
)

warnings.warn(
    "`titans.novelty_detector` has moved to `titans.checkpointing.novelty_detector` "
    "(or use `from titans.checkpointing import ...` for the public API). "
    "This shim will be removed in 0.8.",
    DeprecationWarning,
    stacklevel=2,
)
