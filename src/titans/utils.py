# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Utility helpers for Titans.

This module holds cross-cutting utilities that do not belong to any single
subsystem. Keep its surface area small — if a helper is feature-specific,
put it with the feature, not here.

The first (and currently only) helper is :func:`seed_everything`, the single
entry point for RNG seeding across every training script. See
:doc:`../../docs/reproducibility.md` for what is (and isn't) bit-identical
under which conditions.
"""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np
import torch

__all__ = ["seed_everything"]

_CUBLAS_WORKSPACE_CONFIG: Final[str] = ":4096:8"


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    This is the single entry point for seeding in OpenTitans. Every
    training script calls it exactly once, immediately after config
    parsing, before any model construction, data loading, or
    ``torch.cuda.*`` call — otherwise earlier RNG draws escape the seed.

    Args:
        seed: Integer seed applied to :mod:`random`, :mod:`numpy.random`,
            :func:`torch.manual_seed`, and :func:`torch.cuda.manual_seed_all`
            (no-op if CUDA is unavailable).
        deterministic: When ``True``, also enable PyTorch's deterministic
            algorithms via :func:`torch.use_deterministic_algorithms` and
            export ``CUBLAS_WORKSPACE_CONFIG=:4096:8``. This makes CUDA
            kernel selection deterministic at the cost of some speed and
            occasional unsupported-op ``RuntimeError`` for kernels without
            a deterministic equivalent. CPU-only runs are already
            deterministic for nearly all ops; the flag is mostly a
            belt-and-suspenders switch for CUDA.

    Notes:
        ``CUBLAS_WORKSPACE_CONFIG`` must be set *before* PyTorch initializes
        CUDA. Calling :func:`seed_everything` right after argparse — before
        any model construction — guarantees that ordering.

    See Also:
        ``docs/reproducibility.md`` for the contract on what is bit-identical
        across runs under each combination of ``--seed`` and
        ``--deterministic``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True, warn_only=False)
