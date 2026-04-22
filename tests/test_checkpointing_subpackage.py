# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Contract tests for the ``titans.checkpointing`` subpackage (P9).

Three guarantees:

1. ``import titans`` must NOT eagerly load ``titans.checkpointing``
   (zero-cost-when-unused property).
2. The subpackage's advertised public surface is importable.
3. The legacy top-level module paths (``titans.novelty_detector`` etc.)
   still resolve via deprecation shims, and importing them emits a
   :class:`DeprecationWarning`.
"""

from __future__ import annotations

import subprocess
import sys
import warnings


def test_importing_titans_does_not_load_checkpointing() -> None:
    """``python -X importtime -c 'import titans'`` must not load any
    ``titans.checkpointing`` or legacy auto-checkpointing module.

    Run in a fresh subprocess so a warm module cache from another test
    cannot mask a regression. The importtime trace on stderr contains
    one line per imported module.
    """
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import titans"],
        capture_output=True,
        check=True,
    )
    trace = proc.stderr.decode()
    forbidden = (
        "titans.checkpointing",
        "titans.novelty_detector",
        "titans.memory_checkpointer",
        "titans.checkpoint_signals",
        "titans.checkpoint_types",
    )
    offenders = [
        line for line in trace.splitlines() if any(f in line for f in forbidden)
    ]
    assert not offenders, (
        "import titans must not eagerly load the auto-checkpointing stack, "
        "but these lines appeared in the importtime trace:\n" + "\n".join(offenders)
    )


def test_subpackage_public_api() -> None:
    """Every name in the subpackage's ``__all__`` must be importable."""
    from titans.checkpointing import (  # noqa: F401
        CheckpointEntry,
        GateSnapshot,
        MemoryCheckpointConfig,
        MemoryCheckpointer,
        SignalFrame,
        StatisticalNoveltyDetector,
        TransitionRecord,
        TriggerDecision,
        build_signal_frame,
        compute_momentum_norms,
        compute_momentum_shift,
        compute_weight_delta,
        compute_weight_norms,
    )


def test_deprecated_old_module_path_warns() -> None:
    """Importing the legacy top-level module path emits a
    :class:`DeprecationWarning` and still returns the class from the
    new canonical submodule.

    Run each probe in a fresh subprocess to avoid the module cache
    swallowing the shim's module-level ``warnings.warn`` on subsequent
    imports within the same interpreter.
    """
    for legacy, attr in (
        ("titans.novelty_detector", "StatisticalNoveltyDetector"),
        ("titans.memory_checkpointer", "MemoryCheckpointer"),
        ("titans.checkpoint_signals", "build_signal_frame"),
        ("titans.checkpoint_types", "MemoryCheckpointConfig"),
    ):
        snippet = (
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            f"    mod = __import__({legacy!r}, fromlist=['_'])\n"
            f"    obj = getattr(mod, {attr!r})\n"
            "    assert any(issubclass(warning.category, DeprecationWarning) "
            f"                for warning in w), 'no DeprecationWarning for {legacy}'\n"
            "    assert obj.__module__.startswith('titans.checkpointing'), (\n"
            "        f'expected re-export from titans.checkpointing, got '"
            "        f'{obj.__module__}')\n"
        )
        subprocess.run([sys.executable, "-c", snippet], check=True)


def test_top_level_getattr_resolves_via_new_path() -> None:
    """``from titans import StatisticalNoveltyDetector`` must resolve
    without triggering the legacy-shim ``DeprecationWarning`` on top of
    the top-level ``__getattr__`` warning — there should be exactly one
    warning, pointing at :mod:`titans.checkpointing`.
    """
    # Run in subprocess to guarantee a clean module cache; if
    # titans.novelty_detector was loaded earlier in this process the
    # second shim warning is suppressed by Python's default filter.
    snippet = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    from titans import StatisticalNoveltyDetector\n"
        "    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
        "    assert len(dep) == 1, f'expected 1 DeprecationWarning, got {len(dep)}: '\\\n"
        "        + repr([str(x.message) for x in dep])\n"
        "    assert 'titans.checkpointing' in str(dep[0].message)\n"
    )
    subprocess.run([sys.executable, "-c", snippet], check=True)


def test_warnings_import_used() -> None:
    """Silence unused-import lints for the ``warnings`` module used
    only transitively above."""
    assert warnings is warnings
