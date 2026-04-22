"""Helpers for tests that spawn Python subprocesses which import ``titans``.

Background
----------
``sys.executable`` is not always a Python interpreter that has ``titans``
installed. Under ``uv run pytest`` on some macOS setups (notably when the
host Python is the framework build at ``/Library/Frameworks/Python.framework``),
``sys.executable`` points at the system Python while the project's
dependencies — including ``titans`` itself — live in ``.venv``. Spawning
``[sys.executable, "-c", "import titans"]`` then fails with ``ModuleNotFoundError``,
even though the in-process pytest run is perfectly happy.

Resolution
----------
:func:`subprocess_python` returns an ``argv`` prefix that is known to have
``titans`` importable. It probes, in order:

1. ``sys.executable`` (the interpreter running pytest).
2. ``.venv/bin/python`` at the repo root, if present.
3. Whatever ``python3`` / ``python`` ``shutil.which`` picks up on ``PATH``.

If none can import ``titans``, the caller should ``pytest.skip`` with a
clear reason — cross-interpreter spawning is the feature under test in
those modules, not the project behaviour we care about here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _resolve_interpreter() -> str | None:
    """Return the first interpreter path that can ``import titans``.

    Cached so we pay the probe cost at most once per pytest session
    (each probe spawns a subprocess). Returns ``None`` if no candidate
    works; callers are expected to translate that into a skip.
    """
    candidates: list[str] = [sys.executable]

    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        candidates.append(str(venv_python))

    for which_name in ("python3", "python"):
        resolved = shutil.which(which_name)
        if resolved and resolved not in candidates:
            candidates.append(resolved)

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-c", "import titans"],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate

    return None


def subprocess_python() -> list[str]:
    """Return an ``argv`` prefix launching a Python that imports ``titans``.

    Skips the calling test via ``pytest.skip`` when no candidate
    interpreter can import ``titans`` — we cannot exercise the spawn-side
    behaviour without one.
    """
    interpreter = _resolve_interpreter()
    if interpreter is None:
        pytest.skip("no Python interpreter on this host can `import titans`")
    return [interpreter]
