# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Internal structured-logging helpers for training scripts.

Wraps ``rich.logging.RichHandler`` so scripts get colored, level-aware log
output with minimal ceremony. Also provides a ``metrics_table`` helper for
building tidy per-step / per-epoch metric tables via ``rich.table``.

Not part of the public API -- import paths may change without deprecation.
Library code under ``src/titans`` **must not** import from this module; it
is intended exclusively for the training / inference scripts shipped with
the repo. Rich is already a runtime dependency of ``titans``, so a stray
import would not add a new one, but the expectation is that only script
entry points reach for these helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

_CONFIGURED: bool = False
_CONSOLE: Console | None = None


def get_console() -> Console:
    """Return the process-wide shared :class:`rich.console.Console`.

    Created lazily on first call so importing this module has no side
    effects on stdout/stderr streams.
    """
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Install a :class:`rich.logging.RichHandler` on the root logger.

    Idempotent -- calling this more than once leaves the handler stack
    unchanged but still updates the root logger level to ``level``.

    Args:
        level: Root-logger level. Accepts either a :mod:`logging` int
            constant (``logging.DEBUG``, ``logging.INFO``, ...) or a
            case-insensitive string name (``"DEBUG"``, ``"INFO"``, ...).

    Returns:
        The root logger, for convenience. Scripts typically keep using
        ``logging.getLogger(__name__)`` afterwards and rely on
        propagation.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = RichHandler(
            console=get_console(),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Replace any pre-existing handlers (e.g. a stray
        # ``logging.basicConfig`` call earlier in the process) so we do
        # not double-print every log record.
        root.handlers.clear()
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(level)
    return root


def metrics_table(
    rows: Iterable[Mapping[str, object]],
    *,
    title: str | None = None,
    columns: Iterable[str] | None = None,
) -> Table:
    """Build a :class:`rich.table.Table` from an iterable of metric dicts.

    Args:
        rows: Iterable of ``{metric_name: value}`` mappings. Consumed
            once; an empty iterable produces a zero-row table.
        title: Optional table title. Passed through to
            :class:`rich.table.Table`.
        columns: Column order. When ``None`` (the default) columns are
            inferred from the keys of the first row.

    Returns:
        A fully populated :class:`rich.table.Table`. Values are rendered
        with ``str(...)`` so callers can pre-format floats / tensors to
        whatever precision they want before adding the row.
    """
    rows_list = list(rows)
    if not rows_list:
        return Table(title=title)
    cols = list(columns) if columns is not None else list(rows_list[0].keys())
    table = Table(title=title)
    for col in cols:
        table.add_column(col)
    for row in rows_list:
        table.add_row(*(str(row.get(col, "")) for col in cols))
    return table


def print_metrics_table(
    rows: Iterable[Mapping[str, object]],
    *,
    title: str | None = None,
    columns: Iterable[str] | None = None,
) -> None:
    """Build a metrics table and print it on the shared console.

    Convenience wrapper around :func:`metrics_table` + ``console.print``.
    """
    get_console().print(metrics_table(rows, title=title, columns=columns))


__all__ = [
    "get_console",
    "metrics_table",
    "print_metrics_table",
    "setup_logging",
]
