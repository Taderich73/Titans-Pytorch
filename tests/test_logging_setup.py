# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Unit tests for ``titans._logging`` (rich-backed structured logging)."""

from __future__ import annotations

import logging

import pytest
from rich.logging import RichHandler

import titans._logging as logmod
from titans._logging import (
    metrics_table,
    print_metrics_table,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _isolate_logging_state():
    """Snapshot + restore root-logger handlers and module singletons.

    Without this, ``setup_logging``'s first-time ``root.handlers.clear()``
    wipes pytest's own ``LogCaptureHandler``, breaking every subsequent
    test that relies on ``caplog``.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_configured = logmod._CONFIGURED
    saved_console = logmod._CONSOLE

    # Start every test from a clean slate so ``setup_logging`` exercises
    # its first-call install path.
    logmod._CONFIGURED = False
    logmod._CONSOLE = None
    root.handlers = [h for h in saved_handlers if not isinstance(h, RichHandler)]

    try:
        yield
    finally:
        # Remove anything we added, then restore the original handler
        # list so caplog / other fixtures see the world they set up.
        for h in list(root.handlers):
            if isinstance(h, RichHandler) and h not in saved_handlers:
                root.removeHandler(h)
        root.handlers = list(saved_handlers)
        root.setLevel(saved_level)
        logmod._CONFIGURED = saved_configured
        logmod._CONSOLE = saved_console


def test_setup_logging_is_idempotent() -> None:
    """Calling twice does not stack handlers."""
    r1 = setup_logging(logging.INFO)
    count_after_first = len(r1.handlers)
    r2 = setup_logging(logging.DEBUG)
    assert len(r2.handlers) == count_after_first, "setup_logging should be idempotent"
    assert r1.level == logging.DEBUG  # level was updated on second call


def test_setup_logging_installs_rich_handler() -> None:
    root = setup_logging()
    assert any(isinstance(h, RichHandler) for h in root.handlers)


def test_setup_logging_accepts_string_level() -> None:
    root = setup_logging("WARNING")
    assert root.level == logging.WARNING


def test_metrics_table_builds_table() -> None:
    rows = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.3}]
    t = metrics_table(rows, title="Metrics")
    assert t.title == "Metrics"
    assert t.row_count == 2
    # Columns are inferred from the first row when not given.
    col_headers = [c.header for c in t.columns]
    assert col_headers == ["step", "loss"]


def test_metrics_table_explicit_columns() -> None:
    rows = [{"step": 1, "loss": 0.5, "lr": 1e-4}]
    t = metrics_table(rows, title="Metrics", columns=["loss", "step"])
    col_headers = [c.header for c in t.columns]
    assert col_headers == ["loss", "step"]


def test_metrics_table_empty_rows() -> None:
    t = metrics_table([], title="Empty")
    assert t.row_count == 0


def test_metrics_table_missing_key_renders_blank() -> None:
    """A missing key in a later row should render as an empty string."""
    rows = [
        {"step": 1, "loss": 0.5},
        {"step": 2},  # no loss key
    ]
    t = metrics_table(rows, columns=["step", "loss"])
    assert t.row_count == 2


def test_print_metrics_table_does_not_raise() -> None:
    """Smoke test that the print helper works end-to-end."""
    print_metrics_table(
        [{"step": 1, "loss": 0.5}],
        title="Smoke",
        columns=["step", "loss"],
    )
