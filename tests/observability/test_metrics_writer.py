"""Tests for MetricsWriter and NullMetricsWriter."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from titans.observability.metrics_writer import (
    MetricsWriter,
    NullMetricsWriter,
    build_metrics_writer,
)


def test_metrics_writer_writes_jsonl(tmp_path: Path) -> None:
    """Rows written via log() should round-trip through json.loads."""
    jsonl_path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(jsonl_path)
    writer.log(step=1, loss=3.14, lr=1e-4)
    writer.log(step=2, loss=2.71, lr=1e-4)
    writer.close()

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["step"] == 1
    assert row0["loss"] == pytest.approx(3.14)
    assert row0["lr"] == pytest.approx(1e-4)
    assert "wall_time" in row0


def test_metrics_writer_supports_namespaced_keys(tmp_path: Path) -> None:
    """Feature modules add keys with '/' namespace; writer preserves them."""
    jsonl_path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(jsonl_path)
    writer.log(step=10, **{"grad/global_norm": 1.82, "eval/loss": 2.61})
    writer.close()

    row = json.loads(jsonl_path.read_text().strip())
    assert row["grad/global_norm"] == pytest.approx(1.82)
    assert row["eval/loss"] == pytest.approx(2.61)


def test_null_writer_log_and_close_are_noop() -> None:
    """NullMetricsWriter.log and close must be silent no-ops."""
    writer = NullMetricsWriter()
    writer.log(step=1, loss=3.14)
    writer.close()  # must not raise


def test_null_writer_tqdm_summary_still_updates_postfix() -> None:
    """Disabling JSONL must not silence the tqdm postfix.

    Regression test: an earlier version of ``NullMetricsWriter`` treated
    ``tqdm_summary`` as a no-op, which meant that the documented "empty
    --metrics-jsonl disables JSONL and keeps tqdm-only logging"
    behaviour was broken -- training under defaults produced a raw
    progress meter with no curated keys. The Null writer must project
    the same curated subset as the real writer.
    """
    writer = NullMetricsWriter()
    pbar = MagicMock()

    writer.tqdm_summary(
        pbar,
        step=100,
        loss=2.57,
        lr=2.8e-4,
        **{"grad/global_norm": 1.82},
    )

    pbar.set_postfix.assert_called_once()
    # The curated short labels must include at least 'loss', 'lr', '|g|'.
    call_kwargs = pbar.set_postfix.call_args.kwargs
    assert "loss" in call_kwargs
    assert "lr" in call_kwargs
    assert "|g|" in call_kwargs


def test_build_returns_null_when_path_empty(tmp_path: Path) -> None:
    """Empty jsonl_path disables JSONL; factory returns NullMetricsWriter."""
    accelerator = MagicMock()
    accelerator.is_main_process = True

    writer = build_metrics_writer("", accelerator)
    assert isinstance(writer, NullMetricsWriter)


def test_build_returns_null_on_non_main_process(tmp_path: Path) -> None:
    """Non-main-process ranks must receive NullMetricsWriter."""
    accelerator = MagicMock()
    accelerator.is_main_process = False

    writer = build_metrics_writer(str(tmp_path / "m.jsonl"), accelerator)
    assert isinstance(writer, NullMetricsWriter)


def test_build_returns_real_writer_on_main(tmp_path: Path) -> None:
    """Main process with a path must get a real writer."""
    accelerator = MagicMock()
    accelerator.is_main_process = True
    jsonl_path = tmp_path / "m.jsonl"

    writer = build_metrics_writer(str(jsonl_path), accelerator)
    assert isinstance(writer, MetricsWriter)
    writer.close()


def test_tqdm_summary_updates_postfix(tmp_path: Path) -> None:
    """tqdm_summary should call set_postfix on the pbar with curated keys."""
    jsonl_path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(jsonl_path)
    pbar = MagicMock()

    writer.tqdm_summary(
        pbar,
        step=100,
        loss=2.57,
        lr=2.8e-4,
        **{"grad/global_norm": 1.82},
    )

    pbar.set_postfix.assert_called_once()
    writer.close()


def test_failure_to_open_falls_back_to_null(tmp_path: Path) -> None:
    """If the path cannot be opened, factory emits a warning and returns null."""
    accelerator = MagicMock()
    accelerator.is_main_process = True
    # A path whose parent cannot be created (a file used as parent dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad_path = str(blocker / "sub" / "metrics.jsonl")

    writer = build_metrics_writer(bad_path, accelerator)
    assert isinstance(writer, NullMetricsWriter)
