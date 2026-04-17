"""Regression: `inference.py --save-memory foo.npz` silently did nothing when
the model returned no final states. Users saw no error, no log — just a
missing file. Require an explicit warning/error."""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

_INFERENCE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
_spec = importlib.util.spec_from_file_location("scripts_inference_mod", _INFERENCE_PATH)
_inference_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_inference_mod"] = _inference_mod
_spec.loader.exec_module(_inference_mod)


def test_handle_save_memory_logs_on_empty(caplog):
    # We can't easily invoke inference.main without a model, so test the helper
    # that the fix adds: a function `_save_final_memory(path, states, logger)`.
    _save_final_memory = _inference_mod._save_final_memory

    with caplog.at_level(logging.WARNING):
        _save_final_memory("/tmp/unused_path.npz", [], logger=logging.getLogger())
    assert any(
        "no memory" in rec.message.lower() or "empty" in rec.message.lower()
        for rec in caplog.records
    ), "Expected a warning when --save-memory is set but states is empty"
