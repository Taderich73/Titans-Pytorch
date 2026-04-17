# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for the ``scripts._common`` helpers that don't have dedicated files."""

from __future__ import annotations

import logging
import pathlib
import sys

import torch

# Allow `from scripts._common import ...` under pytest; the repo root is
# not on sys.path by default.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._common import maybe_compile  # noqa: E402


def test_maybe_compile_noop_on_cpu() -> None:
    """CPU device type always skips torch.compile."""
    model = torch.nn.Linear(4, 4)
    out = maybe_compile(model, enabled=True, device_type="cpu", use_attn_res=False)
    assert out is model


def test_maybe_compile_disabled_flag() -> None:
    """``enabled=False`` is a no-op regardless of device/config."""
    model = torch.nn.Linear(4, 4)
    out = maybe_compile(model, enabled=False, device_type="cuda", use_attn_res=False)
    assert out is model


def test_maybe_compile_warns_and_skips_attn_res(caplog) -> None:
    """``use_attn_res=True`` auto-disables compile and logs a warning."""
    model = torch.nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="scripts._common"):
        out = maybe_compile(
            model, enabled=True, device_type="cuda", use_attn_res=True
        )
    assert out is model
    assert any("attn_res" in r.message.lower() for r in caplog.records)
