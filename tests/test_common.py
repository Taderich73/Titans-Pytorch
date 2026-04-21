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

from scripts._common import (  # noqa: E402
    make_dataloader,
    make_optimizer,
    maybe_compile,
)


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


def test_maybe_compile_attn_res_no_longer_auto_disables(caplog) -> None:
    """``use_attn_res=True`` must NOT emit the legacy auto-disable warning.

    The AttnRes sub-layer schedule is now compile-compatible
    (see ``src/titans/models.py::_build_attnres_schedule``), so the old
    warning + skip behavior has been removed. We still return ``model``
    unchanged on CPU because ``torch.compile`` only activates on CUDA.
    """
    model = torch.nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="scripts._common"):
        out = maybe_compile(
            model, enabled=True, device_type="cpu", use_attn_res=True
        )
    assert out is model  # CPU path is a no-op
    assert not any("attn_res" in r.message.lower() for r in caplog.records)


def test_make_optimizer_cpu_defaults_to_foreach() -> None:
    """On CPU, fused=True must NOT be requested."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = make_optimizer(params, lr=1e-3, weight_decay=0.1, device_type="cpu")
    assert isinstance(opt, torch.optim.AdamW)
    # foreach default is True on CPU; fused must NOT be set.
    assert opt.defaults.get("fused") in (False, None)


def test_make_optimizer_cuda_uses_fused() -> None:
    """When device_type=='cuda', fused=True is requested."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    # The _force_fused_flag test hook bypasses torch.cuda.is_available()
    # so we can verify the fused kwarg is passed through on CPU runners.
    opt = make_optimizer(
        params,
        lr=1e-3,
        weight_decay=0.1,
        device_type="cuda",
        _force_fused_flag=True,
    )
    assert opt.defaults.get("fused") is True


def test_make_dataloader_multiworker_flags_cuda() -> None:
    """Multi-worker DataLoader enables pin_memory and persistent_workers on CUDA."""
    from torch.utils.data import TensorDataset

    ds = TensorDataset(torch.arange(128))
    dl = make_dataloader(
        ds,
        batch_size=8,
        num_workers=4,
        device_type="cuda",
        shuffle=True,
    )
    assert dl.num_workers == 4
    assert dl.pin_memory is True
    assert dl.persistent_workers is True


def test_make_dataloader_streaming_forces_zero_workers() -> None:
    """Streaming datasets force num_workers=0 (unsafe with iterable)."""

    class FakeStream:  # no __len__, no __getitem__
        def __iter__(self):
            yield torch.zeros(4)

    ds = FakeStream()
    dl = make_dataloader(
        ds,
        batch_size=1,
        num_workers=4,
        device_type="cuda",
        shuffle=False,
        streaming=True,
    )
    assert dl.num_workers == 0
    assert dl.pin_memory is True  # pin_memory still OK without workers


def test_make_dataloader_zero_workers_no_persistent() -> None:
    """num_workers=0 must not set persistent_workers/prefetch_factor."""
    from torch.utils.data import TensorDataset

    ds = TensorDataset(torch.arange(32))
    dl = make_dataloader(
        ds,
        batch_size=4,
        num_workers=0,
        device_type="cpu",
        shuffle=False,
    )
    assert dl.num_workers == 0
    assert dl.pin_memory is False
    # persistent_workers default is False when num_workers=0.
    assert dl.persistent_workers is False
