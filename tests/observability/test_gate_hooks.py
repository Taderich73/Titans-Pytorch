"""Tests for GateHookRegistry."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from titans.observability.gate_hooks import GateHookRegistry


def _make_non_tnt_block(dim: int = 8) -> SimpleNamespace:
    """Fake non-TNT block exposing block.memory.gate_decay_proj."""
    gate_proj = nn.Linear(dim, 1)
    nn.init.constant_(gate_proj.bias, -2.0)
    return SimpleNamespace(memory=SimpleNamespace(gate_decay_proj=gate_proj))


def _make_tnt_block(dim: int = 8) -> SimpleNamespace:
    """Fake TNT block: block.memory.global_memory.memory.gate_decay_proj."""
    gate_proj = nn.Linear(dim, 1)
    nn.init.constant_(gate_proj.bias, -2.0)
    return SimpleNamespace(
        memory=SimpleNamespace(
            global_memory=SimpleNamespace(
                memory=SimpleNamespace(gate_decay_proj=gate_proj)
            ),
            gate_decay_proj=None,
        )
    )


def test_snapshot_empty_before_any_forward() -> None:
    """Calling snapshot() before any forward pass returns an empty dict."""
    model = SimpleNamespace(blocks=[_make_non_tnt_block(), _make_non_tnt_block()])
    registry = GateHookRegistry(model)

    assert registry.snapshot() == {}
    registry.remove()


def test_hooks_fire_on_forward_non_tnt() -> None:
    """After a forward on the gate proj, snapshot returns populated stats."""
    torch.manual_seed(0)
    b1, b2 = _make_non_tnt_block(8), _make_non_tnt_block(8)
    model = SimpleNamespace(blocks=[b1, b2])
    registry = GateHookRegistry(model)

    # Simulate one forward per block (the training loop would do this via
    # the real forward pass).
    x = torch.randn(4, 8)  # batch=4
    _ = b1.memory.gate_decay_proj(x)
    _ = b2.memory.gate_decay_proj(x)

    snap = registry.snapshot()
    assert "gate/alpha_mean" in snap
    assert "gate/alpha_std" in snap
    assert "gate/alpha_min" in snap
    assert "gate/alpha_max" in snap
    assert "gate/alpha_per_block" in snap
    assert len(snap["gate/alpha_per_block"]) == 2
    for v in snap["gate/alpha_per_block"]:
        assert 0.0 < v < 1.0

    registry.remove()


def test_hooks_fire_on_forward_tnt() -> None:
    """TNT path is detected and hooked correctly."""
    torch.manual_seed(0)
    b = _make_tnt_block(8)
    model = SimpleNamespace(blocks=[b])
    registry = GateHookRegistry(model)

    gate_proj = b.memory.global_memory.memory.gate_decay_proj
    _ = gate_proj(torch.randn(4, 8))

    snap = registry.snapshot()
    assert len(snap["gate/alpha_per_block"]) == 1
    assert 0.0 < snap["gate/alpha_per_block"][0] < 1.0
    registry.remove()


def test_clear_empties_stash() -> None:
    """clear() removes stashed tensors; next snapshot() returns empty."""
    b = _make_non_tnt_block(4)
    model = SimpleNamespace(blocks=[b])
    registry = GateHookRegistry(model)
    _ = b.memory.gate_decay_proj(torch.randn(2, 4))

    assert registry.snapshot() != {}
    registry.clear()
    assert registry.snapshot() == {}
    registry.remove()


def test_remove_deregisters_hooks() -> None:
    """After remove(), new forwards do not accumulate into the registry."""
    b = _make_non_tnt_block(4)
    model = SimpleNamespace(blocks=[b])
    registry = GateHookRegistry(model)
    registry.remove()

    _ = b.memory.gate_decay_proj(torch.randn(2, 4))
    assert registry.snapshot() == {}


def test_mean_is_within_range_of_per_block() -> None:
    """Mean across blocks lies between min and max of per_block values."""
    torch.manual_seed(1)
    blocks = [_make_non_tnt_block(4) for _ in range(3)]
    model = SimpleNamespace(blocks=blocks)
    registry = GateHookRegistry(model)
    for b in blocks:
        _ = b.memory.gate_decay_proj(torch.randn(4, 4))

    snap = registry.snapshot()
    m = snap["gate/alpha_mean"]
    per_block = snap["gate/alpha_per_block"]
    assert min(per_block) - 1e-6 <= m <= max(per_block) + 1e-6
    registry.remove()
