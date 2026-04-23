"""Tests for collect_layer_stats and LayerStats."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from titans.observability.layer_stats import LayerStats, collect_layer_stats


def _make_fake_state(weight_tensor: torch.Tensor) -> SimpleNamespace:
    """Build a non-TNT-style fake state with .weights attribute."""
    return SimpleNamespace(weights=[weight_tensor])


def _make_fake_block(inner_weight: torch.Tensor) -> SimpleNamespace:
    """Build a non-TNT-style fake block: block.memory.memory.layers[0].weight."""
    layer = nn.Linear(inner_weight.shape[1], inner_weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(inner_weight)
    memory_inner = SimpleNamespace(layers=[layer])
    return SimpleNamespace(memory=SimpleNamespace(memory=memory_inner))


def test_collect_four_blocks_non_tnt() -> None:
    """Four-block non-TNT model yields 4 state_norms and 4 weight_norms."""
    states = [
        _make_fake_state(torch.full((2, 2), 1.0)),  # norm = 2.0
        _make_fake_state(torch.full((2, 2), 2.0)),  # norm = 4.0
        _make_fake_state(torch.full((2, 2), 3.0)),  # norm = 6.0
        _make_fake_state(torch.full((2, 2), 4.0)),  # norm = 8.0
    ]
    blocks = [
        _make_fake_block(torch.full((2, 2), 1.0)),
        _make_fake_block(torch.full((2, 2), 2.0)),
        _make_fake_block(torch.full((2, 2), 3.0)),
        _make_fake_block(torch.full((2, 2), 4.0)),
    ]
    model = SimpleNamespace(blocks=blocks)

    stats = collect_layer_stats(model, states)

    assert stats.state_norms == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert stats.weight_norms == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_state_mean_std_min_max() -> None:
    """Aggregate properties compute correctly on known input."""
    stats = LayerStats(
        state_norms=[1.0, 3.0, 5.0, 7.0],
        weight_norms=[10.0, 20.0, 30.0, 40.0],
    )
    assert stats.state_mean == pytest.approx(4.0)
    assert stats.state_min == 1.0
    assert stats.state_max == 7.0
    # std (population or sample — implementation defines; assert > 0)
    assert stats.state_std > 0

    assert stats.weight_mean == pytest.approx(25.0)
    assert stats.weight_min == 10.0
    assert stats.weight_max == 40.0


def test_state_none_is_handled() -> None:
    """A block with state=None must not crash; corresponding norm is None."""
    states = [
        _make_fake_state(torch.ones(2, 2)),
        None,
        _make_fake_state(torch.ones(2, 2) * 2.0),
    ]
    blocks = [
        _make_fake_block(torch.ones(2, 2)),
        _make_fake_block(torch.ones(2, 2)),
        _make_fake_block(torch.ones(2, 2)),
    ]
    model = SimpleNamespace(blocks=blocks)

    stats = collect_layer_stats(model, states)

    assert stats.state_norms[0] is not None
    assert stats.state_norms[1] is None
    assert stats.state_norms[2] is not None


def test_tnt_path_via_global_memory() -> None:
    """TNT-style block: block.memory.global_memory.memory.layers[0].weight."""
    layer = nn.Linear(3, 3)
    with torch.no_grad():
        layer.weight.fill_(2.0)  # norm = sqrt(9 * 4) = 6.0
    tnt_inner = SimpleNamespace(layers=[layer])
    tnt_global = SimpleNamespace(memory=tnt_inner)
    tnt_block = SimpleNamespace(
        memory=SimpleNamespace(global_memory=tnt_global, memory=None)
    )
    model = SimpleNamespace(blocks=[tnt_block])

    # TNT-style state: state.global_state.weights[0]
    tnt_state = SimpleNamespace(
        global_state=SimpleNamespace(weights=[torch.full((2, 2), 3.0)]),  # norm = 6.0
        weights=None,
    )

    stats = collect_layer_stats(model, [tnt_state])

    assert stats.state_norms[0] == pytest.approx(6.0)
    assert stats.weight_norms[0] == pytest.approx(6.0)


def test_to_dict_contains_all_keys() -> None:
    """to_dict() returns the JSONL-ready payload with expected keys."""
    stats = LayerStats(
        state_norms=[1.0, 2.0, 3.0],
        weight_norms=[10.0, 20.0, 30.0],
    )
    d = stats.to_dict()
    assert "layer/state_norm_mean" in d
    assert "layer/state_norm_std" in d
    assert "layer/state_norm_min" in d
    assert "layer/state_norm_max" in d
    assert "layer/state_norm_per_block" in d
    assert "layer/weight_norm_mean" in d
    assert "layer/weight_norm_std" in d
    assert "layer/weight_norm_per_block" in d
    assert d["layer/state_norm_per_block"] == [1.0, 2.0, 3.0]


def test_to_dict_skips_none_entries_in_aggregates() -> None:
    """None state norms do not pollute mean/std aggregates."""
    stats = LayerStats(
        state_norms=[1.0, None, 3.0],
        weight_norms=[10.0, 20.0, 30.0],
    )
    d = stats.to_dict()
    assert d["layer/state_norm_mean"] == pytest.approx(2.0)  # (1+3)/2
    # Per-block list retains the None to preserve positional meaning
    assert d["layer/state_norm_per_block"] == [1.0, None, 3.0]
