# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for checkpoint_signals — pure signal extraction functions."""

from __future__ import annotations


import pytest
import torch

from titans.memory import MemoryState, TNTMemoryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_state(
    n_layers: int,
    dim: int,
    device: torch.device,
    fill_value: float | None = None,
) -> MemoryState:
    """Create a MemoryState with deterministic or random tensors."""
    if fill_value is not None:
        weights = [
            torch.full((dim, dim), fill_value, device=device) for _ in range(n_layers)
        ]
        momentum = [
            torch.full((dim, dim), fill_value, device=device) for _ in range(n_layers)
        ]
    else:
        torch.manual_seed(42)
        weights = [torch.randn(dim, dim, device=device) for _ in range(n_layers)]
        momentum = [torch.randn(dim, dim, device=device) for _ in range(n_layers)]
    return MemoryState(weights=weights, momentum=momentum)


def _make_tnt_state(
    n_layers: int,
    n_local: int,
    dim: int,
    device: torch.device,
    fill_value: float | None = None,
) -> TNTMemoryState:
    """Create a TNTMemoryState with deterministic or random tensors."""
    global_state = _make_memory_state(n_layers, dim, device, fill_value=fill_value)
    local_states = [
        _make_memory_state(n_layers, dim, device, fill_value=fill_value)
        for _ in range(n_local)
    ]
    return TNTMemoryState(
        global_state=global_state,
        local_states=local_states,
        qk_projections=[torch.eye(dim, device=device) for _ in range(n_local)],
        local_step_counters=[0] * n_local,
    )


def _make_gate_snapshot(n_layers: int, device: torch.device) -> object:
    from titans.checkpointing.types import GateSnapshot

    return GateSnapshot(
        alpha=[torch.tensor(0.9, device=device) for _ in range(n_layers)],
        theta=[torch.tensor(0.1, device=device) for _ in range(n_layers)],
        eta=[torch.tensor(0.85, device=device) for _ in range(n_layers)],
        delta=None,
        input_activation_norm=1.0,
        chunk_index=0,
    )


def _frobenius(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.float(), ord="fro").item())


# ---------------------------------------------------------------------------
# TestComputeWeightDelta
# ---------------------------------------------------------------------------


class TestComputeWeightDelta:
    """Tests for compute_weight_delta()."""

    def test_identical_states_delta_near_zero(self, device: torch.device) -> None:
        """Identical old and new states produce delta norms close to zero."""
        from titans.checkpointing.signals import compute_weight_delta

        state = _make_memory_state(2, 8, device)
        deltas = compute_weight_delta(state, state)

        assert isinstance(deltas, list)
        assert len(deltas) == 2
        for d in deltas:
            assert isinstance(d, float)
            assert d == pytest.approx(0.0, abs=1e-6)

    def test_different_states_delta_positive(self, device: torch.device) -> None:
        """Different old and new states produce positive delta norms."""
        from titans.checkpointing.signals import compute_weight_delta

        old_state = _make_memory_state(1, 8, device, fill_value=0.0)
        new_state = _make_memory_state(1, 8, device, fill_value=1.0)
        deltas = compute_weight_delta(old_state, new_state)

        assert len(deltas) == 1
        assert deltas[0] > 0.0

    def test_delta_matches_frobenius_norm(self, device: torch.device) -> None:
        """Delta norm equals ||W_new - W_old||_F for a known pair."""
        from titans.checkpointing.signals import compute_weight_delta

        dim = 4
        w_old = torch.zeros(dim, dim, device=device)
        w_new = torch.ones(dim, dim, device=device)
        old_state = MemoryState(weights=[w_old], momentum=[torch.zeros_like(w_old)])
        new_state = MemoryState(weights=[w_new], momentum=[torch.zeros_like(w_new)])

        expected = _frobenius(w_new - w_old)
        deltas = compute_weight_delta(old_state, new_state)

        assert len(deltas) == 1
        assert deltas[0] == pytest.approx(expected, rel=1e-5)

    def test_multi_layer_states(self, device: torch.device) -> None:
        """Returns one delta per layer for multi-layer states."""
        from titans.checkpointing.signals import compute_weight_delta

        n_layers = 3
        old_state = _make_memory_state(n_layers, 8, device, fill_value=0.0)
        new_state = _make_memory_state(n_layers, 8, device, fill_value=2.0)
        deltas = compute_weight_delta(old_state, new_state)

        assert len(deltas) == n_layers
        for d in deltas:
            assert d > 0.0

    def test_tnt_state_uses_global_weights(self, device: torch.device) -> None:
        """For TNTMemoryState, uses global_state.weights."""
        from titans.checkpointing.signals import compute_weight_delta

        dim = 4
        w_old = torch.zeros(dim, dim, device=device)
        w_new = torch.ones(dim, dim, device=device)

        old_tnt = _make_tnt_state(1, 1, dim, device, fill_value=0.0)
        new_tnt = _make_tnt_state(1, 1, dim, device, fill_value=1.0)

        expected = _frobenius(w_new - w_old)
        deltas = compute_weight_delta(old_tnt, new_tnt)

        assert len(deltas) == 1
        assert deltas[0] == pytest.approx(expected, rel=1e-5)

    def test_return_type_is_list_of_float(self, device: torch.device) -> None:
        """Return value is a list of Python floats (not tensors)."""
        from titans.checkpointing.signals import compute_weight_delta

        state = _make_memory_state(2, 4, device)
        deltas = compute_weight_delta(state, state)

        assert all(isinstance(v, float) for v in deltas)


# ---------------------------------------------------------------------------
# TestComputeMomentumShift
# ---------------------------------------------------------------------------


class TestComputeMomentumShift:
    """Tests for compute_momentum_shift()."""

    def test_identical_momentum_near_zero(self, device: torch.device) -> None:
        """Identical states produce momentum shift norms close to zero."""
        from titans.checkpointing.signals import compute_momentum_shift

        state = _make_memory_state(2, 8, device)
        shifts = compute_momentum_shift(state, state)

        assert isinstance(shifts, list)
        assert len(shifts) == 2
        for s in shifts:
            assert s == pytest.approx(0.0, abs=1e-6)

    def test_different_momentum_positive(self, device: torch.device) -> None:
        """Different momentum states produce positive shift norms."""
        from titans.checkpointing.signals import compute_momentum_shift

        old_state = MemoryState(
            weights=[torch.zeros(4, 4, device=device)],
            momentum=[torch.zeros(4, 4, device=device)],
        )
        new_state = MemoryState(
            weights=[torch.zeros(4, 4, device=device)],
            momentum=[torch.ones(4, 4, device=device)],
        )
        shifts = compute_momentum_shift(old_state, new_state)

        assert len(shifts) == 1
        assert shifts[0] > 0.0

    def test_shift_matches_frobenius_norm(self, device: torch.device) -> None:
        """Shift norm equals ||m_new - m_old||_F for a known pair."""
        from titans.checkpointing.signals import compute_momentum_shift

        dim = 4
        m_old = torch.zeros(dim, dim, device=device)
        m_new = torch.full((dim, dim), 3.0, device=device)
        old_state = MemoryState(weights=[torch.zeros_like(m_old)], momentum=[m_old])
        new_state = MemoryState(weights=[torch.zeros_like(m_new)], momentum=[m_new])

        expected = _frobenius(m_new - m_old)
        shifts = compute_momentum_shift(old_state, new_state)

        assert shifts[0] == pytest.approx(expected, rel=1e-5)

    def test_tnt_state_uses_global_momentum(self, device: torch.device) -> None:
        """For TNTMemoryState, uses global_state.momentum."""
        from titans.checkpointing.signals import compute_momentum_shift

        old_tnt = _make_tnt_state(1, 1, 4, device, fill_value=0.0)
        new_tnt = _make_tnt_state(1, 1, 4, device, fill_value=2.0)

        shifts = compute_momentum_shift(old_tnt, new_tnt)
        assert len(shifts) == 1
        assert shifts[0] > 0.0

    def test_return_type_is_list_of_float(self, device: torch.device) -> None:
        """Return value is a list of Python floats."""
        from titans.checkpointing.signals import compute_momentum_shift

        state = _make_memory_state(2, 4, device)
        shifts = compute_momentum_shift(state, state)

        assert all(isinstance(v, float) for v in shifts)


# ---------------------------------------------------------------------------
# TestComputeNorms
# ---------------------------------------------------------------------------


class TestComputeNorms:
    """Tests for compute_weight_norms() and compute_momentum_norms()."""

    def test_weight_norms_match_frobenius(self, device: torch.device) -> None:
        """compute_weight_norms() matches torch.linalg.norm(..., 'fro')."""
        from titans.checkpointing.signals import compute_weight_norms

        dim = 4
        w = torch.full((dim, dim), 2.0, device=device)
        state = MemoryState(weights=[w], momentum=[torch.zeros_like(w)])
        norms = compute_weight_norms(state)

        expected = _frobenius(w)
        assert len(norms) == 1
        assert norms[0] == pytest.approx(expected, rel=1e-5)

    def test_momentum_norms_match_frobenius(self, device: torch.device) -> None:
        """compute_momentum_norms() matches torch.linalg.norm(..., 'fro')."""
        from titans.checkpointing.signals import compute_momentum_norms

        dim = 4
        m = torch.full((dim, dim), 3.0, device=device)
        state = MemoryState(weights=[torch.zeros_like(m)], momentum=[m])
        norms = compute_momentum_norms(state)

        expected = _frobenius(m)
        assert len(norms) == 1
        assert norms[0] == pytest.approx(expected, rel=1e-5)

    def test_zero_weights_give_zero_norm(self, device: torch.device) -> None:
        """Zero weight tensors produce zero norms."""
        from titans.checkpointing.signals import compute_weight_norms

        state = MemoryState(
            weights=[torch.zeros(4, 4, device=device)],
            momentum=[torch.zeros(4, 4, device=device)],
        )
        norms = compute_weight_norms(state)
        assert norms[0] == pytest.approx(0.0, abs=1e-8)

    def test_multi_layer_norms_length(self, device: torch.device) -> None:
        """Returns one norm per layer."""
        from titans.checkpointing.signals import (
            compute_weight_norms,
            compute_momentum_norms,
        )

        n_layers = 4
        state = _make_memory_state(n_layers, 8, device)
        weight_norms = compute_weight_norms(state)
        momentum_norms = compute_momentum_norms(state)

        assert len(weight_norms) == n_layers
        assert len(momentum_norms) == n_layers

    def test_weight_norms_tnt_uses_global(self, device: torch.device) -> None:
        """compute_weight_norms() on TNTMemoryState uses global_state.weights."""
        from titans.checkpointing.signals import compute_weight_norms

        tnt = _make_tnt_state(2, 1, 4, device, fill_value=1.0)
        plain = _make_memory_state(2, 4, device, fill_value=1.0)

        tnt_norms = compute_weight_norms(tnt)
        plain_norms = compute_weight_norms(plain)

        assert tnt_norms == pytest.approx(plain_norms, rel=1e-5)

    def test_momentum_norms_tnt_uses_global(self, device: torch.device) -> None:
        """compute_momentum_norms() on TNTMemoryState uses global_state.momentum."""
        from titans.checkpointing.signals import compute_momentum_norms

        tnt = _make_tnt_state(1, 1, 4, device, fill_value=2.0)
        plain = _make_memory_state(1, 4, device, fill_value=2.0)

        tnt_norms = compute_momentum_norms(tnt)
        plain_norms = compute_momentum_norms(plain)

        assert tnt_norms == pytest.approx(plain_norms, rel=1e-5)

    def test_norms_are_non_negative(self, device: torch.device) -> None:
        """Frobenius norms are always non-negative."""
        from titans.checkpointing.signals import (
            compute_weight_norms,
            compute_momentum_norms,
        )

        state = _make_memory_state(3, 8, device)
        for norm in compute_weight_norms(state) + compute_momentum_norms(state):
            assert norm >= 0.0

    def test_return_type_is_list_of_float(self, device: torch.device) -> None:
        """Return values are lists of Python floats."""
        from titans.checkpointing.signals import (
            compute_weight_norms,
            compute_momentum_norms,
        )

        state = _make_memory_state(2, 4, device)
        assert all(isinstance(v, float) for v in compute_weight_norms(state))
        assert all(isinstance(v, float) for v in compute_momentum_norms(state))


# ---------------------------------------------------------------------------
# TestBuildSignalFrame
# ---------------------------------------------------------------------------


class TestBuildSignalFrame:
    """Tests for build_signal_frame()."""

    def test_basic_frame_from_memory_state(self, device: torch.device) -> None:
        """build_signal_frame() returns a SignalFrame for a MemoryState."""
        from titans.checkpointing.signals import build_signal_frame
        from titans.checkpointing.types import SignalFrame

        n_layers = 2
        old_state = _make_memory_state(n_layers, 8, device, fill_value=0.0)
        new_state = _make_memory_state(n_layers, 8, device, fill_value=1.0)
        gates = _make_gate_snapshot(n_layers, device)

        frame = build_signal_frame(old_state, new_state, gates, chunk_index=5)

        assert isinstance(frame, SignalFrame)

    def test_chunk_index_stored_correctly(self, device: torch.device) -> None:
        """chunk_index is propagated to the SignalFrame."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(1, 4, device)
        gates = _make_gate_snapshot(1, device)

        frame = build_signal_frame(state, state, gates, chunk_index=42)
        assert frame.chunk_index == 42

    def test_tnt_fields_none_for_memory_state(self, device: torch.device) -> None:
        """TNT-only fields are None when state is MemoryState."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(state, state, gates, chunk_index=0)

        assert frame.local_signal_norms is None

    def test_all_list_fields_have_correct_length(self, device: torch.device) -> None:
        """All per-layer list fields have length == n_layers."""
        from titans.checkpointing.signals import build_signal_frame

        n_layers = 3
        old_state = _make_memory_state(n_layers, 8, device, fill_value=0.0)
        new_state = _make_memory_state(n_layers, 8, device, fill_value=1.0)
        gates = _make_gate_snapshot(n_layers, device)

        frame = build_signal_frame(old_state, new_state, gates, chunk_index=0)

        assert len(frame.weight_delta_norms) == n_layers
        assert len(frame.momentum_shift_norms) == n_layers
        assert len(frame.weight_norms) == n_layers
        assert len(frame.momentum_norms) == n_layers
        assert len(frame.gate_alpha_means) == n_layers
        assert len(frame.gate_theta_means) == n_layers
        assert len(frame.gate_eta_means) == n_layers

    def test_weight_delta_nonzero_when_states_differ(
        self, device: torch.device
    ) -> None:
        """weight_delta_norms > 0 when old and new states differ."""
        from titans.checkpointing.signals import build_signal_frame

        old_state = _make_memory_state(1, 4, device, fill_value=0.0)
        new_state = _make_memory_state(1, 4, device, fill_value=1.0)
        gates = _make_gate_snapshot(1, device)

        frame = build_signal_frame(old_state, new_state, gates, chunk_index=0)
        assert frame.weight_delta_norms[0] > 0.0

    def test_prediction_error_norms_default_to_zeros(
        self, device: torch.device
    ) -> None:
        """prediction_error_norms defaults to a list of zeros."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(state, state, gates, chunk_index=0)

        assert frame.prediction_error_norms == [0.0, 0.0]

    def test_gradient_norms_default_to_zeros(self, device: torch.device) -> None:
        """gradient_norms defaults to a list of zeros."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(state, state, gates, chunk_index=0)

        assert frame.gradient_norms == [0.0, 0.0]

    def test_optional_prediction_error_norms_accepted(
        self, device: torch.device
    ) -> None:
        """Passing prediction_error_norms stores them in the frame."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(
            state, state, gates, chunk_index=0, prediction_error_norms=[0.5, 0.7]
        )

        assert frame.prediction_error_norms == pytest.approx([0.5, 0.7])

    def test_optional_gradient_norms_accepted(self, device: torch.device) -> None:
        """Passing gradient_norms stores them in the frame."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(
            state, state, gates, chunk_index=0, gradient_norms=[1.1, 2.2]
        )

        assert frame.gradient_norms == pytest.approx([1.1, 2.2])

    def test_batch_variance_stored(self, device: torch.device) -> None:
        """batch_variance is passed through to the frame."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(1, 4, device)
        gates = _make_gate_snapshot(1, device)
        frame = build_signal_frame(
            state, state, gates, chunk_index=0, batch_variance=0.042
        )

        assert frame.batch_variance == pytest.approx(0.042)

    def test_batch_variance_defaults_to_none(self, device: torch.device) -> None:
        """batch_variance defaults to None when not provided."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(1, 4, device)
        gates = _make_gate_snapshot(1, device)
        frame = build_signal_frame(state, state, gates, chunk_index=0)

        assert frame.batch_variance is None

    def test_gate_means_are_scalar_floats(self, device: torch.device) -> None:
        """Gate mean fields are lists of scalar Python floats."""
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(state, state, gates, chunk_index=0)

        for val in (
            frame.gate_alpha_means + frame.gate_theta_means + frame.gate_eta_means
        ):
            assert isinstance(val, float)

    def test_tnt_state_populates_tnt_fields(self, device: torch.device) -> None:
        """build_signal_frame() on TNTMemoryState populates TNT-only fields."""
        from titans.checkpointing.signals import build_signal_frame

        n_layers = 1
        n_local = 2
        dim = 4
        old_tnt = _make_tnt_state(n_layers, n_local, dim, device, fill_value=0.0)
        new_tnt = _make_tnt_state(n_layers, n_local, dim, device, fill_value=1.0)
        gates = _make_gate_snapshot(n_layers, device)

        frame = build_signal_frame(old_tnt, new_tnt, gates, chunk_index=3)

        assert frame.local_signal_norms is not None
        assert len(frame.local_signal_norms) == n_local

    def test_tnt_local_signal_norms_shape(self, device: torch.device) -> None:
        """local_signal_norms outer dim equals n_local, inner equals n_layers."""
        from titans.checkpointing.signals import build_signal_frame

        n_layers = 2
        n_local = 3
        dim = 4
        old_tnt = _make_tnt_state(n_layers, n_local, dim, device, fill_value=0.0)
        new_tnt = _make_tnt_state(n_layers, n_local, dim, device, fill_value=1.0)
        gates = _make_gate_snapshot(n_layers, device)

        frame = build_signal_frame(old_tnt, new_tnt, gates, chunk_index=0)

        assert len(frame.local_signal_norms) == n_local
        for inner in frame.local_signal_norms:
            assert len(inner) == n_layers

    def test_frame_to_dict_is_serializable(self, device: torch.device) -> None:
        """Frame produced by build_signal_frame() survives JSON round-trip."""
        import json
        from titans.checkpointing.signals import build_signal_frame

        state = _make_memory_state(2, 4, device)
        gates = _make_gate_snapshot(2, device)
        frame = build_signal_frame(state, state, gates, chunk_index=1)

        d = frame.to_dict()
        serialized = json.dumps(d)
        recovered = json.loads(serialized)
        assert recovered["chunk_index"] == 1


def test_signal_frame_no_duplicate_tnt_weight_norms():
    """Regression: global_signal_norms was populated with exactly the same
    values as weight_norms on the TNT path. Either remove the duplicate field
    or populate with distinct semantics. We choose removal."""
    from titans.checkpointing.types import SignalFrame
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(SignalFrame)}
    assert "global_signal_norms" not in field_names, (
        "SignalFrame still has global_signal_norms — remove the duplicate "
        "of weight_norms"
    )


def test_signal_frame_no_local_reset_flags_field():
    """local_reset_flags was hard-coded to [False] * N, making the consumer
    branch in memory_checkpointer unreachable. Remove the field."""
    from titans.checkpointing.types import SignalFrame
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(SignalFrame)}
    assert "local_reset_flags" not in field_names, (
        "SignalFrame still has dead local_reset_flags field"
    )


def test_build_signal_frame_single_cpu_sync(monkeypatch) -> None:
    """All per-tensor .item() calls must be batched into <= 1 CPU sync.

    The old implementation issued one ``.item()`` per Frobenius norm per layer
    (weight_delta, momentum_shift, weight, momentum) plus one per gate mean,
    i.e. roughly ``4 * L + 3 * L`` syncs per frame for a plain MemoryState.
    After batching we should collapse that into ~1 sync via ``.tolist()`` on
    a single stacked tensor. Budget is generous (<= 4) to accommodate a few
    small scalar tolists without regressing the core contract.
    """
    from titans.checkpointing.signals import build_signal_frame
    from titans.checkpointing.types import GateSnapshot

    item_calls = {"count": 0}
    tolist_calls = {"count": 0}
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist

    def counting_item(self):  # type: ignore[no-untyped-def]
        item_calls["count"] += 1
        return orig_item(self)

    def counting_tolist(self):  # type: ignore[no-untyped-def]
        tolist_calls["count"] += 1
        return orig_tolist(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)

    n_layers = 5
    torch.manual_seed(0)
    old = MemoryState(
        weights=[torch.randn(8, 8) for _ in range(n_layers)],
        momentum=[torch.randn(8, 8) for _ in range(n_layers)],
    )
    new = MemoryState(
        weights=[torch.randn(8, 8) for _ in range(n_layers)],
        momentum=[torch.randn(8, 8) for _ in range(n_layers)],
    )
    gates = GateSnapshot(
        alpha=[torch.randn(1) for _ in range(n_layers)],
        theta=[torch.randn(1) for _ in range(n_layers)],
        eta=[torch.randn(1) for _ in range(n_layers)],
        delta=None,
        input_activation_norm=0.0,
        chunk_index=0,
    )
    _ = build_signal_frame(old, new, gates, chunk_index=0)
    # Budget: at most 4 scalar .item() calls and a small handful of .tolist()s
    # combined total (prior impl was ~35).
    total_syncs = item_calls["count"] + tolist_calls["count"]
    assert total_syncs <= 4, (
        f"Expected <= 4 CPU syncs, got .item()={item_calls['count']} "
        f".tolist()={tolist_calls['count']}"
    )


def test_build_signal_frame_numeric_parity_after_batching() -> None:
    """Batched implementation must match element-wise the naive per-tensor one."""
    from titans.checkpointing.signals import (
        build_signal_frame,
        compute_momentum_norms,
        compute_momentum_shift,
        compute_weight_delta,
        compute_weight_norms,
    )
    from titans.checkpointing.types import GateSnapshot

    n_layers = 3
    torch.manual_seed(17)
    old = MemoryState(
        weights=[torch.randn(6, 6) for _ in range(n_layers)],
        momentum=[torch.randn(6, 6) for _ in range(n_layers)],
    )
    new = MemoryState(
        weights=[torch.randn(6, 6) for _ in range(n_layers)],
        momentum=[torch.randn(6, 6) for _ in range(n_layers)],
    )
    gates = GateSnapshot(
        alpha=[torch.tensor([0.3, 0.4]) for _ in range(n_layers)],
        theta=[torch.tensor([0.1, 0.2]) for _ in range(n_layers)],
        eta=[torch.tensor([0.5, 0.6]) for _ in range(n_layers)],
        delta=None,
        input_activation_norm=0.0,
        chunk_index=0,
    )
    frame = build_signal_frame(old, new, gates, chunk_index=0)

    assert frame.weight_delta_norms == pytest.approx(
        compute_weight_delta(old, new), rel=1e-6
    )
    assert frame.momentum_shift_norms == pytest.approx(
        compute_momentum_shift(old, new), rel=1e-6
    )
    assert frame.weight_norms == pytest.approx(compute_weight_norms(new), rel=1e-6)
    assert frame.momentum_norms == pytest.approx(compute_momentum_norms(new), rel=1e-6)
    assert frame.gate_alpha_means == pytest.approx(
        [float(t.float().mean().item()) for t in gates.alpha], rel=1e-6
    )
    assert frame.gate_theta_means == pytest.approx(
        [float(t.float().mean().item()) for t in gates.theta], rel=1e-6
    )
    assert frame.gate_eta_means == pytest.approx(
        [float(t.float().mean().item()) for t in gates.eta], rel=1e-6
    )


def test_build_signal_frame_tnt_parity_and_batching(monkeypatch) -> None:
    """TNT path must also be batched and numerically equivalent."""
    from titans.checkpointing.signals import build_signal_frame, compute_weight_norms
    from titans.checkpointing.types import GateSnapshot

    n_layers = 3
    n_local = 2
    dim = 6
    torch.manual_seed(5)
    old = _make_tnt_state(n_layers, n_local, dim, torch.device("cpu"))
    new = _make_tnt_state(n_layers, n_local, dim, torch.device("cpu"))
    gates = GateSnapshot(
        alpha=[torch.tensor([0.2, 0.3]) for _ in range(n_layers)],
        theta=[torch.tensor([0.1, 0.2]) for _ in range(n_layers)],
        eta=[torch.tensor([0.5, 0.6]) for _ in range(n_layers)],
        delta=None,
        input_activation_norm=0.0,
        chunk_index=0,
    )

    item_calls = {"count": 0}
    orig_item = torch.Tensor.item

    def counting_item(self):  # type: ignore[no-untyped-def]
        item_calls["count"] += 1
        return orig_item(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)

    frame = build_signal_frame(old, new, gates, chunk_index=0)

    # With n_layers=3 + 2 locals * 3 layers, the old impl did ~21 .item()s.
    # Post-batch we expect 0-3 direct .item() calls (all routed through tolist()).
    assert item_calls["count"] <= 4, f".item() count too high: {item_calls['count']}"

    # Local signal norms parity
    assert frame.local_signal_norms is not None
    assert len(frame.local_signal_norms) == n_local
    for i, local_state in enumerate(new.local_states):
        assert frame.local_signal_norms[i] == pytest.approx(
            compute_weight_norms(local_state), rel=1e-6
        )


def test_memory_checkpointer_docstring_lists_three_states() -> None:
    """The module docstring must not reference the removed TRIGGERED state."""
    import pathlib

    src = pathlib.Path("src/titans/memory_checkpointer.py").read_text()
    # Isolate the module docstring (first triple-quoted block).
    head = src.split('"""', 2)[1]
    assert "TRIGGERED" not in head, (
        "Plan 1 Task 12 removed CheckpointerState.TRIGGERED but the "
        "module docstring still references it. The current enum has "
        "three states: MONITORING, CAPTURING_AFTER, COOLDOWN."
    )
