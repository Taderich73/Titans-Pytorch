"""Tests for checkpoint_types data structures."""

from __future__ import annotations

import json

import pytest
import torch

from titans.memory import MemoryState, TNTMemoryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensors(n: int, shape: tuple[int, ...], device: torch.device) -> list[torch.Tensor]:
    return [torch.randn(*shape, device=device) for _ in range(n)]


def _make_memory_state(n_layers: int, dim: int, device: torch.device) -> MemoryState:
    return MemoryState(
        weights=_make_tensors(n_layers, (dim, dim), device),
        momentum=_make_tensors(n_layers, (dim, dim), device),
    )


def _make_tnt_memory_state(
    n_layers: int, n_local: int, dim: int, device: torch.device
) -> TNTMemoryState:
    return TNTMemoryState(
        global_state=_make_memory_state(n_layers, dim, device),
        local_states=[_make_memory_state(n_layers, dim, device) for _ in range(n_local)],
        local_inits=[[torch.zeros(dim, dim, device=device)] for _ in range(n_local)],
        qk_projections=[torch.eye(dim, device=device) for _ in range(n_local)],
        local_step_counters=[0] * n_local,
    )


# ---------------------------------------------------------------------------
# GateSnapshot
# ---------------------------------------------------------------------------


class TestGateSnapshot:
    """Tests for the GateSnapshot dataclass."""

    def test_construction_without_delta(self, device: torch.device) -> None:
        """GateSnapshot can be constructed with delta=None."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.tensor(0.9, device=device)],
            theta=[torch.tensor(0.1, device=device)],
            eta=[torch.tensor(0.85, device=device)],
            delta=None,
            input_activation_norm=1.23,
            chunk_index=0,
        )
        assert snap.chunk_index == 0
        assert snap.delta is None
        assert snap.input_activation_norm == pytest.approx(1.23)

    def test_construction_with_delta(self, device: torch.device) -> None:
        """GateSnapshot can be constructed with delta set."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.tensor(0.9, device=device)],
            theta=[torch.tensor(0.1, device=device)],
            eta=[torch.tensor(0.85, device=device)],
            delta=[torch.tensor(0.5, device=device)],
            input_activation_norm=2.0,
            chunk_index=3,
        )
        assert snap.delta is not None
        assert len(snap.delta) == 1

    def test_detach_removes_grad(self, device: torch.device) -> None:
        """detach() returns new GateSnapshot with all tensors detached."""
        from titans.checkpoint_types import GateSnapshot

        alpha = torch.tensor(0.9, device=device, requires_grad=True)
        theta = torch.tensor(0.1, device=device, requires_grad=True)
        eta = torch.tensor(0.85, device=device, requires_grad=True)
        delta = torch.tensor(0.5, device=device, requires_grad=True)

        snap = GateSnapshot(
            alpha=[alpha],
            theta=[theta],
            eta=[eta],
            delta=[delta],
            input_activation_norm=1.0,
            chunk_index=1,
        )
        detached = snap.detach()

        assert not detached.alpha[0].requires_grad
        assert not detached.theta[0].requires_grad
        assert not detached.eta[0].requires_grad
        assert detached.delta is not None
        assert not detached.delta[0].requires_grad
        # Original is unchanged
        assert snap.alpha[0].requires_grad

    def test_detach_preserves_values(self, device: torch.device) -> None:
        """detach() preserves tensor values."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.tensor(0.77, device=device)],
            theta=[torch.tensor(0.11, device=device)],
            eta=[torch.tensor(0.55, device=device)],
            delta=None,
            input_activation_norm=3.14,
            chunk_index=5,
        )
        detached = snap.detach()
        assert detached.alpha[0].item() == pytest.approx(0.77)
        assert detached.chunk_index == 5
        assert detached.input_activation_norm == pytest.approx(3.14)

    def test_detach_none_delta(self, device: torch.device) -> None:
        """detach() handles delta=None without error."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.zeros(1, device=device)],
            theta=[torch.zeros(1, device=device)],
            eta=[torch.zeros(1, device=device)],
            delta=None,
            input_activation_norm=0.0,
            chunk_index=0,
        )
        detached = snap.detach()
        assert detached.delta is None

    def test_to_device_cpu(self, device: torch.device) -> None:
        """to() moves all tensors to the given device."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.tensor(0.5, device=device)],
            theta=[torch.tensor(0.5, device=device)],
            eta=[torch.tensor(0.5, device=device)],
            delta=[torch.tensor(0.5, device=device)],
            input_activation_norm=1.0,
            chunk_index=0,
        )
        moved = snap.to(torch.device("cpu"))
        assert moved.alpha[0].device.type == "cpu"
        assert moved.theta[0].device.type == "cpu"
        assert moved.eta[0].device.type == "cpu"
        assert moved.delta is not None
        assert moved.delta[0].device.type == "cpu"

    def test_to_preserves_delta_none(self) -> None:
        """to() keeps delta as None when original is None."""
        from titans.checkpoint_types import GateSnapshot

        snap = GateSnapshot(
            alpha=[torch.zeros(1)],
            theta=[torch.zeros(1)],
            eta=[torch.zeros(1)],
            delta=None,
            input_activation_norm=0.0,
            chunk_index=0,
        )
        moved = snap.to(torch.device("cpu"))
        assert moved.delta is None

    def test_multiple_layers(self, device: torch.device) -> None:
        """GateSnapshot supports multi-layer gate lists."""
        from titans.checkpoint_types import GateSnapshot

        n = 3
        snap = GateSnapshot(
            alpha=_make_tensors(n, (1,), device),
            theta=_make_tensors(n, (1,), device),
            eta=_make_tensors(n, (1,), device),
            delta=None,
            input_activation_norm=1.0,
            chunk_index=0,
        )
        assert len(snap.alpha) == n
        assert len(snap.theta) == n
        assert len(snap.eta) == n


# ---------------------------------------------------------------------------
# SignalFrame
# ---------------------------------------------------------------------------


class TestSignalFrame:
    """Tests for the SignalFrame dataclass."""

    def _make_frame(self, chunk_index: int = 0, n_layers: int = 2) -> object:
        from titans.checkpoint_types import SignalFrame

        return SignalFrame(
            chunk_index=chunk_index,
            prediction_error_norms=[0.1] * n_layers,
            weight_delta_norms=[0.2] * n_layers,
            momentum_shift_norms=[0.3] * n_layers,
            gradient_norms=[0.4] * n_layers,
            weight_norms=[1.0] * n_layers,
            momentum_norms=[0.5] * n_layers,
            gate_alpha_means=[0.9] * n_layers,
            gate_theta_means=[0.1] * n_layers,
            gate_eta_means=[0.85] * n_layers,
            batch_variance=0.05,
            local_signal_norms=None,
        )

    def test_construction_mac_mode(self) -> None:
        """SignalFrame constructs correctly with TNT-only fields as None."""
        from titans.checkpoint_types import SignalFrame

        frame = SignalFrame(
            chunk_index=7,
            prediction_error_norms=[0.1, 0.2],
            weight_delta_norms=[0.3, 0.4],
            momentum_shift_norms=[0.5, 0.6],
            gradient_norms=[0.7, 0.8],
            weight_norms=[2.0, 2.1],
            momentum_norms=[0.1, 0.1],
            gate_alpha_means=[0.9, 0.9],
            gate_theta_means=[0.1, 0.1],
            gate_eta_means=[0.85, 0.85],
            batch_variance=None,
            local_signal_norms=None,
        )
        assert frame.chunk_index == 7
        assert frame.local_signal_norms is None

    def test_construction_tnt_mode(self) -> None:
        """SignalFrame constructs with TNT-only fields populated."""
        from titans.checkpoint_types import SignalFrame

        frame = SignalFrame(
            chunk_index=2,
            prediction_error_norms=[0.1],
            weight_delta_norms=[0.2],
            momentum_shift_norms=[0.3],
            gradient_norms=[0.4],
            weight_norms=[1.5],
            momentum_norms=[0.6],
            gate_alpha_means=[0.9],
            gate_theta_means=[0.1],
            gate_eta_means=[0.8],
            batch_variance=0.01,
            local_signal_norms=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert len(frame.local_signal_norms) == 2

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict() output is JSON-serializable."""
        frame = self._make_frame(chunk_index=3)
        d = frame.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        round_tripped = json.loads(serialized)
        assert round_tripped["chunk_index"] == 3

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict() includes all SignalFrame fields."""
        frame = self._make_frame()
        d = frame.to_dict()
        expected_keys = {
            "chunk_index",
            "prediction_error_norms",
            "weight_delta_norms",
            "momentum_shift_norms",
            "gradient_norms",
            "weight_norms",
            "momentum_norms",
            "gate_alpha_means",
            "gate_theta_means",
            "gate_eta_means",
            "batch_variance",
            "local_signal_norms",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_tnt_fields(self) -> None:
        """to_dict() serializes TNT-specific fields correctly."""
        from titans.checkpoint_types import SignalFrame

        frame = SignalFrame(
            chunk_index=0,
            prediction_error_norms=[0.1],
            weight_delta_norms=[0.2],
            momentum_shift_norms=[0.3],
            gradient_norms=[0.4],
            weight_norms=[1.0],
            momentum_norms=[0.5],
            gate_alpha_means=[0.9],
            gate_theta_means=[0.1],
            gate_eta_means=[0.8],
            batch_variance=0.02,
            local_signal_norms=[[0.1]],
        )
        d = frame.to_dict()
        assert d["local_signal_norms"] == [[0.1]]


# ---------------------------------------------------------------------------
# CheckpointEntry
# ---------------------------------------------------------------------------


class TestCheckpointEntry:
    """Tests for the CheckpointEntry dataclass."""

    def _make_entry(
        self, device: torch.device, trigger_phase: str = "before"
    ) -> object:
        from titans.checkpoint_types import CheckpointEntry

        state = _make_memory_state(2, 8, device)
        return CheckpointEntry(
            state=state,
            gates=None,
            metadata={"step": 10},
            trigger_phase=trigger_phase,
            weight_norms=[1.0, 1.1],
            momentum_norms=[0.5, 0.6],
            config_hash="abc123",
        )

    def test_construction_basic(self, device: torch.device) -> None:
        """CheckpointEntry stores all fields correctly."""
        entry = self._make_entry(device)
        assert entry.trigger_phase == "before"
        assert entry.config_hash == "abc123"
        assert entry.metadata["step"] == 10
        assert entry.gates is None

    def test_valid_trigger_phases(self, device: torch.device) -> None:
        """All four trigger_phase values can be stored."""
        for phase in ("before", "during", "after", "periodic"):
            entry = self._make_entry(device, trigger_phase=phase)
            assert entry.trigger_phase == phase

    def test_with_gates(self, device: torch.device) -> None:
        """CheckpointEntry accepts a GateSnapshot for gates."""
        from titans.checkpoint_types import CheckpointEntry, GateSnapshot

        gates = GateSnapshot(
            alpha=[torch.tensor(0.9, device=device)],
            theta=[torch.tensor(0.1, device=device)],
            eta=[torch.tensor(0.85, device=device)],
            delta=None,
            input_activation_norm=1.0,
            chunk_index=0,
        )
        entry = CheckpointEntry(
            state=_make_memory_state(1, 8, device),
            gates=gates,
            metadata={},
            trigger_phase="periodic",
            weight_norms=[1.0],
            momentum_norms=[0.5],
            config_hash="xyz",
        )
        assert entry.gates is not None

    def test_tnt_state(self, device: torch.device) -> None:
        """CheckpointEntry accepts TNTMemoryState."""
        from titans.checkpoint_types import CheckpointEntry

        state = _make_tnt_memory_state(2, 2, 8, device)
        entry = CheckpointEntry(
            state=state,
            gates=None,
            metadata={},
            trigger_phase="after",
            weight_norms=[1.0],
            momentum_norms=[0.5],
            config_hash="tnt",
        )
        assert isinstance(entry.state, TNTMemoryState)


# ---------------------------------------------------------------------------
# TransitionRecord
# ---------------------------------------------------------------------------


class TestTransitionRecord:
    """Tests for the TransitionRecord dataclass."""

    def _make_entry(self, device: torch.device) -> object:
        from titans.checkpoint_types import CheckpointEntry

        return CheckpointEntry(
            state=_make_memory_state(1, 8, device),
            gates=None,
            metadata={},
            trigger_phase="before",
            weight_norms=[1.0],
            momentum_norms=[0.5],
            config_hash="h",
        )

    def _make_signal_frame(self, idx: int) -> object:
        from titans.checkpoint_types import SignalFrame

        return SignalFrame(
            chunk_index=idx,
            prediction_error_norms=[0.1],
            weight_delta_norms=[0.2],
            momentum_shift_norms=[0.3],
            gradient_norms=[0.4],
            weight_norms=[1.0],
            momentum_norms=[0.5],
            gate_alpha_means=[0.9],
            gate_theta_means=[0.1],
            gate_eta_means=[0.8],
            batch_variance=None,
            local_signal_norms=None,
        )

    def test_construction(self, device: torch.device) -> None:
        """TransitionRecord stores all sub-structures."""
        from titans.checkpoint_types import TransitionRecord

        before = self._make_entry(device)
        during = self._make_entry(device)
        after = [self._make_entry(device) for _ in range(3)]
        signal_window = [self._make_signal_frame(i) for i in range(5)]

        record = TransitionRecord(
            before=before,
            during=during,
            after=after,
            signal_window=signal_window,
            transition_id="t-001",
            transition_magnitude=3.45,
            duration_chunks=7,
        )
        assert record.transition_id == "t-001"
        assert record.transition_magnitude == pytest.approx(3.45)
        assert record.duration_chunks == 7
        assert len(record.after) == 3
        assert len(record.signal_window) == 5

    def test_after_can_be_empty(self, device: torch.device) -> None:
        """TransitionRecord.after may be an empty list (in-progress)."""
        from titans.checkpoint_types import TransitionRecord

        record = TransitionRecord(
            before=self._make_entry(device),
            during=self._make_entry(device),
            after=[],
            signal_window=[],
            transition_id="t-002",
            transition_magnitude=0.0,
            duration_chunks=0,
        )
        assert record.after == []
        assert record.signal_window == []


# ---------------------------------------------------------------------------
# MemoryCheckpointConfig
# ---------------------------------------------------------------------------


class TestMemoryCheckpointConfig:
    """Tests for the MemoryCheckpointConfig dataclass."""

    def test_defaults(self) -> None:
        """Default values match specification."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        cfg = MemoryCheckpointConfig()
        assert cfg.checkpoint_dir == "memory_checkpoints"
        assert cfg.ring_size == 25
        assert cfg.sigma_threshold == pytest.approx(2.0)
        assert cfg.window_size == 50
        assert cfg.min_observations == 10
        assert cfg.cooldown_chunks == 20
        assert cfg.after_capture_count == 5
        assert cfg.keep_last_n_transitions == 10
        assert cfg.signal_log_enabled is False
        assert cfg.signal_log_format == "jsonl"
        assert cfg.signal_log_max_frames == 100_000
        assert cfg.quantize_checkpoints is False
        assert cfg.checkpoint_weight_bits == 8
        assert cfg.checkpoint_momentum_bits == 8

    def test_custom_construction(self) -> None:
        """MemoryCheckpointConfig accepts custom values."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        cfg = MemoryCheckpointConfig(
            checkpoint_dir="/tmp/ckpts",
            ring_size=50,
            sigma_threshold=3.0,
            signal_log_enabled=True,
        )
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.ring_size == 50
        assert cfg.sigma_threshold == pytest.approx(3.0)
        assert cfg.signal_log_enabled is True

    def test_to_dict_contains_all_keys(self) -> None:
        """to_dict() includes every config field."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        cfg = MemoryCheckpointConfig()
        d = cfg.to_dict()
        expected_keys = {
            "checkpoint_dir",
            "ring_size",
            "sigma_threshold",
            "window_size",
            "min_observations",
            "cooldown_chunks",
            "after_capture_count",
            "keep_last_n_transitions",
            "signal_log_enabled",
            "signal_log_format",
            "signal_log_max_frames",
            "quantize_checkpoints",
            "checkpoint_weight_bits",
            "checkpoint_momentum_bits",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self) -> None:
        """to_dict() values match the config fields."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        cfg = MemoryCheckpointConfig(ring_size=99, sigma_threshold=1.5)
        d = cfg.to_dict()
        assert d["ring_size"] == 99
        assert d["sigma_threshold"] == pytest.approx(1.5)

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict() output is JSON-serializable."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        cfg = MemoryCheckpointConfig()
        d = cfg.to_dict()
        serialized = json.dumps(d)
        assert json.loads(serialized)["ring_size"] == 25

    def test_from_dict_roundtrip(self) -> None:
        """from_dict(cfg.to_dict()) reconstructs an equivalent config."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        original = MemoryCheckpointConfig(
            checkpoint_dir="/ckpts",
            ring_size=15,
            sigma_threshold=1.8,
            signal_log_enabled=True,
            quantize_checkpoints=True,
            checkpoint_weight_bits=4,
        )
        d = original.to_dict()
        restored = MemoryCheckpointConfig.from_dict(d)

        assert restored.checkpoint_dir == original.checkpoint_dir
        assert restored.ring_size == original.ring_size
        assert restored.sigma_threshold == pytest.approx(original.sigma_threshold)
        assert restored.signal_log_enabled == original.signal_log_enabled
        assert restored.quantize_checkpoints == original.quantize_checkpoints
        assert restored.checkpoint_weight_bits == original.checkpoint_weight_bits

    def test_from_dict_missing_keys_use_defaults(self) -> None:
        """from_dict() uses default values for keys absent from the dict."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        # Partial dict — only override ring_size
        restored = MemoryCheckpointConfig.from_dict({"ring_size": 7})
        assert restored.ring_size == 7
        assert restored.sigma_threshold == pytest.approx(2.0)  # default
        assert restored.checkpoint_dir == "memory_checkpoints"  # default

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """from_dict() does not raise on unknown keys."""
        from titans.checkpoint_types import MemoryCheckpointConfig

        # Should not raise
        cfg = MemoryCheckpointConfig.from_dict({"ring_size": 5, "unknown_key": "foo"})
        assert cfg.ring_size == 5
