"""Tests for memory state serialization."""

import pytest
import torch

from titans.checkpoint_types import CheckpointEntry, GateSnapshot
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.memory_dump import (
    load_checkpoint_entry,
    load_memory_states,
    save_checkpoint_entry,
    save_memory_states,
)


class TestMemoryDump:
    def test_round_trip(self, default_config, tmp_path, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        states = [
            mem.init_state(batch_size=2) for _ in range(default_config.num_layers)
        ]
        path = tmp_path / "states.npz"
        save_memory_states(states, path)
        loaded = load_memory_states(path, device=device)

        assert len(loaded) == len(states)
        for orig, restored in zip(states, loaded):
            for w_orig, w_loaded in zip(orig.weights, restored.weights):
                torch.testing.assert_close(w_orig, w_loaded)
            for m_orig, m_loaded in zip(orig.momentum, restored.momentum):
                torch.testing.assert_close(m_orig, m_loaded)

    def test_load_to_cpu(self, default_config, tmp_path):
        mem = NeuralLongTermMemory(default_config)
        states = [mem.init_state(batch_size=2)]
        path = tmp_path / "states.npz"
        save_memory_states(states, path)
        loaded = load_memory_states(path, device=torch.device("cpu"))
        assert loaded[0].weights[0].device.type == "cpu"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_memory_states(tmp_path / "nonexistent.npz")

    def test_npz_suffix_auto(self, default_config, tmp_path):
        """Loading without .npz suffix should still work."""
        mem = NeuralLongTermMemory(default_config)
        states = [mem.init_state(batch_size=2)]
        path = tmp_path / "states"
        save_memory_states(states, path)
        loaded = load_memory_states(path)
        assert len(loaded) == 1


def _make_simple_state(num_sublayers: int = 2) -> MemoryState:
    """Build a small MemoryState with known values for testing."""
    weights = [torch.full((4, 4), float(j + 1)) for j in range(num_sublayers)]
    momentum = [torch.full((4, 4), float(j + 0.5)) for j in range(num_sublayers)]
    return MemoryState(weights=weights, momentum=momentum)


def _make_gate_snapshot(num_layers: int = 2, *, with_delta: bool = False) -> GateSnapshot:
    """Build a GateSnapshot with deterministic values for testing."""
    alpha = [torch.tensor([0.1 * (k + 1)]) for k in range(num_layers)]
    theta = [torch.tensor([0.2 * (k + 1)]) for k in range(num_layers)]
    eta = [torch.tensor([0.3 * (k + 1)]) for k in range(num_layers)]
    delta = [torch.tensor([0.4 * (k + 1)]) for k in range(num_layers)] if with_delta else None
    return GateSnapshot(
        alpha=alpha,
        theta=theta,
        eta=eta,
        delta=delta,
        input_activation_norm=3.14,
        chunk_index=7,
    )


class TestCheckpointEntrySerialization:
    def test_round_trip_without_gates(self, tmp_path):
        """Save and load a CheckpointEntry with gates=None."""
        state = _make_simple_state(num_sublayers=2)
        entry = CheckpointEntry(
            state=state,
            gates=None,
            metadata={"step": 42, "loss": 0.5},
            trigger_phase="periodic",
            weight_norms=[1.0, 2.0],
            momentum_norms=[0.5, 1.0],
            config_hash="abc123",
        )
        path = tmp_path / "entry.npz"
        save_checkpoint_entry(entry, path)
        loaded = load_checkpoint_entry(path)

        # State tensors must match
        for w_orig, w_loaded in zip(state.weights, loaded.state.weights):
            torch.testing.assert_close(w_orig, w_loaded)
        for m_orig, m_loaded in zip(state.momentum, loaded.state.momentum):
            torch.testing.assert_close(m_orig, m_loaded)

        # Metadata fields preserved
        assert loaded.trigger_phase == "periodic"
        assert loaded.config_hash == "abc123"
        assert loaded.gates is None

        # Norms preserved
        assert loaded.weight_norms == pytest.approx([1.0, 2.0])
        assert loaded.momentum_norms == pytest.approx([0.5, 1.0])

    def test_round_trip_with_gates(self, tmp_path):
        """Save and load a CheckpointEntry with a GateSnapshot (no delta)."""
        state = _make_simple_state(num_sublayers=2)
        gates = _make_gate_snapshot(num_layers=2, with_delta=False)
        entry = CheckpointEntry(
            state=state,
            gates=gates,
            metadata={"step": 100},
            trigger_phase="before",
            weight_norms=[1.5, 2.5],
            momentum_norms=[0.7, 1.2],
            config_hash="def456",
        )
        path = tmp_path / "entry_gates.npz"
        save_checkpoint_entry(entry, path)
        loaded = load_checkpoint_entry(path)

        assert loaded.gates is not None
        for k in range(2):
            torch.testing.assert_close(loaded.gates.alpha[k], gates.alpha[k])
            torch.testing.assert_close(loaded.gates.theta[k], gates.theta[k])
            torch.testing.assert_close(loaded.gates.eta[k], gates.eta[k])

        assert loaded.gates.delta is None
        assert loaded.gates.input_activation_norm == pytest.approx(3.14, rel=1e-5)
        assert loaded.gates.chunk_index == 7
        assert loaded.trigger_phase == "before"
        assert loaded.config_hash == "def456"

    def test_round_trip_with_gates_and_delta(self, tmp_path):
        """Save and load a CheckpointEntry with a GateSnapshot including delta."""
        state = _make_simple_state(num_sublayers=2)
        gates = _make_gate_snapshot(num_layers=2, with_delta=True)
        entry = CheckpointEntry(
            state=state,
            gates=gates,
            metadata={},
            trigger_phase="during",
            weight_norms=[1.0],
            momentum_norms=[0.5],
            config_hash="ghi789",
        )
        path = tmp_path / "entry_delta.npz"
        save_checkpoint_entry(entry, path)
        loaded = load_checkpoint_entry(path)

        assert loaded.gates is not None
        assert loaded.gates.delta is not None
        for k in range(2):
            torch.testing.assert_close(loaded.gates.delta[k], gates.delta[k])

    def test_backward_compatible_load(self, tmp_path):
        """Checkpoint entry files can still be loaded by load_memory_states.

        The state portion must use the same key format so the existing reader
        ignores the gate/meta keys without error.
        """
        state = _make_simple_state(num_sublayers=2)
        entry = CheckpointEntry(
            state=state,
            gates=None,
            metadata={},
            trigger_phase="after",
            weight_norms=[1.0, 2.0],
            momentum_norms=[0.5, 1.0],
            config_hash="compat",
        )
        path = tmp_path / "compat.npz"
        save_checkpoint_entry(entry, path)

        # load_memory_states must be able to read the file, ignoring gate keys
        loaded_states = load_memory_states(path, device=torch.device("cpu"))
        assert len(loaded_states) == 1
        for w_orig, w_loaded in zip(state.weights, loaded_states[0].weights):
            torch.testing.assert_close(w_orig, w_loaded)
