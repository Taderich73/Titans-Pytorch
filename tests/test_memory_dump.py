"""Tests for memory state serialization."""

import pytest
import torch

from titans.memory import NeuralLongTermMemory
from titans.memory_dump import load_memory_states, save_memory_states


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
