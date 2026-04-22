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


def test_load_memory_states_preserves_qk_projections_by_default(tmp_path):
    """Round-trip a TNTMemoryState and verify qk_projections + counters survive.

    Previously load_memory_states defaulted reset_for_inference=True, which
    silently zeroed qk_projections and local_step_counters — corrupting state
    for training resume callers that used the default.
    """
    import torch
    from titans.memory import MemoryState, TNTMemoryState
    from titans.memory_dump import save_memory_states, load_memory_states

    qk = torch.randn(3, 4)
    counter = 7
    global_state = MemoryState(
        weights=[torch.randn(4, 4)],
        momentum=[torch.zeros(4, 4)],
    )
    local_state = MemoryState(
        weights=[torch.randn(4, 4)],
        momentum=[torch.zeros(4, 4)],
    )
    tnt = TNTMemoryState(
        global_state=global_state,
        local_states=[local_state],
        qk_projections=[qk],
        local_step_counters=[counter],
    )

    path = tmp_path / "mem.npz"
    save_memory_states([tnt], path)

    # Default behaviour MUST preserve qk_projections and counters.
    loaded = load_memory_states(path)
    assert isinstance(loaded[0], TNTMemoryState)
    assert loaded[0].local_step_counters == [counter], (
        "local_step_counters zeroed — TNT state wipe regression"
    )
    assert torch.allclose(loaded[0].qk_projections[0], qk), (
        "qk_projections zeroed — TNT state wipe regression"
    )

    # Explicit reset_for_inference=True still resets (inference callers).
    loaded_reset = load_memory_states(path, reset_for_inference=True)
    assert loaded_reset[0].local_step_counters == [0]
    assert torch.allclose(
        loaded_reset[0].qk_projections[0], torch.zeros_like(qk)
    )


def test_load_memory_states_ignores_legacy_local_inits_key(tmp_path):
    """Legacy .npz files written before the local_inits field was removed
    must still load cleanly - the unknown keys should be ignored."""
    import numpy as np

    from titans.memory import MemoryState, TNTMemoryState
    from titans.memory_dump import load_memory_states, save_memory_states

    # Build a valid TNTMemoryState npz via save, then re-open and inject
    # legacy local_inits_* keys as a new npz to simulate pre-removal files.
    qk = torch.randn(3, 4)
    global_state = MemoryState(
        weights=[torch.randn(4, 4)],
        momentum=[torch.zeros(4, 4)],
    )
    local_state = MemoryState(
        weights=[torch.randn(4, 4)],
        momentum=[torch.zeros(4, 4)],
    )
    tnt = TNTMemoryState(
        global_state=global_state,
        local_states=[local_state],
        qk_projections=[qk],
        local_step_counters=[5],
    )

    base_path = tmp_path / "base.npz"
    save_memory_states([tnt], base_path)

    # Load raw arrays and add legacy local_inits_* keys.
    with np.load(str(base_path)) as data:
        arrays = {k: data[k] for k in data.files}

    # Inject legacy-format local_inits keys for layer 0, local 0.
    arrays["layer_0_local_init_0_count"] = np.array([1])
    arrays["layer_0_local_init_0_0"] = np.zeros((4, 4), dtype=np.float32)

    legacy_path = tmp_path / "legacy.npz"
    np.savez(str(legacy_path), **arrays)

    # Must load without crashing; legacy key ignored.
    loaded = load_memory_states(legacy_path)
    assert len(loaded) == 1
    assert isinstance(loaded[0], TNTMemoryState)
    assert loaded[0].local_step_counters == [5]
    assert torch.allclose(loaded[0].qk_projections[0], qk)
    # Sanity: field really gone.
    assert not hasattr(loaded[0], "local_inits")
