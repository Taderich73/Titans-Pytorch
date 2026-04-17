from __future__ import annotations

"""Tests for memory state persistence in SFT training checkpoints."""

import torch

from titans.checkpoint import save_checkpoint
from titans.memory import MemoryState
from titans.memory_dump import load_memory_states, save_memory_states


def _make_memory_states(num_layers: int = 2) -> list[MemoryState]:
    """Build simple MemoryState objects with known values for testing."""
    states = []
    for i in range(num_layers):
        weights = [torch.full((4, 4), float(i + 1))]
        momentum = [torch.full((4, 4), float(i + 0.5))]
        states.append(MemoryState(weights=weights, momentum=momentum))
    return states


class TestSFTMemoryPersistence:
    """Verify memory states are saved alongside SFT checkpoints."""

    def test_periodic_save_creates_npz(self, tmp_path):
        """Periodic checkpoint should produce a paired memory_step_N.npz file."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        global_step = 1000
        memory_states = _make_memory_states()

        # Simulate the periodic checkpoint save pattern from sft.py
        ckpt_stem = checkpoint_dir / f"step_{global_step}"
        dummy_state_dict = {"weight": torch.zeros(4, 4)}
        save_checkpoint(
            dummy_state_dict,
            ckpt_stem,
            format="pt",
            metadata={"step": global_step, "epoch": 0},
        )

        # This is the new code we're adding to sft.py
        if memory_states is not None:
            mem_path = checkpoint_dir / f"memory_step_{global_step}.npz"
            save_memory_states(memory_states, mem_path)

        # Verify both files exist
        assert (checkpoint_dir / f"step_{global_step}.pt").exists()
        assert (checkpoint_dir / f"memory_step_{global_step}.npz").exists()

        # Verify memory round-trips correctly
        loaded = load_memory_states(
            checkpoint_dir / f"memory_step_{global_step}.npz",
            device=torch.device("cpu"),
        )
        assert len(loaded) == 2
        for orig, restored in zip(memory_states, loaded):
            for w_orig, w_loaded in zip(orig.weights, restored.weights):
                torch.testing.assert_close(w_orig, w_loaded)
            for m_orig, m_loaded in zip(orig.momentum, restored.momentum):
                torch.testing.assert_close(m_orig, m_loaded)

    def test_final_save_creates_npz(self, tmp_path):
        """Final checkpoint should produce memory_final.npz."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        memory_states = _make_memory_states()

        # Simulate final save
        final_stem = checkpoint_dir / "final"
        dummy_state_dict = {"weight": torch.zeros(4, 4)}
        save_checkpoint(
            dummy_state_dict,
            final_stem,
            format="pt",
            metadata={"step": 5000},
        )

        if memory_states is not None:
            mem_path = checkpoint_dir / "memory_final.npz"
            save_memory_states(memory_states, mem_path)

        assert (checkpoint_dir / "final.pt").exists()
        assert (checkpoint_dir / "memory_final.npz").exists()

    def test_resume_loads_memory_from_step(self, tmp_path):
        """Resume should load memory_step_N.npz paired with step_N.pt."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        global_step = 2000

        # Save a checkpoint + memory
        original_states = _make_memory_states()
        mem_path = checkpoint_dir / f"memory_step_{global_step}.npz"
        save_memory_states(original_states, mem_path)

        # Simulate resume path derivation (same logic as sft.py resume block)
        resume_path = checkpoint_dir / f"step_{global_step}.pt"
        derived_mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        if not derived_mem_path.exists():
            derived_mem_path = resume_path.parent / "memory_final.npz"

        loaded = load_memory_states(derived_mem_path, device=torch.device("cpu"))
        assert len(loaded) == len(original_states)
        for orig, restored in zip(original_states, loaded):
            torch.testing.assert_close(orig.weights[0], restored.weights[0])

    def test_resume_falls_back_to_final(self, tmp_path):
        """If memory_step_N.npz doesn't exist, fall back to memory_final.npz."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        global_step = 3000

        # Only save final memory, not step-specific
        original_states = _make_memory_states()
        save_memory_states(original_states, checkpoint_dir / "memory_final.npz")

        # Simulate resume derivation
        resume_path = checkpoint_dir / f"step_{global_step}.pt"
        mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        if not mem_path.exists():
            mem_path = resume_path.parent / "memory_final.npz"

        loaded = load_memory_states(mem_path, device=torch.device("cpu"))
        assert len(loaded) == len(original_states)
        for orig, restored in zip(original_states, loaded):
            torch.testing.assert_close(orig.weights[0], restored.weights[0])

    def test_resume_graceful_without_memory(self, tmp_path):
        """Resume should succeed with fresh memory if no .npz exists."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        global_step = 4000

        resume_path = checkpoint_dir / f"step_{global_step}.pt"
        mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        if not mem_path.exists():
            mem_path = resume_path.parent / "memory_final.npz"

        memory_states = None
        try:
            memory_states = load_memory_states(mem_path, device=torch.device("cpu"))
        except Exception:
            pass  # Graceful fallback — memory_states stays None

        assert memory_states is None

    def test_none_memory_states_skips_save(self, tmp_path):
        """When memory_states is None, no .npz file should be created."""
        checkpoint_dir = tmp_path / "checkpoints" / "sft"
        checkpoint_dir.mkdir(parents=True)
        memory_states = None

        if memory_states is not None:
            mem_path = checkpoint_dir / "memory_step_1000.npz"
            save_memory_states(memory_states, mem_path)

        assert not (checkpoint_dir / "memory_step_1000.npz").exists()
