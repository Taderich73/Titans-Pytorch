"""Tests for memory state persistence in LoRA training checkpoints."""

from __future__ import annotations

import contextlib

import torch

from titans.checkpoint import load_checkpoint, save_checkpoint
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


class TestLoRAMemoryPersistence:
    """Verify memory states are saved alongside LoRA checkpoints."""

    def test_periodic_save_creates_all_files(self, tmp_path):
        """Periodic checkpoint should produce full ckpt and memory files."""
        checkpoint_dir = tmp_path / "checkpoints" / "lora"
        checkpoint_dir.mkdir(parents=True)
        global_step = 5000
        memory_states = _make_memory_states()

        # Simulate the full checkpoint save (new behavior)
        ckpt_stem = checkpoint_dir / f"step_{global_step}"
        dummy_state_dict = {"weight": torch.zeros(4, 4)}
        save_checkpoint(
            dummy_state_dict,
            ckpt_stem,
            format="pt",
            metadata={
                "optimizer": {"state": {}},
                "scheduler": {"last_epoch": 0},
                "step": global_step,
                "epoch": 0,
            },
        )

        if memory_states is not None:
            mem_path = checkpoint_dir / f"memory_step_{global_step}.npz"
            save_memory_states(memory_states, mem_path)

        assert (checkpoint_dir / f"step_{global_step}.pt").exists()
        assert (checkpoint_dir / f"memory_step_{global_step}.npz").exists()

        loaded = load_memory_states(
            checkpoint_dir / f"memory_step_{global_step}.npz",
            device=torch.device("cpu"),
        )
        assert len(loaded) == 2

    def test_resume_loads_model_and_memory(self, tmp_path):
        """Resume should restore model state dict and paired memory states."""
        checkpoint_dir = tmp_path / "checkpoints" / "lora"
        checkpoint_dir.mkdir(parents=True)
        global_step = 5000

        # Save a full checkpoint (the kind our new code produces)
        dummy_state_dict = {"weight": torch.ones(4, 4)}
        ckpt_stem = checkpoint_dir / f"step_{global_step}"
        save_checkpoint(
            dummy_state_dict,
            ckpt_stem,
            format="pt",
            metadata={
                "optimizer": {"state": {}, "param_groups": [{"lr": 1e-4}]},
                "scheduler": {"last_epoch": global_step},
                "step": global_step,
                "epoch": 1,
            },
        )

        original_states = _make_memory_states()
        save_memory_states(
            original_states,
            checkpoint_dir / f"memory_step_{global_step}.npz",
        )

        # Simulate resume
        resume_path = checkpoint_dir / f"step_{global_step}.pt"
        checkpoint = load_checkpoint(resume_path, weights_only=False)
        assert checkpoint["step"] == global_step
        assert checkpoint["epoch"] == 1
        assert "model" in checkpoint

        # Memory load derivation
        mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        loaded = load_memory_states(mem_path, device=torch.device("cpu"))
        assert len(loaded) == len(original_states)
        for orig, restored in zip(original_states, loaded):
            torch.testing.assert_close(orig.weights[0], restored.weights[0])

    def test_final_save_creates_npz(self, tmp_path):
        """Final checkpoint should produce memory_final.npz."""
        checkpoint_dir = tmp_path / "checkpoints" / "lora"
        checkpoint_dir.mkdir(parents=True)
        memory_states = _make_memory_states()

        if memory_states is not None:
            mem_path = checkpoint_dir / "memory_final.npz"
            save_memory_states(memory_states, mem_path)

        assert (checkpoint_dir / "memory_final.npz").exists()

    def test_resume_graceful_without_memory(self, tmp_path):
        """Resume should succeed with fresh memory if no .npz exists."""
        checkpoint_dir = tmp_path / "checkpoints" / "lora"
        checkpoint_dir.mkdir(parents=True)

        resume_path = checkpoint_dir / "step_5000.pt"
        mem_path = resume_path.parent / "memory_step_5000.npz"
        if not mem_path.exists():
            mem_path = resume_path.parent / "memory_final.npz"

        memory_states = None
        with contextlib.suppress(Exception):
            memory_states = load_memory_states(mem_path, device=torch.device("cpu"))

        assert memory_states is None
