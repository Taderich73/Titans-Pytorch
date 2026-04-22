"""Integration test for full auto-checkpointing pipeline."""

import json

import torch

from titans import TitansConfig, TitansMAC
from titans.checkpointing.memory_checkpointer import MemoryCheckpointer
from titans.checkpointing.types import MemoryCheckpointConfig


class TestAutoCheckpointIntegration:
    def test_end_to_end_no_trigger(self, tmp_path):
        """Run a short generation with checkpointing — no trigger expected."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=16,
            window_size=16,
            num_memory_layers=1,
            auto_checkpoint=True,
        )
        model = TitansMAC(config)
        model.eval()

        ckpt_config = MemoryCheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            ring_size=5,
            min_observations=3,
        )
        checkpointer = MemoryCheckpointer(ckpt_config, config_hash="test")

        states = None
        with torch.no_grad():
            for i in range(10):
                x = torch.randint(0, 256, (1, 16))
                logits, states, gate_snapshots = model(x, states=states)
                if states is not None:
                    states = [s.detach() for s in states]
                checkpointer.on_chunk_commit(states, gate_snapshots, chunk_index=i)

        checkpointer.flush()

        session_path = tmp_path / "ckpt" / "session.json"
        assert session_path.exists()
        data = json.loads(session_path.read_text())
        assert data["total_chunks_processed"] == 10
        assert (tmp_path / "ckpt" / "ring_buffer_final.npz").exists()

    def test_end_to_end_with_signal_log(self, tmp_path):
        """Run with signal log enabled — verify log files created."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=16,
            window_size=16,
            num_memory_layers=1,
            auto_checkpoint=True,
        )
        model = TitansMAC(config)
        model.eval()

        ckpt_config = MemoryCheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            signal_log_enabled=True,
            signal_log_max_frames=100,
        )
        checkpointer = MemoryCheckpointer(ckpt_config, config_hash="test")

        states = None
        with torch.no_grad():
            for i in range(5):
                x = torch.randint(0, 256, (1, 16))
                logits, states, gate_snapshots = model(x, states=states)
                if states is not None:
                    states = [s.detach() for s in states]
                checkpointer.on_chunk_commit(states, gate_snapshots, chunk_index=i)

        checkpointer.flush()

        log_dir = tmp_path / "ckpt" / "signal_log"
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.jsonl.gz"))
        assert len(log_files) >= 1

    def test_gate_snapshots_populated(self, tmp_path):
        """Verify gate snapshots are actually populated when auto_checkpoint=True."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            chunk_size=16,
            window_size=16,
            num_memory_layers=1,
            auto_checkpoint=True,
        )
        model = TitansMAC(config)
        model.eval()

        with torch.no_grad():
            x = torch.randint(0, 256, (1, 16))
            logits, states, gate_snapshots = model(x)

        # Gate snapshots should be populated (not all None)
        assert any(gs is not None for gs in gate_snapshots)
