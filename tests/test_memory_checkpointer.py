# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for the MemoryCheckpointer state machine and ring buffer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from titans.checkpointing.types import (
    GateSnapshot,
    MemoryCheckpointConfig,
    MemoryState,
)
from titans.checkpointing.memory_checkpointer import (
    CheckpointerState,
    MemoryCheckpointer,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_state(val: float = 1.0) -> list[MemoryState]:
    """Build a single-block list of MemoryState with given weight value.

    Args:
        val: Value to fill weight tensors with.

    Returns:
        List with one MemoryState containing a 4x4 weight and zero momentum.
    """
    return [MemoryState(weights=[torch.ones(4, 4) * val], momentum=[torch.zeros(4, 4)])]


def _make_gates(chunk_index: int = 0) -> list[GateSnapshot | None]:
    """Build a single-block list of GateSnapshot.

    Args:
        chunk_index: Chunk index to embed in the snapshot.

    Returns:
        List with one GateSnapshot.
    """
    return [
        GateSnapshot(
            alpha=[torch.tensor(0.1)],
            theta=[torch.tensor(0.05)],
            eta=[torch.tensor(0.9)],
            delta=None,
            input_activation_norm=1.5,
            chunk_index=chunk_index,
        )
    ]


def _make_config(tmp_path: Path, **kwargs) -> MemoryCheckpointConfig:
    """Build a minimal MemoryCheckpointConfig for tests.

    Args:
        tmp_path: pytest tmp_path fixture for the checkpoint directory.
        **kwargs: Additional fields to override defaults.

    Returns:
        A MemoryCheckpointConfig.
    """
    defaults = {
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "ring_size": 5,
        "sigma_threshold": 2.0,
        "window_size": 10,
        "min_observations": 3,
        "cooldown_chunks": 3,
        "after_capture_count": 2,
        "keep_last_n_transitions": 5,
        "signal_log_enabled": False,
    }
    defaults.update(kwargs)
    return MemoryCheckpointConfig(**defaults)


# ---------------------------------------------------------------------------
# TestCheckpointerStates
# ---------------------------------------------------------------------------


class TestCheckpointerStates:
    """State machine: initial state and warmup behaviour."""

    def test_starts_in_monitoring(self, tmp_path: Path) -> None:
        """Checkpointer should start in MONITORING state."""
        cfg = _make_config(tmp_path)
        cp = MemoryCheckpointer(cfg)
        assert cp.state == CheckpointerState.MONITORING

    def test_stays_monitoring_during_warmup(self, tmp_path: Path) -> None:
        """During min_observations warmup the state stays MONITORING."""
        cfg = _make_config(tmp_path, min_observations=5)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(4):  # one less than min_observations
            cp.on_chunk_commit(state, gates, chunk_index=i)
            assert cp.state == CheckpointerState.MONITORING

    def test_checkpoint_dir_created(self, tmp_path: Path) -> None:
        """Constructor should ensure checkpoint_dir exists."""
        cfg = _make_config(tmp_path)
        MemoryCheckpointer(cfg)
        assert Path(cfg.checkpoint_dir).exists()


# ---------------------------------------------------------------------------
# TestRingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:
    """Ring buffer: fill, FIFO eviction, and size limit."""

    def test_buffer_fills_to_ring_size(self, tmp_path: Path) -> None:
        """Ring buffer should accumulate up to ring_size entries."""
        cfg = _make_config(tmp_path, ring_size=4, min_observations=100)  # no triggers
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(4):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        assert len(cp._ring_buffer) == 4

    def test_buffer_does_not_exceed_ring_size(self, tmp_path: Path) -> None:
        """Pushing beyond ring_size should cap at ring_size (FIFO eviction)."""
        cfg = _make_config(tmp_path, ring_size=3, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(10):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        assert len(cp._ring_buffer) == 3

    def test_buffer_fifo_order(self, tmp_path: Path) -> None:
        """Oldest entries should fall off; newest entries remain."""
        cfg = _make_config(tmp_path, ring_size=3, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        # Use distinct weight values so we can identify entries
        for i in range(5):
            cp.on_chunk_commit(_make_state(float(i + 1)), _make_gates(i), chunk_index=i)
        # ring_size=3 so we expect the last 3 entries (val 3.0, 4.0, 5.0)
        norms = [e.weight_norms[0] for e in cp._ring_buffer]
        # weight_norms for val=3 → 4x4 tensor of 3s → norm = 3 * 4 = 12.0
        assert len(norms) == 3
        # norms should be strictly increasing (newer entries have larger weight values)
        assert norms[0] < norms[1] < norms[2]


# ---------------------------------------------------------------------------
# TestTransitionCapture
# ---------------------------------------------------------------------------


class TestTransitionCapture:
    """Spike triggers a capture; flush produces expected output files."""

    def test_spike_triggers_capture(self, tmp_path: Path) -> None:
        """A large weight spike should cause a state transition to CAPTURING_AFTER."""
        cfg = _make_config(
            tmp_path,
            ring_size=10,
            min_observations=3,
            sigma_threshold=1.5,
            window_size=5,
            after_capture_count=2,
        )
        cp = MemoryCheckpointer(cfg)
        gates = _make_gates(0)
        # Build a stable baseline
        for i in range(6):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        assert cp.state == CheckpointerState.MONITORING
        # Inject a massive spike (weight jumps from 1.0 to 100.0)
        cp.on_chunk_commit(_make_state(100.0), gates, chunk_index=6)
        # Should have triggered; state is either CAPTURING_AFTER or COOLDOWN
        assert cp.state in (
            CheckpointerState.CAPTURING_AFTER,
            CheckpointerState.COOLDOWN,
        )

    def test_flush_writes_ring_buffer_final(self, tmp_path: Path) -> None:
        """flush() should write ring_buffer_final.npz."""
        cfg = _make_config(tmp_path, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(3):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        cp.flush()
        assert (Path(cfg.checkpoint_dir) / "ring_buffer_final.npz").exists()

    def test_flush_writes_session_json(self, tmp_path: Path) -> None:
        """flush() should write session.json with required keys."""
        cfg = _make_config(tmp_path, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(3):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        cp.flush()
        session_path = Path(cfg.checkpoint_dir) / "session.json"
        assert session_path.exists()
        data = json.loads(session_path.read_text())
        for key in (
            "session_start",
            "session_end",
            "total_chunks_processed",
            "transitions_recorded",
            "config",
        ):
            assert key in data, f"Missing key: {key}"

    def test_transition_writes_disk_files(self, tmp_path: Path) -> None:
        """A complete transition (before+during+after) should write disk files."""
        cfg = _make_config(
            tmp_path,
            ring_size=10,
            min_observations=3,
            sigma_threshold=1.5,
            window_size=5,
            after_capture_count=2,
        )
        cp = MemoryCheckpointer(cfg)
        gates = _make_gates(0)
        # Build baseline
        for i in range(6):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        # Spike to trigger
        cp.on_chunk_commit(_make_state(100.0), gates, chunk_index=6)
        # Feed after_capture_count chunks to complete the transition
        for i in range(7, 7 + cfg.after_capture_count + 2):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        # At least one transition should have been recorded
        assert cp._transitions_recorded >= 1
        transitions_dir = Path(cfg.checkpoint_dir) / "transitions"
        assert transitions_dir.exists()
        tr_dirs = list(transitions_dir.iterdir())
        assert len(tr_dirs) >= 1
        # Check required files exist
        td = tr_dirs[0]
        assert (td / "before.npz").exists()
        assert (td / "during.npz").exists()
        assert (td / "metadata.json").exists()
        assert (td / "signal_window.jsonl.gz").exists()


# ---------------------------------------------------------------------------
# TestCooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    """After a capture completes, the checkpointer enters COOLDOWN."""

    def test_enters_cooldown_after_capture(self, tmp_path: Path) -> None:
        """After collecting all after-capture chunks, state should be COOLDOWN."""
        cfg = _make_config(
            tmp_path,
            ring_size=10,
            min_observations=3,
            sigma_threshold=1.5,
            window_size=5,
            after_capture_count=2,
            cooldown_chunks=5,
        )
        cp = MemoryCheckpointer(cfg)
        gates = _make_gates(0)
        for i in range(6):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        # Spike
        cp.on_chunk_commit(_make_state(100.0), gates, chunk_index=6)
        # Complete after capture
        for i in range(7, 7 + cfg.after_capture_count):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        assert cp.state == CheckpointerState.COOLDOWN

    def test_no_trigger_during_cooldown(self, tmp_path: Path) -> None:
        """A spike during COOLDOWN must not trigger a second transition."""
        cfg = _make_config(
            tmp_path,
            ring_size=10,
            min_observations=3,
            sigma_threshold=1.5,
            window_size=5,
            after_capture_count=1,
            cooldown_chunks=10,
        )
        cp = MemoryCheckpointer(cfg)
        gates = _make_gates(0)
        # Build baseline
        for i in range(6):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        # First spike → trigger
        cp.on_chunk_commit(_make_state(100.0), gates, chunk_index=6)
        # Complete after capture to enter COOLDOWN
        for i in range(7, 7 + cfg.after_capture_count):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        assert cp.state == CheckpointerState.COOLDOWN
        transitions_before = cp._transitions_recorded
        # Another spike during cooldown — should not record a new transition
        cp.on_chunk_commit(_make_state(200.0), gates, chunk_index=100)
        assert cp.state == CheckpointerState.COOLDOWN
        assert cp._transitions_recorded == transitions_before

    def test_cooldown_expires_to_monitoring(self, tmp_path: Path) -> None:
        """After cooldown_chunks chunks COOLDOWN should revert to MONITORING."""
        cfg = _make_config(
            tmp_path,
            ring_size=10,
            min_observations=3,
            sigma_threshold=1.5,
            window_size=5,
            after_capture_count=1,
            cooldown_chunks=3,
        )
        cp = MemoryCheckpointer(cfg)
        gates = _make_gates(0)
        for i in range(6):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        cp.on_chunk_commit(_make_state(100.0), gates, chunk_index=6)
        for i in range(7, 7 + cfg.after_capture_count):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=i)
        assert cp.state == CheckpointerState.COOLDOWN
        # Push through cooldown
        base = 7 + cfg.after_capture_count
        for i in range(cfg.cooldown_chunks):
            cp.on_chunk_commit(_make_state(1.0), gates, chunk_index=base + i)
        assert cp.state == CheckpointerState.MONITORING


# ---------------------------------------------------------------------------
# TestSignalLog
# ---------------------------------------------------------------------------


class TestSignalLog:
    """Signal log: files created when signal_log_enabled=True."""

    def test_signal_log_dir_created(self, tmp_path: Path) -> None:
        """signal_log/ directory should be created when enabled."""
        cfg = _make_config(tmp_path, signal_log_enabled=True, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        cp.on_chunk_commit(state, gates, chunk_index=0)
        log_dir = Path(cfg.checkpoint_dir) / "signal_log"
        assert log_dir.exists()

    def test_signal_log_file_created(self, tmp_path: Path) -> None:
        """At least one signals_*.jsonl.gz file should appear after commits."""
        cfg = _make_config(tmp_path, signal_log_enabled=True, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(5):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        cp.flush()
        log_dir = Path(cfg.checkpoint_dir) / "signal_log"
        gz_files = list(log_dir.glob("signals_*.jsonl.gz"))
        assert len(gz_files) >= 1

    def test_signal_log_readable_jsonl(self, tmp_path: Path) -> None:
        """Each line in the gzip log should parse as valid JSON with chunk_index."""
        cfg = _make_config(tmp_path, signal_log_enabled=True, min_observations=100)
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        for i in range(3):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        cp.flush()
        log_dir = Path(cfg.checkpoint_dir) / "signal_log"
        gz_files = sorted(log_dir.glob("signals_*.jsonl.gz"))
        lines = []
        for gz in gz_files:
            with gzip.open(gz, "rt") as fh:
                lines.extend(fh.readlines())
        # We cannot log the very first chunk (no previous state), so expect >= 2
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert "chunk_index" in record

    def test_signal_log_rotation(self, tmp_path: Path) -> None:
        """When max_frames is small, the log should rotate to a second file."""
        cfg = _make_config(
            tmp_path,
            signal_log_enabled=True,
            signal_log_max_frames=3,
            min_observations=100,
        )
        cp = MemoryCheckpointer(cfg)
        state = _make_state(1.0)
        gates = _make_gates(0)
        # Commit enough chunks to force at least one rotation (max_frames=3, commit 8)
        for i in range(8):
            cp.on_chunk_commit(state, gates, chunk_index=i)
        cp.flush()
        log_dir = Path(cfg.checkpoint_dir) / "signal_log"
        gz_files = list(log_dir.glob("signals_*.jsonl.gz"))
        assert len(gz_files) >= 2


def test_metadata_signal_source_preserves_full_name(tmp_path):
    """Regression: signal_source was being truncated to the last underscore
    segment, turning 'weight_delta' into 'delta'. Fix uses the decision object
    directly."""
    from titans.checkpointing.types import CheckpointEntry, MemoryCheckpointConfig
    from titans.checkpointing.memory_checkpointer import MemoryCheckpointer
    from titans.checkpointing.novelty_detector import TriggerDecision
    from titans.memory import MemoryState

    cfg = MemoryCheckpointConfig(
        checkpoint_dir=str(tmp_path),
        ring_size=2,
        after_capture_count=1,
        cooldown_chunks=1,
    )
    cp = MemoryCheckpointer(cfg, config_hash="test-hash")

    # Inject a synthetic decision whose signal_source contains underscores.
    fake_decision = TriggerDecision(
        triggered=True,
        reason="weight_delta anomaly (z=3.0)",
        confidence=0.9,
        signal_source="weight_delta",
    )

    state = MemoryState(
        weights=[torch.zeros(4, 4)],
        momentum=[torch.zeros(4, 4)],
    )

    # Manually set up a transition with signal_source = "weight_delta"
    entry = CheckpointEntry(
        state=state,
        gates=None,
        metadata={"chunk_index": 0},
        trigger_phase="during",
        weight_norms=[0.0],
        momentum_norms=[0.0],
        config_hash="test-hash",
    )
    cp._before_entry = entry
    cp._during_entry = entry
    cp._after_entries = [entry]
    cp._current_decision = fake_decision
    # Build a synthetic transition record by calling the finalize path.
    cp._finalize_transition()

    transitions = list((tmp_path / "transitions").iterdir())
    assert len(transitions) == 1
    meta = json.loads((transitions[0] / "metadata.json").read_text())
    assert meta["trigger"]["signal_source"] == "weight_delta", (
        f"signal_source was truncated: {meta['trigger']['signal_source']!r}"
    )


def test_checkpointer_state_enum_has_no_triggered():
    """TRIGGERED was set then immediately overwritten in the same method, so
    it was never observable externally. Remove it from the enum."""
    from titans.checkpointing.memory_checkpointer import CheckpointerState

    names = {s.name for s in CheckpointerState}
    assert "TRIGGERED" not in names, "Dead CheckpointerState.TRIGGERED still present"
    # Expected surviving states:
    assert {"MONITORING", "CAPTURING_AFTER", "COOLDOWN"} <= names
