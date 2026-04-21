# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""MemoryCheckpointer state machine and ring buffer for memory auto-checkpointing.

Paper alignment: N/A — novel engineering. The three-state machine
(MONITORING → CAPTURING_AFTER → COOLDOWN) and ring buffer are
project-specific plumbing, not derived from any Titans / TNT / AttnRes paper.

Provides the high-level orchestrator that ties together:
- :class:`StatisticalNoveltyDetector` for anomaly detection,
- A FIFO ring buffer of :class:`CheckpointEntry` snapshots,
- A three-state machine (MONITORING → CAPTURING_AFTER → COOLDOWN),
- Disk I/O for transition records and optional gzip-compressed signal logs.
"""

from __future__ import annotations

import enum
import gzip
import json
import shutil
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from titans.checkpoint_signals import (
    build_signal_frame,
    compute_momentum_norms,
    compute_weight_norms,
)
from titans.checkpoint_types import (
    CheckpointEntry,
    GateSnapshot,
    MemoryCheckpointConfig,
    SignalFrame,
    TransitionRecord,
)
from titans.memory import MemoryState, TNTMemoryState
from titans.memory_dump import save_memory_states
from titans.novelty_detector import StatisticalNoveltyDetector, TriggerDecision


# ---------------------------------------------------------------------------
# CheckpointerState
# ---------------------------------------------------------------------------


class CheckpointerState(enum.Enum):
    """State machine states for :class:`MemoryCheckpointer`.

    Attributes:
        MONITORING: Normal operation; ring buffer fills; detector is armed.
        CAPTURING_AFTER: Collecting post-transition snapshots.
        COOLDOWN: Transition written; ignoring further triggers for a fixed
            number of chunks.
    """

    MONITORING = "monitoring"
    CAPTURING_AFTER = "capturing_after"
    COOLDOWN = "cooldown"


# ---------------------------------------------------------------------------
# _SignalLogWriter
# ---------------------------------------------------------------------------


class _SignalLogWriter:
    """Gzip-compressed JSONL signal log writer with file rotation.

    Writes :class:`SignalFrame` records as newline-delimited JSON entries in
    gzip-compressed files.  Rotates to a new file every *max_frames* entries.
    Flushes the gzip stream every 50 frames for crash resilience.

    Args:
        log_dir: Directory where log files are stored.
        max_frames: Number of frames per file before rotation.
        fmt: Serialisation format (currently only ``"jsonl"`` is supported).
    """

    _FLUSH_INTERVAL: int = 50

    def __init__(self, log_dir: Path, max_frames: int, fmt: str) -> None:
        self._log_dir = log_dir
        self._max_frames = max_frames
        self._fmt = fmt
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file_index: int = 0
        self._frame_count: int = 0
        self._fh: gzip.GzipFile | None = None
        self._open_new_file()

    def _open_new_file(self) -> None:
        """Open the next rotation file."""
        if self._fh is not None:
            self._fh.close()
        self._file_index += 1
        filename = f"signals_{self._file_index:06d}.jsonl.gz"
        path = self._log_dir / filename
        self._fh = gzip.open(str(path), "wt", encoding="utf-8")
        self._frame_count = 0

    def write(self, frame: SignalFrame) -> None:
        """Append one :class:`SignalFrame` to the current log file.

        Args:
            frame: The signal record to serialise.
        """
        if self._fh is None:
            return
        self._fh.write(json.dumps(frame.to_dict()) + "\n")
        self._frame_count += 1
        if self._frame_count % self._FLUSH_INTERVAL == 0:
            self._fh.flush()
        if self._frame_count >= self._max_frames:
            self._open_new_file()

    def close(self) -> None:
        """Flush and close the current log file."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            finally:
                self._fh = None


# ---------------------------------------------------------------------------
# MemoryCheckpointer
# ---------------------------------------------------------------------------


class MemoryCheckpointer:
    """Orchestrates per-chunk memory checkpointing with novelty-triggered captures.

    Maintains a FIFO ring buffer of recent :class:`CheckpointEntry` snapshots
    and a state machine that transitions on novelty events detected by a
    :class:`StatisticalNoveltyDetector`.  When a transition is detected:

    1. The *calmest* entry (lowest summed weight + momentum norms) in the ring
       buffer is saved as the ``before`` snapshot.
    2. The triggering entry is saved as ``during``.
    3. The next ``after_capture_count`` entries are collected as ``after``.
    4. The bundle is written to disk as a :class:`TransitionRecord`.

    Args:
        config: Checkpointing configuration.
        config_hash: Optional hash of the config for staleness detection.
    """

    def __init__(self, config: MemoryCheckpointConfig, config_hash: str = "") -> None:
        self._config = config
        self._config_hash = config_hash
        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Novelty detector
        self._detector = StatisticalNoveltyDetector(
            window_size=config.window_size,
            sigma_threshold=config.sigma_threshold,
            min_observations=config.min_observations,
        )

        # Ring buffer (maxlen enforces FIFO eviction)
        self._ring_buffer: deque[CheckpointEntry] = deque(maxlen=config.ring_size)

        # Signal log writer (optional)
        self._signal_log: _SignalLogWriter | None = None
        if config.signal_log_enabled:
            log_dir = self._checkpoint_dir / "signal_log"
            self._signal_log = _SignalLogWriter(
                log_dir=log_dir,
                max_frames=config.signal_log_max_frames,
                fmt=config.signal_log_format,
            )

        # State machine
        self.state: CheckpointerState = CheckpointerState.MONITORING

        # Transition capture state
        self._before_entry: CheckpointEntry | None = None
        self._during_entry: CheckpointEntry | None = None
        self._after_entries: list[CheckpointEntry] = []
        self._after_remaining: int = 0
        self._cooldown_remaining: int = 0
        self._signal_window: list[SignalFrame] = []
        self._current_decision: TriggerDecision | None = None

        # Counters
        self._total_chunks: int = 0
        self._transitions_recorded: int = 0

        # Session metadata
        self._session_start: str = datetime.now(UTC).isoformat()

        # Previous state for signal computation (one per block)
        self._prev_states: list[MemoryState | TNTMemoryState] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_chunk_commit(
        self,
        committed_state: list[MemoryState | TNTMemoryState],
        gates: list[GateSnapshot | None],
        chunk_index: int,
        prediction_errors: list[list[float]] | None = None,
    ) -> None:
        """Process one committed chunk.

        Args:
            committed_state: Per-block memory states (one per model block).
            gates: Per-block gate snapshots (may contain ``None`` entries).
            chunk_index: Global chunk index.
            prediction_errors: Optional per-block per-layer prediction-error
                norms from the NLTM inner-loop. When ``None``, the signal
                builder falls back to zeros (legacy behaviour) and the
                novelty detector cascades past the primary signal.
        """
        self._total_chunks += 1

        # Use the FIRST block for signal computation
        current = committed_state[0]
        gate = gates[0]

        # Build signal frame (requires a previous state)
        frame: SignalFrame | None = None
        if self._prev_states is not None and gate is not None:
            frame = self._build_frame(
                self._prev_states[0], current, gate, chunk_index,
                prediction_error_norms=(
                    prediction_errors[0] if prediction_errors else None
                ),
            )
        elif self._prev_states is not None and gate is None:
            # Build a minimal frame without gate signals when gate is absent
            frame = None  # skip this chunk if gates unavailable

        self._prev_states = list(committed_state)

        # Build checkpoint entry regardless (ring buffer uses it for "before" selection)
        weight_norms = compute_weight_norms(current)
        momentum_norms = compute_momentum_norms(current)
        entry = CheckpointEntry(
            state=current,
            gates=gate,
            metadata={"chunk_index": chunk_index},
            trigger_phase="monitoring",
            weight_norms=weight_norms,
            momentum_norms=momentum_norms,
            config_hash=self._config_hash,
        )

        # Write signal frame to log if enabled
        if frame is not None and self._signal_log is not None:
            self._signal_log.write(frame)

        # Track signal window for transition records
        if frame is not None:
            self._signal_window.append(frame)
            # Keep sliding window bounded to ring_size
            if len(self._signal_window) > self._config.ring_size * 2:
                self._signal_window = self._signal_window[-self._config.ring_size * 2:]

        # State machine dispatch
        if self.state == CheckpointerState.MONITORING:
            self._ring_buffer.append(entry)
            if frame is not None:
                decision = self._detector.observe(frame)
                if decision.triggered:
                    self._handle_trigger(entry, decision, chunk_index)

        elif self.state == CheckpointerState.CAPTURING_AFTER:
            entry = CheckpointEntry(
                state=current,
                gates=gate,
                metadata={"chunk_index": chunk_index},
                trigger_phase="after",
                weight_norms=weight_norms,
                momentum_norms=momentum_norms,
                config_hash=self._config_hash,
            )
            self._after_entries.append(entry)
            self._after_remaining -= 1
            if self._after_remaining <= 0:
                self._finalize_transition()
                self._enter_cooldown()

        elif self.state == CheckpointerState.COOLDOWN:
            self._ring_buffer.append(entry)
            if frame is not None:
                self._detector.observe(frame)  # observe but ignore trigger
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self.state = CheckpointerState.MONITORING

    def flush(self) -> None:
        """Write final ring buffer and session metadata to disk.

        Saves:
        - ``ring_buffer_final.npz``: States from all entries in the ring buffer.
        - ``session.json``: Session-level metadata.

        Also closes any open signal log file.
        """
        # Write ring_buffer_final.npz
        if self._ring_buffer:
            states = [e.state for e in self._ring_buffer]
            ring_path = self._checkpoint_dir / "ring_buffer_final.npz"
            save_memory_states(states, ring_path)

        # Write session.json
        session_data = {
            "session_start": self._session_start,
            "session_end": datetime.now(UTC).isoformat(),
            "total_chunks_processed": self._total_chunks,
            "transitions_recorded": self._transitions_recorded,
            "config": self._config.to_dict(),
            "config_hash": self._config_hash,
        }
        session_path = self._checkpoint_dir / "session.json"
        session_path.write_text(json.dumps(session_data, indent=2))

        # Close signal log
        if self._signal_log is not None:
            self._signal_log.close()

    # ------------------------------------------------------------------
    # Private state machine helpers
    # ------------------------------------------------------------------

    def _handle_trigger(
        self,
        entry: CheckpointEntry,
        decision: TriggerDecision,
        chunk_index: int,
    ) -> None:
        """React to a novelty detection trigger.

        Selects the calmest before-entry from the ring buffer, marks the
        current entry as ``during``, and enters CAPTURING_AFTER.

        Args:
            entry: The entry for the triggering chunk.
            decision: The trigger decision from the detector.
            chunk_index: Current chunk index.
        """
        # Select calmest "before" entry from ring buffer
        self._before_entry = self._select_calmest_entry()

        # Mark triggering chunk as "during"
        self._during_entry = CheckpointEntry(
            state=entry.state,
            gates=entry.gates,
            metadata={**entry.metadata, "trigger": decision.signal_source},
            trigger_phase="during",
            weight_norms=entry.weight_norms,
            momentum_norms=entry.momentum_norms,
            config_hash=self._config_hash,
        )
        self._current_decision = decision

        # Prepare after-capture
        self._after_entries = []
        self._after_remaining = self._config.after_capture_count

        self.state = CheckpointerState.CAPTURING_AFTER

    def _select_calmest_entry(self) -> CheckpointEntry | None:
        """Return the ring buffer entry with the lowest total weight + momentum norms.

        Returns:
            The calmest :class:`CheckpointEntry`, or ``None`` if ring is empty.
        """
        if not self._ring_buffer:
            return None
        return min(
            self._ring_buffer,
            key=lambda e: sum(e.weight_norms) + sum(e.momentum_norms),
        )

    def _resolve_signal_source(self) -> str:
        """Return the canonical signal_source string for the active decision,
        or ``"unknown"`` if no decision is active (defensive -- in practice
        this should never happen during ``_finalize_transition`` because the
        state machine guarantees ``_current_decision`` is set when entering
        the CAPTURING_AFTER branch).
        """
        decision = self._current_decision
        return decision.signal_source if decision is not None else "unknown"

    def _finalize_transition(self) -> None:
        """Bundle before/during/after into a TransitionRecord and write to disk."""
        if self._before_entry is None or self._during_entry is None:
            return

        before_chunk = self._before_entry.metadata.get("chunk_index", 0)
        after_chunks = [e.metadata.get("chunk_index", 0) for e in self._after_entries]
        duration = (after_chunks[-1] - before_chunk) if after_chunks else 0

        decision = self._current_decision
        magnitude = decision.confidence if decision else 0.0

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        transition_id = f"tr_{timestamp}_{self._resolve_signal_source()}"

        record = TransitionRecord(
            before=self._before_entry,
            during=self._during_entry,
            after=list(self._after_entries),
            signal_window=list(self._signal_window),
            transition_id=transition_id,
            transition_magnitude=magnitude,
            duration_chunks=duration,
        )

        self._write_transition(record)
        self._transitions_recorded += 1
        self._update_session_json()

        # Reset capture state
        self._before_entry = None
        self._during_entry = None
        self._after_entries = []
        self._current_decision = None

    def _enter_cooldown(self) -> None:
        """Switch into COOLDOWN state for cooldown_chunks chunks."""
        self._cooldown_remaining = self._config.cooldown_chunks
        self.state = CheckpointerState.COOLDOWN

    def _write_transition(self, record: TransitionRecord) -> None:
        """Persist a :class:`TransitionRecord` to the transitions directory.

        Creates a subdirectory named ``tr_{timestamp}_{signal_source}`` and
        writes:
        - ``before.npz``
        - ``during.npz``
        - ``after_001.npz`` ... ``after_NNN.npz``
        - ``signal_window.jsonl.gz``
        - ``metadata.json``

        Then enforces the retention policy.

        Args:
            record: The complete transition record to write.
        """
        transitions_root = self._checkpoint_dir / "transitions"
        transitions_root.mkdir(parents=True, exist_ok=True)
        td = transitions_root / record.transition_id
        td.mkdir(parents=True, exist_ok=True)

        # Save before and during states
        if record.before is not None:
            save_memory_states([record.before.state], td / "before.npz")
        if record.during is not None:
            save_memory_states([record.during.state], td / "during.npz")

        # Save after states
        for idx, after_entry in enumerate(record.after, start=1):
            save_memory_states([after_entry.state], td / f"after_{idx:03d}.npz")

        # Save signal window as gzip JSONL
        sig_path = td / "signal_window.jsonl.gz"
        with gzip.open(str(sig_path), "wt", encoding="utf-8") as fh:
            for frame in record.signal_window:
                fh.write(json.dumps(frame.to_dict()) + "\n")

        # Build and save metadata.json
        before_chunk = record.before.metadata.get("chunk_index", 0) if record.before else 0
        during_chunk = record.during.metadata.get("chunk_index", 0) if record.during else 0
        after_range: list[int] = []
        if record.after:
            first_after = record.after[0].metadata.get("chunk_index", 0)
            last_after = record.after[-1].metadata.get("chunk_index", 0)
            after_range = [first_after, last_after]

        decision = self._current_decision
        metadata: dict[str, Any] = {
            "transition_id": record.transition_id,
            "trigger": {
                "signal_source": self._resolve_signal_source(),
                "confidence": record.transition_magnitude,
                "reason": decision.reason if decision else "",
            },
            "transition_magnitude": record.transition_magnitude,
            "duration_chunks": record.duration_chunks,
            "before_chunk_index": before_chunk,
            "during_chunk_index": during_chunk,
            "after_chunk_range": after_range,
            "config_hash": self._config_hash,
            "session_start": self._session_start,
        }
        (td / "metadata.json").write_text(json.dumps(metadata, indent=2))

        self._enforce_retention()

    def _enforce_retention(self) -> None:
        """Delete oldest transition directories beyond keep_last_n_transitions."""
        transitions_root = self._checkpoint_dir / "transitions"
        if not transitions_root.exists():
            return
        dirs = sorted(
            [d for d in transitions_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        while len(dirs) > self._config.keep_last_n_transitions:
            oldest = dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def _update_session_json(self) -> None:
        """Incrementally update session.json after each transition write."""
        session_path = self._checkpoint_dir / "session.json"
        if session_path.exists():
            try:
                data = json.loads(session_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}

        data.update(
            {
                "session_start": self._session_start,
                "last_updated": datetime.now(UTC).isoformat(),
                "total_chunks_processed": self._total_chunks,
                "transitions_recorded": self._transitions_recorded,
                "config": self._config.to_dict(),
                "config_hash": self._config_hash,
            }
        )
        session_path.write_text(json.dumps(data, indent=2))

    def _build_frame(
        self,
        old_state: MemoryState | TNTMemoryState,
        new_state: MemoryState | TNTMemoryState,
        gates: GateSnapshot,
        chunk_index: int,
        prediction_error_norms: list[float] | None = None,
    ) -> SignalFrame:
        """Build a :class:`SignalFrame` from consecutive states.

        Args:
            old_state: Memory state from the previous chunk.
            new_state: Memory state from the current chunk.
            gates: Gate snapshot for the current chunk.
            chunk_index: Current chunk index.
            prediction_error_norms: Optional per-layer prediction-error norms
                from the NLTM inner-loop. When ``None``, the builder fills
                zeros (legacy fallback).

        Returns:
            A fully-populated :class:`SignalFrame`.
        """
        return build_signal_frame(
            old_state=old_state,
            new_state=new_state,
            gates=gates,
            chunk_index=chunk_index,
            prediction_error_norms=prediction_error_norms,
        )
