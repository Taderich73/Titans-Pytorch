# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Data structures for the memory auto-checkpointing system.

Provides dataclasses that capture gate values, per-chunk signal records,
saved checkpoint snapshots, phase-transition records, and configuration for
the auto-checkpointing subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from titans.memory import MemoryState, TNTMemoryState


# ---------------------------------------------------------------------------
# GateSnapshot
# ---------------------------------------------------------------------------


@dataclass
class GateSnapshot:
    """Captures data-dependent gate values from a NeuralLongTermMemory forward().

    Each list field contains one tensor per memory layer, matching the number
    of layers in the MemoryMLP.

    Attributes:
        alpha: Decay gate values per memory layer.
        theta: Learning-rate gate values per memory layer.
        eta: Momentum gate values per memory layer.
        delta: Huber knee gate per memory layer, or None when the memory
            objective is not ``"huber"``.
        input_activation_norm: Frobenius norm of the input activation mean
            (scalar float computed before gating).
        chunk_index: Which chunk produced these gate values.
    """

    alpha: list[torch.Tensor]
    theta: list[torch.Tensor]
    eta: list[torch.Tensor]
    delta: list[torch.Tensor] | None
    input_activation_norm: float
    chunk_index: int

    def detach(self) -> GateSnapshot:
        """Return a new GateSnapshot with all tensors detached from the graph.

        Returns:
            A shallow copy with every tensor replaced by its detached version.
            Non-tensor fields (``input_activation_norm``, ``chunk_index``) are
            copied as-is.
        """
        return GateSnapshot(
            alpha=[t.detach() for t in self.alpha],
            theta=[t.detach() for t in self.theta],
            eta=[t.detach() for t in self.eta],
            delta=[t.detach() for t in self.delta] if self.delta is not None else None,
            input_activation_norm=self.input_activation_norm,
            chunk_index=self.chunk_index,
        )

    def to(self, device: torch.device) -> GateSnapshot:
        """Return a new GateSnapshot with all tensors moved to *device*.

        Args:
            device: Target device.

        Returns:
            A new GateSnapshot whose tensors live on *device*.
        """
        return GateSnapshot(
            alpha=[t.to(device) for t in self.alpha],
            theta=[t.to(device) for t in self.theta],
            eta=[t.to(device) for t in self.eta],
            delta=[t.to(device) for t in self.delta] if self.delta is not None else None,
            input_activation_norm=self.input_activation_norm,
            chunk_index=self.chunk_index,
        )


# ---------------------------------------------------------------------------
# SignalFrame
# ---------------------------------------------------------------------------


@dataclass
class SignalFrame:
    """Lightweight per-chunk signal record for novelty detection and logging.

    All list fields are per-layer (not aggregated). TNT-only fields
    (``local_signal_norms``, ``local_reset_flags``) are ``None`` for
    non-TNT memory variants (MAC, MAG, MAL).

    Attributes:
        chunk_index: Which chunk this frame corresponds to.
        prediction_error_norms: L2 norm of prediction errors per layer.
        weight_delta_norms: Norm of weight changes per layer.
        momentum_shift_norms: Norm of momentum changes per layer.
        gradient_norms: Norm of gradients per layer.
        weight_norms: Frobenius norm of current weights per layer.
        momentum_norms: Frobenius norm of current momentum per layer.
        gate_alpha_means: Mean alpha gate value per layer.
        gate_theta_means: Mean theta gate value per layer.
        gate_eta_means: Mean eta gate value per layer.
        batch_variance: Variance across the batch dimension, or None if not
            computed.
        local_signal_norms: TNT only — per-local-memory signal norms.
            Outer list indexes local memories; inner list indexes time steps.
            None for MAC/MAG/MAL.
        local_reset_flags: TNT only — whether each local memory was reset this
            chunk. None for MAC/MAG/MAL.
    """

    chunk_index: int
    prediction_error_norms: list[float]
    weight_delta_norms: list[float]
    momentum_shift_norms: list[float]
    gradient_norms: list[float]
    weight_norms: list[float]
    momentum_norms: list[float]
    gate_alpha_means: list[float]
    gate_theta_means: list[float]
    gate_eta_means: list[float]
    batch_variance: float | None
    local_signal_norms: list[list[float]] | None
    local_reset_flags: list[bool] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation of this frame.

        Returns:
            Dict with all fields serialised to Python primitives (no tensors).
        """
        return {
            "chunk_index": self.chunk_index,
            "prediction_error_norms": self.prediction_error_norms,
            "weight_delta_norms": self.weight_delta_norms,
            "momentum_shift_norms": self.momentum_shift_norms,
            "gradient_norms": self.gradient_norms,
            "weight_norms": self.weight_norms,
            "momentum_norms": self.momentum_norms,
            "gate_alpha_means": self.gate_alpha_means,
            "gate_theta_means": self.gate_theta_means,
            "gate_eta_means": self.gate_eta_means,
            "batch_variance": self.batch_variance,
            "local_signal_norms": self.local_signal_norms,
            "local_reset_flags": self.local_reset_flags,
        }


# ---------------------------------------------------------------------------
# CheckpointEntry
# ---------------------------------------------------------------------------


@dataclass
class CheckpointEntry:
    """A single saved snapshot pairing memory state with its context.

    Attributes:
        state: The committed memory state at the time of capture.
        gates: Optional gate snapshot captured alongside the state.  May be
            ``None`` when gate capture was disabled.
        metadata: Arbitrary JSON-compatible metadata (e.g. step, loss).
        trigger_phase: When the snapshot was taken relative to a detected
            transition.  One of ``"before"``, ``"during"``, ``"after"``, or
            ``"periodic"``.
        weight_norms: Frobenius norm of memory weights per layer at capture
            time.
        momentum_norms: Frobenius norm of memory momentum per layer at capture
            time.
        config_hash: Hash of the :class:`MemoryCheckpointConfig` that produced
            this entry, for staleness detection on reload.
    """

    state: MemoryState | TNTMemoryState
    gates: GateSnapshot | None
    metadata: dict[str, Any]
    trigger_phase: str
    weight_norms: list[float]
    momentum_norms: list[float]
    config_hash: str


# ---------------------------------------------------------------------------
# TransitionRecord
# ---------------------------------------------------------------------------


@dataclass
class TransitionRecord:
    """A complete phase-transition capture spanning before / during / after.

    Attributes:
        before: Checkpoint taken immediately before the transition was
            detected.
        during: Checkpoint taken at the peak of the detected transition.
        after: Checkpoints captured in the ``after_capture_count`` chunks
            following the transition.  May be empty while still in progress.
        signal_window: The sliding window of :class:`SignalFrame` records that
            surrounded the transition, for post-hoc analysis.
        transition_id: Unique string identifier for this transition (e.g. a
            UUID or monotonic counter string).
        transition_magnitude: The signal magnitude that triggered detection,
            measured in standard deviations above the rolling mean.
        duration_chunks: Number of chunks elapsed between the ``before`` and
            the last ``after`` entry (or 0 if ``after`` is empty).
    """

    before: CheckpointEntry
    during: CheckpointEntry
    after: list[CheckpointEntry]
    signal_window: list[SignalFrame]
    transition_id: str
    transition_magnitude: float
    duration_chunks: int


# ---------------------------------------------------------------------------
# MemoryCheckpointConfig
# ---------------------------------------------------------------------------


@dataclass
class MemoryCheckpointConfig:
    """Configuration for the memory auto-checkpointing system.

    Attributes:
        checkpoint_dir: Directory where checkpoints are persisted.
        ring_size: Maximum number of periodic checkpoints kept in the ring
            buffer before the oldest is evicted.
        sigma_threshold: Number of standard deviations above the rolling mean
            required to declare a novelty transition.
        window_size: Sliding window length (in chunks) used to compute the
            rolling mean and standard deviation for novelty detection.
        min_observations: Minimum number of chunks observed before novelty
            detection is armed.
        cooldown_chunks: Minimum number of chunks between consecutive
            transition detections to avoid double-triggering.
        after_capture_count: Number of post-transition chunks to capture as
            ``after`` entries in a :class:`TransitionRecord`.
        keep_last_n_transitions: Maximum number of :class:`TransitionRecord`
            objects retained in memory.
        signal_log_enabled: Whether to write :class:`SignalFrame` records to
            disk.
        signal_log_format: Serialisation format for signal logs.  Currently
            only ``"jsonl"`` is supported.
        signal_log_max_frames: Maximum number of :class:`SignalFrame` records
            to buffer/write before rolling over.
        quantize_checkpoints: Whether to quantize checkpoint tensors before
            writing to disk to reduce storage.
        checkpoint_weight_bits: Bit-width used when quantising weight tensors.
        checkpoint_momentum_bits: Bit-width used when quantising momentum
            tensors.
    """

    checkpoint_dir: str = "memory_checkpoints"
    ring_size: int = 25
    sigma_threshold: float = 2.0
    window_size: int = 50
    min_observations: int = 10
    cooldown_chunks: int = 20
    after_capture_count: int = 5
    keep_last_n_transitions: int = 10
    signal_log_enabled: bool = False
    signal_log_format: str = "jsonl"
    signal_log_max_frames: int = 100_000
    quantize_checkpoints: bool = False
    checkpoint_weight_bits: int = 8
    checkpoint_momentum_bits: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a JSON-compatible dict.

        Returns:
            Dict mapping every field name to its current value.
        """
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "ring_size": self.ring_size,
            "sigma_threshold": self.sigma_threshold,
            "window_size": self.window_size,
            "min_observations": self.min_observations,
            "cooldown_chunks": self.cooldown_chunks,
            "after_capture_count": self.after_capture_count,
            "keep_last_n_transitions": self.keep_last_n_transitions,
            "signal_log_enabled": self.signal_log_enabled,
            "signal_log_format": self.signal_log_format,
            "signal_log_max_frames": self.signal_log_max_frames,
            "quantize_checkpoints": self.quantize_checkpoints,
            "checkpoint_weight_bits": self.checkpoint_weight_bits,
            "checkpoint_momentum_bits": self.checkpoint_momentum_bits,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryCheckpointConfig:
        """Construct a config from a (possibly partial) dict.

        Unknown keys are silently ignored so that configs serialised by a
        newer version of the code can be loaded by an older version without
        error.

        Args:
            d: Dict of field values, typically from :meth:`to_dict` or a JSON
                file.  Missing keys fall back to their dataclass defaults.

        Returns:
            A :class:`MemoryCheckpointConfig` instance.
        """
        known_fields = {
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
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
