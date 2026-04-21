# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""NoveltyDetector protocol and statistical implementation for auto-checkpointing.

Paper alignment: N/A — novel engineering. No reference paper specifies a
novelty-detection interface or a z-score cascade trigger. See
``docs/memory_auto_checkpointing.md`` for rationale.

This module provides:
- ``TriggerDecision`` — the result dataclass returned by any novelty detector.
- ``NoveltyDetector`` — a structural Protocol defining the detector interface.
- ``StatisticalNoveltyDetector`` — sliding-window z-score detection with signal
  cascade and bidirectional (spike + drop) detection.

The Protocol is the seam where a future learned RNN-based detector can be
plugged in without changing callers.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from titans.checkpoint_types import SignalFrame


# ---------------------------------------------------------------------------
# TriggerDecision
# ---------------------------------------------------------------------------


@dataclass
class TriggerDecision:
    """Result of a single novelty-detection observation.

    Attributes:
        triggered: Whether a novelty event was detected.
        reason: Human-readable description of the trigger.
        confidence: Scaled confidence in [0.0, 1.0].
        signal_source: Which signal caused the trigger.  One of
            ``"prediction_error"``, ``"weight_delta"``, or
            ``"momentum_shift"``.  Empty string when not triggered.
    """

    triggered: bool
    reason: str
    confidence: float
    signal_source: str


# Convenient default for a non-trigger result.
_NO_TRIGGER = TriggerDecision(triggered=False, reason="", confidence=0.0, signal_source="")

_AVAILABILITY_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# NoveltyDetector Protocol
# ---------------------------------------------------------------------------


class NoveltyDetector(Protocol):
    """Structural protocol for novelty detectors.

    Any class that implements ``observe`` and ``reset`` with these exact
    signatures is compatible, regardless of inheritance.
    """

    def observe(self, frame: SignalFrame) -> TriggerDecision:
        """Ingest one signal frame and decide whether a novelty event occurred.

        Args:
            frame: The per-chunk signal record to evaluate.

        Returns:
            A :class:`TriggerDecision` describing the outcome.
        """
        ...

    def reset(self) -> None:
        """Clear all internal state and restart from scratch.

        After calling this the detector is back in its warmup period.
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_unavailable(values: list[float], history: deque[list[float]]) -> bool:
    """Return True when a signal is effectively absent.

    A signal is considered unavailable if every current value AND every value
    in the sliding history is below *_AVAILABILITY_EPS*.

    Args:
        values: Current per-layer signal values.
        history: Sliding window of previous per-layer values.

    Returns:
        True when the signal is not carrying useful information.
    """
    if any(v >= _AVAILABILITY_EPS for v in values):
        return False
    for past_values in history:
        if any(v >= _AVAILABILITY_EPS for v in past_values):
            return False
    return True


class WelfordStats:
    """Online mean + M2 accumulator with a bounded-window evict path.

    Uses Welford's algorithm for numerically stable incremental updates and a
    closed-form reverse update when the oldest observation is evicted, giving
    O(1) per operation while tracking population variance over the retained
    window.

    Attributes:
        count: Number of observations currently represented.
        mean: Running mean of the retained observations.
        M2: Running sum of squared deltas (not divided by count).

    Example:
        >>> stats = WelfordStats()
        >>> stats.push(1.0)
        >>> stats.push(2.0)
        >>> round(stats.population_variance, 2)
        0.25
    """

    __slots__ = ("_buffer", "count", "mean", "M2")

    def __init__(self) -> None:
        self._buffer: deque[float] = deque()
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def push(self, x: float) -> None:
        """Add a new observation using Welford's incremental update.

        Args:
            x: The new observation value.
        """
        self._buffer.append(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def _evict_oldest(self) -> None:
        """Reverse-apply the oldest observation to shrink the window by one."""
        if not self._buffer:
            return
        x = self._buffer.popleft()
        if self.count <= 1:
            self.count = 0
            self.mean = 0.0
            self.M2 = 0.0
            return
        new_count = self.count - 1
        new_mean = (self.mean * self.count - x) / new_count
        # Closed-form reverse update: subtract (x - new_mean) * (x - old_mean).
        self.M2 -= (x - new_mean) * (x - self.mean)
        # Guard against tiny negative values from floating-point drift.
        if self.M2 < 0.0:
            self.M2 = 0.0
        self.count = new_count
        self.mean = new_mean

    def push_with_evict(self, x: float, window_max: int) -> None:
        """Push *x* and evict oldest entries until ``count <= window_max``.

        Args:
            x: New observation.
            window_max: Maximum number of observations to retain.
        """
        self.push(x)
        while self.count > window_max:
            self._evict_oldest()

    def clear(self) -> None:
        """Reset the accumulator to its empty initial state."""
        self._buffer.clear()
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    @property
    def population_variance(self) -> float:
        """Return the population variance (M2/count), or 0.0 when empty."""
        if self.count == 0:
            return 0.0
        return self.M2 / self.count


def _z_score_spike(
    value: float, stats: WelfordStats, sigma_threshold: float
) -> float | None:
    """Compute a spike z-score for *value* against a Welford-backed window.

    Args:
        value: The current observation.
        stats: Running Welford stats for the value window.
        sigma_threshold: Threshold multiplier for sigma.

    Returns:
        The z-score if it exceeds the threshold; ``None`` otherwise.
    """
    if stats.count < 2:
        return None
    std = math.sqrt(stats.population_variance)
    if std < _AVAILABILITY_EPS:
        return None
    z = (value - stats.mean) / std
    if z > sigma_threshold:
        return z
    return None


def _z_score_drop(
    roc: float, stats: WelfordStats, sigma_threshold: float
) -> float | None:
    """Check for a sudden downward cliff in rate-of-change.

    Args:
        roc: Current rate-of-change (current - previous value).
        stats: Running Welford stats for the rate-of-change window.
        sigma_threshold: Threshold multiplier for sigma.

    Returns:
        The magnitude of the z-score if it drops below mean - k*sigma; else None.
    """
    if stats.count < 2:
        return None
    std = math.sqrt(stats.population_variance)
    if std < _AVAILABILITY_EPS:
        return None
    z = (roc - stats.mean) / std
    if z < -sigma_threshold:
        return abs(z)
    return None


def _compute_confidence(z_magnitude: float, sigma_threshold: float) -> float:
    """Scale z-score magnitude to a [0, 1] confidence.

    Confidence is 0 at exactly the threshold and 1 at 2× the threshold,
    clamped to [0, 1].

    Args:
        z_magnitude: Absolute z-score value (already above threshold).
        sigma_threshold: Detection threshold.

    Returns:
        Confidence float in [0.0, 1.0].
    """
    return min(1.0, z_magnitude / (sigma_threshold * 2.0))


# ---------------------------------------------------------------------------
# Per-layer window state
# ---------------------------------------------------------------------------


class _LayerWindows:
    """Sliding-window state for a single layer's signal.

    Maintains:
    - A value window for spike detection (raw values).
    - A rate-of-change window for drop detection.
    - The previous raw value to compute the current rate-of-change.

    Args:
        window_size: Maximum number of historical values to retain.
    """

    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)
        self.roc_values: deque[float] = deque(maxlen=window_size)
        self.value_stats: WelfordStats = WelfordStats()
        self.roc_stats: WelfordStats = WelfordStats()
        self._prev: float | None = None

    def push(self, value: float) -> None:
        """Append a new observation and update rate-of-change history.

        Maintains Welford running stats in lockstep with the bounded
        ``values`` / ``roc_values`` deques so spike and drop z-scores update
        in O(1) per call.

        Args:
            value: The latest raw signal value for this layer.
        """
        if self._prev is not None:
            roc = value - self._prev
            self.roc_values.append(roc)
            self.roc_stats.push_with_evict(roc, self._window_size)
        self.values.append(value)
        self.value_stats.push_with_evict(value, self._window_size)
        self._prev = value

    def reset(self) -> None:
        """Clear all window history for this layer."""
        self.values.clear()
        self.roc_values.clear()
        self.value_stats.clear()
        self.roc_stats.clear()
        self._prev = None


# ---------------------------------------------------------------------------
# Aggregated ring buffer
# ---------------------------------------------------------------------------


class _AggregatedWindow:
    """Ring buffer of per-step layer-means + Welford stats, for aggregation.

    Lives on the detector and updates in O(1) per observation instead of
    rebuilding the aggregated window from every layer's deque on every call.

    Attributes:
        values: Bounded deque of per-step layer-mean values.
        roc_values: Bounded deque of per-step layer-mean rate-of-change.
        value_stats: Welford stats for ``values``.
        roc_stats: Welford stats for ``roc_values``.
    """

    __slots__ = ("values", "roc_values", "value_stats", "roc_stats", "_prev")

    def __init__(self, window_size: int) -> None:
        self.values: deque[float] = deque(maxlen=window_size)
        self.roc_values: deque[float] = deque(maxlen=window_size)
        self.value_stats: WelfordStats = WelfordStats()
        self.roc_stats: WelfordStats = WelfordStats()
        self._prev: float | None = None

    def push(self, step_mean: float, window_size: int) -> None:
        """Append a new aggregated observation and update RoC state.

        Args:
            step_mean: The mean across layers for the current step.
            window_size: Upper bound for the underlying Welford state so it
                stays in lockstep with ``values`` / ``roc_values``.
        """
        if self._prev is not None:
            roc = step_mean - self._prev
            self.roc_values.append(roc)
            self.roc_stats.push_with_evict(roc, window_size)
        self.values.append(step_mean)
        self.value_stats.push_with_evict(step_mean, window_size)
        self._prev = step_mean

    def reset(self) -> None:
        """Clear the ring buffer and Welford state."""
        self.values.clear()
        self.roc_values.clear()
        self.value_stats.clear()
        self.roc_stats.clear()
        self._prev = None


# ---------------------------------------------------------------------------
# StatisticalNoveltyDetector
# ---------------------------------------------------------------------------


class StatisticalNoveltyDetector:
    """Sliding-window z-score novelty detector with signal cascade.

    Detection uses bidirectional analysis:
    - **Spikes**: z-score of raw value against its window.
    - **Drops**: z-score of first-derivative (rate-of-change) against a
      separate window — detects sudden cliffs (e.g. grokking) while
      ignoring gradual decline.

    Signal cascade order:
    1. ``prediction_error_norms`` (primary).
    2. ``weight_delta_norms`` (fallback when primary is unavailable).
    3. ``momentum_shift_norms`` (fallback when primary AND secondary are
       unavailable).

    The cascade only fires when the *primary* signal is unavailable.  When
    the primary is present but below threshold, no cascade occurs.

    Args:
        window_size: Number of historical frames in each sliding window.
        sigma_threshold: Number of standard deviations required to declare a
            novelty event.
        min_observations: Minimum frames observed before detection is armed.
        per_layer: When ``True``, evaluate each layer independently (a spike
            in any single layer triggers).  When ``False``, aggregate by
            taking the mean across layers before evaluating.
    """

    def __init__(
        self,
        window_size: int = 50,
        sigma_threshold: float = 2.0,
        min_observations: int = 10,
        per_layer: bool = True,
    ) -> None:
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self.min_observations = min_observations
        self.per_layer = per_layer

        # observation counter — resets with reset()
        self._n_observations: int = 0

        # per-signal per-layer windows; keyed by signal name, then layer index
        # Lazily initialised on first observe() call once we know n_layers
        self._windows: dict[str, list[_LayerWindows]] = {}

        # For unavailability checks we also need raw history per signal
        # (used in _is_unavailable).  We store the last window_size raw value
        # lists per signal.
        self._raw_history: dict[str, deque[list[float]]] = {}

        # Cached aggregated (per_layer=False) ring buffers per signal.  Keeps
        # the aggregated sliding window live across observe() calls instead of
        # rebuilding it from the per-layer deques on every call.
        self._aggregated_stream: dict[str, _AggregatedWindow] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, frame: SignalFrame) -> TriggerDecision:
        """Ingest one signal frame and return a trigger decision.

        Args:
            frame: The per-chunk signal record to evaluate.

        Returns:
            A :class:`TriggerDecision` with ``triggered=True`` when novelty
            is detected.
        """
        signals = {
            "prediction_error": frame.prediction_error_norms,
            "weight_delta": frame.weight_delta_norms,
            "momentum_shift": frame.momentum_shift_norms,
        }
        n_layers = len(frame.prediction_error_norms)

        # Lazy init of per-layer windows
        self._ensure_windows(signals, n_layers)

        # Update raw history (needed for availability checks)
        for name, values in signals.items():
            self._raw_history[name].append(list(values))

        # Increment observation counter
        self._n_observations += 1

        # Push values into layer windows for all signals
        for name, values in signals.items():
            for layer_idx, val in enumerate(values):
                self._windows[name][layer_idx].push(val)

        # Still in warmup — update windows but do not trigger
        if self._n_observations < self.min_observations:
            return _NO_TRIGGER

        # Determine which signal to evaluate using cascade logic
        error_values = frame.prediction_error_norms
        error_history = self._raw_history["prediction_error"]

        if not _is_unavailable(error_values, error_history):
            # Primary signal is available — evaluate only it
            return self._evaluate_signal("prediction_error", error_values)

        # Primary unavailable — try weight_delta
        wd_values = frame.weight_delta_norms
        wd_history = self._raw_history["weight_delta"]
        if not _is_unavailable(wd_values, wd_history):
            return self._evaluate_signal("weight_delta", wd_values)

        # Both unavailable — try momentum_shift
        ms_values = frame.momentum_shift_norms
        ms_history = self._raw_history["momentum_shift"]
        if not _is_unavailable(ms_values, ms_history):
            return self._evaluate_signal("momentum_shift", ms_values)

        return _NO_TRIGGER

    def reset(self) -> None:
        """Clear all sliding window history and reset the observation counter.

        After calling this the detector re-enters its warmup period.
        """
        self._n_observations = 0
        for name in self._windows:
            for lw in self._windows[name]:
                lw.reset()
        for name in self._raw_history:
            self._raw_history[name].clear()
        for agg in self._aggregated_stream.values():
            agg.reset()

    def reset_local_windows(self, reset_flags: list[bool]) -> None:
        """Clear signal windows for specific layers (TNT shard-boundary resets).

        For each layer index ``i`` where ``reset_flags[i]`` is ``True``,
        clears all window history for that layer across every signal.

        Args:
            reset_flags: One boolean per layer.  ``True`` means reset that
                layer's windows.
        """
        for name, layer_windows in self._windows.items():
            for layer_idx, should_reset in enumerate(reset_flags):
                if should_reset and layer_idx < len(layer_windows):
                    layer_windows[layer_idx].reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_windows(self, signals: dict[str, list[float]], n_layers: int) -> None:
        """Lazily initialise per-signal per-layer window structures.

        Args:
            signals: Mapping from signal name to per-layer value list.
            n_layers: Number of layers detected from the current frame.
        """
        for name in signals:
            if name not in self._windows:
                self._windows[name] = [
                    _LayerWindows(self.window_size) for _ in range(n_layers)
                ]
                self._raw_history[name] = deque(maxlen=self.window_size)

    def _evaluate_signal(self, signal_name: str, values: list[float]) -> TriggerDecision:
        """Run spike + drop detection on a chosen signal.

        When ``per_layer=True`` evaluates each layer independently; a single
        triggering layer is sufficient.  When ``per_layer=False`` aggregates
        to a scalar mean before evaluating.

        Args:
            signal_name: Key into ``self._windows``.
            values: Current per-layer signal values.

        Returns:
            A :class:`TriggerDecision`.
        """
        layer_windows = self._windows[signal_name]

        if self.per_layer:
            return self._evaluate_per_layer(signal_name, values, layer_windows)
        else:
            return self._evaluate_aggregated(signal_name, values, layer_windows)

    def _evaluate_per_layer(
        self,
        signal_name: str,
        values: list[float],
        layer_windows: list[_LayerWindows],
    ) -> TriggerDecision:
        """Evaluate layers independently; first trigger wins.

        Args:
            signal_name: Source label for TriggerDecision.
            values: Current per-layer values.
            layer_windows: Per-layer window objects.

        Returns:
            A :class:`TriggerDecision`.
        """
        best_abs_z: float | None = None
        best_signed_z: float | None = None
        best_direction: str = "spike"

        for _layer_idx, (val, lw) in enumerate(zip(values, layer_windows)):
            # Spike check — signed z-score filtered above +sigma.
            z_spike = _z_score_spike(val, lw.value_stats, self.sigma_threshold)
            if z_spike is not None:
                abs_z = abs(z_spike)
                if best_abs_z is None or abs_z > best_abs_z:
                    best_abs_z = abs_z
                    best_signed_z = z_spike
                    best_direction = "spike"

            # Drop check via rate-of-change.
            # The current RoC was already appended to lw.roc_values during push();
            # we read it directly rather than recomputing from lw._prev (which has
            # already been updated to the current value by push()).
            if lw.roc_values:
                current_roc = lw.roc_values[-1]
                z_drop = _z_score_drop(current_roc, lw.roc_stats, self.sigma_threshold)
                if z_drop is not None:
                    abs_z = abs(z_drop)
                    if best_abs_z is None or abs_z > best_abs_z:
                        best_abs_z = abs_z
                        # _z_score_drop returns magnitude; the underlying z was
                        # strongly negative, so the signed value is -abs_z.
                        best_signed_z = -abs_z
                        best_direction = "drop"

        if best_abs_z is not None and best_signed_z is not None:
            confidence = _compute_confidence(best_abs_z, self.sigma_threshold)
            return TriggerDecision(
                triggered=True,
                reason=(
                    f"{signal_name} {best_direction} "
                    f"(|z|={best_abs_z:.2f}, signed={best_signed_z:+.2f})"
                ),
                confidence=confidence,
                signal_source=signal_name,
            )
        return _NO_TRIGGER

    def _evaluate_aggregated(
        self,
        signal_name: str,
        values: list[float],
        layer_windows: list[_LayerWindows],
    ) -> TriggerDecision:
        """Aggregate layers to a scalar mean then evaluate.

        Args:
            signal_name: Source label for TriggerDecision.
            values: Current per-layer values.
            layer_windows: Per-layer window objects.

        Returns:
            A :class:`TriggerDecision`.
        """
        n = len(values)
        if n == 0:
            return _NO_TRIGGER

        # Push the current step's layer-mean into the cached aggregated ring
        # buffer so Welford stats update in O(1) per observation instead of
        # rebuilding the window from every layer's deque on every call.
        current_mean = sum(values) / n
        agg = self._aggregated_stream.get(signal_name)
        if agg is None:
            agg = _AggregatedWindow(self.window_size)
            self._aggregated_stream[signal_name] = agg
        agg.push(current_mean, self.window_size)

        # layer_windows retained in the signature for API parity with
        # _evaluate_per_layer; unused here since the aggregate is cached.
        del layer_windows

        best_abs_z: float | None = None
        best_signed_z: float | None = None
        best_direction: str = "spike"

        # Spike check on aggregated scalar — stats already include current_mean.
        z_spike = _z_score_spike(current_mean, agg.value_stats, self.sigma_threshold)
        if z_spike is not None:
            abs_z = abs(z_spike)
            if best_abs_z is None or abs_z > best_abs_z:
                best_abs_z = abs_z
                best_signed_z = z_spike
                best_direction = "spike"

        # Drop check on aggregated rate-of-change.  The current RoC was appended
        # by agg.push() above; read it directly.
        if agg.roc_values:
            current_agg_roc = agg.roc_values[-1]
            z_drop = _z_score_drop(current_agg_roc, agg.roc_stats, self.sigma_threshold)
            if z_drop is not None:
                abs_z = abs(z_drop)
                if best_abs_z is None or abs_z > best_abs_z:
                    best_abs_z = abs_z
                    # _z_score_drop returns magnitude; the underlying z was
                    # strongly negative, so the signed value is -abs_z.
                    best_signed_z = -abs_z
                    best_direction = "drop"

        if best_abs_z is not None and best_signed_z is not None:
            confidence = _compute_confidence(best_abs_z, self.sigma_threshold)
            return TriggerDecision(
                triggered=True,
                reason=(
                    f"{signal_name} {best_direction} "
                    f"(|z|={best_abs_z:.2f}, signed={best_signed_z:+.2f})"
                ),
                confidence=confidence,
                signal_source=signal_name,
            )
        return _NO_TRIGGER
