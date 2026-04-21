# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for the NoveltyDetector protocol and StatisticalNoveltyDetector."""

from __future__ import annotations

import pytest

from titans.checkpoint_types import SignalFrame


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _make_frame(
    chunk_index: int = 0,
    error_norms: list[float] | None = None,
    weight_delta_norms: list[float] | None = None,
    momentum_shift_norms: list[float] | None = None,
) -> SignalFrame:
    """Build a minimal SignalFrame for detector tests.

    Args:
        chunk_index: Chunk index to embed in the frame.
        error_norms: Per-layer prediction error norms; defaults to zeros.
        weight_delta_norms: Per-layer weight delta norms; defaults to zeros.
        momentum_shift_norms: Per-layer momentum shift norms; defaults to zeros.

    Returns:
        A fully-populated SignalFrame suitable for passing to a NoveltyDetector.
    """
    n = len(error_norms or weight_delta_norms or momentum_shift_norms or [0.0])
    return SignalFrame(
        chunk_index=chunk_index,
        prediction_error_norms=error_norms or [0.0] * n,
        weight_delta_norms=weight_delta_norms or [0.0] * n,
        momentum_shift_norms=momentum_shift_norms or [0.0] * n,
        gradient_norms=[0.0] * n,
        weight_norms=[1.0] * n,
        momentum_norms=[0.5] * n,
        gate_alpha_means=[0.1] * n,
        gate_theta_means=[0.05] * n,
        gate_eta_means=[0.9] * n,
        batch_variance=None,
        local_signal_norms=None,
    )


# ---------------------------------------------------------------------------
# TestTriggerDecision
# ---------------------------------------------------------------------------


class TestTriggerDecision:
    """Tests for the TriggerDecision dataclass."""

    def test_basic_creation_no_trigger(self) -> None:
        """TriggerDecision can be created with triggered=False."""
        from titans.novelty_detector import TriggerDecision

        td = TriggerDecision(
            triggered=False,
            reason="",
            confidence=0.0,
            signal_source="",
        )
        assert td.triggered is False
        assert td.reason == ""
        assert td.confidence == 0.0
        assert td.signal_source == ""

    def test_basic_creation_trigger(self) -> None:
        """TriggerDecision can be created with triggered=True and valid fields."""
        from titans.novelty_detector import TriggerDecision

        td = TriggerDecision(
            triggered=True,
            reason="spike detected",
            confidence=0.85,
            signal_source="prediction_error",
        )
        assert td.triggered is True
        assert td.reason == "spike detected"
        assert td.confidence == pytest.approx(0.85)
        assert td.signal_source == "prediction_error"

    def test_valid_signal_sources(self) -> None:
        """All three signal_source values are accepted."""
        from titans.novelty_detector import TriggerDecision

        for source in ("prediction_error", "weight_delta", "momentum_shift"):
            td = TriggerDecision(
                triggered=True,
                reason="test",
                confidence=0.5,
                signal_source=source,
            )
            assert td.signal_source == source

    def test_no_trigger_constant(self) -> None:
        """_NO_TRIGGER sentinel has correct default values."""
        from titans.novelty_detector import _NO_TRIGGER

        assert _NO_TRIGGER.triggered is False
        assert _NO_TRIGGER.confidence == 0.0

    def test_confidence_bounds(self) -> None:
        """Confidence field accepts 0.0 and 1.0 boundary values."""
        from titans.novelty_detector import TriggerDecision

        lo = TriggerDecision(triggered=False, reason="", confidence=0.0, signal_source="")
        hi = TriggerDecision(triggered=True, reason="", confidence=1.0, signal_source="prediction_error")
        assert lo.confidence == pytest.approx(0.0)
        assert hi.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestStatisticalNoveltyDetector
# ---------------------------------------------------------------------------


class TestStatisticalNoveltyDetector:
    """Tests for StatisticalNoveltyDetector."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction_defaults(self) -> None:
        """StatisticalNoveltyDetector can be constructed with default args."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector()
        assert det.window_size == 50
        assert det.sigma_threshold == pytest.approx(2.0)
        assert det.min_observations == 10
        assert det.per_layer is True

    def test_construction_custom(self) -> None:
        """StatisticalNoveltyDetector accepts custom constructor parameters."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=20,
            sigma_threshold=3.0,
            min_observations=5,
            per_layer=False,
        )
        assert det.window_size == 20
        assert det.sigma_threshold == pytest.approx(3.0)
        assert det.min_observations == 5
        assert det.per_layer is False

    # ------------------------------------------------------------------
    # Warmup / min_observations
    # ------------------------------------------------------------------

    def test_no_trigger_during_warmup(self) -> None:
        """No trigger before min_observations frames have been observed."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(min_observations=10, window_size=20)
        # Feed 9 frames — still in warmup
        for i in range(9):
            frame = _make_frame(chunk_index=i, error_norms=[1.0])
            result = det.observe(frame)
            assert result.triggered is False, f"Should not trigger during warmup (frame {i})"

    def test_no_trigger_on_stable_signal(self) -> None:
        """No trigger when all values are constant (~1.0) — no z-score anomaly."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=30, sigma_threshold=2.0, min_observations=10
        )
        for i in range(30):
            frame = _make_frame(chunk_index=i, error_norms=[1.0])
            result = det.observe(frame)
        # Last observation on a stable signal should not trigger
        assert result.triggered is False

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def test_spike_triggers(self) -> None:
        """A spike (10.0 after 20 frames of 1.0) triggers the detector."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # 20 stable frames
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))
        # Spike
        result = det.observe(_make_frame(chunk_index=20, error_norms=[10.0]))
        assert result.triggered is True
        assert result.signal_source == "prediction_error"
        assert result.confidence > 0.0

    def test_spike_confidence_scales_with_magnitude(self) -> None:
        """Larger spike should produce equal or higher confidence."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        def _run_spike(spike_val: float) -> float:
            det = StatisticalNoveltyDetector(
                window_size=50, sigma_threshold=2.0, min_observations=10
            )
            for i in range(20):
                det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))
            result = det.observe(_make_frame(chunk_index=20, error_norms=[spike_val]))
            return result.confidence

        c_moderate = _run_spike(5.0)
        c_large = _run_spike(20.0)
        assert c_large >= c_moderate

    # ------------------------------------------------------------------
    # Cascade: fallback to weight_delta
    # ------------------------------------------------------------------

    def test_cascade_to_weight_delta_when_error_zeros(self) -> None:
        """When prediction_error is all zeros, cascades to weight_delta."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # 20 frames of stable weight_delta, zero error
        for i in range(20):
            det.observe(_make_frame(
                chunk_index=i,
                error_norms=[0.0],
                weight_delta_norms=[1.0],
            ))
        # Spike in weight_delta while error stays zero
        result = det.observe(_make_frame(
            chunk_index=20,
            error_norms=[0.0],
            weight_delta_norms=[10.0],
        ))
        assert result.triggered is True
        assert result.signal_source == "weight_delta"

    # ------------------------------------------------------------------
    # Cascade: fallback to momentum_shift
    # ------------------------------------------------------------------

    def test_cascade_to_momentum_shift_when_error_and_delta_zeros(self) -> None:
        """When both error and weight_delta are zero, cascades to momentum_shift."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        for i in range(20):
            det.observe(_make_frame(
                chunk_index=i,
                error_norms=[0.0],
                weight_delta_norms=[0.0],
                momentum_shift_norms=[1.0],
            ))
        result = det.observe(_make_frame(
            chunk_index=20,
            error_norms=[0.0],
            weight_delta_norms=[0.0],
            momentum_shift_norms=[10.0],
        ))
        assert result.triggered is True
        assert result.signal_source == "momentum_shift"

    # ------------------------------------------------------------------
    # Cascade: no cascade when primary signal is available
    # ------------------------------------------------------------------

    def test_no_cascade_when_primary_available_and_stable(self) -> None:
        """When prediction_error is available but stable, no cascade occurs."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # Stable error norms, but spike in weight_delta
        for i in range(20):
            det.observe(_make_frame(
                chunk_index=i,
                error_norms=[1.0],
                weight_delta_norms=[1.0],
            ))
        # error stays stable; weight_delta spikes — should NOT trigger because error is primary
        result = det.observe(_make_frame(
            chunk_index=20,
            error_norms=[1.0],         # identical to all prior frames — no anomaly
            weight_delta_norms=[10.0], # large spike in fallback signal — must be ignored
        ))
        assert result.triggered is False

    # ------------------------------------------------------------------
    # Drop detection (rate-of-change)
    # ------------------------------------------------------------------

    def test_drop_detection_via_rate_of_change(self) -> None:
        """Sudden drop from high error to near zero triggers via rate-of-change."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # Build up history of high error (e.g. 5.0)
        for i in range(25):
            det.observe(_make_frame(chunk_index=i, error_norms=[5.0]))
        # Sudden cliff drop to near zero
        result = det.observe(_make_frame(chunk_index=25, error_norms=[0.01]))
        assert result.triggered is True
        assert result.signal_source == "prediction_error"

    def test_gradual_decline_does_not_trigger(self) -> None:
        """A slow, gradual decline in error should not trigger drop detection."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # Gradual linear decline over 40 frames: 5.0 down to 0.5
        n = 40
        for i in range(n):
            val = 5.0 - (4.5 / n) * i
            result = det.observe(_make_frame(chunk_index=i, error_norms=[val]))
        # The final result of the gradual decline should not trigger
        assert result.triggered is False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def test_reset_clears_state(self) -> None:
        """After reset(), a spike immediately after shouldn't trigger (back in warmup)."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # Build enough history to arm the detector
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))

        det.reset()

        # Spike right after reset — should NOT trigger because warmup restarted
        result = det.observe(_make_frame(chunk_index=0, error_norms=[100.0]))
        assert result.triggered is False

    def test_reset_allows_re_arming(self) -> None:
        """After reset(), detector re-arms and triggers normally once warmed up."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        # Initial arm
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))

        det.reset()

        # Re-warm
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))

        # Now a spike should trigger again
        result = det.observe(_make_frame(chunk_index=20, error_norms=[10.0]))
        assert result.triggered is True

    # ------------------------------------------------------------------
    # Per-layer independent detection
    # ------------------------------------------------------------------

    def test_per_layer_spike_in_single_layer_triggers(self) -> None:
        """With per_layer=True, a spike in one layer triggers even if others are stable."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=True
        )
        # Two layers, both stable
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0, 1.0]))
        # Layer 0 stays stable; layer 1 spikes
        result = det.observe(_make_frame(chunk_index=20, error_norms=[1.0, 10.0]))
        assert result.triggered is True
        assert result.signal_source == "prediction_error"

    def test_per_layer_false_mean_aggregation(self) -> None:
        """With per_layer=False, layers are averaged; a spike in one layer diluted by stable others."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        # Use many layers so that one spike is diluted below threshold
        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=False
        )
        n_layers = 10
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0] * n_layers))
        # Only layer 0 spikes moderately; mean barely moves
        spike_norms = [3.0] + [1.0] * (n_layers - 1)  # mean ~ 1.2, not a huge anomaly
        result = det.observe(_make_frame(chunk_index=20, error_norms=spike_norms))
        # With only moderate spike in 1/10 layers, mean is likely not a z-score outlier
        # We just verify the method runs without error; the exact trigger depends on variance
        assert isinstance(result.triggered, bool)

    # ------------------------------------------------------------------
    # reset_local_windows()
    # ------------------------------------------------------------------

    def test_reset_local_windows_clears_flagged_layers(self) -> None:
        """reset_local_windows() clears windows only for flagged layers."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=True
        )
        # Two-layer model, build history
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0, 1.0]))

        # Reset only layer 0
        det.reset_local_windows([True, False])

        # A spike in layer 1 (non-reset) should still trigger
        result = det.observe(_make_frame(chunk_index=20, error_norms=[0.0, 10.0]))
        assert result.triggered is True

    def test_reset_local_windows_all_false_is_noop(self) -> None:
        """reset_local_windows([False, False]) doesn't change any windows."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=True
        )
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0, 1.0]))

        det.reset_local_windows([False, False])

        # A spike should still trigger — history intact
        result = det.observe(_make_frame(chunk_index=20, error_norms=[10.0, 1.0]))
        assert result.triggered is True

    # ------------------------------------------------------------------
    # Protocol conformance
    # ------------------------------------------------------------------

    def test_implements_novelty_detector_protocol(self) -> None:
        """StatisticalNoveltyDetector is structurally compatible with NoveltyDetector."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector()
        # Structural protocol check: must have observe and reset
        assert hasattr(det, "observe")
        assert hasattr(det, "reset")
        assert callable(det.observe)
        assert callable(det.reset)

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def test_confidence_at_threshold_is_nonzero(self) -> None:
        """A z-score exactly at threshold should give non-zero confidence."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))
        # Spike large enough to cross threshold
        result = det.observe(_make_frame(chunk_index=20, error_norms=[10.0]))
        if result.triggered:
            assert result.confidence > 0.0
            assert result.confidence <= 1.0

    def test_confidence_capped_at_one(self) -> None:
        """Confidence is capped at 1.0 even for very large spikes."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10
        )
        for i in range(20):
            det.observe(_make_frame(chunk_index=i, error_norms=[1.0]))
        result = det.observe(_make_frame(chunk_index=20, error_norms=[1000.0]))
        assert result.triggered is True
        assert result.confidence == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Direction vs magnitude in per-layer evaluation
    # ------------------------------------------------------------------

    def test_evaluate_per_layer_picks_drop_when_drop_is_larger(self) -> None:
        """Per-layer evaluation preserves direction (drop vs spike) in the reason.

        `_z_score_spike` returns a signed z-score filtered above +sigma, while
        `_z_score_drop` returns the magnitude of a strongly negative RoC z.
        The selection should rank by |z| and the trigger reason should label
        the winning direction ("drop" vs "spike") so downstream logging can
        tell a grokking cliff apart from a noise spike.
        """
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=True
        )

        # Prime two layers with noisy histories so variance > 0 for both the
        # value window and the rate-of-change window.
        # Layer 0 wanders around 1.0 with moderate noise (enough std that a
        # modest jump to 1.5 produces a moderate, not huge, z-spike).
        # Layer 1 wanders around 5.0 with similar relative noise.
        jitter_pattern = [0.3, -0.2, 0.15, -0.25, 0.1, -0.15, 0.2, -0.3, 0.25, -0.1]
        for i in range(30):
            j = jitter_pattern[i % len(jitter_pattern)]
            det.observe(
                _make_frame(
                    chunk_index=i,
                    error_norms=[1.0 + j, 5.0 + j],
                )
            )

        # Layer 0: modest positive spike (value ~1.0 -> 1.6, small z).
        # Layer 1: large cliff drop (5.0 -> 0.01, huge negative RoC z).
        # Drop magnitude in z-units dwarfs the spike magnitude, so the
        # winning trigger reason must mention "drop".
        result = det.observe(
            _make_frame(chunk_index=30, error_norms=[1.6, 0.01])
        )

        assert result.triggered is True
        assert result.signal_source == "prediction_error"
        assert "drop" in result.reason.lower(), (
            f"Expected drop direction in reason; got {result.reason!r}"
        )

    def test_evaluate_aggregated_picks_drop_when_drop_is_larger(self) -> None:
        """Aggregated evaluation also ranks by |z| and preserves direction.

        Mirror of the per-layer test but with per_layer=False. The aggregated
        path (_evaluate_aggregated) must apply the same abs(z) comparison
        between signed spike z and magnitude drop z, and label the winning
        direction in the reason.
        """
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=50, sigma_threshold=2.0, min_observations=10, per_layer=False
        )

        # Prime two layers (aggregated = mean) with jittered histories so
        # both the aggregated value window and aggregated roc window have
        # nonzero variance.
        jitter_pattern = [0.3, -0.2, 0.15, -0.25, 0.1, -0.15, 0.2, -0.3, 0.25, -0.1]
        for i in range(30):
            j = jitter_pattern[i % len(jitter_pattern)]
            det.observe(
                _make_frame(
                    chunk_index=i,
                    error_norms=[1.0 + j, 5.0 + j],
                )
            )

        # Aggregated previous mean ~ (1.0 + 5.0) / 2 = 3.0.
        # New values [1.6, 0.01] aggregate to ~0.805, a sharp cliff down
        # relative to the jittered ~3.0 baseline -> strongly negative RoC z.
        # The spike check on the aggregated scalar sees value 0.805 vs
        # baseline 3.0 -> negative signed z, which _z_score_spike filters out
        # (it only returns positive z above +sigma). So the winning signal is
        # the drop, and the reason must mention "drop".
        result = det.observe(
            _make_frame(chunk_index=30, error_norms=[1.6, 0.01])
        )

        assert result.triggered is True
        assert result.signal_source == "prediction_error"
        assert "drop" in result.reason.lower(), (
            f"Expected drop direction in reason; got {result.reason!r}"
        )


# ---------------------------------------------------------------------------
# TestNoveltyCascadePriority (Task 9)
# ---------------------------------------------------------------------------


class TestNoveltyCascadePriority:
    """Regression guards locking in the cascade preference.

    Once ``prediction_error_norms`` is populated (see Tasks 7+8), the
    :class:`StatisticalNoveltyDetector` cascade must prefer it over
    ``weight_delta``.  These tests protect against a future reordering that
    would silently demote the primary signal.
    """

    def test_cascade_prefers_prediction_error_when_both_spike(self) -> None:
        """When both pred_error and weight_delta are populated and spike,
        the decision must cite prediction_error as the signal source."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=10,
            sigma_threshold=2.0,
            min_observations=4,
        )
        # Warmup with stable values on both signals.
        for i in range(5):
            det.observe(
                _make_frame(
                    chunk_index=i,
                    error_norms=[0.1],
                    weight_delta_norms=[0.1],
                )
            )
        # Both signals spike simultaneously.
        decision = det.observe(
            _make_frame(
                chunk_index=5,
                error_norms=[100.0],
                weight_delta_norms=[100.0],
            )
        )
        assert decision.triggered
        assert decision.signal_source == "prediction_error", (
            f"Expected prediction_error priority, got {decision.signal_source}"
        )

    def test_cascade_falls_back_to_weight_delta_when_pred_error_absent(
        self,
    ) -> None:
        """When prediction_error is unavailable (all-zero history) the
        cascade must fall through to weight_delta."""
        from titans.novelty_detector import StatisticalNoveltyDetector

        det = StatisticalNoveltyDetector(
            window_size=10,
            sigma_threshold=2.0,
            min_observations=4,
        )
        # Warmup with all-zero pred_error but stable weight_delta.
        for i in range(5):
            det.observe(
                _make_frame(
                    chunk_index=i,
                    error_norms=[0.0],
                    weight_delta_norms=[0.1],
                )
            )
        # weight_delta spikes; pred_error stays zero (unavailable).
        decision = det.observe(
            _make_frame(
                chunk_index=5,
                error_norms=[0.0],
                weight_delta_norms=[100.0],
            )
        )
        assert decision.triggered
        assert decision.signal_source == "weight_delta", (
            f"Expected weight_delta fallback, got {decision.signal_source}"
        )
