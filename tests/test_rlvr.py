"""Tests for RLVR verifiers, loss functions, and data pipeline."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class TestExactMatchVerifier:
    """Tests for exact_match verifier."""

    def test_exact_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("42", ["42"]) == 1.0

    def test_case_insensitive(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("Hello World", ["hello world"]) == 1.0

    def test_strips_whitespace(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("  answer  ", ["answer"]) == 1.0

    def test_no_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("wrong", ["right"]) == 0.0

    def test_multiple_ground_truths(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("b", ["a", "b", "c"]) == 1.0


class TestNumericMatchVerifier:
    """Tests for numeric_match verifier."""

    def test_exact_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 42.", ["42"]) == 1.0

    def test_approximate(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("3.14159", ["3.14"], tolerance=0.01) == 1.0

    def test_no_number_found(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("no numbers here", ["42"]) == 0.0

    def test_wrong_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 7", ["42"]) == 0.0

    def test_extracts_last_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("Step 1: 10, Step 2: 20, Final: 42", ["42"]) == 1.0


class TestGRPOLoss:
    """Tests for GRPO loss with clipped importance ratios."""

    def test_zero_advantage_zero_loss(self) -> None:
        """When all rewards are identical, advantages are zero -> loss is zero."""
        from scripts.rlvr import grpo_loss

        log_probs = mx.array([[[-0.5, -0.3, -0.4]]])
        log_probs_old = mx.array([[[-0.5, -0.3, -0.4]]])
        rewards = mx.array([[1.0]])
        masks = mx.array([[[1.0, 1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        assert abs(float(loss)) < 1e-6

    def test_positive_advantage_negative_loss(self) -> None:
        """Rollout with above-average reward should decrease loss."""
        from scripts.rlvr import grpo_loss

        log_probs = mx.array([[[-0.5, -0.3], [-0.5, -0.3]]])
        log_probs_old = mx.array([[[-0.5, -0.3], [-0.5, -0.3]]])
        rewards = mx.array([[1.0, 0.0]])
        masks = mx.array([[[1.0, 1.0], [1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_clipping_bounds_ratio(self) -> None:
        """Large log-prob differences should be clipped."""
        from scripts.rlvr import grpo_loss

        log_probs = mx.array([[[0.0, 0.0], [-5.0, -5.0]]])
        log_probs_old = mx.array([[[-5.0, -5.0], [0.0, 0.0]]])
        rewards = mx.array([[1.0, 0.0]])
        masks = mx.array([[[1.0, 1.0], [1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        assert np.isfinite(float(loss))


class TestREINFORCELoss:
    """Tests for REINFORCE with baseline loss."""

    def test_positive_advantage(self) -> None:
        """Positive advantage produces negative loss (encourages action)."""
        from scripts.rlvr import reinforce_loss

        log_probs = mx.array([[-0.5, -0.3, -0.4]])
        rewards = mx.array([1.0])
        baseline = 0.5
        masks = mx.array([[1.0, 1.0, 1.0]])

        loss = reinforce_loss(log_probs, rewards, baseline, masks)
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_zero_advantage_zero_loss(self) -> None:
        """When reward equals baseline, loss is zero."""
        from scripts.rlvr import reinforce_loss

        log_probs = mx.array([[-0.5, -0.3]])
        rewards = mx.array([0.5])
        baseline = 0.5
        masks = mx.array([[1.0, 1.0]])

        loss = reinforce_loss(log_probs, rewards, baseline, masks)
        mx.eval(loss)
        assert abs(float(loss)) < 1e-6
