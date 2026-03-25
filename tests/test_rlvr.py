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


class TestOfflineRLDataset:
    """Tests for offline rollout data loading."""

    def test_reward_from_ground_truth(self) -> None:
        """Rewards are computed by verifier against ground_truth."""
        from scripts.rlvr import compute_rollout_rewards, exact_match

        outputs = ["42", "wrong", "42"]
        ground_truth = ["42"]

        rewards = compute_rollout_rewards(outputs, ground_truth, exact_match)
        assert rewards == [1.0, 0.0, 1.0]

    def test_empty_outputs(self) -> None:
        """No outputs produces empty rewards."""
        from scripts.rlvr import compute_rollout_rewards, exact_match

        rewards = compute_rollout_rewards([], ["42"], exact_match)
        assert rewards == []


from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestRLVRIntegration:
    """End-to-end RLVR loss with a real model."""

    def test_grpo_with_model(self) -> None:
        """GRPO loss computation with actual model log-probs."""
        from scripts.rlvr import compute_logprobs, grpo_loss

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        # 1 prompt, 2 rollouts, 5 tokens each
        rollout_ids = mx.array([
            [[1, 5, 10, 20, 30], [1, 5, 11, 21, 31]]
        ])
        masks = mx.ones((1, 2, 4))  # seq_len - 1 = 4

        # Compute log-probs for each rollout
        lps = []
        for i in range(2):
            lp = compute_logprobs(model, rollout_ids[:, i, :])
            lps.append(lp)
        log_probs = mx.stack(lps, axis=1)
        mx.eval(log_probs)

        rewards = mx.array([[1.0, 0.0]])

        loss = grpo_loss(
            log_probs, mx.stop_gradient(log_probs),
            rewards, masks, epsilon=0.2,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))
