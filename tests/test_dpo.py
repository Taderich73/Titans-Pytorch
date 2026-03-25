"""Tests for DPO data pipeline and log-probability computation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestExtractMessages:
    """Tests for extracting role/content from Dolci-style message dicts."""

    def test_strips_metadata_fields(self) -> None:
        """Only role and content are kept from message dicts."""
        from scripts.dpo import extract_messages

        raw_messages = [
            {
                "role": "user",
                "content": "Hello",
                "country": "US",
                "hashed_ip": "abc123",
                "toxic": False,
                "redacted": False,
                "turn_identifier": 1,
                "header": {"accept-language": "en"},
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "country": None,
                "hashed_ip": None,
                "toxic": False,
                "redacted": False,
                "turn_identifier": 2,
                "header": {},
            },
        ]

        cleaned = extract_messages(raw_messages)
        assert len(cleaned) == 2
        assert cleaned[0] == {"role": "user", "content": "Hello"}
        assert cleaned[1] == {"role": "assistant", "content": "Hi there!"}

    def test_handles_minimal_messages(self) -> None:
        """Messages with only role/content pass through unchanged."""
        from scripts.dpo import extract_messages

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        cleaned = extract_messages(messages)
        assert cleaned == messages


class TestComputeLogprobs:
    """Tests for per-token log-probability computation."""

    def test_output_shape(self) -> None:
        """Log-probs shape is (batch, seq_len - 1)."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20], [2, 6, 11, 21]])
        log_probs = compute_logprobs(model, input_ids)
        mx.eval(log_probs)

        assert log_probs.shape == (2, 3)  # (batch=2, seq_len-1=3)

    def test_values_are_negative(self) -> None:
        """Log-probs should all be <= 0."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20]])
        log_probs = compute_logprobs(model, input_ids)
        mx.eval(log_probs)

        assert np.all(np.array(log_probs) <= 0.0 + 1e-6)

    def test_mask_zeroes_padding(self) -> None:
        """Masked positions produce zero log-probs."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20]])
        # Mask: first token real, rest padded
        mask = mx.array([[1, 1, 0, 0]])
        log_probs = compute_logprobs(model, input_ids, mask=mask)
        mx.eval(log_probs)

        lp = np.array(log_probs)
        # mask[:, 1:] = [1, 0, 0] — positions 1,2 should be zero
        assert lp[0, 1] == 0.0
        assert lp[0, 2] == 0.0
        assert lp[0, 0] != 0.0  # first position is unmasked


class TestDPOLoss:
    """Tests for DPO loss computation."""

    def test_prefers_chosen(self) -> None:
        """Loss is lower when policy assigns higher log-prob to chosen."""
        from scripts.dpo import dpo_loss

        chosen_logps = mx.array([-1.0])
        rejected_logps = mx.array([-5.0])
        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-2.0])

        loss_good = dpo_loss(chosen_logps, rejected_logps,
                             ref_chosen_logps, ref_rejected_logps, beta=0.1)

        loss_bad = dpo_loss(rejected_logps, chosen_logps,
                            ref_chosen_logps, ref_rejected_logps, beta=0.1)
        mx.eval(loss_good, loss_bad)

        assert float(loss_good) < float(loss_bad)

    def test_beta_scaling(self) -> None:
        """Higher beta amplifies the loss signal."""
        from scripts.dpo import dpo_loss

        chosen_logps = mx.array([-1.0])
        rejected_logps = mx.array([-3.0])
        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-2.0])

        loss_low_beta = dpo_loss(chosen_logps, rejected_logps,
                                  ref_chosen_logps, ref_rejected_logps, beta=0.01)
        loss_high_beta = dpo_loss(chosen_logps, rejected_logps,
                                   ref_chosen_logps, ref_rejected_logps, beta=1.0)
        mx.eval(loss_low_beta, loss_high_beta)

        assert float(loss_low_beta) != float(loss_high_beta)


class TestSimPOLoss:
    """Tests for SimPO loss computation."""

    def test_prefers_chosen(self) -> None:
        """Loss is lower when avg log-prob of chosen exceeds rejected."""
        from scripts.dpo import simpo_loss

        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-3.0])

        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps,
                          beta=0.1, gamma=1.0)
        mx.eval(loss)

        assert np.isfinite(float(loss))
        assert float(loss) > 0

    def test_no_reference_model(self) -> None:
        """SimPO loss takes only policy log-probs, no reference."""
        from scripts.dpo import simpo_loss

        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-2.0])
        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps,
                          beta=0.1, gamma=0.5)
        mx.eval(loss)
        assert np.isfinite(float(loss))
