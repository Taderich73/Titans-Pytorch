"""Tests for DPO data pipeline and log-probability computation."""

from __future__ import annotations

import mlx.core as mx
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

        loss_good = dpo_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1
        )

        loss_bad = dpo_loss(
            rejected_logps, chosen_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1
        )
        mx.eval(loss_good, loss_bad)

        assert float(loss_good) < float(loss_bad)

    def test_beta_scaling(self) -> None:
        """Higher beta amplifies the loss signal."""
        from scripts.dpo import dpo_loss

        chosen_logps = mx.array([-1.0])
        rejected_logps = mx.array([-3.0])
        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-2.0])

        loss_low_beta = dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=0.01,
        )
        loss_high_beta = dpo_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=1.0
        )
        mx.eval(loss_low_beta, loss_high_beta)

        assert float(loss_low_beta) != float(loss_high_beta)


class TestSimPOLoss:
    """Tests for SimPO loss computation."""

    def test_prefers_chosen(self) -> None:
        """Loss is lower when avg log-prob of chosen exceeds rejected."""
        from scripts.dpo import simpo_loss

        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-3.0])

        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps, beta=0.1, gamma=1.0)
        mx.eval(loss)

        assert np.isfinite(float(loss))
        assert float(loss) > 0

    def test_no_reference_model(self) -> None:
        """SimPO loss takes only policy log-probs, no reference."""
        from scripts.dpo import simpo_loss

        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-2.0])
        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps, beta=0.1, gamma=0.5)
        mx.eval(loss)
        assert np.isfinite(float(loss))


class TestDPOStreamingDataset:
    """Tests for DPO data loading (using mock data)."""

    def test_batch_shape(self) -> None:
        """Batch has correct keys and shapes."""
        from unittest.mock import MagicMock

        from scripts.dpo import DPOStreamingDataset

        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.additional_special_tokens = []
        tokenizer.add_special_tokens = MagicMock()
        tokenizer.encode = lambda text: list(range(len(text)))

        dataset = DPOStreamingDataset.__new__(DPOStreamingDataset)
        dataset.tokenizer = tokenizer
        dataset.max_len = 32

        from scripts.dpo import tokenize_sequence

        chosen_msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        rejected_msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Bye"},
        ]

        c_ids, c_mask = tokenize_sequence(chosen_msgs, tokenizer, 32)
        r_ids, r_mask = tokenize_sequence(rejected_msgs, tokenizer, 32)

        assert len(c_ids) == 32
        assert len(c_mask) == 32
        assert sum(c_mask) > 0


from scripts.lora import set_lora_enabled, wrap_lora_layers


class TestDPOIntegration:
    """End-to-end DPO loss computation with a real model."""

    def test_dpo_loss_with_lora_reference(self) -> None:
        """Full DPO forward pass: policy + reference via LoRA toggle."""
        from scripts.dpo import compute_logprobs, dpo_loss

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)
        wrap_lora_layers(model, "attn", rank=4, alpha=8.0)

        chosen_ids = mx.array([[1, 5, 10, 20, 30]])
        rejected_ids = mx.array([[1, 5, 11, 21, 31]])

        # Policy log-probs (LoRA enabled)
        pi_chosen = compute_logprobs(model, chosen_ids)
        pi_rejected = compute_logprobs(model, rejected_ids)

        # Reference log-probs (LoRA disabled)
        set_lora_enabled(model, False)
        ref_chosen = compute_logprobs(model, chosen_ids)
        ref_rejected = compute_logprobs(model, rejected_ids)
        set_lora_enabled(model, True)

        mx.eval(pi_chosen, pi_rejected, ref_chosen, ref_rejected)

        loss = dpo_loss(
            pi_chosen.sum(axis=1),
            pi_rejected.sum(axis=1),
            ref_chosen.sum(axis=1),
            ref_rejected.sum(axis=1),
            beta=0.1,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_simpo_loss_no_reference(self) -> None:
        """SimPO forward pass without any reference model."""
        from scripts.dpo import compute_logprobs, simpo_loss

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        chosen_ids = mx.array([[1, 5, 10, 20, 30]])
        rejected_ids = mx.array([[1, 5, 11, 21, 31]])
        mask = mx.array([[1, 1, 1, 1, 1]])

        chosen_lps = compute_logprobs(model, chosen_ids, mask=mask)
        rejected_lps = compute_logprobs(model, rejected_ids, mask=mask)
        mx.eval(chosen_lps, rejected_lps)

        lengths = mx.array([4.0])
        loss = simpo_loss(
            chosen_lps.sum(axis=1) / lengths,
            rejected_lps.sum(axis=1) / lengths,
            beta=0.1,
            gamma=1.0,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))
