# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for SFT data pipeline: ChatML formatting and loss mask generation."""

from scripts.sft import build_loss_mask, format_chatml

# ============================================================================
# ChatML Formatting
# ============================================================================


class TestFormatChatML:
    """Tests for format_chatml()."""

    def test_single_turn(self) -> None:
        """Single user/assistant turn produces correct ChatML."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_chatml(messages)
        expected = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>\n"
        )
        assert result == expected

    def test_multi_turn(self) -> None:
        """Multi-turn conversation has correct token counts."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        result = format_chatml(messages)
        assert result.count("<|im_start|>user") == 2
        assert result.count("<|im_start|>assistant") == 2
        assert result.count("<|im_end|>") == 4

    def test_system_message(self) -> None:
        """System messages are formatted with system role."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        result = format_chatml(messages)
        expected = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        assert result == expected


# ============================================================================
# Loss Mask Generation
# ============================================================================


class TestBuildLossMask:
    """Tests for build_loss_mask()."""

    def test_assistant_only_mask(self) -> None:
        """Mask is 1 only for assistant content tokens and EOS."""
        # seq: [user_tok0..5=0, asst_prefix_tok5=0, content_tok6-8=1, eos_tok9=1]
        mask = build_loss_mask(
            seq_len=10,
            assistant_content_spans=[(6, 9)],
            eos_positions=[9],
            include_eos=True,
            train_on_all=False,
        )
        assert len(mask) == 10
        # user tokens 0-5 are 0
        assert all(mask[i] == 0 for i in range(6)), (
            f"Expected 0s at positions 0-5, got {mask[:6]}"
        )
        # assistant content tokens 6-8 are 1
        assert all(mask[i] == 1 for i in range(6, 9)), (
            f"Expected 1s at positions 6-8, got {mask[6:9]}"
        )
        # EOS token 9 is 1
        assert mask[9] == 1, f"Expected 1 at position 9, got {mask[9]}"

    def test_train_on_all(self) -> None:
        """When train_on_all=True, entire mask is 1s."""
        mask = build_loss_mask(
            seq_len=8,
            assistant_content_spans=[(2, 5)],
            eos_positions=[5],
            include_eos=True,
            train_on_all=True,
        )
        assert mask == [1] * 8

    def test_multi_turn_mask(self) -> None:
        """Two assistant turns each masked correctly; total sum is 6."""
        # spans: (3,5) -> positions 3,4 = 2 tokens
        #        (8,10) -> positions 8,9 = 2 tokens
        # eos:  5, 10 -> positions 5, 10 = 2 tokens
        # total = 6
        mask = build_loss_mask(
            seq_len=11,
            assistant_content_spans=[(3, 5), (8, 10)],
            eos_positions=[5, 10],
            include_eos=True,
            train_on_all=False,
        )
        assert len(mask) == 11
        assert sum(mask) == 6
        # First assistant turn: content at 3,4; eos at 5
        assert mask[3] == 1
        assert mask[4] == 1
        assert mask[5] == 1
        # Second assistant turn: content at 8,9; eos at 10
        assert mask[8] == 1
        assert mask[9] == 1
        assert mask[10] == 1
        # Non-assistant positions are 0
        assert mask[0] == 0
        assert mask[1] == 0
        assert mask[2] == 0
        assert mask[6] == 0
        assert mask[7] == 0
