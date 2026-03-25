# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Supervised Fine-Tuning (SFT) data pipeline for Titans MLX models.

Supports:
- ChatML formatting for conversation templates
- Per-token loss masking (assistant-only or train-on-all)
- Tokenization with HuggingFace tokenizers (optional)

Usage:
    from scripts.sft import format_chatml, build_loss_mask, tokenize_chat
"""

from __future__ import annotations

import logging
from typing import Any

# Optional imports
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    import datasets

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML string.

    Each message becomes:
        <|im_start|>{role}\\n{content}<|im_end|>\\n

    Args:
        messages: List of dicts with "role" and "content" keys.

    Returns:
        Formatted ChatML string.
    """
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def build_loss_mask(
    seq_len: int,
    assistant_content_spans: list[tuple[int, int]],
    include_eos: bool = True,
    eos_positions: list[int] | None = None,
    train_on_all: bool = False,
) -> list[int]:
    """Build a per-token loss mask.

    Args:
        seq_len: Total sequence length.
        assistant_content_spans: List of (start, end) ranges (end exclusive)
            indicating assistant content token positions.
        include_eos: Whether to include EOS positions in the mask.
        eos_positions: List of EOS token positions to mask in.
        train_on_all: If True, return all 1s (train on every token).

    Returns:
        List of ints (0 or 1) of length seq_len.
    """
    if train_on_all:
        return [1] * seq_len

    mask = [0] * seq_len

    for start, end in assistant_content_spans:
        for i in range(start, end):
            if 0 <= i < seq_len:
                mask[i] = 1

    if include_eos and eos_positions:
        for pos in eos_positions:
            if 0 <= pos < seq_len:
                mask[pos] = 1

    return mask


def tokenize_chat(
    messages: list[dict],
    tokenizer: Any,
    max_len: int,
    train_on_all: bool = False,
) -> dict:
    """Tokenize a chat messages list and build a loss mask.

    If the tokenizer has a chat_template, uses apply_chat_template.
    Otherwise falls back to format_chatml with ChatML special tokens.

    Builds loss mask by tokenizing each turn individually to track
    assistant content token boundaries.

    Args:
        messages: List of dicts with "role" and "content" keys.
        tokenizer: HuggingFace tokenizer or compatible object.
        max_len: Maximum sequence length (truncates if needed).
        train_on_all: If True, mask is all 1s.

    Returns:
        Dict with keys:
            - "input_ids": List[int], token ids (length up to max_len - 1)
            - "labels": List[int], next-token targets (shifted by 1)
            - "loss_mask": List[int], per-token mask aligned with labels
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        input_ids: list[int] = tokenizer.apply_chat_template(
            messages, tokenize=True
        )
    else:
        # Ensure ChatML special tokens are registered
        special_tokens = []
        existing = set(tokenizer.additional_special_tokens or [])
        if IM_START not in existing:
            special_tokens.append(IM_START)
        if IM_END not in existing:
            special_tokens.append(IM_END)
        if special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )

        formatted = format_chatml(messages)
        input_ids = tokenizer.encode(formatted)

    # Truncate to max_len
    input_ids = input_ids[:max_len]

    # Build assistant content spans by tokenizing each turn individually
    assistant_content_spans: list[tuple[int, int]] = []
    eos_positions: list[int] = []

    if not train_on_all and not has_chat_template:
        cursor = 0
        for message in messages:
            role = message["role"]
            content = message["content"]

            # Tokenize prefix: <|im_start|>{role}\n
            prefix = f"{IM_START}{role}\n"
            prefix_ids = tokenizer.encode(prefix)
            prefix_len = len(prefix_ids)

            # Tokenize content
            content_ids = tokenizer.encode(content)
            content_len = len(content_ids)

            # Tokenize suffix: <|im_end|>\n
            suffix = f"{IM_END}\n"
            suffix_ids = tokenizer.encode(suffix)
            suffix_len = len(suffix_ids)

            turn_start = cursor
            content_start = turn_start + prefix_len
            content_end = content_start + content_len
            eos_pos = content_end  # first token of suffix is <|im_end|>

            if role == "assistant":
                span_end = min(content_end, len(input_ids))
                if content_start < span_end:
                    assistant_content_spans.append(
                        (content_start, span_end)
                    )
                if eos_pos < len(input_ids):
                    eos_positions.append(eos_pos)

            cursor += prefix_len + content_len + suffix_len

    elif not train_on_all and has_chat_template:
        # For chat_template tokenizers, we do a best-effort span detection
        # by encoding each assistant message's content and searching for it
        for message in messages:
            if message["role"] != "assistant":
                continue
            content_ids = tokenizer.encode(
                message["content"], add_special_tokens=False
            )
            # Search for this subsequence in input_ids
            clen = len(content_ids)
            for i in range(len(input_ids) - clen + 1):
                if input_ids[i : i + clen] == content_ids:
                    assistant_content_spans.append((i, i + clen))
                    if i + clen < len(input_ids):
                        eos_positions.append(i + clen)
                    break

    seq_len = len(input_ids)
    loss_mask = build_loss_mask(
        seq_len=seq_len,
        assistant_content_spans=assistant_content_spans,
        include_eos=True,
        eos_positions=eos_positions,
        train_on_all=train_on_all,
    )

    # Shift for next-token prediction:
    # input_ids[:-1] -> labels[1:], loss_mask[1:]
    inputs = input_ids[:-1]
    labels = input_ids[1:]
    mask = loss_mask[1:]

    return {
        "input_ids": inputs,
        "labels": labels,
        "loss_mask": mask,
    }
