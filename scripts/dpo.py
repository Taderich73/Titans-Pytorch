#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Direct Preference Optimization (DPO) training for Titans MLX models.

Supports:
- Standard DPO (Rafailov et al.) with reference model
- SimPO (reference-free, length-normalized)
- LoRA mode with base-model-as-reference trick
- Streaming HuggingFace preference datasets
- Gradient accumulation, cosine LR, checkpointing

Usage:
    # DPO with LoRA (recommended — base model serves as reference)
    uv run python scripts/dpo.py --model mac --dataset allenai/Dolci-Instruct-DPO \\
        --tokenizer gpt2 --dim 256 --num-layers 4 --lora

    # SimPO (no reference model needed)
    uv run python scripts/dpo.py --model mac --dataset allenai/Dolci-Instruct-DPO \\
        --method simpo --tokenizer gpt2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tqdm import tqdm

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Optional imports
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# =============================================================================
# Data Utilities
# =============================================================================


def extract_messages(raw_messages: list[dict]) -> list[dict]:
    """Extract only role and content from message dicts.

    Dolci-Instruct-DPO messages contain metadata fields (country, hashed_ip,
    toxic, header, etc.) that we discard.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in raw_messages
    ]


def format_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML string."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def tokenize_sequence(
    messages: list[dict],
    tokenizer: Any,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a message list into input_ids and an attention mask.

    Returns:
        (token_ids, attention_mask) where mask is 1 for real, 0 for padding.
        Both are lists of length max_len.
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        input_ids: list[int] = tokenizer.apply_chat_template(
            messages, tokenize=True
        )
    else:
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

    # Truncate
    input_ids = input_ids[:max_len]
    real_len = len(input_ids)

    # Pad to max_len
    pad_len = max_len - real_len
    attention_mask = [1] * real_len + [0] * pad_len
    input_ids = input_ids + [0] * pad_len

    return input_ids, attention_mask


# =============================================================================
# Log-Probability Computation
# =============================================================================


def compute_logprobs(
    model: nn.Module,
    input_ids: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute per-token log-probabilities of the actual next tokens.

    Args:
        model: Titans model returning (logits, states).
        input_ids: (batch, seq_len) token IDs.
        mask: (batch, seq_len) attention mask. If provided, positions where
            mask[:, 1:] == 0 will have their log-probs zeroed out.

    Returns:
        (batch, seq_len - 1) per-token log-probabilities.
    """
    logits, _ = model(input_ids)
    # Log-softmax via MLX primitives
    log_probs = logits[:, :-1] - mx.logsumexp(
        logits[:, :-1], axis=-1, keepdims=True
    )
    # Gather log-probs for actual next tokens
    token_log_probs = mx.take_along_axis(
        log_probs, input_ids[:, 1:, None], axis=-1
    ).squeeze(-1)
    if mask is not None:
        token_log_probs = token_log_probs * mask[:, 1:]
    return token_log_probs


# =============================================================================
# Loss Functions
# =============================================================================


def dpo_loss(
    chosen_logps: mx.array,
    rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """Standard DPO loss (Rafailov et al.).

    Args:
        chosen_logps: Sum of per-token log-probs for chosen, (batch,).
        rejected_logps: Sum of per-token log-probs for rejected, (batch,).
        ref_chosen_logps: Reference model log-probs for chosen, (batch,).
        ref_rejected_logps: Reference model log-probs for rejected, (batch,).
        beta: KL penalty strength.

    Returns:
        Scalar loss.
    """
    log_ratio_chosen = chosen_logps - ref_chosen_logps
    log_ratio_rejected = rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    # Numerically stable log-sigmoid: log(sigmoid(x)) = x - softplus(x)
    return -mx.mean(logits - mx.logaddexp(mx.zeros_like(logits), logits))


def simpo_loss(
    chosen_avg_logps: mx.array,
    rejected_avg_logps: mx.array,
    beta: float = 0.1,
    gamma: float = 1.0,
) -> mx.array:
    """SimPO loss (reference-free, length-normalized).

    Args:
        chosen_avg_logps: Mean per-token log-prob for chosen, (batch,).
        rejected_avg_logps: Mean per-token log-prob for rejected, (batch,).
        beta: Scaling factor.
        gamma: Reward margin.

    Returns:
        Scalar loss.
    """
    logits = beta * (chosen_avg_logps - rejected_avg_logps - gamma)
    return -mx.mean(logits - mx.logaddexp(mx.zeros_like(logits), logits))


# =============================================================================
# DPO Streaming Dataset
# =============================================================================


class DPOStreamingDataset:
    """Streaming dataset for DPO preference pairs.

    Expects HuggingFace datasets with 'chosen' and 'rejected' fields,
    each containing a list of message dicts with 'role' and 'content'.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        buffer_size: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.buffer_size = buffer_size
        self._iterator: Any = None

    def _create_iterator(self):
        """Create a fresh streaming iterator of preference pairs."""
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for example in ds:
            chosen_raw = example.get(self.chosen_field)
            rejected_raw = example.get(self.rejected_field)
            if chosen_raw is None or rejected_raw is None:
                continue

            try:
                chosen_msgs = extract_messages(chosen_raw)
                rejected_msgs = extract_messages(rejected_raw)

                c_ids, c_mask = tokenize_sequence(
                    chosen_msgs, self.tokenizer, self.max_len
                )
                r_ids, r_mask = tokenize_sequence(
                    rejected_msgs, self.tokenizer, self.max_len
                )
            except Exception:
                continue

            yield {
                "chosen_ids": c_ids,
                "chosen_mask": c_mask,
                "rejected_ids": r_ids,
                "rejected_mask": r_mask,
            }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Return a batch of preference pairs as mx.arrays.

        Returns:
            Dict with "chosen_ids", "chosen_mask", "rejected_ids",
            "rejected_mask" as mx.arrays of shape (batch, max_len),
            or None if exhausted.
        """
        if self._iterator is None:
            self._iterator = self._create_iterator()

        batch_items: list[dict] = []
        for _ in range(batch_size):
            try:
                item = next(self._iterator)
                batch_items.append(item)
            except StopIteration:
                self._iterator = self._create_iterator()
                if batch_items:
                    break
                return None

        if not batch_items:
            return None

        return {
            "chosen_ids": mx.array(
                np.array([item["chosen_ids"] for item in batch_items])
            ),
            "chosen_mask": mx.array(
                np.array(
                    [item["chosen_mask"] for item in batch_items],
                    dtype=np.float32,
                )
            ),
            "rejected_ids": mx.array(
                np.array([item["rejected_ids"] for item in batch_items])
            ),
            "rejected_mask": mx.array(
                np.array(
                    [item["rejected_mask"] for item in batch_items],
                    dtype=np.float32,
                )
            ),
        }
