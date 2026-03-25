#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Reinforcement Learning with Verifiable Rewards (RLVR) for Titans MLX models.

Supports:
- GRPO (Group Relative Policy Optimization) with clipped importance ratios
- REINFORCE with EMA baseline
- Offline mode (pre-computed rollouts) and live mode (generate + verify)
- Pluggable verifier framework (exact_match, numeric_match, custom)
- LoRA and full-parameter training

Usage:
    # GRPO with offline rollouts
    uv run python scripts/rlvr.py --model mac --dataset allenai/Dolci-Think-RL-7B \\
        --mode offline --method grpo --tokenizer gpt2

    # REINFORCE with live generation
    uv run python scripts/rlvr.py --model mac --dataset my/prompts \\
        --mode live --method reinforce --verifier exact_match
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# =============================================================================
# Verifiers
# =============================================================================


def exact_match(response: str, ground_truth: list[str]) -> float:
    """Exact match verifier (case-insensitive, whitespace-stripped).

    Returns 1.0 if response matches any ground truth, else 0.0.
    """
    normalized = response.strip().lower()
    for gt in ground_truth:
        if normalized == gt.strip().lower():
            return 1.0
    return 0.0


def numeric_match(
    response: str,
    ground_truth: list[str],
    tolerance: float = 0.01,
) -> float:
    """Extract the last number from response and compare to ground truth.

    Returns 1.0 if within tolerance of any ground truth number, else 0.0.
    """
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if not numbers:
        return 0.0

    extracted = float(numbers[-1])

    for gt in ground_truth:
        try:
            gt_num = float(gt.strip())
            if abs(extracted - gt_num) <= tolerance:
                return 1.0
        except ValueError:
            continue

    return 0.0


def load_custom_verifier(spec: str) -> Callable:
    """Load a custom verifier from 'path/to/module.py:function_name'."""
    module_path, func_name = spec.rsplit(":", 1)
    spec_obj = importlib.util.spec_from_file_location("custom_verifier", module_path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return getattr(module, func_name)


BUILTIN_VERIFIERS = {
    "exact_match": exact_match,
    "numeric_match": numeric_match,
}


# =============================================================================
# Log-Probability Computation
# =============================================================================


def compute_logprobs(
    model: nn.Module,
    input_ids: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute per-token log-probabilities of actual next tokens.

    Args:
        model: Titans model returning (logits, states).
        input_ids: (batch, seq_len) token IDs.
        mask: (batch, seq_len) attention mask.

    Returns:
        (batch, seq_len - 1) per-token log-probabilities.
    """
    logits, _ = model(input_ids)
    log_probs = logits[:, :-1] - mx.logsumexp(
        logits[:, :-1], axis=-1, keepdims=True
    )
    token_log_probs = mx.take_along_axis(
        log_probs, input_ids[:, 1:, None], axis=-1
    ).squeeze(-1)
    if mask is not None:
        token_log_probs = token_log_probs * mask[:, 1:]
    return token_log_probs


# =============================================================================
# Loss Functions
# =============================================================================


def grpo_loss(
    log_probs: mx.array,
    log_probs_old: mx.array,
    rewards: mx.array,
    masks: mx.array,
    epsilon: float = 0.2,
    kl_beta: float = 0.0,
    ref_log_probs: mx.array | None = None,
) -> mx.array:
    """GRPO loss with clipped importance ratios (per DeepSeekMath).

    Args:
        log_probs: (batch, num_rollouts, seq_len-1) current policy log-probs.
        log_probs_old: (batch, num_rollouts, seq_len-1) old policy log-probs (no grad).
        rewards: (batch, num_rollouts) per-rollout rewards.
        masks: (batch, num_rollouts, seq_len-1) token masks.
        epsilon: Clipping range for importance ratios.
        kl_beta: KL penalty coefficient (0 = disabled).
        ref_log_probs: (batch, num_rollouts, seq_len-1) reference model log-probs for KL.

    Returns:
        Scalar loss.
    """
    mean_reward = mx.mean(rewards, axis=1, keepdims=True)
    std_reward = mx.sqrt(mx.var(rewards, axis=1, keepdims=True) + 1e-8)
    advantages = (rewards - mean_reward) / std_reward

    seq_log_probs = (log_probs * masks).sum(axis=-1)
    seq_log_probs_old = (log_probs_old * masks).sum(axis=-1)

    ratio = mx.exp(seq_log_probs - mx.stop_gradient(seq_log_probs_old))
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -mx.mean(mx.minimum(surr1, surr2))

    kl_loss = mx.array(0.0)
    if kl_beta > 0 and ref_log_probs is not None:
        kl = (log_probs - ref_log_probs) * masks
        kl_loss = kl_beta * mx.mean(kl.sum(axis=-1))

    return policy_loss + kl_loss


def reinforce_loss(
    log_probs: mx.array,
    rewards: mx.array,
    baseline: float,
    masks: mx.array,
) -> mx.array:
    """REINFORCE loss with EMA baseline.

    Args:
        log_probs: (batch, seq_len-1) per-token log-probs.
        rewards: (batch,) per-example rewards.
        baseline: EMA baseline value.
        masks: (batch, seq_len-1) token masks.

    Returns:
        Scalar loss.
    """
    advantages = rewards - baseline
    seq_log_probs = (log_probs * masks).sum(axis=-1)
    return -mx.mean(advantages * seq_log_probs)


# =============================================================================
# Data Utilities
# =============================================================================


def format_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML string."""
    parts: list[str] = []
    for message in messages:
        parts.append(f"{IM_START}{message['role']}\n{message['content']}{IM_END}\n")
    return "".join(parts)


def compute_rollout_rewards(
    outputs: list[str],
    ground_truth: list[str],
    verifier: Callable,
) -> list[float]:
    """Score each rollout output against ground truth using the verifier."""
    return [verifier(output, ground_truth) for output in outputs]


def tokenize_and_pad(
    text: str,
    tokenizer: Any,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize text, truncate/pad to max_len. Returns (token_ids, attention_mask)."""
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        ids = tokenizer.encode(text)
    else:
        special_tokens = []
        existing = set(tokenizer.additional_special_tokens or [])
        if IM_START not in existing:
            special_tokens.append(IM_START)
        if IM_END not in existing:
            special_tokens.append(IM_END)
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        ids = tokenizer.encode(text)

    ids = ids[:max_len]
    real_len = len(ids)
    pad_len = max_len - real_len
    mask = [1] * real_len + [0] * pad_len
    ids = ids + [0] * pad_len
    return ids, mask


# =============================================================================
# Offline RL Dataset
# =============================================================================


class OfflineRLDataset:
    """Streaming dataset for offline RL with pre-computed rollouts.

    Expects HuggingFace datasets with 'prompt', 'ground_truth', and
    'outputs' fields (e.g., allenai/Dolci-Think-RL-7B).
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_len: int,
        num_rollouts: int = 8,
        verifier: Callable = exact_match,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        prompt_field: str = "prompt",
        ground_truth_field: str = "ground_truth",
        outputs_field: str = "outputs",
        buffer_size: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_rollouts = num_rollouts
        self.verifier = verifier
        self.subset = subset
        self.split = split
        self.seed = seed
        self.prompt_field = prompt_field
        self.ground_truth_field = ground_truth_field
        self.outputs_field = outputs_field
        self.buffer_size = buffer_size
        self._iterator: Any = None

    def _create_iterator(self):
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for example in ds:
            prompt = example.get(self.prompt_field, "")
            ground_truth = example.get(self.ground_truth_field, [])
            outputs = example.get(self.outputs_field, [])

            if not outputs or not ground_truth:
                continue

            rollouts = outputs[:self.num_rollouts]
            if len(rollouts) < 2:
                continue

            rewards = compute_rollout_rewards(rollouts, ground_truth, self.verifier)

            prompt_ids, _ = tokenize_and_pad(prompt, self.tokenizer, self.max_len)

            rollout_ids_list = []
            rollout_masks_list = []
            for rollout_text in rollouts:
                full_text = prompt + rollout_text
                r_ids, r_mask = tokenize_and_pad(full_text, self.tokenizer, self.max_len)
                rollout_ids_list.append(r_ids)
                rollout_masks_list.append(r_mask)

            yield {
                "prompt_ids": prompt_ids,
                "rollout_ids": rollout_ids_list,
                "rollout_masks": rollout_masks_list,
                "rewards": rewards,
            }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Return a batch of rollout groups."""
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

        max_rollouts = max(len(item["rewards"]) for item in batch_items)

        padded_rollout_ids = []
        padded_rollout_masks = []
        padded_rewards = []

        for item in batch_items:
            n = len(item["rewards"])
            pad_n = max_rollouts - n

            r_ids = item["rollout_ids"] + [[0] * self.max_len] * pad_n
            r_masks = item["rollout_masks"] + [[0] * self.max_len] * pad_n
            rews = item["rewards"] + [0.0] * pad_n

            padded_rollout_ids.append(r_ids)
            padded_rollout_masks.append(r_masks)
            padded_rewards.append(rews)

        return {
            "prompt_ids": mx.array(
                np.array([item["prompt_ids"] for item in batch_items])
            ),
            "rollout_ids": mx.array(np.array(padded_rollout_ids)),
            "rollout_masks": mx.array(
                np.array(padded_rollout_masks, dtype=np.float32)
            ),
            "rewards": mx.array(
                np.array(padded_rewards, dtype=np.float32)
            ),
        }
