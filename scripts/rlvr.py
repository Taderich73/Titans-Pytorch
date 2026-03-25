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
