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
