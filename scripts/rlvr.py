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


# =============================================================================
# Live Generation
# =============================================================================


def generate_rollout(
    model: nn.Module,
    prompt_ids: mx.array,
    max_new_tokens: int,
    temperature: float = 0.7,
    eos_token_id: int | None = None,
) -> mx.array:
    """Generate a single rollout via temperature sampling.

    Args:
        model: Titans model.
        prompt_ids: (1, prompt_len) token IDs.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        eos_token_id: Stop token (e.g., <|im_end|>).

    Returns:
        (1, prompt_len + generated_len) full sequence.
    """
    tokens = prompt_ids
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        next_logits = logits[:, -1, :] / temperature
        next_token = mx.random.categorical(next_logits)
        next_token = next_token.reshape(1, 1)
        tokens = mx.concatenate([tokens, next_token], axis=1)
        mx.eval(tokens)

        if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
            break

    return tokens


# =============================================================================
# RLVR Configuration
# =============================================================================


@dataclass
class RLVRConfig:
    """RLVR training hyperparameters."""

    # Model
    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    use_tnt: bool = False
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    global_chunk_size: int = 2048
    tnt_stage: int = 1
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global: bool = True
    attnres_modulate_local: bool = False

    # RLVR-specific
    method: str = "grpo"
    mode: str = "offline"
    num_rollouts: int = 8
    epsilon: float = 0.2
    kl_beta: float = 0.0
    ema_decay: float = 0.99
    temperature: float = 0.7
    max_new_tokens: int = 2048
    verifier: str = "exact_match"

    # LoRA
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: str = "attn,ffn"
    lora_dropout: float = 0.0

    # Data
    dataset: str | None = None
    dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    max_len: int = 2048
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"
    outputs_field: str = "outputs"
    chat_template: str = "auto"

    # Training
    max_steps: int = 5000
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lr: float = 1e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    eval_dataset: str | None = None
    eval_split: str = "train"
    resume: str | None = None
    init_weights: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-mlx-rlvr"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    dtype: str = "float16"


# =============================================================================
# Model Creation
# =============================================================================


def create_model(model_type: str, config: "TitansConfig") -> nn.Module:
    """Create Titans model based on type."""
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](config)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""

    def _count(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count(v)
        elif isinstance(params, (list, tuple)):
            for item in params:
                total += _count(item)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    return _count(model.parameters())


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


def get_lr_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))


# =============================================================================
# Gradient Accumulation Utilities
# =============================================================================


def _tree_add(
    acc: dict | list | mx.array,
    new: dict | list | mx.array,
) -> dict | list | mx.array:
    """Recursively add two gradient trees element-wise."""
    if isinstance(acc, mx.array):
        return acc + new
    elif isinstance(acc, dict):
        return {k: _tree_add(acc[k], new[k]) for k in acc}
    elif isinstance(acc, list):
        return [_tree_add(a, n) for a, n in zip(acc, new)]
    return acc


def _tree_scale(
    tree: dict | list | mx.array,
    scale: mx.array,
) -> dict | list | mx.array:
    """Recursively scale a gradient tree by a scalar."""
    if isinstance(tree, mx.array):
        return tree * scale
    elif isinstance(tree, dict):
        return {k: _tree_scale(v, scale) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_scale(v, scale) for v in tree]
    return tree


def _eval_grads(grads: dict, *extra: mx.array) -> None:
    """Evaluate (materialize) all arrays in a gradient tree.

    Must be called after each micro-step to prevent the lazy computation
    graph from growing unboundedly across accumulation steps.
    """
    arrays = [v for _, v in tree_flatten(grads)]
    mx.eval(*arrays, *extra)


# =============================================================================
# Gradient Computation and Application
# =============================================================================


def sanitize_and_clip_grads(grads: dict, grad_clip: float) -> dict:
    """Sanitize NaN gradients and apply global norm clipping.

    Args:
        grads: Gradient tree (nested dict of mx.array)
        grad_clip: Maximum gradient norm (0 to disable)

    Returns:
        Sanitized and clipped gradient tree.
    """
    nan_paths: list[str] = []

    def sanitize(g, path=""):
        if isinstance(g, mx.array):
            has_nan = mx.any(mx.isnan(g))
            if has_nan:
                nan_count = mx.sum(mx.isnan(g).astype(mx.int32))
                nan_paths.append(f"{path} [{g.shape}] ({nan_count}/{g.size} NaN)")
            return mx.where(mx.isnan(g), mx.zeros_like(g), g)
        elif isinstance(g, dict):
            return {k: sanitize(v, f"{path}.{k}") for k, v in g.items()}
        elif isinstance(g, list):
            return [sanitize(v, f"{path}[{i}]") for i, v in enumerate(g)]
        return g

    grads = {k: sanitize(v, k) for k, v in grads.items()}

    if nan_paths:
        logger.warning(
            "NaN detected in gradients — replacing with zeros. "
            f"Affected params ({len(nan_paths)}):\n  " + "\n  ".join(nan_paths[:20])
        )

    if grad_clip > 0:
        max_norm = mx.array(grad_clip)

        def compute_norm_sq(g):
            if isinstance(g, mx.array):
                return mx.sum(g * g)
            elif isinstance(g, dict):
                total = mx.array(0.0)
                for v in g.values():
                    total = total + compute_norm_sq(v)
                return total
            elif isinstance(g, list):
                total = mx.array(0.0)
                for v in g:
                    total = total + compute_norm_sq(v)
                return total
            return mx.array(0.0)

        total_norm_sq = mx.array(0.0)
        for g in grads.values():
            total_norm_sq = total_norm_sq + compute_norm_sq(g)

        total_norm = mx.sqrt(total_norm_sq + 1e-8)
        clip_coef = mx.minimum(max_norm / total_norm, mx.array(1.0))

        def clip_grad(g):
            if isinstance(g, mx.array):
                return g * clip_coef
            elif isinstance(g, dict):
                return {k: clip_grad(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [clip_grad(v) for v in g]
            return g

        grads = {k: clip_grad(v) for k, v in grads.items()}

    return grads


def apply_gradients(
    model: nn.Module,
    optimizer: optim.Optimizer,
    grads: dict,
    grad_clip: float = 1.0,
) -> None:
    """Sanitize, clip, and apply accumulated gradients."""
    grads = sanitize_and_clip_grads(grads, grad_clip)
    optimizer.update(model, grads)

    # Re-tie head and embedding weights after optimizer step.
    if hasattr(model, "head") and hasattr(model, "embed"):
        model.head.weight = model.embed.weight

    mx.eval(model.parameters(), optimizer.state)


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: RLVRConfig,
    model_config: "TitansConfig",
    step: int,
    epoch: int,
    best_val_loss: float,
    path: Path,
) -> None:
    """Save checkpoint in MLX format."""
    metadata = {
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_type": config.model_type,
        "dim": model_config.dim,
        "num_heads": model_config.num_heads,
        "num_layers": model_config.num_layers,
        "vocab_size": model_config.vocab_size,
        "chunk_size": model_config.chunk_size,
        "window_size": model_config.window_size,
        "num_persistent_tokens": model_config.num_persistent_tokens,
        "num_memory_layers": model_config.num_memory_layers,
        "use_tnt": model_config.use_tnt,
        "global_chunk_size": model_config.global_chunk_size,
        "local_chunk_sizes": ",".join(str(c) for c in model_config.local_chunk_sizes),
        "local_shard_length": model_config.local_shard_length,
        "use_qk_projection": model_config.use_qk_projection,
        "tnt_stage": model_config.tnt_stage,
        "use_attn_res": model_config.use_attn_res,
        "num_attnres_blocks": model_config.num_attnres_blocks,
        "attnres_warmup_steps": model_config.attnres_warmup_steps,
        "attnres_modulate_global_memory": model_config.attnres_modulate_global_memory,
        "attnres_modulate_local_memory": model_config.attnres_modulate_local_memory,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "tokenizer_name": config.tokenizer,
        "chat_template": config.chat_template,
        "training_stage": "rlvr",
    }

    # Save using safetensors format via mlx
    model.save_weights(str(path.with_suffix(".safetensors")))

    # Also save metadata separately
    np.savez(
        str(path.with_suffix(".meta.npz")),
        **{k: np.array([v]) for k, v in metadata.items()},
    )

    logger.info(f"Saved checkpoint to {path}")


def prune_checkpoints(checkpoint_dir: Path, keep: int = 3) -> None:
    """Keep only the most recent `keep` step_N checkpoints, delete the rest."""
    step_files = sorted(
        checkpoint_dir.glob("step_*.safetensors"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(step_files) <= keep:
        return
    for old in step_files[:-keep]:
        old.unlink(missing_ok=True)
        old.with_suffix(".meta.npz").unlink(missing_ok=True)
        logger.info(f"Pruned old checkpoint: {old.stem}")


def _remap_tnt_keys(weights: dict) -> dict:
    """Remap old TNT checkpoint keys to consolidated format."""
    remapped = {}
    for k, v in weights.items():
        new_key = k.replace(".hierarchical_memory.", ".memory.")
        remapped[new_key] = v
    return remapped


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[int, int, float]:
    """Load checkpoint."""
    stem = path
    if stem.name.endswith(".meta.npz"):
        stem = stem.with_name(stem.name.removesuffix(".meta.npz"))
    else:
        stem = stem.with_suffix("")

    weights_path = stem.with_suffix(".safetensors")
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        weights = _remap_tnt_keys(weights)
        model.load_weights(list(weights.items()))
    else:
        npz_path = stem.with_suffix(".npz")
        checkpoint = np.load(str(npz_path), allow_pickle=True)
        weights = {}
        for k in checkpoint.files:
            if not k.startswith("_"):
                arr = checkpoint[k]
                if arr.dtype == object:
                    arr = arr.item()
                weights[k] = mx.array(arr)
        weights = _remap_tnt_keys(weights)
        model.update(weights)

    # Re-tie head and embedding weights
    if hasattr(model, "head") and hasattr(model, "embed"):
        model.head.weight = model.embed.weight

    meta_path = stem.with_suffix(".meta.npz")
    if meta_path.exists():
        meta = np.load(str(meta_path))
        step = int(meta["step"][0])
        epoch = int(meta["epoch"][0])
        best_val_loss = float(meta["best_val_loss"][0])
    else:
        step, epoch, best_val_loss = 0, 0, float("inf")

    logger.info(f"Loaded checkpoint from {path} (step {step})")
    return step, epoch, best_val_loss


# =============================================================================
# RLVR Gradient Computation
# =============================================================================


def compute_rlvr_grads(
    model: nn.Module,
    rollout_ids: mx.array,
    rollout_masks: mx.array,
    rewards: mx.array,
    config: RLVRConfig,
    ema_baseline: float = 0.0,
) -> tuple[mx.array, dict, float]:
    """Compute GRPO or REINFORCE gradients.

    Args:
        model: Titans model.
        rollout_ids: (batch, num_rollouts, seq_len) token IDs.
        rollout_masks: (batch, num_rollouts, seq_len) attention masks.
        rewards: (batch, num_rollouts) per-rollout rewards.
        config: RLVRConfig with method, epsilon, kl_beta, ema_decay.
        ema_baseline: Current EMA baseline for REINFORCE.

    Returns:
        Tuple of (loss, grads, new_ema_baseline).
        grads is empty dict if batch was skipped (all-same-reward).
    """
    batch_size, num_rollouts, seq_len = rollout_ids.shape

    # Slice masks to match compute_logprobs output (seq_len - 1)
    shifted_masks = rollout_masks[:, :, 1:]

    if config.method == "grpo":
        reward_std = mx.sqrt(mx.var(rewards, axis=1) + 1e-8)
        if float(mx.max(reward_std)) < 1e-6:
            logger.warning("All rollout rewards identical — skipping batch")
            return mx.array(0.0), {}, ema_baseline

        # Compute old log-probs without gradient
        log_probs_old_list = []
        for i in range(num_rollouts):
            lp = compute_logprobs(model, rollout_ids[:, i, :])
            log_probs_old_list.append(lp)
        log_probs_old = mx.stop_gradient(mx.stack(log_probs_old_list, axis=1))
        mx.eval(log_probs_old)

        def grpo_loss_fn(m):
            lp_list = []
            for i in range(num_rollouts):
                lp = compute_logprobs(m, rollout_ids[:, i, :])
                lp_list.append(lp)
            log_probs = mx.stack(lp_list, axis=1)
            return grpo_loss(
                log_probs,
                log_probs_old,
                rewards,
                shifted_masks,
                epsilon=config.epsilon,
                kl_beta=config.kl_beta,
            )

        loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads, ema_baseline

    else:  # reinforce
        total_loss = mx.array(0.0)
        all_grads = None

        for i in range(num_rollouts):
            r_ids = rollout_ids[:, i, :]
            r_mask = shifted_masks[:, i, :]  # already seq_len - 1
            r_rewards = rewards[:, i]
            advantage = r_rewards - ema_baseline

            def reinforce_loss_fn(
                m,
                _r_ids=r_ids,
                _r_mask=r_mask,
                _adv=advantage,
            ):
                lp = compute_logprobs(m, _r_ids)  # (batch, seq_len-1)
                masked_lp = lp * _r_mask
                seq_lp = masked_lp.sum(axis=-1)
                return -mx.mean(_adv * seq_lp)

            loss_and_grad_fn = nn.value_and_grad(model, reinforce_loss_fn)
            loss, grads = loss_and_grad_fn(model)
            total_loss = total_loss + loss

            if all_grads is None:
                all_grads = grads
            else:
                all_grads = _tree_add(all_grads, grads)

        scale = mx.array(1.0 / num_rollouts)
        avg_grads = _tree_scale(all_grads, scale)
        avg_loss = total_loss / num_rollouts

        batch_mean_reward = float(mx.mean(rewards))
        new_baseline = (
            config.ema_decay * ema_baseline
            + (1 - config.ema_decay) * batch_mean_reward
        )

        return avg_loss, avg_grads, new_baseline


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataset: OfflineRLDataset,
    config: RLVRConfig,
    model_config: "TitansConfig",
) -> None:
    """Main RLVR training loop (streaming only)."""
    total_steps = config.max_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Method: {config.method}  Mode: {config.mode}")

    # State
    global_step = 0
    epoch = 0
    best_val_loss = float("inf")
    running_loss = 0.0
    running_reward = 0.0
    running_count = 0
    ema_baseline = 0.0

    # Resume if specified
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            global_step, epoch, best_val_loss = load_checkpoint(
                resume_path, model, optimizer
            )

    # Load pretrained weights only (fresh step/epoch/schedule)
    if config.init_weights:
        init_path = Path(config.init_weights)
        if init_path.exists() or init_path.with_suffix(".safetensors").exists():
            load_checkpoint(init_path, model)
            logger.info("Loaded pretrained weights (step/epoch/schedule reset to 0)")
        else:
            logger.warning(f"init-weights path not found: {init_path}")

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    if config.wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "model_type": config.model_type,
                "dim": config.dim,
                "num_layers": config.num_layers,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "max_len": config.max_len,
                "method": config.method,
                "mode": config.mode,
                "epsilon": config.epsilon,
                "kl_beta": config.kl_beta,
                "num_rollouts": config.num_rollouts,
                "verifier": config.verifier,
                "lora": config.lora,
            },
        )

    start_time = time.time()
    pbar = tqdm(total=total_steps, initial=global_step, desc="RLVR Training")
    pbar.refresh()

    # Gradient accumulation state
    accumulated_grads: dict | None = None
    accumulation_loss = 0.0
    accumulation_step = 0

    while global_step < total_steps:
        epoch += 1

        # Streaming loop
        while global_step < total_steps:
            batch = train_dataset.get_batch(config.batch_size)
            if batch is None:
                # Dataset exhausted — reset iterator for next epoch
                train_dataset._iterator = None
                batch = train_dataset.get_batch(config.batch_size)
                if batch is None:
                    break

            rollout_ids = batch["rollout_ids"]
            rollout_masks = batch["rollout_masks"]
            rewards = batch["rewards"]

            batch_reward_mean = float(mx.mean(rewards))

            # --- Micro-step: compute gradients only ---
            micro_start = time.time()
            loss, grads, ema_baseline = compute_rlvr_grads(
                model,
                rollout_ids,
                rollout_masks,
                rewards,
                config,
                ema_baseline=ema_baseline,
            )

            # Skip if grads is empty (all-same-reward batch)
            if not grads:
                continue

            _eval_grads(grads, loss)
            micro_elapsed = time.time() - micro_start

            # Show micro-step progress
            pbar.set_postfix({
                "micro": f"{accumulation_step + 1}/{config.gradient_accumulation_steps}",
                "uloss": f"{float(loss):.4f}",
                "utime": f"{micro_elapsed:.1f}s",
            })

            loss_val = float(loss)
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(
                    f"Skipping micro-step with invalid loss: {loss_val}"
                )
                continue

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = _tree_add(accumulated_grads, grads)

            accumulation_loss += loss_val
            accumulation_step += 1
            running_loss += loss_val
            running_reward += batch_reward_mean
            running_count += 1

            # --- Optimizer step after full accumulation window ---
            if accumulation_step >= config.gradient_accumulation_steps:
                scale = mx.array(
                    1.0 / config.gradient_accumulation_steps
                )
                avg_grads = _tree_scale(accumulated_grads, scale)

                lr = get_lr_schedule(
                    global_step, total_steps, warmup_steps, config.lr
                )
                optimizer.learning_rate = lr

                apply_gradients(
                    model, optimizer, avg_grads, config.grad_clip
                )

                global_step += 1
                accumulation_step = 0
                accumulated_grads = None
                accumulation_loss = 0.0

                # Periodic checkpoint
                if (
                    config.save_every > 0
                    and global_step % config.save_every == 0
                ):
                    save_checkpoint(
                        model,
                        optimizer,
                        config,
                        model_config,
                        global_step,
                        epoch,
                        best_val_loss,
                        checkpoint_dir / f"step_{global_step}",
                    )
                    prune_checkpoints(checkpoint_dir, keep=3)

                # Logging
                if global_step % config.log_every == 0:
                    avg_loss = running_loss / running_count
                    avg_reward = running_reward / running_count

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/reward_mean": avg_reward,
                        "train/lr": lr,
                        "train/step": global_step,
                    }

                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "reward": f"{avg_reward:.4f}",
                            "lr": f"{lr:.2e}",
                        }
                    )

                    if config.wandb and HAS_WANDB:
                        wandb.log(log_dict, step=global_step)

                    running_loss = 0.0
                    running_reward = 0.0
                    running_count = 0

                pbar.update(1)

    pbar.close()

    # Final checkpoint
    save_checkpoint(
        model,
        optimizer,
        config,
        model_config,
        global_step,
        epoch,
        best_val_loss,
        checkpoint_dir / "final_model",
    )

    elapsed = time.time() - start_time
    logger.info(f"RLVR completed in {elapsed / 3600:.2f} hours")
    logger.info(f"Total steps: {global_step}")

    if config.wandb and HAS_WANDB:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLVR training for Titans MLX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="mac",
        choices=["mac", "mag", "mal", "lmm"],
        help="Model variant",
    )
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size (MAC)")
    parser.add_argument(
        "--window-size", type=int, default=512, help="Window size (MAG/MAL)"
    )

    # TNT Hierarchical Memory
    parser.add_argument(
        "--use-tnt", action="store_true", help="Enable TNT hierarchical memory"
    )
    parser.add_argument(
        "--local-chunk-sizes", type=int, nargs="+", default=[8, 16],
        help="Chunk sizes for local memories",
    )
    parser.add_argument(
        "--local-shard-length", type=int, default=2048,
        help="Local memory reset period (tokens)",
    )
    parser.add_argument(
        "--global-chunk-size", type=int, default=2048,
        help="Global memory chunk size",
    )
    parser.add_argument(
        "--tnt-stage", type=int, default=1, choices=[1, 2],
        help="TNT training stage (1=pretrain, 2=finetune)",
    )

    # AttnRes
    parser.add_argument(
        "--use-attn-res", action="store_true", help="Enable Attention Residuals"
    )
    parser.add_argument(
        "--num-attnres-blocks", type=int, default=8, help="AttnRes block count"
    )
    parser.add_argument(
        "--attnres-warmup-steps", type=int, default=0,
        help="Steps before AttnRes gating activates",
    )
    parser.add_argument(
        "--attnres-modulate-global", action="store_true", default=True,
        help="Gate global memory LR with AttnRes",
    )
    parser.add_argument(
        "--no-attnres-modulate-global", dest="attnres_modulate_global",
        action="store_false",
    )
    parser.add_argument(
        "--attnres-modulate-local", action="store_true", default=False,
        help="Gate local memory LR with AttnRes",
    )

    # RLVR-specific
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "live"],
        help="Rollout mode: offline (pre-computed) or live (generate + verify)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grpo",
        choices=["grpo", "reinforce"],
        help="RL optimization method",
    )
    parser.add_argument("--num-rollouts", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="GRPO importance ratio clipping"
    )
    parser.add_argument("--kl-beta", type=float, default=0.0, help="KL penalty coefficient")
    parser.add_argument(
        "--ema-decay", type=float, default=0.99, help="REINFORCE EMA baseline decay"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature for live mode"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Max tokens to generate per rollout"
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="exact_match",
        help="Verifier: 'exact_match', 'numeric_match', or 'path/to/file.py:function'",
    )

    # LoRA
    parser.add_argument(
        "--lora", action="store_true",
        help="Use LoRA adapters",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument(
        "--lora-targets", type=str, default="attn,ffn",
        help="Comma-separated LoRA target groups: attn,ffn,memory,all",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset (e.g., allenai/Dolci-Think-RL-7B)",
    )
    parser.add_argument(
        "--dataset-subset", type=str, default=None, help="Dataset subset"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer",
    )
    parser.add_argument("--max-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument(
        "--prompt-field", type=str, default="prompt",
        help="Field name for prompts in dataset",
    )
    parser.add_argument(
        "--ground-truth-field", type=str, default="ground_truth",
        help="Field name for ground truth in dataset",
    )
    parser.add_argument(
        "--outputs-field", type=str, default="outputs",
        help="Field name for pre-computed rollout outputs in dataset",
    )

    # Training
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Max training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save every N steps"
    )
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Eval every N steps"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset for evaluation",
    )
    parser.add_argument(
        "--eval-split", type=str, default="train",
        help="Split for eval dataset",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Load pretrained weights without restoring step/epoch/schedule",
    )

    # Logging
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="titans-mlx-rlvr")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Mixed precision
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for training",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Build config
    config = RLVRConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        method=args.method,
        mode=args.mode,
        num_rollouts=args.num_rollouts,
        epsilon=args.epsilon,
        kl_beta=args.kl_beta,
        ema_decay=args.ema_decay,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        verifier=args.verifier,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        lora_dropout=args.lora_dropout,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        max_len=args.max_len,
        prompt_field=args.prompt_field,
        ground_truth_field=args.ground_truth_field,
        outputs_field=args.outputs_field,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_dataset=args.eval_dataset,
        eval_split=args.eval_split,
        resume=args.resume,
        init_weights=args.init_weights,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        dtype=args.dtype,
        use_tnt=args.use_tnt,
        tnt_stage=args.tnt_stage,
        local_chunk_sizes=args.local_chunk_sizes,
        local_shard_length=args.local_shard_length,
        global_chunk_size=args.global_chunk_size,
        use_attn_res=args.use_attn_res,
        num_attnres_blocks=args.num_attnres_blocks,
        attnres_warmup_steps=args.attnres_warmup_steps,
        attnres_modulate_global=args.attnres_modulate_global,
        attnres_modulate_local=args.attnres_modulate_local,
    )

    # Check dependencies
    if not HAS_DATASETS:
        logger.error(
            "Install 'datasets' for HuggingFace datasets: pip install datasets"
        )
        return

    if config.wandb and not HAS_WANDB:
        logger.warning("wandb not installed, disabling logging")
        config.wandb = False

    # Load tokenizer
    if not HAS_TRANSFORMERS:
        logger.error(
            "Install 'transformers' for tokenizer support: pip install transformers"
        )
        return

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer vocab size: {config.vocab_size}")

    # Resolve verifier
    if config.verifier in BUILTIN_VERIFIERS:
        verifier_fn = BUILTIN_VERIFIERS[config.verifier]
        logger.info(f"Using built-in verifier: {config.verifier}")
    else:
        # Custom verifier: path/to/module.py:function_name
        verifier_fn = load_custom_verifier(config.verifier)
        logger.info(f"Loaded custom verifier from: {config.verifier}")

    # Create model config
    from titans_mlx import TitansConfig
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
        dropout=0.0,
        use_conv=False,
        use_tnt=config.use_tnt,
        tnt_stage=config.tnt_stage,
        local_chunk_sizes=config.local_chunk_sizes,
        local_shard_length=config.local_shard_length,
        global_chunk_size=config.global_chunk_size,
        use_attn_res=config.use_attn_res,
        num_attnres_blocks=config.num_attnres_blocks,
        attnres_warmup_steps=config.attnres_warmup_steps,
        attnres_modulate_global_memory=config.attnres_modulate_global,
        attnres_modulate_local_memory=config.attnres_modulate_local,
    )

    # Configure dtype for mixed precision
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    train_dtype = dtype_map[config.dtype]

    # Create model
    model = create_model(config.model_type, model_config)

    # Convert model to target dtype for mixed precision
    if config.dtype != "float32":
        logger.info(f"Converting model to {config.dtype} for mixed precision training")

        def convert_dtype(params, parent_key=""):
            if isinstance(params, dict):
                result = {}
                for k, v in params.items():
                    result[k] = convert_dtype(v, parent_key=k)
                return result
            elif isinstance(params, (list, tuple)):
                return type(params)(convert_dtype(item) for item in params)
            elif isinstance(params, mx.array):
                if "embed" in parent_key:
                    return params
                return params.astype(train_dtype)
            return params

        model.update(convert_dtype(model.parameters()))
        mx.eval(model.parameters())

    total_params = count_parameters(model)
    logger.info(f"Model: Titans{config.model_type.upper()} (MLX)")
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    logger.info(f"Training dtype: {config.dtype}")

    # LoRA setup
    if config.lora:
        from scripts.lora import wrap_lora_layers
        wrapped = wrap_lora_layers(
            model,
            targets=config.lora_targets,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        logger.info(f"LoRA enabled: wrapped {len(wrapped)} layers")

    # Create optimizer (AdamW)
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        betas=[0.9, 0.95],
    )

    # Create dataset
    logger.info(f"Using HuggingFace dataset: {config.dataset}")
    train_dataset = OfflineRLDataset(
        config.dataset,
        tokenizer,
        config.max_len,
        num_rollouts=config.num_rollouts,
        verifier=verifier_fn,
        subset=config.dataset_subset,
        split="train",
        seed=config.seed,
        prompt_field=config.prompt_field,
        ground_truth_field=config.ground_truth_field,
        outputs_field=config.outputs_field,
    )

    # Train
    train(
        model,
        optimizer,
        train_dataset,
        config,
        model_config,
    )


if __name__ == "__main__":
    main()
