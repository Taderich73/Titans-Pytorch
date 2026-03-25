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


# =============================================================================
# DPO Configuration
# =============================================================================


@dataclass
class DPOConfig:
    """DPO training hyperparameters."""

    # Model (same fields as SFTConfig)
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

    # DPO-specific
    method: str = "dpo"
    beta: float = 0.1
    gamma: float = 1.0

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
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    chat_template: str = "auto"

    # Training
    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lr: float = 5e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

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
    wandb_project: str = "titans-mlx-dpo"
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
    config: DPOConfig,
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
        "training_stage": "dpo",
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
# DPO Gradient Computation
# =============================================================================


def compute_dpo_grads(
    model: nn.Module,
    ref_model: nn.Module | None,
    chosen_ids: mx.array,
    chosen_mask: mx.array,
    rejected_ids: mx.array,
    rejected_mask: mx.array,
    config: DPOConfig,
    use_lora_ref: bool = False,
) -> tuple[mx.array, dict]:
    """Compute DPO/SimPO loss and gradients."""
    if config.method == "simpo":
        def simpo_loss_fn(m):
            chosen_lps = compute_logprobs(m, chosen_ids, mask=chosen_mask)
            rejected_lps = compute_logprobs(m, rejected_ids, mask=rejected_mask)
            chosen_lengths = mx.clip(chosen_mask[:, 1:].sum(axis=1), a_min=1, a_max=None)
            rejected_lengths = mx.clip(rejected_mask[:, 1:].sum(axis=1), a_min=1, a_max=None)
            chosen_avg = chosen_lps.sum(axis=1) / chosen_lengths
            rejected_avg = rejected_lps.sum(axis=1) / rejected_lengths
            return simpo_loss(chosen_avg, rejected_avg, beta=config.beta, gamma=config.gamma)

        loss_and_grad_fn = nn.value_and_grad(model, simpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads
    else:
        from scripts.lora import set_lora_enabled

        if use_lora_ref:
            set_lora_enabled(model, False)
            ref_chosen_lps = compute_logprobs(model, chosen_ids, mask=chosen_mask)
            ref_rejected_lps = compute_logprobs(model, rejected_ids, mask=rejected_mask)
            mx.eval(ref_chosen_lps, ref_rejected_lps)
            ref_chosen_sum = mx.stop_gradient(ref_chosen_lps.sum(axis=1))
            ref_rejected_sum = mx.stop_gradient(ref_rejected_lps.sum(axis=1))
            set_lora_enabled(model, True)
        elif ref_model is not None:
            ref_chosen_lps = compute_logprobs(ref_model, chosen_ids, mask=chosen_mask)
            ref_rejected_lps = compute_logprobs(ref_model, rejected_ids, mask=rejected_mask)
            mx.eval(ref_chosen_lps, ref_rejected_lps)
            ref_chosen_sum = mx.stop_gradient(ref_chosen_lps.sum(axis=1))
            ref_rejected_sum = mx.stop_gradient(ref_rejected_lps.sum(axis=1))
        else:
            raise ValueError("Standard DPO requires --lora or a reference model")

        def dpo_loss_fn(m):
            c_lps = compute_logprobs(m, chosen_ids, mask=chosen_mask)
            r_lps = compute_logprobs(m, rejected_ids, mask=rejected_mask)
            return dpo_loss(c_lps.sum(axis=1), r_lps.sum(axis=1),
                           ref_chosen_sum, ref_rejected_sum, beta=config.beta)

        loss_and_grad_fn = nn.value_and_grad(model, dpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    model: nn.Module,
    ref_model: nn.Module | None,
    optimizer: optim.Optimizer,
    train_dataset: DPOStreamingDataset,
    config: DPOConfig,
    model_config: "TitansConfig",
    use_lora_ref: bool = False,
) -> None:
    """Main DPO training loop (streaming only)."""
    # Calculate total steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        total_steps = 100000  # Default for streaming

    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Method: {config.method}")

    # State
    global_step = 0
    epoch = 0
    best_val_loss = float("inf")
    running_loss = 0.0
    running_count = 0

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
                "beta": config.beta,
                "gamma": config.gamma,
                "lora": config.lora,
            },
        )

    start_time = time.time()
    pbar = tqdm(total=total_steps, initial=global_step, desc="DPO Training")
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

            chosen_ids = batch["chosen_ids"]
            chosen_mask = batch["chosen_mask"]
            rejected_ids = batch["rejected_ids"]
            rejected_mask = batch["rejected_mask"]

            # --- Micro-step: compute gradients only ---
            micro_start = time.time()
            loss, grads = compute_dpo_grads(
                model,
                ref_model,
                chosen_ids,
                chosen_mask,
                rejected_ids,
                rejected_mask,
                config,
                use_lora_ref=use_lora_ref,
            )
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

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": global_step,
                    }

                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        }
                    )

                    if config.wandb and HAS_WANDB:
                        wandb.log(log_dict, step=global_step)

                    running_loss = 0.0
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
    logger.info(f"DPO completed in {elapsed / 3600:.2f} hours")
    logger.info(f"Total steps: {global_step}")

    if config.wandb and HAS_WANDB:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO/SimPO training for Titans MLX models",
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

    # DPO-specific
    parser.add_argument(
        "--method",
        type=str,
        default="dpo",
        choices=["dpo", "simpo"],
        help="Preference optimization method",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty / scaling factor")
    parser.add_argument("--gamma", type=float, default=1.0, help="SimPO reward margin")

    # LoRA
    parser.add_argument(
        "--lora", action="store_true",
        help="Use LoRA (base model serves as reference for DPO)",
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
        help="HuggingFace preference dataset (e.g., allenai/Dolci-Instruct-DPO)",
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
        "--chosen-field", type=str, default="chosen",
        help="Field name for chosen responses in dataset",
    )
    parser.add_argument(
        "--rejected-field", type=str, default="rejected",
        help="Field name for rejected responses in dataset",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1=epochs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")

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
    parser.add_argument("--wandb-project", type=str, default="titans-mlx-dpo")
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
    config = DPOConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        method=args.method,
        beta=args.beta,
        gamma=args.gamma,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        lora_dropout=args.lora_dropout,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        max_len=args.max_len,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        epochs=args.epochs,
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

    # Create model config
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
    ref_model: nn.Module | None = None
    use_lora_ref = False

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
        if config.method == "dpo":
            use_lora_ref = True
            logger.info("DPO: using base model (LoRA disabled) as reference")
    elif config.method == "dpo":
        # Standard DPO: create a frozen reference model
        logger.warning(
            "Standard DPO without LoRA requires a second frozen reference model. "
            "This doubles memory usage. Consider --lora for memory efficiency."
        )
        ref_model = create_model(config.model_type, model_config)
        if config.dtype != "float32":
            ref_model.update(convert_dtype(ref_model.parameters()))
        mx.eval(ref_model.parameters())
        logger.info("Created frozen reference model for standard DPO")

    # Create optimizer (AdamW)
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        betas=[0.9, 0.95],
    )

    # Create dataset
    logger.info(f"Using HuggingFace dataset: {config.dataset}")
    train_dataset = DPOStreamingDataset(
        config.dataset,
        tokenizer,
        config.max_len,
        subset=config.dataset_subset,
        split="train",
        seed=config.seed,
        chosen_field=config.chosen_field,
        rejected_field=config.rejected_field,
    )

    # Train
    train(
        model,
        ref_model,
        optimizer,
        train_dataset,
        config,
        model_config,
        use_lora_ref=use_lora_ref,
    )


if __name__ == "__main__":
    main()
