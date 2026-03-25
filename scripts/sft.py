#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Supervised Fine-Tuning (SFT) script for Titans MLX models on Apple Silicon.

Supports:
- ChatML formatting for conversation templates
- Per-token loss masking (assistant-only or train-on-all)
- Streaming HuggingFace chat datasets
- Gradient accumulation, cosine LR, checkpointing
- Weights & Biases logging (optional)

Usage:
    # SFT with a chat dataset
    uv run python scripts/sft.py --model mac --dataset allenai/Dolci-Instruct-SFT \
        --tokenizer gpt2 --dim 256 --num-layers 4

    # Train on all tokens (not just assistant)
    uv run python scripts/sft.py --model mac --dataset allenai/Dolci-Instruct-SFT \
        --tokenizer gpt2 --train-on-all
"""

from __future__ import annotations

import argparse
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


# =============================================================================
# SFT Configuration
# =============================================================================


@dataclass
class SFTConfig:
    """SFT training hyperparameters."""

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

    # Data
    dataset: str | None = None  # HuggingFace dataset name (required)
    dataset_subset: str | None = None  # Dataset subset/config
    tokenizer: str = "gpt2"  # HuggingFace tokenizer
    seq_len: int = 2048
    messages_field: str = "messages"
    train_on_all: bool = False
    chat_template: str = "auto"

    # Training (SFT defaults: lower LR, more accumulation)
    epochs: int = 1
    max_steps: int = -1  # -1 = use epochs
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Mixed precision
    dtype: str = "float16"

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
    wandb_project: str = "titans-mlx-sft"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42


# =============================================================================
# Streaming SFT Dataset
# =============================================================================


class SFTStreamingDataset:
    """Streaming dataset from HuggingFace chat datasets.

    Streams examples, tokenizes each via tokenize_chat(), and yields
    batches with input_ids, labels, and loss_mask.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        messages_field: str = "messages",
        train_on_all: bool = False,
        buffer_size: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self.messages_field = messages_field
        self.train_on_all = train_on_all
        self.buffer_size = buffer_size
        self._iterator: Any = None

    def _create_iterator(self):
        """Create a fresh streaming iterator."""
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for example in ds:
            messages = example.get(self.messages_field)
            if messages is None:
                continue

            try:
                result = tokenize_chat(
                    messages,
                    self.tokenizer,
                    self.max_len,
                    train_on_all=self.train_on_all,
                )
            except Exception:
                continue

            # Skip empty sequences
            if len(result["input_ids"]) == 0:
                continue

            yield result

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Return a batch of tokenized examples, padded to longest in batch.

        Returns:
            Dict with "input_ids", "labels", "loss_mask" as mx.arrays,
            or None if the dataset is exhausted.
        """
        if self._iterator is None:
            self._iterator = self._create_iterator()

        batch_items: list[dict] = []
        for _ in range(batch_size):
            try:
                item = next(self._iterator)
                batch_items.append(item)
            except StopIteration:
                # Reset iterator for next epoch
                self._iterator = self._create_iterator()
                if batch_items:
                    break
                return None

        if not batch_items:
            return None

        # Pad to longest sequence in batch
        max_seq = max(len(item["input_ids"]) for item in batch_items)
        pad_id = 0  # Pad token ID

        input_ids_batch = []
        labels_batch = []
        mask_batch = []

        for item in batch_items:
            seq_len = len(item["input_ids"])
            pad_len = max_seq - seq_len

            input_ids_batch.append(
                item["input_ids"] + [pad_id] * pad_len
            )
            labels_batch.append(
                item["labels"] + [pad_id] * pad_len
            )
            mask_batch.append(
                item["loss_mask"] + [0] * pad_len
            )

        return {
            "input_ids": mx.array(np.array(input_ids_batch)),
            "labels": mx.array(np.array(labels_batch)),
            "loss_mask": mx.array(np.array(mask_batch, dtype=np.float32)),
        }


# =============================================================================
# Model Creation
# =============================================================================


def create_model(model_type: str, config: TitansConfig) -> nn.Module:
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
# Loss and Gradient Functions
# =============================================================================


def masked_loss_fn(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compute masked cross-entropy loss (only on loss_mask=1 tokens)."""
    logits, _ = model(input_ids)  # Titans models return (logits, states)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)
    per_token = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="none")
    loss = (per_token * mask_flat).sum() / mx.clip(mask_flat.sum(), a_min=1, a_max=None)
    return loss, logits


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


def compute_grads(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
) -> tuple[mx.array, dict]:
    """Compute masked loss and gradients without updating parameters."""
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: masked_loss_fn(m, input_ids, labels, loss_mask)[0]
    )
    loss, grads = loss_and_grad_fn(model)
    return loss, grads


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
# Evaluation
# =============================================================================


def evaluate(
    model: nn.Module,
    dataset: SFTStreamingDataset,
    batch_size: int,
    num_batches: int = 50,
) -> dict[str, float]:
    """Evaluate by consuming num_batches from streaming dataset."""
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        batch = dataset.get_batch(batch_size)
        if batch is None:
            break

        loss, _ = masked_loss_fn(
            model, batch["input_ids"], batch["labels"], batch["loss_mask"]
        )
        mx.eval(loss)

        masked_count = int(batch["loss_mask"].sum())
        total_loss += float(loss) * masked_count
        total_tokens += masked_count

    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 100))}


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: SFTConfig,
    model_config: TitansConfig,
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
# Main Training Loop
# =============================================================================


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataset: SFTStreamingDataset,
    val_dataset: SFTStreamingDataset | None,
    config: SFTConfig,
    model_config: TitansConfig,
) -> None:
    """Main SFT training loop (streaming only)."""
    # Calculate total steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        total_steps = 100000  # Default for streaming

    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

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
                "seq_len": config.seq_len,
                "train_on_all": config.train_on_all,
                "chat_template": config.chat_template,
            },
        )

    start_time = time.time()
    pbar = tqdm(total=total_steps, initial=global_step, desc="SFT Training")
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

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]

            # --- Micro-step: compute gradients only ---
            micro_start = time.time()
            loss, grads = compute_grads(model, input_ids, labels, loss_mask)
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
                    avg_ppl = math.exp(min(avg_loss, 100))
                    masked_ratio = float(loss_mask.sum()) / max(
                        float(loss_mask.size), 1
                    )

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/ppl": avg_ppl,
                        "train/lr": lr,
                        "train/step": global_step,
                        "train/masked_ratio": masked_ratio,
                    }

                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ppl": f"{avg_ppl:.2f}",
                            "lr": f"{lr:.2e}",
                            "mask%": f"{masked_ratio:.1%}",
                        }
                    )

                    if config.wandb and HAS_WANDB:
                        wandb.log(log_dict, step=global_step)

                    running_loss = 0.0
                    running_count = 0

                # Evaluation
                if (
                    config.eval_every > 0
                    and global_step % config.eval_every == 0
                    and val_dataset is not None
                ):
                    val_metrics = evaluate(
                        model, val_dataset, config.batch_size
                    )
                    logger.info(
                        f"Step {global_step}: "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"val_ppl={val_metrics['val_ppl']:.2f}"
                    )

                    if config.wandb and HAS_WANDB:
                        wandb.log(
                            {
                                f"val/{k}": v
                                for k, v in val_metrics.items()
                            },
                            step=global_step,
                        )

                    # Save eval checkpoint and prune
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

                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        save_checkpoint(
                            model,
                            optimizer,
                            config,
                            model_config,
                            global_step,
                            epoch,
                            best_val_loss,
                            checkpoint_dir / "best_model",
                        )

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
    logger.info(f"SFT completed in {elapsed / 3600:.2f} hours")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    if config.wandb and HAS_WANDB:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for Titans MLX models",
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

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace chat dataset (e.g., allenai/Dolci-Instruct-SFT)",
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
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--messages-field", type=str, default="messages",
        help="Field name for messages in dataset",
    )
    parser.add_argument(
        "--train-on-all", action="store_true",
        help="Train on all tokens (not just assistant)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1=epochs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
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
    parser.add_argument("--wandb-project", type=str, default="titans-mlx-sft")
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
    config = SFTConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        messages_field=args.messages_field,
        train_on_all=args.train_on_all,
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

    # Create optimizer (AdamW)
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        betas=[0.9, 0.95],
    )

    # Create datasets
    logger.info(f"Using HuggingFace dataset: {config.dataset}")
    train_dataset = SFTStreamingDataset(
        config.dataset,
        tokenizer,
        config.seq_len,
        subset=config.dataset_subset,
        split="train",
        seed=config.seed,
        messages_field=config.messages_field,
        train_on_all=config.train_on_all,
    )

    val_dataset: SFTStreamingDataset | None = None
    if config.eval_dataset:
        val_dataset = SFTStreamingDataset(
            config.eval_dataset,
            tokenizer,
            config.seq_len,
            subset=config.dataset_subset,
            split=config.eval_split,
            seed=config.seed + 1,
            messages_field=config.messages_field,
            train_on_all=config.train_on_all,
        )
    elif config.eval_every > 0:
        # Use the same dataset with different seed for eval
        val_dataset = SFTStreamingDataset(
            config.dataset,
            tokenizer,
            config.seq_len,
            subset=config.dataset_subset,
            split="train",
            seed=config.seed + 1,
            messages_field=config.messages_field,
            train_on_all=config.train_on_all,
        )

    # Train
    train(model, optimizer, train_dataset, val_dataset, config, model_config)


if __name__ == "__main__":
    main()
