#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans MLX models on Apple Silicon.

Optimized for M1/M2/M3/M4 GPUs with unified memory architecture.

Supports:
- HuggingFace tokenizers (LLaMA 2, GPT-2, etc.)
- HuggingFace datasets with streaming
- Cosine annealing with warmup
- Gradient accumulation
- Weights & Biases logging (optional)

Usage:
    # Demo with synthetic data
    uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

    # Train with FineWeb-Edu (streaming)
    uv run python scripts/pretrain.py --model mac --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer meta-llama/Llama-2-7b-hf --dim 512 --num-layers 12

    # Train with local text file
    uv run python scripts/pretrain.py --model mag --data path/to/data.txt

    # Resume from checkpoint
    uv run python scripts/pretrain.py --model mac --resume checkpoints/latest.npz
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
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


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

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
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global: bool = True
    attnres_modulate_local: bool = False

    # Data
    dataset: str | None = None  # HuggingFace dataset name
    dataset_subset: str | None = None  # Dataset subset/config
    data_path: str | None = None  # Local text file
    tokenizer: str = "gpt2"  # HuggingFace tokenizer
    seq_len: int = 4096  # Paper uses 4K

    # Training (following paper Section 5.1)
    epochs: int = 1
    max_steps: int = -1  # -1 = use epochs
    batch_size: int = 4  # Per-device batch size
    gradient_accumulation_steps: int = 32  # Effective batch ~0.5M tokens
    lr: float = 4e-4  # Paper: 4e-4
    weight_decay: float = 0.1  # Paper: 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Mixed precision
    dtype: str = "float16"  # float32, float16, bfloat16

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10000  # Save every N steps
    eval_every: int = 500  # Evaluate every N steps
    eval_dataset: str | None = None  # Separate HF dataset for eval
    eval_split: str = "train"  # Split for eval dataset
    eval_buffer_size: int = 100  # Sequences to buffer for eval
    resume: str | None = None
    init_weights: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-mlx"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    synthetic_samples: int = 10000  # For demo mode


# =============================================================================
# Datasets
# =============================================================================


class SyntheticDataset:
    """Synthetic dataset for testing/demo purposes."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        np.random.seed(seed)
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        return {
            "input_ids": mx.array(self.data[idx, :-1]),
            "labels": mx.array(self.data[idx, 1:]),
        }

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_data = self.data[indices]
        return {
            "input_ids": mx.array(batch_data[:, :-1]),
            "labels": mx.array(batch_data[:, 1:]),
        }


class TextFileDataset:
    """Dataset from a local text file with HuggingFace tokenizer."""

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        self.tokens = np.array(
            tokenizer.encode(text, add_special_tokens=False), dtype=np.int32
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": mx.array(x), "labels": mx.array(y)}

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_x = []
        batch_y = []
        for idx in indices:
            batch_x.append(self.tokens[idx : idx + self.seq_len])
            batch_y.append(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


class CharLevelDataset:
    """Simple character-level dataset (fallback when no tokenizer)."""

    def __init__(
        self,
        path: Path,
        vocab_size: int,
        seq_len: int,
    ) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        char_to_idx = {c: i % vocab_size for i, c in enumerate(chars)}
        self.tokens = np.array([char_to_idx.get(c, 0) for c in text], dtype=np.int32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": mx.array(x), "labels": mx.array(y)}

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_x = []
        batch_y = []
        for idx in indices:
            batch_x.append(self.tokens[idx : idx + self.seq_len])
            batch_y.append(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


class StreamingDataset:
    """Streaming dataset from HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self._iterator = None
        self._buffer: list[int] = []

    def __iter__(self):
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)

        buffer: list[int] = []
        for example in ds:
            # Get text from example (try common field names)
            text = example.get("text") or example.get("content") or str(example)

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            # Yield complete sequences
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]  # Overlap by 1 for next prediction

                yield {
                    "input_ids": mx.array(chunk[:-1]),
                    "labels": mx.array(chunk[1:]),
                }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Get a batch from streaming dataset."""
        if self._iterator is None:
            self._iterator = iter(self)

        batch_x = []
        batch_y = []

        for _ in range(batch_size):
            try:
                sample = next(self._iterator)
                batch_x.append(np.array(sample["input_ids"]))
                batch_y.append(np.array(sample["labels"]))
            except StopIteration:
                self._iterator = iter(self)
                if batch_x:
                    break
                return None

        if not batch_x:
            return None

        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


class BufferedEvalDataset:
    """Fixed-size eval set buffered from a streaming source.

    Drains num_sequences samples from a streaming dataset's __iter__
    into memory, then exposes __len__ + get_batch(indices) for
    compatibility with the evaluate() function.
    """

    def __init__(
        self,
        streaming_dataset: StreamingDataset,
        num_sequences: int = 100,
    ) -> None:
        self.seq_len = streaming_dataset.seq_len
        self.input_ids: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

        logger.info(f"Buffering {num_sequences} eval sequences from stream...")
        for sample in streaming_dataset:
            self.input_ids.append(np.array(sample["input_ids"]))
            self.labels.append(np.array(sample["labels"]))
            if len(self.input_ids) >= num_sequences:
                break

        logger.info(f"Eval buffer ready: {len(self.input_ids)} sequences")

    def __len__(self) -> int:
        """Return number of buffered sequences."""
        return len(self.input_ids)

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get batch by indices -- matches evaluate() expected interface."""
        batch_x = [self.input_ids[i] for i in indices]
        batch_y = [self.labels[i] for i in indices]
        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
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
# Training Functions
# =============================================================================


def loss_fn(
    model: nn.Module, input_ids: mx.array, labels: mx.array
) -> tuple[mx.array, mx.array]:
    """Compute cross-entropy loss."""
    logits, _ = model(input_ids)
    # Reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Cross entropy loss
    loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="mean")
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
# Gradient Computation and Application (split for accumulation)
# =============================================================================


def compute_grads(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
) -> tuple[mx.array, dict]:
    """Compute loss and gradients without updating parameters.

    Returns:
        Tuple of (loss, grads) where grads is a nested dict matching
        the model parameter tree.
    """
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: loss_fn(m, input_ids, labels)[0]
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
    # Replace NaN with 0 (global norm clipping below preserves gradient direction)
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

    # Global norm clipping
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
    """Sanitize, clip, and apply accumulated gradients.

    This is the only place where optimizer.update and mx.eval of model
    state should occur during training.
    """
    grads = sanitize_and_clip_grads(grads, grad_clip)
    optimizer.update(model, grads)

    # Re-tie head and embedding weights after optimizer step.
    # MLX optimizer.update() creates new arrays for each parameter entry,
    # breaking the weight tie. Re-tying after each step ensures the head
    # always uses the updated embedding weights.
    if hasattr(model, "head") and hasattr(model, "embed"):
        model.head.weight = model.embed.weight

    mx.eval(model.parameters(), optimizer.state)


def evaluate(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int = 50,
) -> dict[str, float]:
    """Evaluate on validation set."""
    total_loss = 0.0
    total_tokens = 0

    indices = list(range(min(len(dataset), num_batches * batch_size)))
    np.random.shuffle(indices)

    for i in range(0, min(len(indices), num_batches * batch_size), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            continue

        batch = dataset.get_batch(batch_indices)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        loss, _ = loss_fn(model, input_ids, labels)
        mx.eval(loss)

        batch_tokens = labels.size
        total_loss += float(loss) * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 100))}


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    model_config: TitansConfig,
    step: int,
    epoch: int,
    best_val_loss: float,
    path: Path,
) -> None:
    """Save checkpoint in MLX format."""
    # Get model weights as flat dict
    weights = dict(model.parameters())

    # Convert to numpy for saving
    weights_np = {}
    for k, v in weights.items():
        if isinstance(v, mx.array):
            weights_np[k] = np.array(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                weights_np[f"{k}.{k2}"] = np.array(v2)

    # Save metadata as JSON-compatible dict
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
    """Remap old TNT checkpoint keys to consolidated format.

    Old: blocks.N.hierarchical_memory.* -> New: blocks.N.memory.*
    """
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
    # Normalize path: strip .meta.npz or .npz to get the stem
    stem = path
    if stem.name.endswith(".meta.npz"):
        stem = stem.with_name(stem.name.removesuffix(".meta.npz"))
    else:
        stem = stem.with_suffix("")

    # Load weights
    weights_path = stem.with_suffix(".safetensors")
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        weights = _remap_tnt_keys(weights)
        model.load_weights(list(weights.items()))
    else:
        # Try legacy npz format
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

    # Re-tie head and embedding weights (load_weights breaks the reference)
    if hasattr(model, "head") and hasattr(model, "embed"):
        model.head.weight = model.embed.weight

    # Load metadata
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
    train_dataset: Any,
    val_dataset: Any | None,
    config: TrainingConfig,
    model_config: TitansConfig,
) -> None:
    """Main training loop."""
    # Calculate total steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    elif hasattr(train_dataset, "__len__"):
        steps_per_epoch = (
            len(train_dataset)
            // config.batch_size
            // config.gradient_accumulation_steps
        )
        total_steps = max(1, steps_per_epoch * config.epochs)
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

    # Resume if specified (restores step/epoch/schedule)
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
            },
        )

    start_time = time.time()
    pbar = tqdm(total=total_steps, initial=global_step, desc="Training")
    pbar.refresh()  # Force initial render

    # Training loop — gradient accumulation state
    accumulated_grads: dict | None = None
    accumulation_loss = 0.0
    accumulation_step = 0

    while global_step < total_steps:
        epoch += 1

        # Get batches
        if hasattr(train_dataset, "get_batch") and hasattr(train_dataset, "__len__"):
            # Fixed-size dataset
            indices = list(range(len(train_dataset)))
            np.random.shuffle(indices)

            for i in range(0, len(indices), config.batch_size):
                if global_step >= total_steps:
                    break

                batch_indices = indices[i : i + config.batch_size]
                if len(batch_indices) < config.batch_size:
                    continue

                batch = train_dataset.get_batch(batch_indices)
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # --- Micro-step: compute gradients only ---
                micro_start = time.time()
                loss, grads = compute_grads(model, input_ids, labels)

                # Materialize this micro-step's arrays to bound memory.
                # Without this, MLX's lazy graph grows across accumulation
                # steps and eventually OOMs.
                _eval_grads(grads, loss)
                micro_elapsed = time.time() - micro_start

                # Show micro-step progress within accumulation window
                pbar.set_postfix({
                    "micro": f"{accumulation_step + 1}/{config.gradient_accumulation_steps}",
                    "μloss": f"{float(loss):.4f}",
                    "μtime": f"{micro_elapsed:.1f}s",
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
                    # Average the accumulated gradients
                    scale = mx.array(
                        1.0 / config.gradient_accumulation_steps
                    )
                    avg_grads = _tree_scale(accumulated_grads, scale)

                    # Set LR once per optimizer step (not per micro-step)
                    lr = get_lr_schedule(
                        global_step, total_steps, warmup_steps, config.lr
                    )
                    optimizer.learning_rate = lr

                    # Sanitize, clip, update, eval
                    apply_gradients(
                        model, optimizer, avg_grads, config.grad_clip
                    )

                    global_step += 1
                    accumulation_step = 0
                    accumulation_loss = 0.0
                    accumulated_grads = None

                    # Periodic checkpoint (--save-every)
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

                        log_dict = {
                            "train/loss": avg_loss,
                            "train/ppl": avg_ppl,
                            "train/lr": lr,
                            "train/step": global_step,
                        }

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{avg_ppl:.2f}",
                                "lr": f"{lr:.2e}",
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

                        # Save eval checkpoint and prune old ones
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

                        # Update best model only when loss improves
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

        else:
            # Streaming dataset — use get_batch() for proper batching
            # (iterating __iter__ yields single samples, ignoring batch_size)
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

                # --- Micro-step: compute gradients only ---
                micro_start = time.time()
                loss, grads = compute_grads(model, input_ids, labels)
                _eval_grads(grads, loss)
                micro_elapsed = time.time() - micro_start

                # Show micro-step progress within accumulation window
                pbar.set_postfix({
                    "micro": f"{accumulation_step + 1}/{config.gradient_accumulation_steps}",
                    "μloss": f"{float(loss):.4f}",
                    "μtime": f"{micro_elapsed:.1f}s",
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

                    # Periodic checkpoint (--save-every)
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

                    if global_step % config.log_every == 0:
                        avg_loss = running_loss / running_count
                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{math.exp(min(avg_loss, 100)):.2f}",
                                "lr": f"{lr:.2e}",
                            }
                        )

                        if config.wandb and HAS_WANDB:
                            wandb.log(
                                {
                                    "train/loss": avg_loss,
                                    "train/ppl": math.exp(
                                        min(avg_loss, 100)
                                    ),
                                    "train/lr": lr,
                                    "train/step": global_step,
                                },
                                step=global_step,
                            )

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

                        # Save eval checkpoint and prune old ones
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

                        # Update best model only when loss improves
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
    logger.info(f"Training completed in {elapsed / 3600:.2f} hours")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    if config.wandb and HAS_WANDB:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain Titans MLX models",
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
        "--use-tnt", action="store_true", help="Enable TNT hierarchical memory (global + local)"
    )
    parser.add_argument(
        "--local-chunk-sizes", type=int, nargs="+", default=[8, 16],
        help="Chunk sizes for local memories (one per local memory)",
    )
    parser.add_argument(
        "--local-shard-length", type=int, default=2048,
        help="Local memory reset period (tokens)",
    )
    parser.add_argument(
        "--global-chunk-size", type=int, default=2048,
        help="Global memory chunk size",
    )

    # AttnRes
    parser.add_argument(
        "--use-attn-res", action="store_true", help="Enable Attention Residuals"
    )
    parser.add_argument(
        "--num-attnres-blocks", type=int, default=8, help="AttnRes block count (N)"
    )
    parser.add_argument(
        "--attnres-warmup-steps", type=int, default=0,
        help="Steps before AttnRes memory gating activates",
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
        default=None,
        help="HuggingFace dataset (e.g., HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument(
        "--dataset-subset", type=str, default=None, help="Dataset subset"
    )
    parser.add_argument("--data", type=str, default=None, help="Local text file path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")

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
        default=32,
        help="Gradient accumulation",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
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
        help="HuggingFace dataset for evaluation (default: same as --dataset)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="train",
        help="Split for eval dataset (default: train)",
    )
    parser.add_argument(
        "--eval-buffer-size",
        type=int,
        default=100,
        help="Number of sequences to buffer for streaming evaluation (default: 100)",
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
    parser.add_argument("--wandb-project", type=str, default="titans-mlx")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Mixed precision
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for training (float16/bfloat16 for mixed precision)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--synthetic-samples", type=int, default=10000, help="Synthetic samples (demo)"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Build config
    config = TrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        data_path=args.data,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
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
        eval_buffer_size=args.eval_buffer_size,
        resume=args.resume,
        init_weights=args.init_weights,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
        dtype=args.dtype,
        use_tnt=args.use_tnt,
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
    if config.dataset and not HAS_DATASETS:
        logger.error(
            "Install 'datasets' for HuggingFace datasets: pip install datasets"
        )
        return

    if config.wandb and not HAS_WANDB:
        logger.warning("wandb not installed, disabling logging")
        config.wandb = False

    # Load tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS and (config.dataset or config.data_path):
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
        dropout=0.0,  # Usually 0 for pretraining
        use_conv=False,  # Disable conv to avoid dimension issues
        use_tnt=config.use_tnt,
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
            """Recursively convert parameters to target dtype."""
            if isinstance(params, dict):
                result = {}
                for k, v in params.items():
                    result[k] = convert_dtype(v, parent_key=k)
                return result
            elif isinstance(params, (list, tuple)):
                return type(params)(convert_dtype(item) for item in params)
            elif isinstance(params, mx.array):
                # Keep embedding weights in float32 for stability
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

    # Create optimizer (AdamW as in paper)
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        betas=[0.9, 0.95],  # Common for LLMs
    )

    # Create dataset
    train_dataset: Any
    val_dataset: Any | None = None

    if config.dataset:
        # HuggingFace streaming dataset
        logger.info(f"Using HuggingFace dataset: {config.dataset}")
        if tokenizer is None:
            raise ValueError("Tokenizer required for HuggingFace datasets")

        train_dataset = StreamingDataset(
            config.dataset,
            tokenizer,
            config.seq_len,
            subset=config.dataset_subset,
            split="train",
            seed=config.seed,
        )
        # Buffer eval sequences from stream
        if config.eval_dataset:
            # Use a separate dataset for eval
            eval_stream = StreamingDataset(
                config.eval_dataset,
                tokenizer,
                config.seq_len,
                subset=config.dataset_subset,
                split=config.eval_split,
                seed=config.seed + 1,
            )
        else:
            # Buffer from a fresh stream of the same dataset (different seed)
            eval_stream = StreamingDataset(
                config.dataset,
                tokenizer,
                config.seq_len,
                subset=config.dataset_subset,
                split="train",
                seed=config.seed + 1,
            )
        val_dataset = BufferedEvalDataset(eval_stream, config.eval_buffer_size)

    elif config.data_path:
        # Local text file
        logger.info(f"Loading data from: {config.data_path}")
        path = Path(config.data_path)

        if tokenizer is not None:
            full_dataset = TextFileDataset(path, tokenizer, config.seq_len)
        else:
            full_dataset = CharLevelDataset(path, config.vocab_size, config.seq_len)

        # Split into train/val
        train_size = int(0.95 * len(full_dataset))
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subset datasets
        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

            def get_batch(self, batch_indices):
                actual_indices = [self.indices[i] for i in batch_indices]
                return self.dataset.get_batch(actual_indices)

        train_dataset = SubsetDataset(full_dataset, train_indices)
        val_dataset = SubsetDataset(full_dataset, val_indices)
        logger.info(f"Train samples: {train_size}, Val samples: {len(val_indices)}")

    else:
        # Synthetic data (demo)
        logger.info("Using synthetic data (demo mode)")
        train_dataset = SyntheticDataset(
            config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
        )
        val_dataset = SyntheticDataset(
            config.vocab_size,
            config.seq_len,
            config.synthetic_samples // 10,
            config.seed + 1,
        )

    # Log effective batch size
    effective_batch_size = (
        config.batch_size * config.gradient_accumulation_steps * config.seq_len
    )
    logger.info(f"Effective batch size: {effective_batch_size:,} tokens")
    logger.info(f"Sequence length: {config.seq_len}")
    logger.info("Backend: MLX (Apple Silicon optimized)")

    # Train
    train(model, optimizer, train_dataset, val_dataset, config, model_config)


if __name__ == "__main__":
    main()
    os._exit(0)
