#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
LoRA fine-tuning script for Titans PyTorch models.

Loads a pretrained Titans model, wraps targeted linear layers with LoRA
adapters, and fine-tunes on chat-formatted data (ChatML).  Only the LoRA
A and B matrices are updated; the base model remains frozen throughout.

Supports:
- HuggingFace tokenizers and datasets (streaming)
- HuggingFace Accelerate (single/multi-GPU, mixed precision)
- Cosine annealing with warmup
- Gradient accumulation
- WandB logging (optional)
- Optional adapter merging after training

Usage:
    # Fine-tune from a pretrained checkpoint
    python scripts/lora.py \\
        --init-weights checkpoints/final.pt \\
        --data-path data/chat.jsonl \\
        --lora-rank 8 --lora-alpha 16 --lora-targets attn

    # Multi-GPU via accelerate
    accelerate launch scripts/lora.py \\
        --init-weights checkpoints/final.pt \\
        --lora-targets attn,ffn \\
        --merge-and-save merged/model.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states
from titans.lora import (
    count_lora_parameters,
    merge_lora_weights,
    save_adapters,
    set_lora_enabled,
    wrap_lora_layers,
)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from accelerate import Accelerator

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore[misc,assignment]

HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_WANDB = importlib.util.find_spec("wandb") is not None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mac": TitansMAC,
    "mag": TitansMAG,
    "mal": TitansMAL,
    "lmm": TitansLMM,
}


@dataclass
class LoRATrainingConfig:
    """Training hyperparameters for LoRA fine-tuning."""

    # Model
    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    rope_proportion: float = 1.0
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0

    # Data
    data_path: str | None = None
    dataset: str | None = None
    dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    seq_len: int = 2048
    max_seq_len: int = 2048

    # Training
    init_weights: str | None = None
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    mixed_precision: str = "no"

    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: str = "attn"
    merge_and_save: str | None = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints/lora"
    save_every: int = 5000
    save_format: str = "pt"
    eval_every: int = 500
    resume: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-lora"
    wandb_run_name: str | None = None

    # Misc
    seed: int = 42
    synthetic_samples: int = 1000

    # Populated at runtime
    wrapped_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat data utilities (self-contained, no shared utilities)
# ---------------------------------------------------------------------------

CHATML_BOS = "<|im_start|>"
CHATML_EOS = "<|im_end|>"


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Format a list of role/content dicts into a ChatML string.

    Args:
        messages: List of dicts with "role" and "content" keys.

    Returns:
        A single ChatML-formatted string.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{CHATML_BOS}{role}\n{content}{CHATML_EOS}\n")
    return "".join(parts)


def tokenize_chat(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> dict[str, list[int]]:
    """Tokenize a ChatML conversation and produce input_ids + labels.

    Labels use -100 for all prompt tokens so loss is only computed over
    assistant turns.

    Args:
        messages: List of role/content dicts.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum token length (sequences are truncated).

    Returns:
        Dict with keys "input_ids" and "labels" (both lists of ints).
    """
    full_text = format_chatml(messages)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Build labels: mask out non-assistant tokens
    labels: list[int] = []
    input_ids: list[int] = []

    pos = 0
    text_so_far = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        turn_text = f"{CHATML_BOS}{role}\n{content}{CHATML_EOS}\n"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        if role == "assistant":
            input_ids.extend(turn_ids)
            labels.extend(turn_ids)
        else:
            input_ids.extend(turn_ids)
            labels.extend([-100] * len(turn_ids))

        text_so_far += turn_text
        pos += len(turn_ids)

    # Truncate to max_seq_len
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    return {"input_ids": input_ids, "labels": labels}


def build_loss_mask(labels: list[int]) -> list[int]:
    """Build a binary loss mask from a labels list.

    Args:
        labels: Token labels with -100 for masked positions.

    Returns:
        List of 0/1 integers where 1 = compute loss.
    """
    return [0 if tok == -100 else 1 for tok in labels]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SFTStreamingDataset(IterableDataset):  # type: ignore[type-arg]
    """Streaming dataset for SFT/LoRA training from a JSONL chat file.

    Each line must be a JSON object with a "messages" key containing a list
    of role/content dicts in ChatML format.

    Args:
        path: Path to the JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ) -> None:
        import json

        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._json = json

    def __iter__(self):  # type: ignore[override]
        import json

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    if not messages:
                        continue
                    tokenized = tokenize_chat(
                        messages, self.tokenizer, self.max_seq_len
                    )
                    input_ids = tokenized["input_ids"]
                    labels = tokenized["labels"]
                    loss_mask = build_loss_mask(labels)

                    # Replace -100 in labels with 0 for cross_entropy indexing
                    labels_clean = [max(tok, 0) for tok in labels]

                    # Skip samples with no supervised tokens
                    if sum(loss_mask) == 0:
                        continue

                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels_clean, dtype=torch.long),
                        "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
                    }
                except Exception as exc:
                    logger.warning(f"Skipping malformed record: {exc}")
                    continue


class SyntheticChatDataset(Dataset):  # type: ignore[type-arg]
    """Synthetic dataset that mimics the SFT batch schema for quick testing."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.input_ids = torch.from_numpy(
            rng.integers(0, vocab_size, (num_samples, seq_len)).astype("int64")
        )
        # Randomly mask first half as prompt (loss_mask=0), rest as response
        self.loss_mask = torch.zeros(num_samples, seq_len)
        self.loss_mask[:, seq_len // 2 :] = 1.0
        self.labels = self.input_ids.clone()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "loss_mask": self.loss_mask[idx],
        }


def sft_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length SFT samples to the same length.

    Args:
        batch: List of sample dicts with input_ids, labels, loss_mask.

    Returns:
        Padded batch dict.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []

    for item in batch:
        length = item["input_ids"].shape[0]
        pad_len = max_len - length
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.zeros(pad_len, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat([item["loss_mask"], torch.zeros(pad_len, dtype=torch.float)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "loss_mask": torch.stack(mask_list),
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(config: LoRATrainingConfig) -> nn.Module:
    """Build a Titans model from LoRATrainingConfig.

    Args:
        config: Training configuration.

    Returns:
        An instantiated (but not yet LoRA-wrapped) Titans model.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    if config.model_type not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type '{config.model_type}'. "
            f"Valid options: {sorted(_MODEL_CLASSES)}"
        )
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        rope_proportion=config.rope_proportion,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
        memory_objective=config.memory_objective,
        huber_delta_init=config.huber_delta_init,
    )
    return _MODEL_CLASSES[config.model_type](model_config)


def build_dataset(
    config: LoRATrainingConfig,
) -> Dataset | IterableDataset:  # type: ignore[type-arg]
    """Build a training dataset from the config.

    Args:
        config: Training configuration.

    Returns:
        A Dataset or IterableDataset.
    """
    if config.data_path is not None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for chat datasets. "
                "Install with: pip install transformers"
            )
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return SFTStreamingDataset(
            Path(config.data_path), tokenizer, config.max_seq_len
        )

    if config.dataset is not None:
        raise NotImplementedError(
            "Streaming HuggingFace dataset support for LoRA coming soon. "
            "Use --data-path with a local JSONL file, or omit for synthetic data."
        )

    logger.info("No dataset specified — using synthetic data for demo")
    return SyntheticChatDataset(
        config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: LoRATrainingConfig) -> None:
    """Main LoRA training loop.

    Args:
        config: Training configuration (populated from CLI args).

    Raises:
        ImportError: If accelerate is not installed.
    """
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required for training. "
            "Install with: pip install accelerate"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

    if accelerator.is_main_process:
        logger.info(f"LoRA training config: {config}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")

    torch.manual_seed(config.seed)

    # --- 1. Build base model ---
    model = build_model(config)

    # --- 2. Load pretrained weights ---
    if config.init_weights is not None:
        weights_path = Path(config.init_weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"init_weights file not found: {weights_path}"
            )
        state_dict = torch.load(str(weights_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            if missing:
                logger.warning(f"Missing keys when loading weights: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading weights: {unexpected}")
            logger.info(f"Loaded pretrained weights from {weights_path}")
    else:
        if accelerator.is_main_process:
            logger.warning(
                "No --init-weights provided. Training LoRA on randomly "
                "initialized weights."
            )

    # --- 3. Wrap targeted layers with LoRA ---
    wrapped_paths = wrap_lora_layers(
        model,
        targets=config.lora_targets,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    config.wrapped_paths = wrapped_paths

    trainable_params, total_params = count_lora_parameters(model)
    if accelerator.is_main_process:
        logger.info(
            f"LoRA wrapping complete: {len(wrapped_paths)} layers targeted "
            f"({', '.join(wrapped_paths[:5])}"
            f"{'...' if len(wrapped_paths) > 5 else ''})"
        )
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
        )

    # --- 4. Optimizer — only LoRA parameters ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # --- 5. Dataset and dataloader ---
    dataset = build_dataset(config)
    use_collate = isinstance(dataset, SFTStreamingDataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=0,
        drop_last=True,
        collate_fn=sft_collate_fn if use_collate else None,
    )

    # --- 6. LR scheduler ---
    total_steps = (
        config.max_steps
        if config.max_steps > 0
        else (
            len(dataloader) * config.epochs
            if not isinstance(dataset, IterableDataset)
            else 10000  # fallback for streaming datasets
        )
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )

    # --- 7. Accelerate preparation ---
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if config.wandb and HAS_WANDB and accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb_project,
            config=vars(config),
            init_kwargs={"wandb": {"name": config.wandb_run_name}},
        )

    checkpoint_dir = Path(config.checkpoint_dir)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_epoch = 0
    memory_states = None
    vocab_size = config.vocab_size

    # ------------------------------------------------------------------
    # Resume from LoRA checkpoint
    # ------------------------------------------------------------------
    if config.resume is not None:
        resume_path = Path(config.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", 0)

        # Load memory states if available
        mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        if not mem_path.exists():
            mem_path = resume_path.parent / "memory_final.npz"
        try:
            from titans.memory_dump import load_memory_states

            memory_states = load_memory_states(mem_path, device=accelerator.device)
            if accelerator.is_main_process:
                logger.info(f"Loaded memory states from {mem_path}")
        except Exception as e:
            if accelerator.is_main_process:
                logger.info(f"No memory states found ({e}), starting fresh")

        if accelerator.is_main_process:
            logger.info(
                f"Resumed from {resume_path} at step {global_step}, epoch {start_epoch}"
            )
        del checkpoint

    # --- 8. Training loop ---
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            with accelerator.accumulate(model):
                logits, memory_states, _ = model(
                    batch["input_ids"], states=memory_states
                )

                # Masked cross-entropy: only compute loss on assistant tokens
                logits_flat = logits.view(-1, vocab_size)
                labels_flat = batch["labels"].view(-1)
                mask_flat = batch["loss_mask"].view(-1).float()

                per_token = F.cross_entropy(
                    logits_flat, labels_flat, reduction="none"
                )
                loss = (per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if memory_states is not None:
                memory_states = [s.detach() for s in memory_states]

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            if global_step % config.log_every == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                lr_val = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}", lr=f"{lr_val:.2e}", step=global_step
                )
                if config.wandb and HAS_WANDB:
                    accelerator.log(
                        {
                            "train/loss": loss_val,
                            "train/avg_loss": avg_loss,
                            "train/lr": lr_val,
                        },
                        step=global_step,
                    )

            if (
                global_step % config.save_every == 0
                and accelerator.is_main_process
            ):
                unwrapped = accelerator.unwrap_model(model)
                adapter_path = (
                    checkpoint_dir
                    / f"adapters_step_{global_step}.safetensors"
                )
                meta = {
                    "lora_rank": config.lora_rank,
                    "lora_alpha": config.lora_alpha,
                    "lora_targets": config.lora_targets,
                    "global_step": global_step,
                }
                try:
                    save_adapters(unwrapped, adapter_path, meta)
                    logger.info(
                        f"Checkpoint: saved adapters at step "
                        f"{global_step}"
                    )
                except ImportError:
                    ckpt_stem = (
                        checkpoint_dir / f"step_{global_step}"
                    )
                    save_checkpoint(
                        unwrapped.state_dict(),
                        ckpt_stem,
                        format=config.save_format,
                    )
                    logger.info(
                        f"Checkpoint: saved full model at step "
                        f"{global_step}"
                    )

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} complete — avg loss: {avg_loss:.4f}")

    # --- 9. Final save ---
    if accelerator.is_main_process:
        # Save full model checkpoint
        final_stem = checkpoint_dir / "final"
        unwrapped = accelerator.unwrap_model(model)
        paths = save_checkpoint(
            unwrapped.state_dict(),
            final_stem,
            format=config.save_format,
        )
        logger.info(f"Saved full checkpoint to {paths[0]}")

        # Save adapter-only file (small, portable)
        adapter_path = checkpoint_dir / "adapters.safetensors"
        meta = {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_targets": config.lora_targets,
            "wrapped_paths": config.wrapped_paths,
            "model_type": config.model_type,
            "dim": config.dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "vocab_size": config.vocab_size,
            "global_step": global_step,
        }
        try:
            save_adapters(unwrapped, adapter_path, meta)
            logger.info(f"Saved LoRA adapters to {adapter_path}")
        except ImportError:
            logger.warning(
                "safetensors not installed — skipping adapter-only save. "
                "Install with: pip install safetensors"
            )

        # Optionally merge LoRA weights into base and save a standalone model
        if config.merge_and_save is not None:
            merge_path = Path(config.merge_and_save)
            merge_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Merging LoRA weights into base model...")
            merge_lora_weights(unwrapped)
            merge_stem = merge_path.with_suffix("")
            merge_files = save_checkpoint(
                unwrapped.state_dict(),
                merge_stem,
                format=config.save_format,
            )
            logger.info(f"Saved merged model to {merge_files[0]}")

    if config.wandb and HAS_WANDB:
        accelerator.end_training()

    logger.info("LoRA training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> LoRATrainingConfig:
    """Parse command-line arguments and return a LoRATrainingConfig."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Titans PyTorch models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", type=str, default="mac", choices=list(_MODEL_CLASSES)
    )
    model_group.add_argument("--dim", type=int, default=512)
    model_group.add_argument("--num-heads", type=int, default=8)
    model_group.add_argument("--num-layers", type=int, default=12)
    model_group.add_argument("--vocab-size", type=int, default=32000)
    model_group.add_argument("--chunk-size", type=int, default=512)
    model_group.add_argument("--window-size", type=int, default=512)
    model_group.add_argument(
        "--rope-proportion", type=float, default=1.0,
        help="Fraction of head_dim pairs to apply RoPE to (0.0-1.0, default 1.0)",
    )
    model_group.add_argument("--num-persistent-tokens", type=int, default=16)
    model_group.add_argument("--num-memory-layers", type=int, default=2)
    model_group.add_argument(
        "--memory-objective", type=str, default="l2", choices=["l2", "huber"]
    )

    # --- Data ---
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-path", type=str, default=None,
                            help="Path to JSONL file with 'messages' field per line")
    data_group.add_argument("--dataset", type=str, default=None)
    data_group.add_argument("--dataset-subset", type=str, default=None)
    data_group.add_argument("--tokenizer", type=str, default="gpt2")
    data_group.add_argument("--seq-len", type=int, default=2048)
    data_group.add_argument("--max-seq-len", type=int, default=2048)

    # --- Training ---
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--init-weights", type=str, default=None,
        help="Path to pretrained model state_dict (.pt) to load before LoRA wrapping"
    )
    train_group.add_argument("--epochs", type=int, default=1)
    train_group.add_argument("--max-steps", type=int, default=-1)
    train_group.add_argument("--batch-size", type=int, default=4)
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=8
    )
    train_group.add_argument("--lr", type=float, default=1e-4,
                             help="Learning rate (default higher than SFT: 1e-4)")
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--warmup-ratio", type=float, default=0.03)
    train_group.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )

    # --- LoRA ---
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-rank", type=int, default=8,
                            help="LoRA rank r")
    lora_group.add_argument("--lora-alpha", type=float, default=16.0,
                            help="LoRA alpha (scale = alpha / rank)")
    lora_group.add_argument("--lora-dropout", type=float, default=0.05,
                            help="Dropout on LoRA input path")
    lora_group.add_argument(
        "--lora-targets", type=str, default="attn",
        help="Comma-separated target groups: attn, ffn, memory, all"
    )
    lora_group.add_argument(
        "--merge-and-save", type=str, default=None, metavar="PATH",
        help="After training, merge LoRA into base weights and save to PATH"
    )

    # --- Checkpointing ---
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/lora"
    )
    ckpt_group.add_argument("--save-every", type=int, default=5000)
    ckpt_group.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
    )
    ckpt_group.add_argument("--eval-every", type=int, default=500)
    ckpt_group.add_argument("--resume", type=str, default=None)

    # --- Logging ---
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-every", type=int, default=10)
    log_group.add_argument("--wandb", action="store_true")
    log_group.add_argument("--wandb-project", type=str, default="titans-lora")
    log_group.add_argument("--wandb-run-name", type=str, default=None)

    # --- Misc ---
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--seed", type=int, default=42)
    misc_group.add_argument("--synthetic-samples", type=int, default=1000)

    args = parser.parse_args()

    return LoRATrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        rope_proportion=args.rope_proportion,
        num_persistent_tokens=args.num_persistent_tokens,
        num_memory_layers=args.num_memory_layers,
        memory_objective=args.memory_objective,
        data_path=args.data_path,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
        init_weights=args.init_weights,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=args.mixed_precision,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        merge_and_save=args.merge_and_save,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        save_format=args.save_format,
        eval_every=args.eval_every,
        resume=args.resume,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
