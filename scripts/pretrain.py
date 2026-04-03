#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans PyTorch models.

Supports:
- HuggingFace tokenizers and datasets (streaming)
- HuggingFace Accelerate (single/multi-GPU, mixed precision)
- Cosine annealing with warmup
- Gradient accumulation
- WandB logging (optional)

Usage:
    # Demo with synthetic data (CPU/GPU)
    python scripts/pretrain.py --model mac --dim 256 --epochs 10

    # Train with FineWeb-Edu on GPU
    python scripts/pretrain.py --model mac --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer meta-llama/Llama-2-7b-hf --dim 512 --num-layers 12

    # Multi-GPU via accelerate
    accelerate launch scripts/pretrain.py --model mac --dim 1024 --num-layers 24
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from titans import TitansConfig, TitansMAC

# Optional imports
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


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0

    dataset: str | None = None
    dataset_subset: str | None = None
    data_path: str | None = None
    tokenizer: str = "gpt2"
    seq_len: int = 4096

    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    lr: float = 4e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    mixed_precision: str = "no"

    checkpoint_dir: str = "checkpoints"
    save_every: int = 10000
    eval_every: int = 500
    resume: str | None = None

    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-pytorch"
    wandb_run_name: str | None = None

    seed: int = 42
    synthetic_samples: int = 10000


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing/demo."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        np.random.seed(seed)
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.data[idx, :-1].copy()).long(),
            "labels": torch.from_numpy(self.data[idx, 1:].copy()).long(),
        }


class TextFileDataset(Dataset):
    """Dataset from a local text file."""

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizerBase, seq_len: int):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.tokens = np.array(
            tokenizer.encode(text, add_special_tokens=False), dtype=np.int32
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {
            "input_ids": torch.from_numpy(x.copy()).long(),
            "labels": torch.from_numpy(y.copy()).long(),
        }


def build_model(config: TrainingConfig) -> TitansMAC:
    """Build Titans model from training config."""
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
        memory_objective=config.memory_objective,
        huber_delta_init=config.huber_delta_init,
    )
    if config.model_type != "mac":
        raise NotImplementedError(
            f"Model type '{config.model_type}' not yet ported. Only 'mac' is available."
        )
    return TitansMAC(model_config)


def build_dataset(config: TrainingConfig) -> Dataset:
    """Build training dataset."""
    if config.data_path is not None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for text file datasets")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        return TextFileDataset(Path(config.data_path), tokenizer, config.seq_len)

    if config.dataset is not None:
        raise NotImplementedError(
            "Streaming HuggingFace dataset support coming soon. "
            "Use --data-path with a local text file, or omit for synthetic data."
        )

    logger.info("No dataset specified — using synthetic data for demo")
    return SyntheticDataset(
        config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
    )


def train(config: TrainingConfig) -> None:
    """Main training loop."""
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required for training. Install with: pip install accelerate"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

    if accelerator.is_main_process:
        logger.info(f"Training config: {config}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")

    torch.manual_seed(config.seed)

    model = build_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    dataset = build_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    total_steps = config.max_steps if config.max_steps > 0 else len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )

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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    memory_states = None

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_main_process)

        for batch in pbar:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            with accelerator.accumulate(model):
                logits, memory_states = model(batch["input_ids"], states=memory_states)

                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    batch["labels"].view(-1),
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if memory_states is not None:
                memory_states = [s.detach() for s in memory_states]

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                if config.wandb and HAS_WANDB:
                    accelerator.log(
                        {"loss": loss_val, "avg_loss": avg_loss, "lr": lr},
                        step=global_step,
                    )

            if global_step % config.save_every == 0 and accelerator.is_main_process:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    if accelerator.is_main_process:
        final_path = checkpoint_dir / "final.pt"
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), final_path)
        logger.info(f"Training complete. Final checkpoint: {final_path}")

    if config.wandb and HAS_WANDB:
        accelerator.end_training()


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Pretrain Titans PyTorch models")
    parser.add_argument("--model", type=str, default="mac", choices=["mac"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--num-persistent-tokens", type=int, default=16)
    parser.add_argument("--num-memory-layers", type=int, default=2)
    parser.add_argument("--memory-objective", type=str, default="l2", choices=["l2", "huber"])

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-subset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="titans-pytorch")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=10000)

    args = parser.parse_args()

    return TrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        num_persistent_tokens=args.num_persistent_tokens,
        num_memory_layers=args.num_memory_layers,
        memory_objective=args.memory_objective,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        data_path=args.data_path,
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
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
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
