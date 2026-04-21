#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Supervised Fine-Tuning (SFT) script for Titans PyTorch models.

Ports the MLX SFT workflow to PyTorch/Accelerate. Supports chat/instruction
fine-tuning with per-token loss masking so only assistant turns contribute
to the gradient.

Supported model variants: mac, mag, mal, lmm
Supported datasets: any HuggingFace dataset whose rows have a ``messages``
field (or the field named by --messages-field) containing a list of
``{"role": ..., "content": ...}`` dicts.

Usage:
    # Quick smoke-test on synthetic data
    python scripts/sft.py --model mac --dim 256 --epochs 2

    # Fine-tune a pretrained checkpoint on a chat dataset
    python scripts/sft.py \\
        --model mac \\
        --init-weights checkpoints/pretrain/final.pt \\
        --dataset HuggingFaceH4/ultrachat_200k \\
        --tokenizer meta-llama/Llama-2-7b-hf \\
        --dim 512 --num-layers 12 \\
        --mixed-precision bf16

    # Multi-GPU via accelerate
    accelerate launch scripts/sft.py --model mac --dim 1024 --num-layers 24 \\
        --dataset myorg/my-chat-dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states

# scripts/ is imported both as a namespace package ("scripts._common") and as
# a flat directory (when tests add scripts/ onto sys.path and import "sft").
# Try the package-style import first; fall back to sibling-module import.
try:
    from scripts._common import (  # type: ignore[import-not-found]
        base_argparse_parser,
        build_titans_config,
        chunked_forward,
        create_model,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
        tokenize_chat,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in test-only sys.path layouts
    from _common import (  # type: ignore[no-redef]
        base_argparse_parser,
        build_titans_config,
        chunked_forward,
        create_model,
        init_accelerator_and_logging,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
        tokenize_chat,
    )

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from accelerate import Accelerator  # noqa: F401

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_WANDB = importlib.util.find_spec("wandb") is not None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SFT Config
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """All hyperparameters for SFT training.

    Architecture fields mirror TitansConfig. Training fields are tuned for
    fine-tuning (lower LR, smaller grad-acc, shorter sequences vs pretrain).
    """

    # --- Model architecture ---
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

    # TNT / hierarchical memory
    use_tnt: bool = False
    global_chunk_size: int = 2048
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    use_qk_projection: bool = True
    tnt_stage: int = 1
    finetune_local_chunk_sizes: list[int] | None = None

    # Attention residual
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global_memory: bool = True
    attnres_modulate_local_memory: bool = False

    # Adaptive window
    adaptive_window: bool = False
    adaptive_window_min: int = 64
    adaptive_window_max: int | None = None
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01

    # MCA (multi-context attention)
    use_mca: bool = False
    mca_insertion_layers: list[int] | None = None
    mca_num_heads: int = 8
    mca_gate_type: str = "scalar"
    mca_gate_bias_init: float = -3.0

    # Misc architecture
    dropout: float = 0.0
    use_conv: bool = False

    # --- Data ---
    dataset: str | None = None
    dataset_subset: str | None = None
    eval_dataset: str | None = None
    eval_dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    messages_field: str = "messages"
    seq_len: int = 2048
    train_on_all: bool = False

    # --- Training ---
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    mixed_precision: str = "no"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/sft"
    save_every: int = 1000
    save_format: str = "pt"
    eval_every: int = 200
    eval_batches: int = 50
    resume: str | None = None
    init_weights: str | None = None

    # --- Logging ---
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-sft"
    wandb_run_name: str | None = None

    # --- Memory state lifecycle ---
    reset_memory_per_batch: bool = True
    state_carry_warmup_steps: int = 0

    # --- Misc ---
    seed: int = 42
    synthetic_samples: int = 5000


# ---------------------------------------------------------------------------
# Streaming SFT Dataset
# ---------------------------------------------------------------------------


class SFTStreamingDataset(IterableDataset):
    """Stream chat examples from a HuggingFace dataset and tokenize on the fly.

    Each example must have a field (``messages_field``) containing a list of
    message dicts. Examples that fail to produce any assistant-turn tokens are
    skipped.

    Args:
        dataset_name: HuggingFace dataset repo id.
        subset: Optional dataset configuration name.
        tokenizer: HuggingFace tokenizer instance.
        max_len: Maximum sequence length (examples are truncated, not packed).
        messages_field: Name of the field holding the messages list.
        train_on_all: If True, compute loss on all tokens.
        seed: Shuffle buffer seed.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        tokenizer: "PreTrainedTokenizerBase",
        max_len: int,
        messages_field: str = "messages",
        train_on_all: bool = False,
        seed: int = 42,
    ) -> None:
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )
        from datasets import load_dataset

        self.ds = load_dataset(
            dataset_name,
            subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).shuffle(seed=seed, buffer_size=10_000)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.messages_field = messages_field
        self.train_on_all = train_on_all

    def __iter__(self):
        for example in self.ds:
            messages = example.get(self.messages_field)
            if not messages or not isinstance(messages, list):
                continue

            try:
                tokenized = tokenize_chat(
                    messages,
                    self.tokenizer,
                    self.max_len,
                    train_on_all=self.train_on_all,
                )
            except Exception as exc:
                logger.debug(f"Skipping example due to tokenization error: {exc}")
                continue

            # Skip examples where no tokens contribute to the loss
            if not any(tokenized["loss_mask"]):
                continue

            # Skip degenerate examples (< 2 tokens after shifting)
            if len(tokenized["input_ids"]) < 2:
                continue

            yield {
                "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "labels": torch.tensor(tokenized["labels"], dtype=torch.long),
                "loss_mask": torch.tensor(tokenized["loss_mask"], dtype=torch.float),
            }


class SyntheticSFTDataset(Dataset):
    """Synthetic chat dataset for smoke-testing without a real corpus.

    Generates random token sequences with random assistant-span masks.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_ids = rng.integers(0, vocab_size, (num_samples, seq_len))
        # Random assistant span: second half of each sequence
        self.loss_mask = np.zeros((num_samples, seq_len), dtype=np.float32)
        half = seq_len // 2
        self.loss_mask[:, half:] = 1.0

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.input_ids[idx]
        return {
            "input_ids": torch.from_numpy(x[:-1].copy()).long(),
            "labels": torch.from_numpy(x[1:].copy()).long(),
            "loss_mask": torch.from_numpy(self.loss_mask[idx, 1:].copy()),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator: "Accelerator",
    vocab_size: int,
    max_batches: int = 50,
) -> float:
    """Compute masked validation loss over a subset of the eval dataloader.

    Args:
        model: The (possibly wrapped) model in eval mode.
        dataloader: DataLoader over an eval dataset.
        accelerator: Accelerate instance.
        vocab_size: Used to reshape logits.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Mean masked cross-entropy loss as a Python float.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    memory_states = None
    chunk_size = accelerator.unwrap_model(model).config.chunk_size

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            lbl_chunks = batch["labels"].split(chunk_size, dim=1)
            msk_chunks = batch["loss_mask"].split(chunk_size, dim=1)

            batch_loss_num = torch.tensor(0.0, device=batch["input_ids"].device)
            batch_tokens_num = torch.tensor(
                0.0, device=batch["input_ids"].device
            )

            chunk_iter = chunked_forward(
                model,
                batch["input_ids"],
                chunk_size,
                states=memory_states,
                detach_between=True,
            )
            for (logits, _ids_c, new_states), lbl_c, msk_c in zip(
                chunk_iter, lbl_chunks, msk_chunks
            ):
                memory_states = new_states
                logits_flat = logits.reshape(-1, vocab_size)
                labels_flat = lbl_c.reshape(-1)
                mask_flat = msk_c.reshape(-1).float()

                per_token = F.cross_entropy(
                    logits_flat, labels_flat, reduction="none"
                )
                batch_loss_num = batch_loss_num + (per_token * mask_flat).sum()
                batch_tokens_num = batch_tokens_num + mask_flat.sum()

            # Gather across processes
            batch_loss = (
                accelerator.gather(batch_loss_num.unsqueeze(0)).sum().item()
            )
            batch_tokens = (
                accelerator.gather(batch_tokens_num.unsqueeze(0)).sum().item()
            )

            total_loss += batch_loss
            total_tokens += batch_tokens

    model.train()
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: SFTConfig) -> None:
    """Run the full SFT training loop.

    Args:
        config: Populated SFTConfig instance.

    Raises:
        ImportError: If accelerate is not installed.
    """
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required. Install with: pip install accelerate"
        )

    bundle = init_accelerator_and_logging(config)
    accelerator = bundle.accelerator

    if accelerator.is_main_process:
        logger.info(f"SFT config: {config}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(
            f"Gradient accumulation steps: {config.gradient_accumulation_steps}"
        )

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = None
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or 0
            if accelerator.is_main_process:
                logger.info(f"Loaded tokenizer: {config.tokenizer}")
                logger.info(f"Vocab size (tokenizer): {tokenizer.vocab_size}")
        except Exception as exc:
            logger.warning(f"Could not load tokenizer '{config.tokenizer}': {exc}")
            tokenizer = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    titans_config = build_titans_config(config)
    model = create_model(config.model_type, titans_config)
    num_params = sum(p.numel() for p in model.parameters())

    if accelerator.is_main_process:
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Model parameters: {num_params:,}")

    # Load pretrained weights (pretrain -> SFT transfer)
    if config.init_weights is not None:
        init_path = Path(config.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-weights path not found: {init_path}")

        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
        # Support both raw state_dict and wrapped {"model": ..., ...} format
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained weights from {init_path}")
            if missing:
                logger.warning(f"Missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                logger.warning(
                    f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ..."
                )

    # ------------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------------
    if config.dataset is not None and HAS_DATASETS and tokenizer is not None:
        train_dataset: Dataset = SFTStreamingDataset(
            dataset_name=config.dataset,
            subset=config.dataset_subset,
            tokenizer=tokenizer,
            max_len=config.seq_len,
            messages_field=config.messages_field,
            train_on_all=config.train_on_all,
            seed=config.seed,
        )
        if accelerator.is_main_process:
            logger.info(f"Streaming SFT dataset: {config.dataset}")
    else:
        if accelerator.is_main_process:
            if config.dataset is not None:
                logger.warning(
                    "Could not load dataset (missing dependencies or tokenizer). "
                    "Falling back to synthetic data."
                )
            else:
                logger.info("No dataset specified — using synthetic data for demo.")
        train_dataset = SyntheticSFTDataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            num_samples=config.synthetic_samples,
            seed=config.seed,
        )

    is_streaming = isinstance(train_dataset, IterableDataset)
    train_dataloader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=int(os.environ.get("NUM_WORKERS", "4")),
        device_type=accelerator.device.type,
        shuffle=not is_streaming,
        streaming=is_streaming,
        drop_last=True,
    )

    # Optional eval dataloader
    eval_dataloader = None
    eval_dataset_name = config.eval_dataset or config.dataset
    if eval_dataset_name is not None and HAS_DATASETS and tokenizer is not None:
        try:
            from datasets import load_dataset as _load_dataset

            raw_eval = _load_dataset(
                eval_dataset_name,
                config.eval_dataset_subset or config.dataset_subset,
                split="test",
                streaming=True,
                trust_remote_code=True,
            )
            eval_hf = SFTStreamingDataset(
                dataset_name=eval_dataset_name,
                subset=config.eval_dataset_subset or config.dataset_subset,
                tokenizer=tokenizer,
                max_len=config.seq_len,
                messages_field=config.messages_field,
                train_on_all=config.train_on_all,
                seed=config.seed + 1,
            )
            # Override the loaded dataset with the test split
            eval_hf.ds = raw_eval
            eval_dataloader = make_dataloader(
                eval_hf,
                batch_size=config.batch_size,
                num_workers=int(os.environ.get("NUM_WORKERS", "4")),
                device_type=accelerator.device.type,
                shuffle=False,
                streaming=True,
                drop_last=False,
            )
            if accelerator.is_main_process:
                logger.info("Eval dataloader ready (test split).")
        except Exception as exc:
            if accelerator.is_main_process:
                logger.warning(f"Could not build eval dataloader: {exc}")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = make_optimizer(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        device_type=accelerator.device.type,
    )

    if config.max_steps > 0:
        total_steps = config.max_steps
    elif not is_streaming:
        total_steps = (len(train_dataloader) // config.gradient_accumulation_steps) * config.epochs
    else:
        # Streaming dataset — set a large number; training stops at max_steps or
        # when the dataset is exhausted.
        total_steps = 100_000

    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        """Linear warmup then cosine decay."""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    if eval_dataloader is not None:
        model, optimizer, train_dataloader, eval_dataloader, scheduler = (
            accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
        )
    else:
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )

    # Opt-in torch.compile (COMPILE=1). No-op on CPU or when use_attn_res.
    model = maybe_compile(
        model,
        enabled=bool(int(os.environ.get("COMPILE", "0"))),
        device_type=accelerator.device.type,
        use_attn_res=getattr(config, "use_attn_res", False),
    )

    # ------------------------------------------------------------------
    # WandB / tracker init
    # ------------------------------------------------------------------
    if config.wandb and HAS_WANDB and accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb_project,
            config=vars(config),
            init_kwargs={"wandb": {"name": config.wandb_run_name}},
        )

    # ------------------------------------------------------------------
    # Checkpoint directory + optional resume-path validation
    # ------------------------------------------------------------------
    ckpt_setup = setup_checkpoint_dir(config.checkpoint_dir, config.resume)
    checkpoint_dir = ckpt_setup.output_dir

    # ------------------------------------------------------------------
    # Resume from SFT checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    memory_states = None

    if ckpt_setup.resume_path is not None:
        resume_path = ckpt_setup.resume_path
        checkpoint = load_checkpoint(resume_path, weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if accelerator.is_main_process:
            logger.info(
                f"Resumed from {resume_path} at step {global_step}, epoch {start_epoch}"
            )

        # Load memory states if available
        mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
        if not mem_path.exists():
            mem_path = resume_path.parent / "memory_final.npz"
        try:
            from titans.memory_dump import load_memory_states

            memory_states = load_memory_states(
                mem_path,
                device=accelerator.device,
                reset_for_inference=False,
            )
            if accelerator.is_main_process:
                logger.info(f"Loaded memory states from {mem_path}")
        except Exception as e:
            if accelerator.is_main_process:
                logger.info(f"No memory states found ({e}), starting fresh")

        del checkpoint

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()

    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        num_optimizer_steps = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            # Reset memory at batch boundary per lifecycle policy.
            reset_this_batch = (
                config.reset_memory_per_batch
                or global_step < config.state_carry_warmup_steps
            )
            if reset_this_batch:
                memory_states = None

            with accelerator.accumulate(model):
                chunk_size = config.chunk_size
                lbl_chunks = batch["labels"].split(chunk_size, dim=1)
                msk_chunks = batch["loss_mask"].split(chunk_size, dim=1)
                num_chunks = len(lbl_chunks)

                # Aggregate numerator/denominator across chunks so the final
                # reported loss equals the masked-token-weighted mean CE
                # over the full sequence (matches a single-shot reference
                # when seq_len <= chunk_size).
                total_loss_num = torch.tensor(0.0, device=batch["input_ids"].device)
                total_tokens = torch.tensor(0.0, device=batch["input_ids"].device)
                loss_accum_for_backward = torch.tensor(
                    0.0, device=batch["input_ids"].device
                )

                chunk_iter = chunked_forward(
                    model,
                    batch["input_ids"],
                    chunk_size,
                    states=memory_states,
                    detach_between=True,
                )
                for (logits, _ids_c, new_states), lbl_c, msk_c in zip(
                    chunk_iter, lbl_chunks, msk_chunks
                ):
                    # Truncated BPTT: helper already detached new_states.
                    memory_states = new_states

                    logits_flat = logits.reshape(-1, config.vocab_size)
                    labels_flat = lbl_c.reshape(-1)
                    mask_flat = msk_c.reshape(-1).float()

                    per_token = F.cross_entropy(
                        logits_flat, labels_flat, reduction="none"
                    )
                    chunk_loss_num = (per_token * mask_flat).sum()
                    chunk_tokens = mask_flat.sum()

                    total_loss_num = total_loss_num + chunk_loss_num.detach()
                    total_tokens = total_tokens + chunk_tokens.detach()

                    # Per-chunk backward would be more memory-efficient but
                    # changes accumulation semantics; keep full-graph backward
                    # here to match pretrain.py's loss-averaging behavior.
                    chunk_loss = chunk_loss_num / chunk_tokens.clamp(min=1.0)
                    loss_accum_for_backward = loss_accum_for_backward + chunk_loss

                loss = total_loss_num / total_tokens.clamp(min=1.0)
                backward_loss = loss_accum_for_backward / max(num_chunks, 1)

                accelerator.backward(backward_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                    num_optimizer_steps += 1
                    global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Detach memory states to prevent BPTT across batch boundaries
            if memory_states is not None:
                memory_states = [
                    s.detach() if s is not None else None
                    for s in memory_states
                ]

            loss_val = loss.item()
            masked_tokens = total_tokens.item()
            epoch_loss += loss_val * masked_tokens
            epoch_tokens += masked_tokens

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % config.log_every == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / max(epoch_tokens, 1.0)
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    avg=f"{avg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    step=global_step,
                )

                if config.wandb and HAS_WANDB:
                    accelerator.log(
                        {
                            "train/loss": loss_val,
                            "train/avg_loss": avg_loss,
                            "train/lr": lr,
                            "train/masked_tokens": masked_tokens,
                        },
                        step=global_step,
                    )

            # ----------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------
            if (
                global_step > 0
                and global_step % config.eval_every == 0
                and eval_dataloader is not None
            ):
                val_loss = evaluate(
                    model,
                    eval_dataloader,
                    accelerator,
                    config.vocab_size,
                    max_batches=config.eval_batches,
                )
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} — val loss: {val_loss:.4f}")
                    if config.wandb and HAS_WANDB:
                        accelerator.log({"eval/loss": val_loss}, step=global_step)

            # ----------------------------------------------------------
            # Checkpointing
            # ----------------------------------------------------------
            if global_step > 0 and global_step % config.save_every == 0:
                if accelerator.is_main_process:
                    ckpt_stem = checkpoint_dir / f"step_{global_step}"
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint(
                        unwrapped.state_dict(),
                        ckpt_stem,
                        format=config.save_format,
                        metadata={
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "config": vars(config),
                            "titans_config": titans_config.__dict__
                            if hasattr(titans_config, "__dict__")
                            else {},
                            "step": global_step,
                            "epoch": epoch,
                        },
                    )
                    logger.info(f"Saved checkpoint: step {global_step}")
                    should_save_memory = (
                        not config.reset_memory_per_batch
                        and memory_states is not None
                        and any(s is not None for s in memory_states)
                    )
                    if should_save_memory:
                        mem_path = checkpoint_dir / f"memory_step_{global_step}.npz"
                        save_memory_states(memory_states, mem_path)
                        logger.info(f"Saved memory states: step {global_step}")

        # End-of-epoch summary
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1.0)
            logger.info(
                f"Epoch {epoch + 1} complete — avg loss: {avg_epoch_loss:.4f}, "
                f"optimizer steps: {num_optimizer_steps}"
            )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        final_stem = checkpoint_dir / "final"
        unwrapped = accelerator.unwrap_model(model)
        paths = save_checkpoint(
            unwrapped.state_dict(),
            final_stem,
            format=config.save_format,
            metadata={
                "config": vars(config),
                "titans_config": titans_config.__dict__
                if hasattr(titans_config, "__dict__")
                else {},
                "step": global_step,
            },
        )
        logger.info(f"SFT training complete. Final checkpoint: {paths[0]}")
        should_save_memory = (
            not config.reset_memory_per_batch
            and memory_states is not None
            and any(s is not None for s in memory_states)
        )
        if should_save_memory:
            mem_path = checkpoint_dir / "memory_final.npz"
            save_memory_states(memory_states, mem_path)

    if config.wandb and HAS_WANDB:
        accelerator.end_training()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> SFTConfig:
    """Parse command-line arguments and return an SFTConfig."""
    parser = base_argparse_parser(
        description="Supervised Fine-Tuning (SFT) for Titans PyTorch models",
    )
    # Override base defaults to match SFT-specific values.
    parser.set_defaults(
        lr=2e-5,
        checkpoint_dir="checkpoints/sft",
        wandb_project="titans-sft",
    )

    # SFT-only data flags
    data = parser.add_argument_group("SFT data")
    data.add_argument(
        "--dataset", type=str, default=None, help="HuggingFace dataset repo id",
    )
    data.add_argument("--dataset-subset", type=str, default=None)
    data.add_argument("--eval-dataset", type=str, default=None)
    data.add_argument("--eval-dataset-subset", type=str, default=None)
    data.add_argument("--tokenizer", type=str, default="gpt2")
    data.add_argument(
        "--messages-field",
        type=str,
        default="messages",
        help="Name of the dataset field containing the messages list",
    )
    data.add_argument("--seq-len", type=int, default=2048)
    data.add_argument(
        "--train-on-all",
        action="store_true",
        help="Compute loss on all tokens, not just assistant turns",
    )

    # SFT-only eval flags
    ev = parser.add_argument_group("SFT eval")
    ev.add_argument("--eval-every", type=int, default=200)
    ev.add_argument("--eval-batches", type=int, default=50)

    # Memory state lifecycle (SFT-specific)
    mem = parser.add_argument_group("Memory lifecycle")
    mem.add_argument(
        "--reset-memory-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If True (default), reset Titans memory states to None at the "
            "start of each batch. Set --no-reset-memory-per-batch to carry "
            "detached state across batches (streaming / long-context regime)."
        ),
    )
    mem.add_argument(
        "--state-carry-warmup-steps",
        type=int,
        default=0,
        help=(
            "When --no-reset-memory-per-batch is set, still reset memory for "
            "the first N steps (warmup before carrying state)."
        ),
    )

    # Misc SFT-only
    parser.add_argument("--synthetic-samples", type=int, default=5000)

    args = parser.parse_args()

    return SFTConfig(
        # Architecture
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
        huber_delta_init=args.huber_delta_init,
        dropout=args.dropout,
        use_conv=args.use_conv,
        # TNT
        use_tnt=args.use_tnt,
        global_chunk_size=args.global_chunk_size,
        local_chunk_sizes=args.local_chunk_sizes,
        local_shard_length=args.local_shard_length,
        use_qk_projection=args.use_qk_projection,
        tnt_stage=args.tnt_stage,
        finetune_local_chunk_sizes=args.finetune_local_chunk_sizes,
        # Attn res
        use_attn_res=args.use_attn_res,
        num_attnres_blocks=args.num_attnres_blocks,
        attnres_warmup_steps=args.attnres_warmup_steps,
        attnres_modulate_global_memory=args.attnres_modulate_global_memory,
        attnres_modulate_local_memory=args.attnres_modulate_local_memory,
        # Adaptive window
        adaptive_window=args.adaptive_window,
        adaptive_window_min=args.adaptive_window_min,
        adaptive_window_max=args.adaptive_window_max,
        adaptive_window_temperature=args.adaptive_window_temperature,
        adaptive_window_lambda=args.adaptive_window_lambda,
        # MCA
        use_mca=args.use_mca,
        mca_insertion_layers=args.mca_insertion_layers,
        mca_num_heads=args.mca_num_heads,
        mca_gate_type=args.mca_gate_type,
        mca_gate_bias_init=args.mca_gate_bias_init,
        # Data
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        eval_dataset=args.eval_dataset,
        eval_dataset_subset=args.eval_dataset_subset,
        tokenizer=args.tokenizer,
        messages_field=args.messages_field,
        seq_len=args.seq_len,
        train_on_all=args.train_on_all,
        # Training
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=args.mixed_precision,
        # Checkpointing
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        save_format=args.save_format,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        resume=args.resume,
        init_weights=args.init_weights,
        # Logging
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        # Memory lifecycle
        reset_memory_per_batch=args.reset_memory_per_batch,
        state_carry_warmup_steps=args.state_carry_warmup_steps,
        # Misc
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = parse_args()
    train(config)
