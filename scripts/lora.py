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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states

# scripts/ is imported both as a namespace package ("scripts._common") and as
# a flat directory (when tests add scripts/ onto sys.path and import "lora").
try:
    from scripts._common import (  # type: ignore[import-not-found]
        base_argparse_parser,
        build_titans_config,
        chunked_forward,
        create_model,
        init_accelerator_and_logging,
        loss_mask_to_zero_one,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
    )
    from scripts._common import tokenize_chat as _tokenize_chat_canonical
except ModuleNotFoundError:  # pragma: no cover - exercised in test-only sys.path layouts
    from _common import (  # type: ignore[no-redef]
        base_argparse_parser,
        build_titans_config,
        chunked_forward,
        create_model,
        init_accelerator_and_logging,
        loss_mask_to_zero_one,
        make_dataloader,
        make_optimizer,
        maybe_compile,
        setup_checkpoint_dir,
    )
    from _common import tokenize_chat as _tokenize_chat_canonical  # type: ignore[no-redef]
from titans.lora import (
    count_lora_parameters,
    merge_lora_weights,
    save_adapters,
    wrap_lora_layers,
)

# ---------------------------------------------------------------------------
# Optional imports
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
    PreTrainedTokenizerBase = Any  # type: ignore[misc,assignment]

HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_WANDB = importlib.util.find_spec("wandb") is not None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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

    # TNT (Plan 3 additions — unblock --use-tnt flag)
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

    # MCA
    use_mca: bool = False
    mca_insertion_layers: list[int] | None = None
    mca_num_heads: int = 8
    mca_gate_type: str = "scalar"
    mca_gate_bias_init: float = -3.0

    # Extras expected by build_titans_config
    dropout: float = 0.0
    use_conv: bool = False

    # Data
    data_path: str | None = None
    eval_data_path: str | None = None
    dataset: str | None = None
    dataset_subset: str | None = None
    eval_dataset: str | None = None
    eval_dataset_subset: str | None = None
    messages_field: str = "messages"
    train_on_all: bool = False
    eval_split: str = "test"
    tokenizer: str = "gpt2"
    seq_len: int = 2048
    max_seq_len: int = 2048

    # Memory state lifecycle
    reset_memory_per_batch: bool = True
    state_carry_warmup_steps: int = 0

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
    eval_batches: int = 50
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
# Backward-compatible wrappers around consolidated helpers
# ---------------------------------------------------------------------------


def build_model(config: LoRATrainingConfig) -> torch.nn.Module:
    """Build a Titans model from a LoRATrainingConfig.

    Thin wrapper over :func:`scripts._common.build_titans_config` and
    :func:`scripts._common.create_model`. Preserved for backward-compat
    with tests/callers that imported ``lora.build_model`` before the
    _common migration.

    Args:
        config: LoRA training configuration.

    Returns:
        An instantiated (but not yet LoRA-wrapped) Titans model.
    """
    titans_config = build_titans_config(config)
    return create_model(config.model_type, titans_config)


def tokenize_chat(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> dict[str, list[int]]:
    """Thin wrapper around scripts._common.tokenize_chat.

    Differs from the canonical helper by returning
    ``{"input_ids", "labels"}`` in the historical lora format (labels
    uses -100 sentinels instead of loss_mask). Assistant content tokens
    are kept; everything else is -100.

    Args:
        messages: List of role/content dicts.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum token length (sequences are truncated).

    Returns:
        Dict with keys "input_ids" and "labels" (both lists of ints).
    """
    out = _tokenize_chat_canonical(messages, tokenizer, max_seq_len)
    # Rewrite labels to use -100 sentinels in positions where loss_mask=0.
    ids = out["input_ids"]
    labels = [
        tok if m == 1 else -100 for tok, m in zip(out["labels"], out["loss_mask"])
    ]
    return {"input_ids": ids, "labels": labels}


def build_loss_mask(labels: list[int]) -> list[int]:
    """Backward-compatible: accept a labels list with -100 sentinels.

    Args:
        labels: Token labels with -100 for masked positions.

    Returns:
        List of 0/1 integers where 1 = compute loss.
    """
    return loss_mask_to_zero_one(labels)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class JSONLChatDataset(IterableDataset):  # type: ignore[type-arg]
    """Streaming dataset for LoRA training from a local JSONL chat file.

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


class HFChatStreamingDataset(IterableDataset):  # type: ignore[type-arg]
    """Stream chat examples from a HuggingFace dataset repo and tokenize live.

    Mirrors :class:`scripts.sft.SFTStreamingDataset` so the LoRA script can
    consume HF repos (e.g. ``HuggingFaceH4/ultrachat_200k``) in addition to
    local JSONL files. Each example must have a ``messages_field`` containing
    a list of role/content dicts in ChatML format. Examples that produce no
    supervised (assistant-turn) tokens are skipped.

    Args:
        dataset_name: HuggingFace dataset repo id.
        subset: Optional dataset configuration name.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_len: Maximum sequence length (examples are truncated).
        messages_field: Name of the field holding the messages list.
        train_on_all: If ``True`` compute loss on all tokens (not just
            assistant turns).
        split: Dataset split to stream (``"train"``, ``"test"``, ...).
        seed: Shuffle buffer seed.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        messages_field: str = "messages",
        train_on_all: bool = False,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library is required. "
                "Install with: pip install datasets"
            )
        from datasets import load_dataset

        ds = load_dataset(
            dataset_name,
            subset,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        # shuffle() is no-op for test splits that are already deterministic but
        # safe to call; buffer_size is small to avoid memory pressure on eval.
        self.ds = ds.shuffle(seed=seed, buffer_size=10_000)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.messages_field = messages_field
        self.train_on_all = train_on_all

    def __iter__(self):  # type: ignore[override]
        for example in self.ds:
            messages = example.get(self.messages_field)
            if not messages or not isinstance(messages, list):
                continue
            try:
                tokenized = tokenize_chat(
                    messages, self.tokenizer, self.max_seq_len
                )
            except Exception as exc:
                logger.debug(
                    f"Skipping HF example due to tokenization error: {exc}"
                )
                continue

            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            loss_mask = (
                [1] * len(labels)
                if self.train_on_all
                else build_loss_mask(labels)
            )

            if sum(loss_mask) == 0 or len(input_ids) < 2:
                continue

            labels_clean = [max(tok, 0) for tok in labels]

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels_clean, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            }


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
# Dataset construction
# ---------------------------------------------------------------------------


def _load_tokenizer(config: LoRATrainingConfig) -> PreTrainedTokenizerBase:
    """Load a HF tokenizer once, raising a clear error if unavailable."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for chat datasets. "
            "Install with: pip install transformers"
        )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(
    config: LoRATrainingConfig,
) -> Dataset | IterableDataset:  # type: ignore[type-arg]
    """Build a training dataset from the config.

    Routing precedence:
        1. ``data_path`` — local JSONL file (:class:`JSONLChatDataset`).
        2. ``dataset``   — HuggingFace repo id, streamed
           (:class:`HFChatStreamingDataset`).
        3. fallback      — :class:`SyntheticChatDataset` for demo/smoke runs.

    Args:
        config: Training configuration.

    Returns:
        A Dataset or IterableDataset.
    """
    if config.data_path is not None:
        tokenizer = _load_tokenizer(config)
        return JSONLChatDataset(
            Path(config.data_path), tokenizer, config.max_seq_len
        )

    if config.dataset is not None:
        tokenizer = _load_tokenizer(config)
        logger.info(
            f"Streaming HF dataset '{config.dataset}' "
            f"(subset={config.dataset_subset}) for training."
        )
        return HFChatStreamingDataset(
            dataset_name=config.dataset,
            subset=config.dataset_subset,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            messages_field=config.messages_field,
            train_on_all=config.train_on_all,
            split="train",
            seed=config.seed,
        )

    logger.info("No dataset specified — using synthetic data for demo")
    return SyntheticChatDataset(
        config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
    )


def build_eval_dataset(
    config: LoRATrainingConfig,
) -> Dataset | IterableDataset | None:  # type: ignore[type-arg]
    """Build an optional eval dataset mirroring :func:`build_dataset`.

    Returns ``None`` when no eval data is configured so the caller skips
    evaluation cleanly. Routing precedence:

    1. ``eval_data_path`` — local JSONL file (:class:`JSONLChatDataset`).
    2. ``eval_dataset``   — explicit HF repo id for eval
       (:class:`HFChatStreamingDataset`, split=``eval_split``).
    3. ``dataset``        — reuse training HF repo with ``eval_split``. This
       matches :mod:`scripts.sft`'s convenience default: if the caller only
       sets ``--dataset``, we stream its test split for eval. If the split
       does not exist, we surface the error to the caller rather than
       silently disabling eval.
    4. Otherwise, returns ``None``.
    """
    if config.eval_data_path is not None:
        tokenizer = _load_tokenizer(config)
        return JSONLChatDataset(
            Path(config.eval_data_path), tokenizer, config.max_seq_len
        )

    eval_repo = config.eval_dataset or config.dataset
    eval_subset = config.eval_dataset_subset or config.dataset_subset
    if eval_repo is not None:
        tokenizer = _load_tokenizer(config)
        logger.info(
            f"Streaming HF eval dataset '{eval_repo}' "
            f"(subset={eval_subset}, split={config.eval_split})."
        )
        return HFChatStreamingDataset(
            dataset_name=eval_repo,
            subset=eval_subset,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            messages_field=config.messages_field,
            train_on_all=config.train_on_all,
            split=config.eval_split,
            seed=config.seed + 1,
        )

    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: "Accelerator",
    vocab_size: int,
    max_batches: int = 50,
) -> float:
    """Compute masked validation loss over a subset of the eval dataloader.

    Ported from :mod:`scripts.sft` so lora.py remains self-contained. Mirrors
    the training-side masked cross-entropy: per-token CE weighted by
    ``loss_mask``, summed then divided by the total supervised-token count
    gathered across processes.

    Args:
        model: The (possibly wrapped) model; caller's train/eval mode is
            preserved — we flip to eval here and restore train on exit.
        dataloader: DataLoader over an eval dataset.
        accelerator: Accelerate instance used for cross-process gather.
        vocab_size: Used to reshape logits.
        max_batches: Maximum number of batches to evaluate. Guards against
            long-running eval passes when the eval set is large.

    Returns:
        Mean masked cross-entropy loss as a Python float. Returns
        ``float('inf')`` if no supervised tokens were seen.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    memory_states = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            logits, memory_states, _ = model(
                batch["input_ids"], states=memory_states
            )
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = batch["labels"].view(-1)
            mask_flat = batch["loss_mask"].view(-1).float()

            per_token = F.cross_entropy(
                logits_flat, labels_flat, reduction="none"
            )
            batch_loss = (per_token * mask_flat).sum()
            batch_tokens = mask_flat.sum()

            batch_loss = accelerator.gather(
                batch_loss.unsqueeze(0)
            ).sum().item()
            batch_tokens = accelerator.gather(
                batch_tokens.unsqueeze(0)
            ).sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens

            if memory_states is not None:
                memory_states = [
                    s.detach() if s is not None else None for s in memory_states
                ]

    model.train()
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


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

    bundle = init_accelerator_and_logging(config)
    accelerator = bundle.accelerator

    if accelerator.is_main_process:
        logger.info(f"LoRA training config: {config}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")

    torch.manual_seed(config.seed)

    # --- 1. Build base model ---
    titans_config = build_titans_config(config)
    model = create_model(config.model_type, titans_config)

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
    optimizer = make_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        device_type=accelerator.device.type,
    )

    # --- 5. Dataset and dataloader ---
    dataset = build_dataset(config)
    use_collate = isinstance(dataset, (JSONLChatDataset, HFChatStreamingDataset))
    is_streaming = isinstance(dataset, IterableDataset)
    dataloader = make_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=int(os.environ.get("NUM_WORKERS", "4")),
        device_type=accelerator.device.type,
        shuffle=not is_streaming,
        streaming=is_streaming,
        drop_last=True,
        collate_fn=sft_collate_fn if use_collate else None,
    )

    # Optional eval dataloader — None when no eval data is configured so the
    # periodic-eval branch below simply skips without crashing.
    eval_dataset = build_eval_dataset(config)
    eval_dataloader: DataLoader | None = None
    if eval_dataset is not None:
        eval_use_collate = isinstance(
            eval_dataset, (JSONLChatDataset, HFChatStreamingDataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=sft_collate_fn if eval_use_collate else None,
        )
        if accelerator.is_main_process:
            logger.info(
                f"Eval dataloader ready "
                f"(eval_every={config.eval_every}, "
                f"eval_batches={config.eval_batches})."
            )
    elif accelerator.is_main_process:
        logger.info(
            "No --eval-data-path configured — skipping periodic evaluation."
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
    if eval_dataloader is not None:
        model, optimizer, dataloader, eval_dataloader, scheduler = (
            accelerator.prepare(
                model, optimizer, dataloader, eval_dataloader, scheduler
            )
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

    # Opt-in torch.compile (COMPILE=1). No-op on CPU or when use_attn_res.
    model = maybe_compile(
        model,
        enabled=bool(int(os.environ.get("COMPILE", "0"))),
        device_type=accelerator.device.type,
        use_attn_res=getattr(config, "use_attn_res", False),
    )

    if config.wandb and HAS_WANDB and accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb_project,
            config=vars(config),
            init_kwargs={"wandb": {"name": config.wandb_run_name}},
        )

    ckpt_setup = setup_checkpoint_dir(config.checkpoint_dir, config.resume)
    checkpoint_dir = ckpt_setup.output_dir

    global_step = 0
    start_epoch = 0
    memory_states = None
    vocab_size = config.vocab_size

    # ------------------------------------------------------------------
    # Resume from LoRA checkpoint
    # ------------------------------------------------------------------
    if ckpt_setup.resume_path is not None:
        resume_path = ckpt_setup.resume_path
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

                total_loss_num = torch.tensor(
                    0.0, device=batch["input_ids"].device
                )
                total_tokens = torch.tensor(
                    0.0, device=batch["input_ids"].device
                )
                backward_accum = torch.tensor(
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

                    logits_flat = logits.reshape(-1, vocab_size)
                    labels_flat = lbl_c.reshape(-1)
                    mask_flat = msk_c.reshape(-1).float()

                    per_token = F.cross_entropy(
                        logits_flat, labels_flat, reduction="none"
                    )
                    chunk_num = (per_token * mask_flat).sum()
                    chunk_tok = mask_flat.sum()

                    total_loss_num = total_loss_num + chunk_num.detach()
                    total_tokens = total_tokens + chunk_tok.detach()
                    backward_accum = (
                        backward_accum + chunk_num / chunk_tok.clamp(min=1.0)
                    )

                loss = total_loss_num / total_tokens.clamp(min=1.0)
                backward_loss = backward_accum / max(num_chunks, 1)
                accelerator.backward(backward_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                    global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

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

            # ----------------------------------------------------------
            # Periodic evaluation
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
                    vocab_size,
                    max_batches=config.eval_batches,
                )
                # Detach memory states after eval — evaluate() restores
                # train mode but the training-loop states above are still
                # what we propagate forward.
                if accelerator.is_main_process:
                    logger.info(
                        f"Step {global_step} — val loss: {val_loss:.4f}"
                    )
                    if config.wandb and HAS_WANDB:
                        accelerator.log(
                            {"eval/loss": val_loss}, step=global_step
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

                # Full checkpoint for resume support
                ckpt_stem = checkpoint_dir / f"step_{global_step}"
                save_checkpoint(
                    unwrapped.state_dict(),
                    ckpt_stem,
                    format=config.save_format,
                    metadata={
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": vars(config),
                        "step": global_step,
                        "epoch": epoch,
                    },
                )
                logger.info(
                    f"Checkpoint: saved full model at step {global_step}"
                )

                # Memory states
                if (
                    not config.reset_memory_per_batch
                    and memory_states is not None
                    and any(s is not None for s in memory_states)
                ):
                    mem_path = checkpoint_dir / f"memory_step_{global_step}.npz"
                    save_memory_states(memory_states, mem_path)
                    logger.info(f"Saved memory states: step {global_step}")

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

        # Memory states
        if (
            not config.reset_memory_per_batch
            and memory_states is not None
            and any(s is not None for s in memory_states)
        ):
            mem_path = checkpoint_dir / "memory_final.npz"
            save_memory_states(memory_states, mem_path)

    if config.wandb and HAS_WANDB:
        accelerator.end_training()

    logger.info("LoRA training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> LoRATrainingConfig:
    """Parse command-line arguments and return a LoRATrainingConfig."""
    parser = base_argparse_parser(
        description="LoRA fine-tuning for Titans PyTorch models",
    )
    parser.set_defaults(
        lr=1e-4,
        weight_decay=0.01,
        checkpoint_dir="checkpoints/lora",
        save_every=5000,
        wandb_project="titans-lora",
    )

    # LoRA-specific flags
    lora_g = parser.add_argument_group("LoRA")
    lora_g.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank r")
    lora_g.add_argument("--lora-alpha", type=float, default=16.0,
                       help="LoRA alpha (scale = alpha / rank)")
    lora_g.add_argument("--lora-dropout", type=float, default=0.05,
                       help="Dropout on LoRA input path")
    lora_g.add_argument(
        "--lora-targets", type=str, default="attn",
        help="Comma-separated target groups: attn, ffn, memory, all",
    )
    lora_g.add_argument(
        "--merge-and-save", type=str, default=None, metavar="PATH",
        help="After training, merge LoRA into base weights and save to PATH",
    )

    # Data flags (LoRA-specific: JSONL data_path plus tokenizer/seq-len)
    data_g = parser.add_argument_group("LoRA data")
    data_g.add_argument(
        "--data-path", type=str, default=None,
        help="Path to JSONL file with 'messages' field per line",
    )
    data_g.add_argument(
        "--eval-data-path", type=str, default=None,
        help="Optional JSONL eval file (same schema as --data-path). "
             "If unset, periodic eval is skipped.",
    )
    data_g.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset repo id to stream for training.",
    )
    data_g.add_argument("--dataset-subset", type=str, default=None)
    data_g.add_argument(
        "--eval-dataset", type=str, default=None,
        help="Optional HF dataset repo for eval. Defaults to --dataset when "
             "--eval-data-path is unset and --dataset is set.",
    )
    data_g.add_argument("--eval-dataset-subset", type=str, default=None)
    data_g.add_argument(
        "--eval-split", type=str, default="test",
        help="Split to stream from the eval HF dataset (default: test).",
    )
    data_g.add_argument(
        "--messages-field", type=str, default="messages",
        help="Field name containing the messages list in HF examples.",
    )
    data_g.add_argument(
        "--train-on-all", action="store_true",
        help="Compute loss on all tokens instead of assistant-only.",
    )
    data_g.add_argument("--tokenizer", type=str, default="gpt2")
    data_g.add_argument("--seq-len", type=int, default=2048)
    data_g.add_argument("--max-seq-len", type=int, default=2048)

    # Memory lifecycle
    mem_group = parser.add_argument_group("Memory lifecycle")
    mem_group.add_argument(
        "--reset-memory-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    mem_group.add_argument(
        "--state-carry-warmup-steps",
        type=int,
        default=0,
    )

    # Misc
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--eval-batches", type=int, default=50,
        help="Max batches per eval pass (caps wall-time when eval set is large).",
    )

    args = parser.parse_args()

    return LoRATrainingConfig(
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
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        eval_dataset=args.eval_dataset,
        eval_dataset_subset=args.eval_dataset_subset,
        messages_field=args.messages_field,
        train_on_all=args.train_on_all,
        eval_split=args.eval_split,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
        # Memory lifecycle
        reset_memory_per_batch=args.reset_memory_per_batch,
        state_carry_warmup_steps=args.state_carry_warmup_steps,
        # Training
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
        # LoRA
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        merge_and_save=args.merge_and_save,
        # Checkpointing
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        save_format=args.save_format,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        resume=args.resume,
        # Logging
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        # Misc
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
