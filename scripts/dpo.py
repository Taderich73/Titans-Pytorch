#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Direct Preference Optimization (DPO) training script for Titans PyTorch models.

Implements both standard DPO (with a reference model via the LoRA-as-reference
trick) and SimPO (reference-free, length-normalized).  The LoRA-as-reference
trick avoids keeping a second copy of the model in memory: the same single model
serves as both policy (LoRA enabled) and reference (LoRA disabled).

Supported loss types:
- ``dpo``   — standard DPO with beta-scaled log-ratio objective
- ``simpo`` — SimPO (length-normalized, reference-free)

Supported model variants: mac, mag, mal, lmm
Supported datasets: any HuggingFace dataset with ``chosen`` and ``rejected``
fields (either message lists or plain strings).

Usage:
    # Quick smoke-test on synthetic preference data
    python scripts/dpo.py --model mac --dim 256 --epochs 2

    # DPO from a pretrained/SFT checkpoint with LoRA
    python scripts/dpo.py \\
        --model mac \\
        --init-weights checkpoints/sft/final.pt \\
        --dataset trl-lib/ultrafeedback_binarized \\
        --tokenizer meta-llama/Llama-2-7b-hf \\
        --dim 512 --num-layers 12 \\
        --loss-type dpo --beta 0.1 \\
        --lora-rank 8 --lora-targets attn \\
        --mixed-precision bf16

    # SimPO (no reference model needed)
    python scripts/dpo.py \\
        --model mac \\
        --init-weights checkpoints/sft/final.pt \\
        --loss-type simpo --beta 2.0 --gamma 0.5

    # Multi-GPU via accelerate
    accelerate launch scripts/dpo.py --model mac --dim 1024 --num-layers 24 \\
        --dataset myorg/my-preference-dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.lora import (
    count_lora_parameters,
    merge_lora_weights,
    save_adapters,
    set_lora_enabled,
    wrap_lora_layers,
)

# ---------------------------------------------------------------------------
# Optional dependency guards
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChatML constants
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mac": TitansMAC,
    "mag": TitansMAG,
    "mal": TitansMAL,
    "lmm": TitansLMM,
}

# ---------------------------------------------------------------------------
# Chat formatting helpers  (mirrors sft.py)
# ---------------------------------------------------------------------------


def format_chatml(messages: list[dict]) -> str:
    """Format a list of message dicts into a ChatML string.

    Args:
        messages: List of dicts with ``role`` and ``content`` keys.

    Returns:
        A single string with all turns formatted in ChatML markup.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def build_loss_mask(
    seq_len: int,
    assistant_content_spans: list[tuple[int, int]],
    include_eos: bool = True,
    eos_positions: list[int] | None = None,
    train_on_all: bool = False,
) -> list[int]:
    """Build a per-token binary loss mask.

    Args:
        seq_len: Total sequence length (after shifting for next-token prediction).
        assistant_content_spans: List of (start, end) token index pairs that
            correspond to assistant turn content in the *label* (shifted) sequence.
        include_eos: Whether to include EOS tokens that follow assistant turns.
        eos_positions: Token positions of EOS tokens following assistant turns.
        train_on_all: If True, return an all-ones mask regardless of spans.

    Returns:
        A list of ints (0 or 1) of length ``seq_len``.
    """
    if train_on_all:
        return [1] * seq_len

    mask = [0] * seq_len
    for start, end in assistant_content_spans:
        for i in range(start, min(end, seq_len)):
            mask[i] = 1

    if include_eos and eos_positions:
        for pos in eos_positions:
            if 0 <= pos < seq_len:
                mask[pos] = 1

    return mask


def tokenize_chat(
    messages: list[dict],
    tokenizer: "PreTrainedTokenizerBase",
    max_len: int,
    train_on_all: bool = False,
) -> dict[str, list[int]]:
    """Tokenize a conversation and produce per-token loss masks.

    Uses ``tokenizer.apply_chat_template`` when available; otherwise falls
    back to ChatML formatting.  Identifies assistant turns in order to mask
    non-assistant tokens from the loss.

    The output is already shifted for next-token prediction:
    - ``input_ids`` = tokens[:-1]
    - ``labels``    = tokens[1:]
    - ``loss_mask`` = mask[1:]

    Args:
        messages: List of message dicts with ``role`` / ``content`` keys.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum sequence length (sequences are truncated to this).
        train_on_all: If True, mask is all-ones (train on every token).

    Returns:
        Dict with keys ``input_ids``, ``labels``, and ``loss_mask``
        (all lists of ints, length ``<= max_len - 1``).
    """
    use_native_template = (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )

    if use_native_template:
        full_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        assistant_spans: list[tuple[int, int]] = []
        eos_after_assistant: list[int] = []

        if not train_on_all:
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue

                prefix_turns = messages[:i]
                if prefix_turns:
                    prefix_ids: list[int] = tokenizer.apply_chat_template(
                        prefix_turns,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                else:
                    prefix_ids = tokenizer.encode(
                        f"{IM_START}assistant\n", add_special_tokens=False
                    )

                content_start = len(prefix_ids)

                turns_through = messages[: i + 1]
                through_ids: list[int] = tokenizer.apply_chat_template(
                    turns_through,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                content_end = len(through_ids)

                if content_end < len(full_ids):
                    eos_after_assistant.append(content_end)

                assistant_spans.append((content_start, content_end))

    else:
        full_text = format_chatml(messages)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        assistant_spans = []
        eos_after_assistant = []

        if not train_on_all:
            cursor = 0
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                header = f"{IM_START}{role}\n"
                footer = f"{IM_END}\n"

                header_ids = tokenizer.encode(header, add_special_tokens=False)
                content_ids = tokenizer.encode(content, add_special_tokens=False)
                footer_ids = tokenizer.encode(footer, add_special_tokens=False)

                content_start = cursor + len(header_ids)
                content_end = content_start + len(content_ids)
                footer_end = content_end + len(footer_ids)

                if role == "assistant":
                    assistant_spans.append((content_start, content_end))
                    if footer_ids and content_end < len(full_ids):
                        eos_after_assistant.append(content_end)

                cursor = footer_end

    full_ids = full_ids[:max_len]
    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    shifted_spans = [
        (max(0, s - 1), max(0, e - 1)) for s, e in assistant_spans
    ]
    shifted_eos = [max(0, pos - 1) for pos in eos_after_assistant]

    loss_mask = build_loss_mask(
        seq_len=len(labels),
        assistant_content_spans=shifted_spans,
        include_eos=True,
        eos_positions=shifted_eos,
        train_on_all=train_on_all,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }


def tokenize_plain(
    text: str,
    tokenizer: "PreTrainedTokenizerBase",
    max_len: int,
) -> dict[str, list[int]]:
    """Tokenize a plain string (non-chat) with an all-ones loss mask.

    Args:
        text: The input string.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum sequence length.

    Returns:
        Dict with keys ``input_ids``, ``labels``, and ``loss_mask``.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    input_ids = ids[:-1]
    labels = ids[1:]
    loss_mask = [1] * len(labels)
    return {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}


# ---------------------------------------------------------------------------
# DPO and SimPO loss functions
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard DPO loss (Rafailov et al., 2023).

    Loss = -logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))

    Args:
        policy_chosen_logps: Per-example sum log-probs for chosen responses
            under the policy model. Shape: (batch,).
        policy_rejected_logps: Per-example sum log-probs for rejected responses
            under the policy model. Shape: (batch,).
        ref_chosen_logps: Per-example sum log-probs for chosen responses under
            the reference model. Shape: (batch,).
        ref_rejected_logps: Per-example sum log-probs for rejected responses
            under the reference model. Shape: (batch,).
        beta: KL penalty coefficient.

    Returns:
        Tuple of (loss scalar, rewards tensor of shape (batch,)).
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    rewards = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(rewards).mean()
    return loss, rewards.detach()


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SimPO loss — reference-free, length-normalized (Meng et al., 2024).

    Loss = -logsigmoid(beta * (chosen_logps/chosen_len - rejected_logps/rejected_len) - gamma)

    Args:
        policy_chosen_logps: Per-example sum log-probs for chosen responses.
            Shape: (batch,).
        policy_rejected_logps: Per-example sum log-probs for rejected responses.
            Shape: (batch,).
        chosen_lengths: Number of response tokens for chosen examples.
            Shape: (batch,). Used for length normalization.
        rejected_lengths: Number of response tokens for rejected examples.
            Shape: (batch,).
        beta: Reward scaling coefficient.
        gamma: Target reward margin.

    Returns:
        Tuple of (loss scalar, length-normalized reward differences of shape (batch,)).
    """
    chosen_norm = policy_chosen_logps / chosen_lengths.float().clamp(min=1)
    rejected_norm = policy_rejected_logps / rejected_lengths.float().clamp(min=1)
    rewards = beta * (chosen_norm - rejected_norm) - gamma
    loss = -F.logsigmoid(rewards).mean()
    return loss, rewards.detach()


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-example sum log-probabilities and response token counts.

    Args:
        model: The language model.  Must accept ``(input_ids,)`` and return
            ``(logits, memory_states)`` with logits of shape (B, T, V).
        input_ids: Token input ids of shape (B, T).
        labels: Target token ids of shape (B, T).
        loss_mask: Binary float mask of shape (B, T) where 1 = response token.
        vocab_size: Vocabulary size (used to reshape logits).

    Returns:
        Tuple of:
        - ``sum_logps``: Per-example sum of log-probs over response tokens.
          Shape: (B,).
        - ``lengths``: Number of response tokens per example. Shape: (B,).
    """
    logits, _ = model(input_ids)
    logits = logits.float()  # ensure fp32 for numerical stability

    # log-softmax over vocabulary
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather log-prob of the target token at each position
    # labels shape: (B, T) — clamp to avoid index errors from padding
    labels_clamped = labels.clamp(min=0, max=vocab_size - 1)
    token_log_probs = log_probs.gather(
        dim=-1, index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Mask and sum over response tokens only
    masked_log_probs = token_log_probs * loss_mask
    sum_logps = masked_log_probs.sum(dim=-1)  # (B,)
    lengths = loss_mask.sum(dim=-1)  # (B,)

    return sum_logps, lengths


# ---------------------------------------------------------------------------
# DPO Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    """All hyperparameters for DPO/SimPO training.

    Architecture fields mirror TitansConfig / SFTConfig. DPO-specific fields
    control the loss type, beta/gamma, LoRA parameters, and dataset field names.
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

    # --- DPO-specific ---
    loss_type: str = "dpo"          # "dpo" or "simpo"
    beta: float = 0.1               # KL penalty (DPO) or reward scale (SimPO)
    gamma: float = 0.5              # Target reward margin (SimPO only)

    # --- LoRA ---
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: str = "attn"
    merge_and_save: str | None = None

    # Populated at runtime
    wrapped_paths: list[str] = field(default_factory=list)

    # --- Data ---
    dataset: str | None = None
    dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    seq_len: int = 2048

    # --- Training ---
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr: float = 5e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    mixed_precision: str = "no"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/dpo"
    save_every: int = 1000
    save_format: str = "pt"
    resume: str | None = None
    init_weights: str | None = None

    # --- Logging ---
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-dpo"
    wandb_run_name: str | None = None

    # --- Misc ---
    seed: int = 42
    synthetic_samples: int = 2000


# ---------------------------------------------------------------------------
# Preference pair dataset
# ---------------------------------------------------------------------------


def _parse_field(
    value: Any,
    tokenizer: "PreTrainedTokenizerBase",
    max_len: int,
) -> dict[str, list[int]]:
    """Tokenize a chosen/rejected field value.

    Accepts either a list of message dicts (chat format) or a plain string.

    Args:
        value: The field value from the dataset row.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum sequence length.

    Returns:
        Dict with ``input_ids``, ``labels``, ``loss_mask`` (lists of ints).
    """
    if isinstance(value, list):
        return tokenize_chat(value, tokenizer, max_len, train_on_all=False)
    return tokenize_plain(str(value), tokenizer, max_len)


class DPOStreamingDataset(IterableDataset):
    """Stream preference pairs from a HuggingFace dataset.

    Each example must have ``chosen_field`` and ``rejected_field`` columns
    containing either message lists or plain strings.  Examples where either
    side produces zero response tokens are skipped.

    Args:
        dataset_name: HuggingFace dataset repo id.
        subset: Optional dataset configuration name.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum sequence length per side.
        chosen_field: Column name for the chosen response.
        rejected_field: Column name for the rejected response.
        seed: Shuffle buffer seed.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        tokenizer: "PreTrainedTokenizerBase",
        max_len: int,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
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
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field

    def __iter__(self):
        for example in self.ds:
            chosen_raw = example.get(self.chosen_field)
            rejected_raw = example.get(self.rejected_field)

            if chosen_raw is None or rejected_raw is None:
                continue

            try:
                chosen = _parse_field(chosen_raw, self.tokenizer, self.max_len)
                rejected = _parse_field(rejected_raw, self.tokenizer, self.max_len)
            except Exception as exc:
                logger.debug(f"Skipping example due to tokenization error: {exc}")
                continue

            # Skip examples where either side has no response tokens
            if not any(chosen["loss_mask"]) or not any(rejected["loss_mask"]):
                continue

            # Skip degenerate examples (< 2 tokens after shifting)
            if len(chosen["input_ids"]) < 2 or len(rejected["input_ids"]) < 2:
                continue

            yield {
                "chosen_input_ids": torch.tensor(
                    chosen["input_ids"], dtype=torch.long
                ),
                "chosen_labels": torch.tensor(chosen["labels"], dtype=torch.long),
                "chosen_loss_mask": torch.tensor(
                    chosen["loss_mask"], dtype=torch.float
                ),
                "rejected_input_ids": torch.tensor(
                    rejected["input_ids"], dtype=torch.long
                ),
                "rejected_labels": torch.tensor(
                    rejected["labels"], dtype=torch.long
                ),
                "rejected_loss_mask": torch.tensor(
                    rejected["loss_mask"], dtype=torch.float
                ),
            }


class SyntheticDPODataset(Dataset):
    """Synthetic preference pair dataset for smoke-testing without a real corpus.

    Generates random token sequences with random response-span masks.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 2000,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Generate chosen and rejected separately to simulate different responses
        chosen_ids = rng.integers(0, vocab_size, (num_samples, seq_len))
        rejected_ids = rng.integers(0, vocab_size, (num_samples, seq_len))

        self.chosen_input_ids = chosen_ids[:, :-1].copy()
        self.chosen_labels = chosen_ids[:, 1:].copy()
        self.rejected_input_ids = rejected_ids[:, :-1].copy()
        self.rejected_labels = rejected_ids[:, 1:].copy()

        # Loss mask: second half of each sequence is the "response"
        half = (seq_len - 1) // 2
        self.loss_mask = np.zeros((num_samples, seq_len - 1), dtype=np.float32)
        self.loss_mask[:, half:] = 1.0

    def __len__(self) -> int:
        return len(self.chosen_input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "chosen_input_ids": torch.from_numpy(
                self.chosen_input_ids[idx]
            ).long(),
            "chosen_labels": torch.from_numpy(self.chosen_labels[idx]).long(),
            "chosen_loss_mask": torch.from_numpy(self.loss_mask[idx]),
            "rejected_input_ids": torch.from_numpy(
                self.rejected_input_ids[idx]
            ).long(),
            "rejected_labels": torch.from_numpy(self.rejected_labels[idx]).long(),
            "rejected_loss_mask": torch.from_numpy(self.loss_mask[idx]),
        }


def dpo_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length preference pairs.

    Pads each side (chosen, rejected) to the longest example in the batch.
    Padding uses 0 for input_ids/labels and 0.0 for loss_mask.

    Args:
        batch: List of sample dicts from DPOStreamingDataset.

    Returns:
        Padded batch dict.
    """
    chosen_max = max(item["chosen_input_ids"].shape[0] for item in batch)
    rejected_max = max(item["rejected_input_ids"].shape[0] for item in batch)

    chosen_ids, chosen_lbls, chosen_masks = [], [], []
    rejected_ids, rejected_lbls, rejected_masks = [], [], []

    for item in batch:
        c_len = item["chosen_input_ids"].shape[0]
        r_len = item["rejected_input_ids"].shape[0]

        c_pad = chosen_max - c_len
        r_pad = rejected_max - r_len

        chosen_ids.append(
            torch.cat([item["chosen_input_ids"],
                       torch.zeros(c_pad, dtype=torch.long)])
        )
        chosen_lbls.append(
            torch.cat([item["chosen_labels"],
                       torch.zeros(c_pad, dtype=torch.long)])
        )
        chosen_masks.append(
            torch.cat([item["chosen_loss_mask"],
                       torch.zeros(c_pad, dtype=torch.float)])
        )
        rejected_ids.append(
            torch.cat([item["rejected_input_ids"],
                       torch.zeros(r_pad, dtype=torch.long)])
        )
        rejected_lbls.append(
            torch.cat([item["rejected_labels"],
                       torch.zeros(r_pad, dtype=torch.long)])
        )
        rejected_masks.append(
            torch.cat([item["rejected_loss_mask"],
                       torch.zeros(r_pad, dtype=torch.float)])
        )

    return {
        "chosen_input_ids": torch.stack(chosen_ids),
        "chosen_labels": torch.stack(chosen_lbls),
        "chosen_loss_mask": torch.stack(chosen_masks),
        "rejected_input_ids": torch.stack(rejected_ids),
        "rejected_labels": torch.stack(rejected_lbls),
        "rejected_loss_mask": torch.stack(rejected_masks),
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Instantiate a Titans model by variant name.

    Args:
        model_type: One of ``mac``, ``mag``, ``mal``, ``lmm``.
        config: Fully-populated TitansConfig.

    Returns:
        Initialised (but untrained) model.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    if model_type not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_MODEL_CLASSES.keys())}"
        )
    return _MODEL_CLASSES[model_type](config)


def build_titans_config(cfg: DPOConfig) -> TitansConfig:
    """Translate DPOConfig fields into a TitansConfig.

    Args:
        cfg: DPOConfig populated from CLI arguments.

    Returns:
        TitansConfig instance ready to pass to create_model.
    """
    kwargs: dict[str, Any] = dict(
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        vocab_size=cfg.vocab_size,
        chunk_size=cfg.chunk_size,
        window_size=cfg.window_size,
        rope_proportion=cfg.rope_proportion,
        num_persistent_tokens=cfg.num_persistent_tokens,
        num_memory_layers=cfg.num_memory_layers,
        memory_objective=cfg.memory_objective,
        huber_delta_init=cfg.huber_delta_init,
        dropout=cfg.dropout,
        use_conv=cfg.use_conv,
    )

    if cfg.use_tnt:
        kwargs.update(
            use_tnt=cfg.use_tnt,
            global_chunk_size=cfg.global_chunk_size,
            use_qk_projection=cfg.use_qk_projection,
            tnt_stage=cfg.tnt_stage,
            finetune_local_chunk_sizes=cfg.finetune_local_chunk_sizes,
        )
        if cfg.local_chunk_sizes:
            kwargs["local_chunk_sizes"] = cfg.local_chunk_sizes
        if cfg.local_shard_length:
            kwargs["local_shard_length"] = cfg.local_shard_length

    if cfg.use_attn_res:
        kwargs.update(
            use_attn_res=cfg.use_attn_res,
            num_attnres_blocks=cfg.num_attnres_blocks,
            attnres_warmup_steps=cfg.attnres_warmup_steps,
            attnres_modulate_global_memory=cfg.attnres_modulate_global_memory,
            attnres_modulate_local_memory=cfg.attnres_modulate_local_memory,
        )

    if cfg.adaptive_window:
        kwargs.update(
            adaptive_window=cfg.adaptive_window,
            adaptive_window_min=cfg.adaptive_window_min,
            adaptive_window_max=cfg.adaptive_window_max,
            adaptive_window_temperature=cfg.adaptive_window_temperature,
            adaptive_window_lambda=cfg.adaptive_window_lambda,
        )

    if cfg.use_mca:
        kwargs.update(
            use_mca=cfg.use_mca,
            mca_num_heads=cfg.mca_num_heads,
            mca_gate_type=cfg.mca_gate_type,
            mca_gate_bias_init=cfg.mca_gate_bias_init,
        )
        if cfg.mca_insertion_layers:
            kwargs["mca_insertion_layers"] = cfg.mca_insertion_layers

    return TitansConfig(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: DPOConfig) -> None:
    """Run the full DPO/SimPO training loop.

    For standard DPO with LoRA:
    - Policy forward: set_lora_enabled(model, True) — LoRA delta active.
    - Reference forward: set_lora_enabled(model, False) inside torch.no_grad()
      — only the frozen base weights, no LoRA delta.

    For SimPO:
    - No reference forward at all (reference-free).

    Args:
        config: Populated DPOConfig instance.

    Raises:
        ImportError: If accelerate is not installed.
        ValueError: If loss_type is not recognised.
    """
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required. Install with: pip install accelerate"
        )

    if config.loss_type not in ("dpo", "simpo"):
        raise ValueError(
            f"Unknown loss_type '{config.loss_type}'. Choose from: dpo, simpo"
        )

    # Auto-adjust beta default for SimPO if the user did not explicitly change it
    effective_beta = config.beta
    if config.loss_type == "simpo" and config.beta == 0.1:
        effective_beta = 2.0
        logger.info(
            "SimPO detected with default beta=0.1; using beta=2.0 for SimPO. "
            "Override with --beta."
        )

    use_lora = config.use_lora
    is_simpo = config.loss_type == "simpo"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

    if accelerator.is_main_process:
        logger.info(f"DPO config: {config}")
        logger.info(f"Loss type: {config.loss_type}")
        logger.info(f"Effective beta: {effective_beta}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(
            f"Gradient accumulation steps: {config.gradient_accumulation_steps}"
        )
        logger.info(f"LoRA enabled: {use_lora}")

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

    # Load pretrained / SFT weights
    if config.init_weights is not None:
        init_path = Path(config.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-weights path not found: {init_path}")

        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
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
    # LoRA wrapping
    # ------------------------------------------------------------------
    if use_lora:
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
                f"LoRA wrapping: {len(wrapped_paths)} layers targeted "
                f"({', '.join(wrapped_paths[:5])}"
                f"{'...' if len(wrapped_paths) > 5 else ''})"
            )
            logger.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
            )
    else:
        # Full fine-tune: all parameters trainable
        if accelerator.is_main_process and not is_simpo:
            logger.warning(
                "No LoRA specified for DPO. Running full fine-tune with frozen "
                "base as reference is not possible without LoRA. "
                "The reference model will be the initial weights at step 0. "
                "Consider using --use-lora for the LoRA-as-reference trick."
            )

    # ------------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------------
    use_streaming = False
    if config.dataset is not None and HAS_DATASETS and tokenizer is not None:
        train_dataset: Dataset = DPOStreamingDataset(
            dataset_name=config.dataset,
            subset=config.dataset_subset,
            tokenizer=tokenizer,
            max_len=config.seq_len,
            chosen_field=config.chosen_field,
            rejected_field=config.rejected_field,
            seed=config.seed,
        )
        use_streaming = True
        if accelerator.is_main_process:
            logger.info(f"Streaming DPO dataset: {config.dataset}")
    else:
        if accelerator.is_main_process:
            if config.dataset is not None:
                logger.warning(
                    "Could not load dataset (missing dependencies or tokenizer). "
                    "Falling back to synthetic preference data."
                )
            else:
                logger.info(
                    "No dataset specified — using synthetic preference data for demo."
                )
        train_dataset = SyntheticDPODataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            num_samples=config.synthetic_samples,
            seed=config.seed,
        )

    is_iterable = isinstance(train_dataset, IterableDataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=not is_iterable,
        num_workers=0,
        drop_last=True,
        collate_fn=dpo_collate_fn if use_streaming else None,
    )

    # ------------------------------------------------------------------
    # Optimizer (LoRA-only params if LoRA enabled, else all params)
    # ------------------------------------------------------------------
    if use_lora:
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())

    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ------------------------------------------------------------------
    # LR scheduler (linear warmup + cosine decay)
    # ------------------------------------------------------------------
    if config.max_steps > 0:
        total_steps = config.max_steps
    elif not is_iterable:
        total_steps = (
            len(train_dataloader) // config.gradient_accumulation_steps
        ) * config.epochs
    else:
        total_steps = 100_000

    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        """Linear warmup then cosine decay."""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
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
    # Checkpoint directory
    # ------------------------------------------------------------------
    checkpoint_dir = Path(config.checkpoint_dir)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume from DPO checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0

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
        if accelerator.is_main_process:
            logger.info(
                f"Resumed from {resume_path} at step {global_step}, "
                f"epoch {start_epoch}"
            )

    # ------------------------------------------------------------------
    # For full fine-tune DPO (no LoRA), snapshot reference weights once
    # at the start of training.
    # ------------------------------------------------------------------
    ref_model: nn.Module | None = None
    if not use_lora and not is_simpo:
        ref_model = create_model(config.model_type, titans_config)
        # Load same weights as the policy
        unwrapped_policy = accelerator.unwrap_model(model)
        ref_model.load_state_dict(unwrapped_policy.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model = ref_model.to(accelerator.device)
        if accelerator.is_main_process:
            logger.info(
                "Reference model snapshot created (no LoRA mode). "
                "Reference weights are frozen at initialization."
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()

    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        epoch_chosen_rewards = 0.0
        epoch_rejected_rewards = 0.0
        num_optimizer_steps = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            chosen_input_ids = batch["chosen_input_ids"]
            chosen_labels = batch["chosen_labels"]
            chosen_loss_mask = batch["chosen_loss_mask"]
            rejected_input_ids = batch["rejected_input_ids"]
            rejected_labels = batch["rejected_labels"]
            rejected_loss_mask = batch["rejected_loss_mask"]

            with accelerator.accumulate(model):
                # ----------------------------------------------------------
                # Policy forward passes
                # ----------------------------------------------------------
                if use_lora:
                    set_lora_enabled(accelerator.unwrap_model(model), True)

                policy_chosen_logps, chosen_lengths = compute_log_probs(
                    model,
                    chosen_input_ids,
                    chosen_labels,
                    chosen_loss_mask,
                    config.vocab_size,
                )
                policy_rejected_logps, rejected_lengths = compute_log_probs(
                    model,
                    rejected_input_ids,
                    rejected_labels,
                    rejected_loss_mask,
                    config.vocab_size,
                )

                # ----------------------------------------------------------
                # Reference forward passes (DPO only)
                # ----------------------------------------------------------
                if is_simpo:
                    loss, rewards = simpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        chosen_lengths,
                        rejected_lengths,
                        beta=effective_beta,
                        gamma=config.gamma,
                    )
                else:
                    if use_lora:
                        # LoRA-as-reference: disable LoRA, run base model only
                        with torch.no_grad():
                            set_lora_enabled(
                                accelerator.unwrap_model(model), False
                            )
                            ref_chosen_logps, _ = compute_log_probs(
                                model,
                                chosen_input_ids,
                                chosen_labels,
                                chosen_loss_mask,
                                config.vocab_size,
                            )
                            ref_rejected_logps, _ = compute_log_probs(
                                model,
                                rejected_input_ids,
                                rejected_labels,
                                rejected_loss_mask,
                                config.vocab_size,
                            )
                        # Re-enable LoRA for the backward pass
                        set_lora_enabled(accelerator.unwrap_model(model), True)
                    else:
                        # Separate frozen reference model
                        with torch.no_grad():
                            ref_chosen_logps, _ = compute_log_probs(
                                ref_model,
                                chosen_input_ids,
                                chosen_labels,
                                chosen_loss_mask,
                                config.vocab_size,
                            )
                            ref_rejected_logps, _ = compute_log_probs(
                                ref_model,
                                rejected_input_ids,
                                rejected_labels,
                                rejected_loss_mask,
                                config.vocab_size,
                            )

                    loss, rewards = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        beta=effective_beta,
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                    num_optimizer_steps += 1
                    global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ------------------------------------------------------------------
            # Metrics
            # ------------------------------------------------------------------
            loss_val = loss.item()
            # rewards: shape (B,) — mean chosen vs mean rejected reward
            if is_simpo:
                # SimPO rewards are length-normalized differences
                batch_chosen_reward = rewards.mean().item()
                batch_rejected_reward = 0.0  # not separately tracked for SimPO
            else:
                # DPO rewards: (chosen - ref_chosen) - (rejected - ref_rejected)
                # We log the mean reward difference
                batch_chosen_reward = rewards.clamp(min=0).mean().item()
                batch_rejected_reward = rewards.clamp(max=0).abs().mean().item()

            epoch_loss += loss_val
            epoch_chosen_rewards += batch_chosen_reward
            epoch_rejected_rewards += batch_rejected_reward

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % config.log_every == 0 and accelerator.is_main_process:
                n = num_optimizer_steps if num_optimizer_steps > 0 else 1
                avg_loss = epoch_loss / n
                lr_val = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    avg=f"{avg_loss:.4f}",
                    lr=f"{lr_val:.2e}",
                    step=global_step,
                )

                if config.wandb and HAS_WANDB:
                    log_dict: dict[str, float] = {
                        "train/loss": loss_val,
                        "train/avg_loss": avg_loss,
                        "train/lr": lr_val,
                        "train/chosen_reward": batch_chosen_reward,
                    }
                    if not is_simpo:
                        log_dict["train/rejected_reward"] = batch_rejected_reward
                    accelerator.log(log_dict, step=global_step)

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

                    if use_lora:
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
                        except ImportError:
                            logger.warning(
                                "safetensors not installed — skipping "
                                "adapter-only save."
                            )

        # End-of-epoch summary
        if accelerator.is_main_process:
            n = num_optimizer_steps if num_optimizer_steps > 0 else 1
            avg_epoch_loss = epoch_loss / n
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
        logger.info(f"DPO training complete. Final checkpoint: {paths[0]}")

        # Adapter-only save
        if use_lora:
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
                "loss_type": config.loss_type,
                "beta": effective_beta,
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

            if config.merge_and_save is not None:
                merge_path = Path(config.merge_and_save)
                merge_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Merging LoRA weights into base model...")
                merge_lora_weights(unwrapped)
                merge_stem = merge_path.with_suffix("")
                merge_paths = save_checkpoint(
                    unwrapped.state_dict(),
                    merge_stem,
                    format=config.save_format,
                )
                logger.info(f"Saved merged model to {merge_paths[0]}")

    if config.wandb and HAS_WANDB:
        accelerator.end_training()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> DPOConfig:
    """Parse command-line arguments and return a DPOConfig."""
    parser = argparse.ArgumentParser(
        description="Direct Preference Optimization (DPO/SimPO) for Titans models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture
    arch = parser.add_argument_group("Model architecture")
    arch.add_argument(
        "--model",
        type=str,
        default="mac",
        choices=["mac", "mag", "mal", "lmm"],
        help="Titans model variant",
    )
    arch.add_argument("--dim", type=int, default=512)
    arch.add_argument("--num-heads", type=int, default=8)
    arch.add_argument("--num-layers", type=int, default=12)
    arch.add_argument("--vocab-size", type=int, default=32000)
    arch.add_argument("--chunk-size", type=int, default=512)
    arch.add_argument("--window-size", type=int, default=512)
    arch.add_argument(
        "--rope-proportion", type=float, default=1.0,
        help="Fraction of head_dim pairs to apply RoPE to (0.0-1.0, default 1.0)",
    )
    arch.add_argument("--num-persistent-tokens", type=int, default=16)
    arch.add_argument("--num-memory-layers", type=int, default=2)
    arch.add_argument(
        "--memory-objective", type=str, default="l2", choices=["l2", "huber"]
    )
    arch.add_argument("--huber-delta-init", type=float, default=0.0)
    arch.add_argument("--dropout", type=float, default=0.0)
    arch.add_argument("--use-conv", action="store_true")

    # TNT
    tnt = parser.add_argument_group("TNT / hierarchical memory")
    tnt.add_argument("--use-tnt", action="store_true")
    tnt.add_argument("--global-chunk-size", type=int, default=2048)
    tnt.add_argument(
        "--local-chunk-sizes", type=int, nargs="+", default=[8, 16], metavar="N"
    )
    tnt.add_argument("--local-shard-length", type=int, default=2048)
    tnt.add_argument("--use-qk-projection", action="store_true", default=True)
    tnt.add_argument("--tnt-stage", type=int, default=1)
    tnt.add_argument(
        "--finetune-local-chunk-sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
    )

    # Attention residual
    attn = parser.add_argument_group("Attention residual")
    attn.add_argument("--use-attn-res", action="store_true")
    attn.add_argument("--num-attnres-blocks", type=int, default=8)
    attn.add_argument("--attnres-warmup-steps", type=int, default=0)
    attn.add_argument(
        "--attnres-modulate-global-memory", action="store_true", default=True
    )
    attn.add_argument(
        "--no-attnres-modulate-global-memory",
        dest="attnres_modulate_global_memory",
        action="store_false",
    )
    attn.add_argument("--attnres-modulate-local-memory", action="store_true")

    # Adaptive window
    aw = parser.add_argument_group("Adaptive window")
    aw.add_argument("--adaptive-window", action="store_true")
    aw.add_argument("--adaptive-window-min", type=int, default=64)
    aw.add_argument("--adaptive-window-max", type=int, default=None)
    aw.add_argument("--adaptive-window-temperature", type=float, default=10.0)
    aw.add_argument("--adaptive-window-lambda", type=float, default=0.01)

    # MCA
    mca = parser.add_argument_group("Multi-context attention (MCA)")
    mca.add_argument("--use-mca", action="store_true")
    mca.add_argument(
        "--mca-insertion-layers", type=int, nargs="+", default=None, metavar="N"
    )
    mca.add_argument("--mca-num-heads", type=int, default=8)
    mca.add_argument("--mca-gate-type", type=str, default="scalar")
    mca.add_argument("--mca-gate-bias-init", type=float, default=-2.0)

    # DPO-specific
    dpo_g = parser.add_argument_group("DPO / SimPO")
    dpo_g.add_argument(
        "--loss-type",
        type=str,
        default="dpo",
        choices=["dpo", "simpo"],
        help="Loss type: 'dpo' (standard) or 'simpo' (reference-free, length-normalized)",
    )
    dpo_g.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient for DPO (default 0.1) or reward scale for SimPO "
        "(auto-set to 2.0 for SimPO if left at default)",
    )
    dpo_g.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Target reward margin for SimPO (ignored for DPO)",
    )

    # LoRA
    lora_g = parser.add_argument_group("LoRA")
    lora_g.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapters (also enables LoRA-as-reference trick for DPO)",
    )
    lora_g.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r")
    lora_g.add_argument(
        "--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale=alpha/rank)"
    )
    lora_g.add_argument(
        "--lora-dropout", type=float, default=0.05, help="Dropout on LoRA input path"
    )
    lora_g.add_argument(
        "--lora-targets",
        type=str,
        default="attn",
        help="Comma-separated target groups: attn, ffn, memory, all",
    )
    lora_g.add_argument(
        "--merge-and-save",
        type=str,
        default=None,
        metavar="PATH",
        help="After training, merge LoRA into base weights and save to PATH",
    )

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset", type=str, default=None, help="HuggingFace dataset repo id"
    )
    data.add_argument("--dataset-subset", type=str, default=None)
    data.add_argument("--tokenizer", type=str, default="gpt2")
    data.add_argument(
        "--chosen-field",
        type=str,
        default="chosen",
        help="Dataset column name for the chosen response",
    )
    data.add_argument(
        "--rejected-field",
        type=str,
        default="rejected",
        help="Dataset column name for the rejected response",
    )
    data.add_argument("--seq-len", type=int, default=2048)

    # Training
    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--epochs", type=int, default=1)
    train_g.add_argument("--max-steps", type=int, default=-1)
    train_g.add_argument("--batch-size", type=int, default=2)
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train_g.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (lower than SFT; DPO is sensitive to LR)",
    )
    train_g.add_argument("--weight-decay", type=float, default=0.1)
    train_g.add_argument("--grad-clip", type=float, default=1.0)
    train_g.add_argument("--warmup-ratio", type=float, default=0.03)
    train_g.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )

    # Checkpointing
    ckpt = parser.add_argument_group("Checkpointing")
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/dpo")
    ckpt.add_argument("--save-every", type=int, default=1000)
    ckpt.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
    )
    ckpt.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume DPO from a previous DPO checkpoint (.pt)",
    )
    ckpt.add_argument(
        "--init-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Load pretrained or SFT weights before DPO",
    )

    # Logging
    log = parser.add_argument_group("Logging")
    log.add_argument("--log-every", type=int, default=10)
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="titans-dpo")
    log.add_argument("--wandb-run-name", type=str, default=None)

    # Misc
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--synthetic-samples", type=int, default=2000)

    args = parser.parse_args()

    return DPOConfig(
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
        # DPO-specific
        loss_type=args.loss_type,
        beta=args.beta,
        gamma=args.gamma,
        # LoRA
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        merge_and_save=args.merge_and_save,
        # Data
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        seq_len=args.seq_len,
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
        resume=args.resume,
        init_weights=args.init_weights,
        # Logging
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
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
