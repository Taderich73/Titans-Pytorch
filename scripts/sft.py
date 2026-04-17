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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansMAC, TitansMAG, TitansMAL, TitansLMM
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import save_memory_states

# scripts/ is imported both as a namespace package ("scripts._common") and as
# a flat directory (when tests add scripts/ onto sys.path and import "sft").
# Try the package-style import first; fall back to sibling-module import.
try:
    from scripts._common import chunked_forward  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in test-only sys.path layouts
    from _common import chunked_forward  # type: ignore[no-redef]

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
# ChatML constants
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ---------------------------------------------------------------------------
# Chat formatting helpers
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
            correspond to assistant turn content (inclusive start, exclusive end)
            in the *label* (shifted) sequence.
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
    back to ChatML formatting. Identifies assistant turns in order to mask
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
        # Tokenize with native template; track assistant spans manually
        # by tokenizing each prefix and diffing token counts.
        full_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Identify assistant content token spans by tokenizing incrementally.
        # We find where each assistant turn begins and ends in the token stream.
        assistant_spans: list[tuple[int, int]] = []
        eos_after_assistant: list[int] = []

        if not train_on_all:
            # Tokenize prefix up to each turn to find span boundaries.
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue

                # Tokens for conversation up to (but not including) this turn's
                # content — i.e. header tokens.
                prefix_turns = messages[:i]
                if prefix_turns:
                    prefix_ids: list[int] = tokenizer.apply_chat_template(
                        prefix_turns,
                        tokenize=True,
                        add_generation_prompt=True,  # adds the assistant header
                    )
                else:
                    # First message is assistant — add header manually
                    prefix_ids = tokenizer.encode(
                        f"{IM_START}assistant\n", add_special_tokens=False
                    )

                content_start = len(prefix_ids)

                # Tokens through end of this assistant turn
                turns_through = messages[: i + 1]
                through_ids: list[int] = tokenizer.apply_chat_template(
                    turns_through,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                content_end = len(through_ids)

                # The IM_END token (if present) is at content_end - 1 or nearby;
                # include it if it directly follows the content.
                if content_end < len(full_ids):
                    eos_after_assistant.append(content_end)

                assistant_spans.append((content_start, content_end))

    else:
        # ChatML fallback — tokenize the full formatted string, then re-tokenize
        # each turn to identify spans.
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
                    # Mark the EOS token (first token of footer) if present
                    if footer_ids and content_end < len(full_ids):
                        eos_after_assistant.append(content_end)

                cursor = footer_end

    # Truncate to max_len (keep the first max_len tokens)
    full_ids = full_ids[:max_len]

    # Shift: input = [:-1], label = [1:]
    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    # Adjust spans for the label shift (label[i] = full_ids[i+1], so spans stay
    # the same relative to the underlying index but we subtract 1 from each).
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
# Model factory
# ---------------------------------------------------------------------------


def create_model(model_type: str, config: TitansConfig) -> torch.nn.Module:
    """Instantiate a Titans model by variant name.

    Args:
        model_type: One of ``mac``, ``mag``, ``mal``, ``lmm``.
        config: Fully-populated TitansConfig.

    Returns:
        Initialised (but untrained) model.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(models.keys())}"
        )
    return models[model_type](config)


def build_titans_config(cfg: SFTConfig) -> TitansConfig:
    """Translate SFTConfig fields into a TitansConfig.

    Args:
        cfg: SFTConfig populated from CLI arguments.

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

    # TNT fields
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

    # Attention-residual fields
    if cfg.use_attn_res:
        kwargs.update(
            use_attn_res=cfg.use_attn_res,
            num_attnres_blocks=cfg.num_attnres_blocks,
            attnres_warmup_steps=cfg.attnres_warmup_steps,
            attnres_modulate_global_memory=cfg.attnres_modulate_global_memory,
            attnres_modulate_local_memory=cfg.attnres_modulate_local_memory,
        )

    # Adaptive window fields
    if cfg.adaptive_window:
        kwargs.update(
            adaptive_window=cfg.adaptive_window,
            adaptive_window_min=cfg.adaptive_window_min,
            adaptive_window_max=cfg.adaptive_window_max,
            adaptive_window_temperature=cfg.adaptive_window_temperature,
            adaptive_window_lambda=cfg.adaptive_window_lambda,
        )

    # MCA fields
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

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=not is_streaming,
        num_workers=0,
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
            eval_dataloader = DataLoader(
                eval_hf,
                batch_size=config.batch_size,
                num_workers=0,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
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
    # Resume from SFT checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    memory_states = None

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
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning (SFT) for Titans PyTorch models",
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
        "--local-chunk-sizes",
        type=int,
        nargs="+",
        default=[8, 16],
        metavar="N",
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
    attn.add_argument("--attnres-modulate-global-memory", action="store_true", default=True)
    attn.add_argument("--no-attnres-modulate-global-memory", dest="attnres_modulate_global_memory", action="store_false")
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
        "--mca-insertion-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
    )
    mca.add_argument("--mca-num-heads", type=int, default=8)
    mca.add_argument("--mca-gate-type", type=str, default="scalar")
    mca.add_argument("--mca-gate-bias-init", type=float, default=-2.0)

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset repo id")
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

    # Training
    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--epochs", type=int, default=1)
    train_g.add_argument("--max-steps", type=int, default=-1)
    train_g.add_argument("--batch-size", type=int, default=4)
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_g.add_argument("--lr", type=float, default=2e-5)
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
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/sft")
    ckpt.add_argument("--save-every", type=int, default=1000)
    ckpt.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
    )
    ckpt.add_argument("--eval-every", type=int, default=200)
    ckpt.add_argument("--eval-batches", type=int, default=50)
    ckpt.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume SFT from a previous SFT checkpoint (.pt)",
    )
    ckpt.add_argument(
        "--init-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Load pretrained model weights before SFT (e.g. from pretrain.py)",
    )

    # Logging
    log = parser.add_argument_group("Logging")
    log.add_argument("--log-every", type=int, default=10)
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="titans-sft")
    log.add_argument("--wandb-run-name", type=str, default=None)

    # Memory state lifecycle
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

    # Misc
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--synthetic-samples", type=int, default=5000)

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
