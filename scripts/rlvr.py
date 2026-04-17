#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Reinforcement Learning with Verifiable Rewards (RLVR) training script for
Titans PyTorch models using HuggingFace Accelerate.

Supports:
- GRPO (Group Relative Policy Optimization) with clipped importance ratios
- REINFORCE with EMA baseline
- Offline mode (pre-computed rollouts + rewards) and live mode (generate + verify)
- Pluggable verifier framework (exact_match, numeric_match, custom)
- LoRA-as-reference trick (single model, LoRA toggled for policy vs reference)

Supported model variants: mac, mag, mal, lmm
Supported datasets (live mode): any HuggingFace dataset with a prompt field and
    an answer field.
Supported datasets (offline mode): any HuggingFace dataset with pre-computed
    completions and rewards fields.

Usage:
    # GRPO live mode with LoRA
    python scripts/rlvr.py \\
        --model mac \\
        --init-weights checkpoints/sft/final.pt \\
        --dataset my/reasoning-prompts \\
        --tokenizer meta-llama/Llama-2-7b-hf \\
        --loss-type grpo --group-size 4 --beta 0.1 \\
        --lora-rank 8 --lora-targets attn \\
        --mixed-precision bf16

    # REINFORCE live mode
    python scripts/rlvr.py \\
        --model mac \\
        --dataset my/prompts \\
        --loss-type reinforce --verifier numeric_match

    # Offline mode (pre-computed rollouts)
    python scripts/rlvr.py \\
        --model mac \\
        --dataset allenai/Dolci-Think-RL-7B \\
        --offline \\
        --rollout-field outputs --reward-field rewards

    # Multi-GPU via accelerate
    accelerate launch scripts/rlvr.py \\
        --model mac --dim 1024 --num-layers 24 \\
        --dataset my/prompts --loss-type grpo
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory_dump import load_memory_states, save_memory_states
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
# Model class registry
# ---------------------------------------------------------------------------

_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mac": TitansMAC,
    "mag": TitansMAG,
    "mal": TitansMAL,
    "lmm": TitansLMM,
}

# ---------------------------------------------------------------------------
# Verifier framework
# ---------------------------------------------------------------------------


def exact_match_verifier(prediction: str, answer: str) -> float:
    """Return 1.0 if stripped strings match (case-insensitive), 0.0 otherwise.

    Args:
        prediction: Model-generated text to evaluate.
        answer: Ground-truth answer string.

    Returns:
        1.0 if strings match after stripping and lowercasing, else 0.0.
    """
    return 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0


def numeric_match_verifier(
    prediction: str, answer: str, tolerance: float = 1e-6
) -> float:
    """Return 1.0 if the last number in prediction matches answer within tolerance.

    Extracts the last numeric value from the prediction string and compares it
    to the parsed answer value. Returns 0.0 if no numeric value can be parsed
    from either string.

    Args:
        prediction: Model-generated text to evaluate.
        answer: Ground-truth numeric string (e.g. "42" or "3.14").
        tolerance: Absolute tolerance for float comparison.

    Returns:
        1.0 if numeric values match within tolerance, else 0.0.
    """
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", prediction)
    if not numbers:
        return 0.0
    try:
        pred_val = float(numbers[-1])
        ans_val = float(answer.strip())
    except ValueError:
        return 0.0
    return 1.0 if abs(pred_val - ans_val) <= tolerance else 0.0


VERIFIERS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match_verifier,
    "numeric_match": numeric_match_verifier,
}


def load_custom_verifier(spec: str) -> Callable[[str, str], float]:
    """Load a custom verifier function from a file path specification.

    Args:
        spec: String in the form ``path/to/module.py:function_name``.

    Returns:
        The callable verifier function.

    Raises:
        ValueError: If the spec does not contain a colon separator.
        AttributeError: If the function name is not found in the module.
    """
    if ":" not in spec:
        raise ValueError(
            f"Custom verifier spec must be 'path/to/module.py:function_name', "
            f"got: {spec!r}"
        )
    module_path, func_name = spec.rsplit(":", 1)
    spec_obj = importlib.util.spec_from_file_location("custom_verifier", module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, func_name)


# ---------------------------------------------------------------------------
# GRPO and REINFORCE loss functions
# ---------------------------------------------------------------------------


def grpo_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: torch.Tensor,
    beta: float = 0.1,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """GRPO: normalize rewards within each group, compute clipped policy gradient
    with KL penalty.

    Args:
        log_probs: Per-response sum log-probs for the policy. Shape: (batch, group_size).
        rewards: Verifier rewards per response. Shape: (batch, group_size).
        ref_log_probs: Per-response sum log-probs for the reference (base model).
            Shape: (batch, group_size).
        beta: KL penalty weight.
        clip_ratio: PPO-style importance ratio clipping range.

    Returns:
        Scalar loss tensor.
    """
    # Normalize rewards within each group (zero mean, unit variance)
    reward_mean = rewards.mean(dim=-1, keepdim=True)
    reward_std = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    advantages = (rewards - reward_mean) / reward_std

    # Policy ratio (importance weight)
    ratio = (log_probs - ref_log_probs.detach()).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty: E[log(ref) - log(pi)] approximation
    kl = (ref_log_probs.detach() - log_probs).mean()

    return policy_loss + beta * kl


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
) -> tuple[torch.Tensor, float]:
    """REINFORCE with exponential moving average baseline.

    Args:
        log_probs: Per-example sum log-probs for the policy. Shape: (batch,).
        rewards: Verifier rewards per example. Shape: (batch,).
        baseline: Current EMA baseline value.

    Returns:
        Tuple of (loss scalar tensor, updated_baseline float).
    """
    advantages = rewards - baseline
    loss = -(log_probs * advantages).mean()
    new_baseline = 0.99 * baseline + 0.01 * rewards.mean().item()
    return loss, new_baseline


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------


def compute_token_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    vocab_size: int,
    states: list | None = None,
) -> tuple[torch.Tensor, list | None]:
    """Compute per-token log-probs with memory-aware chunking.

    Chunks the sequence along dim=1 by the model's ``config.chunk_size``,
    threading Titans memory state through successive chunks. This both
    avoids the per-forward single-chunk limit and preserves the memory
    accumulation regime the model was trained under.

    Args:
        model: Titans model with ``config.chunk_size``.
        input_ids: Token IDs of shape (B, T).
        vocab_size: Vocabulary size (for target clamping).
        states: Optional initial memory states; ``None`` means start fresh.

    Returns:
        Tuple of:
        - Per-token log-probs of shape (B, T-1).
        - Final memory states after processing the full sequence.
    """
    base_model = model.module if hasattr(model, "module") else model
    chunk_size = base_model.config.chunk_size

    # Chunk input_ids, run model per-chunk, collect logits, then compute
    # shifted log-probs over the concatenated logits.
    chunks = input_ids.split(chunk_size, dim=1)
    all_logits: list[torch.Tensor] = []
    for ids_c in chunks:
        logits, states, _ = model(ids_c, states=states)
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=1).float()  # (B, T, V) fp32 for stability
    # Shift: logits[:, :-1] predicts input_ids[:, 1:]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # (B, T-1, V)
    targets = input_ids[:, 1:].clamp(min=0, max=vocab_size - 1)  # (B, T-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    return token_log_probs, states


def sum_log_probs_for_completion(
    token_log_probs: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum log-probs over the completion portion (after the prompt).

    Args:
        token_log_probs: Per-token log-probs of shape (B, T-1).
        prompt_len: Number of prompt tokens; only tokens at positions
            >= prompt_len are summed.

    Returns:
        Shape (B,) sum of log-probs over completion tokens.
    """
    # token_log_probs[i] = log p(input_ids[i+1] | input_ids[:i+1])
    # Completion starts at position prompt_len in input_ids, which corresponds
    # to index prompt_len - 1 in token_log_probs (since we dropped the last token).
    start = max(0, prompt_len - 1)
    return token_log_probs[:, start:].sum(dim=-1)


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_rollouts(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    num_samples: int = 4,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rollouts with memory-aware prefill->buffer->commit decoding.

    Mirrors the inference.py generation pattern: prefill the prompt once
    per sample to produce an initial committed memory state, then
    autoregressively decode, re-feeding the since-commit buffer each step
    from the committed state so memory sees each completed chunk exactly
    once.

    Args:
        model: Titans model with ``config.chunk_size``. Must accept
            ``(input_ids, states=...)`` and return ``(logits, states, _)``.
        input_ids: Prompt token IDs of shape (B, prompt_len). Must be on the
            correct device already.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (higher = more diverse).
        num_samples: Number of independent samples per prompt (group size).
        eos_token_id: If provided, stop generation when this token is produced.
        pad_token_id: Token used to pad shorter sequences.

    Returns:
        Tuple of:
        - ``generated_ids``: shape (B, num_samples, prompt_len + max_new_tokens)
          full sequences including the prompt, padded to the longest generated
          sequence in the batch.
        - ``completion_log_probs``: shape (B, num_samples) sum log-probs for
          the generated completion (post-prompt) tokens only.
    """
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    chunk_size = base_model.config.chunk_size

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device

    all_sequences: list[torch.Tensor] = []  # each: (B, T)
    all_sum_logps: list[torch.Tensor] = []  # each: (B,)

    for _ in range(num_samples):
        tokens = input_ids.clone()  # (B, prompt_len)
        # Accumulate per-token log-probs for the completion
        completion_logps = torch.zeros(batch_size, device=device)
        # Track which sequences have hit EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # ---- Prefill: iterate only FULL prompt chunks so memory commits
        # are aligned to chunk boundaries. Any partial tail is handled
        # uniformly via the re-feed path below (the same pattern used in
        # the decode loop). Iterating input_ids.split(chunk_size) would
        # commit the partial tail here AND re-feed it, double-counting
        # those tokens once the first decode chunk commits. ----
        n_full = prompt_len // chunk_size
        buffer_start = n_full * chunk_size
        states = None
        last_chunk_logits = None
        for i in range(n_full):
            chunk_ids = input_ids[:, i * chunk_size : (i + 1) * chunk_size]
            last_chunk_logits, states, _ = model(chunk_ids, states=states)
            if states is not None:
                states = [
                    s.detach() if s is not None else None for s in states
                ]

        committed_states = (
            [s.detach() if s is not None else None for s in states]
            if states is not None
            else None
        )

        # Derive next-token logits (for sampling the first generated token).
        # If the prompt has a partial tail, re-feed it from committed_states
        # so memory only ever sees it once (when the chunk completes during
        # decode). If the prompt is chunk-aligned, the last committed chunk's
        # final-position logits are already the next-token prediction.
        if buffer_start < prompt_len:
            buf = tokens[:, buffer_start:]
            logits, _, _ = model(buf, states=committed_states)
        else:
            assert last_chunk_logits is not None, (
                "chunk-aligned prompt must have at least one full chunk"
            )
            logits = last_chunk_logits

        # ---- Decode loop ----
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float() / max(temperature, 1e-8)
            # Multinomial sampling
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Accumulate log-prob for non-finished sequences
            log_p = F.log_softmax(next_logits, dim=-1)
            token_logp = log_p.gather(dim=-1, index=next_token).squeeze(-1)  # (B,)
            completion_logps = completion_logps + token_logp * (~finished).float()

            tokens = torch.cat([tokens, next_token], dim=-1)  # (B, T+1)

            buffer = tokens[:, buffer_start:]
            buffer_len = buffer.shape[1]

            if buffer_len >= chunk_size:
                # Full chunk ready — process and commit memory update
                chunk = buffer[:, :chunk_size]
                logits, states, _ = model(chunk, states=committed_states)
                states = (
                    [s.detach() if s is not None else None for s in states]
                    if states is not None
                    else None
                )
                committed_states = (
                    [s.detach() if s is not None else None for s in states]
                    if states is not None
                    else None
                )
                buffer_start += chunk_size
                if buffer_len > chunk_size:
                    remainder = buffer[:, chunk_size:]
                    logits, states, _ = model(
                        remainder, states=committed_states
                    )
            else:
                # Partial buffer — re-feed from committed state so memory
                # only sees these tokens once when the chunk completes
                logits, states, _ = model(buffer, states=committed_states)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        all_sequences.append(tokens)
        all_sum_logps.append(completion_logps)

    # Pad all sequences to the same length
    max_len = max(seq.shape[-1] for seq in all_sequences)
    padded: list[torch.Tensor] = []
    for seq in all_sequences:
        pad_len = max_len - seq.shape[-1]
        if pad_len > 0:
            pad = torch.full(
                (batch_size, pad_len), pad_token_id, dtype=torch.long, device=device
            )
            seq = torch.cat([seq, pad], dim=-1)
        padded.append(seq)

    # Stack: (B, num_samples, max_len)
    generated_ids = torch.stack(padded, dim=1)
    # Stack: (B, num_samples)
    completion_log_probs = torch.stack(all_sum_logps, dim=1)

    return generated_ids, completion_log_probs


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def decode_tokens(
    token_ids: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_len: int,
) -> list[str]:
    """Decode completion tokens (post-prompt) to strings.

    Args:
        token_ids: Shape (B,) or (T,) token IDs. If 1-D, decoded as-is.
            If the input is a single sequence, only tokens at indices
            >= prompt_len are decoded.
        tokenizer: HuggingFace tokenizer.
        prompt_len: Number of prompt tokens to skip.

    Returns:
        List of decoded completion strings, one per batch item.
    """
    ids_list = token_ids.tolist()
    if isinstance(ids_list[0], list):
        return [
            tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in ids_list
        ]
    return [tokenizer.decode(ids_list[prompt_len:], skip_special_tokens=True)]


def verify_batch(
    generated_ids: torch.Tensor,
    answers: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    verifier: Callable[[str, str], float],
    prompt_len: int,
    num_samples: int,
) -> torch.Tensor:
    """Score each generated completion against the corresponding answer.

    Args:
        generated_ids: Shape (B, num_samples, T) full sequences.
        answers: List of B ground-truth answer strings.
        tokenizer: HuggingFace tokenizer for decoding.
        verifier: Callable(prediction, answer) -> float in [0, 1].
        prompt_len: Number of prompt tokens to skip when decoding.
        num_samples: Number of samples per prompt.

    Returns:
        Rewards tensor of shape (B, num_samples).
    """
    batch_size = generated_ids.shape[0]
    rewards = torch.zeros(batch_size, num_samples, device=generated_ids.device)

    for b in range(batch_size):
        for s in range(num_samples):
            seq = generated_ids[b, s]  # (T,)
            completion = tokenizer.decode(
                seq[prompt_len:].tolist(), skip_special_tokens=True
            )
            rewards[b, s] = verifier(completion, answers[b])

    return rewards


# ---------------------------------------------------------------------------
# Log-prob computation for generated sequences (policy + reference)
# ---------------------------------------------------------------------------


def compute_log_probs_for_generated(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Compute sum log-probs over the completion portion of each generated sequence.

    Processes each sample in the group separately to avoid GPU memory issues with
    large batches. Each sample is processed independently with a fresh memory
    state, matching the RLVR rollout regime where each rollout begins from a
    common initial memory.

    Args:
        model: Language model.
        prompt_ids: Prompt token IDs of shape (B, prompt_len).
        generated_ids: Full sequences of shape (B, num_samples, T).
        vocab_size: Vocabulary size for logit indexing.

    Returns:
        Sum log-probs of shape (B, num_samples).
    """
    batch_size, num_samples, seq_len = generated_ids.shape
    prompt_len = prompt_ids.shape[1]
    all_logps: list[torch.Tensor] = []

    for s in range(num_samples):
        seqs = generated_ids[:, s, :]  # (B, T)
        token_logps, _ = compute_token_log_probs(
            model, seqs, vocab_size, states=None
        )  # (B, T-1)
        sum_logps = sum_log_probs_for_completion(token_logps, prompt_len)  # (B,)
        all_logps.append(sum_logps)

    return torch.stack(all_logps, dim=1)  # (B, num_samples)


# ---------------------------------------------------------------------------
# RLVR Config
# ---------------------------------------------------------------------------


@dataclass
class RLVRConfig:
    """All hyperparameters for RLVR training.

    Architecture fields mirror TitansConfig / DPOConfig. RLVR-specific fields
    control the loss type, group size, rollout generation, and verifier.
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

    # --- RLVR-specific ---
    loss_type: str = "grpo"       # "grpo" or "reinforce"
    beta: float = 0.1             # KL penalty weight (GRPO)
    clip_ratio: float = 0.2       # PPO-style clipping range (GRPO)
    group_size: int = 4           # Samples per prompt (GRPO); set 1 for REINFORCE
    max_new_tokens: int = 256     # Max tokens to generate per rollout
    temperature: float = 0.7      # Sampling temperature
    verifier: str = "exact_match" # Verifier name or "path/to/file.py:fn_name"

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: str = "attn"
    merge_and_save: str | None = None

    # Populated at runtime by wrap_lora_layers
    wrapped_paths: list[str] = field(default_factory=list)

    # --- Data ---
    dataset: str | None = None
    dataset_subset: str | None = None
    prompt_field: str = "prompt"
    answer_field: str = "answer"
    tokenizer: str = "gpt2"
    seq_len: int = 2048

    # Offline mode (pre-computed rollouts)
    offline: bool = False
    rollout_field: str = "completions"
    reward_field: str = "rewards"

    # --- Training ---
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    mixed_precision: str = "no"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/rlvr"
    save_every: int = 1000
    save_format: str = "pt"
    resume: str | None = None
    init_weights: str | None = None

    # --- Memory state lifecycle ---
    reset_memory_per_batch: bool = True
    state_carry_warmup_steps: int = 0

    # --- Logging ---
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-rlvr"
    wandb_run_name: str | None = None

    # --- Misc ---
    seed: int = 42
    synthetic_samples: int = 500


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


def build_titans_config(cfg: RLVRConfig) -> TitansConfig:
    """Translate RLVRConfig fields into a TitansConfig.

    Args:
        cfg: RLVRConfig populated from CLI arguments.

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
# Streaming datasets
# ---------------------------------------------------------------------------


class LiveRLVRDataset(IterableDataset):
    """Stream prompts and answers from a HuggingFace dataset for live RLVR.

    Each example must have a prompt field and an answer field. The model
    generates rollouts at training time and the verifier scores them.

    Args:
        dataset_name: HuggingFace dataset repo id.
        subset: Optional dataset configuration name.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum prompt length (prompts are truncated).
        prompt_field: Dataset column name for the prompt.
        answer_field: Dataset column name for the answer.
        seed: Shuffle buffer seed.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        tokenizer: "PreTrainedTokenizerBase",
        max_len: int,
        prompt_field: str = "prompt",
        answer_field: str = "answer",
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
        self.prompt_field = prompt_field
        self.answer_field = answer_field

    def __iter__(self):
        for example in self.ds:
            prompt = example.get(self.prompt_field)
            answer = example.get(self.answer_field)

            if not prompt or answer is None:
                continue

            try:
                prompt_ids = self.tokenizer.encode(
                    str(prompt), add_special_tokens=False
                )[: self.max_len]
            except Exception as exc:
                logger.debug(f"Skipping example due to tokenization error: {exc}")
                continue

            if len(prompt_ids) < 2:
                continue

            yield {
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "answer": str(answer),
            }


class OfflineRLVRDataset(IterableDataset):
    """Stream pre-computed (prompt, completions, rewards) tuples.

    Each example must have a prompt field, a rollout field (list of completion
    strings), and a reward field (list of floats). The dataset skips examples
    with fewer completions than min_rollouts.

    Args:
        dataset_name: HuggingFace dataset repo id.
        subset: Optional dataset configuration name.
        tokenizer: HuggingFace tokenizer.
        max_len: Maximum sequence length (prompt + completion truncated to this).
        prompt_field: Dataset column name for the prompt text.
        rollout_field: Dataset column name for the list of completions.
        reward_field: Dataset column name for the list of rewards.
        min_rollouts: Minimum number of completions required per example.
        seed: Shuffle buffer seed.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        tokenizer: "PreTrainedTokenizerBase",
        max_len: int,
        prompt_field: str = "prompt",
        rollout_field: str = "completions",
        reward_field: str = "rewards",
        min_rollouts: int = 2,
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
        self.prompt_field = prompt_field
        self.rollout_field = rollout_field
        self.reward_field = reward_field
        self.min_rollouts = min_rollouts

    def __iter__(self):
        for example in self.ds:
            prompt = example.get(self.prompt_field)
            completions = example.get(self.rollout_field)
            rewards = example.get(self.reward_field)

            if not prompt or not completions or not rewards:
                continue

            if not isinstance(completions, list) or not isinstance(rewards, list):
                continue

            if len(completions) < self.min_rollouts or len(rewards) < self.min_rollouts:
                continue

            # Ensure completions and rewards are aligned
            n = min(len(completions), len(rewards))
            completions = completions[:n]
            rewards_list = [float(r) for r in rewards[:n]]

            try:
                prompt_ids = self.tokenizer.encode(
                    str(prompt), add_special_tokens=False
                )
            except Exception as exc:
                logger.debug(f"Skipping example due to tokenization error: {exc}")
                continue

            if len(prompt_ids) < 1:
                continue

            # Tokenize each completion (full sequence = prompt + completion)
            rollout_ids_list: list[torch.Tensor] = []
            for completion in completions:
                try:
                    full_text = str(prompt) + str(completion)
                    full_ids = self.tokenizer.encode(
                        full_text, add_special_tokens=False
                    )[: self.max_len]
                    rollout_ids_list.append(
                        torch.tensor(full_ids, dtype=torch.long)
                    )
                except Exception as exc:
                    logger.debug(f"Skipping completion due to error: {exc}")
                    continue

            if len(rollout_ids_list) < self.min_rollouts:
                continue

            yield {
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "rollout_ids": rollout_ids_list,
                "rewards": torch.tensor(rewards_list[: len(rollout_ids_list)], dtype=torch.float),
            }


class SyntheticRLVRDataset(torch.utils.data.Dataset):
    """Synthetic RLVR dataset for smoke-testing without a real corpus.

    Generates random prompt sequences and random rollouts with random rewards.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        group_size: int = 4,
        num_samples: int = 500,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.prompt_ids = rng.integers(4, vocab_size, (num_samples, seq_len // 4))
        self.rollout_ids = rng.integers(4, vocab_size, (num_samples, group_size, seq_len))
        self.rewards = rng.random((num_samples, group_size)).astype(np.float32)
        self.group_size = group_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "prompt_ids": torch.from_numpy(self.prompt_ids[idx]).long(),
            "rollout_ids": [
                torch.from_numpy(self.rollout_ids[idx, s]).long()
                for s in range(self.group_size)
            ],
            "rewards": torch.from_numpy(self.rewards[idx]),
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def rlvr_collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Pad and collate a batch of RLVR examples.

    Pads prompt_ids and each rollout to the longest sequence in the batch.
    Pads rewards to the maximum group size in the batch.

    Args:
        batch: List of sample dicts with ``prompt_ids``, ``rollout_ids``,
            and ``rewards``.
        pad_token_id: Token id used for padding.

    Returns:
        Collated dict with:
        - ``prompt_ids``: (B, prompt_len) padded prompt ids.
        - ``rollout_ids``: (B, max_group, seq_len) padded rollout ids.
        - ``rewards``: (B, max_group) padded rewards.
        - ``prompt_len``: scalar int (minimum prompt length in batch).
    """
    # Pad prompt ids
    prompt_max_len = max(item["prompt_ids"].shape[0] for item in batch)
    padded_prompts: list[torch.Tensor] = []
    for item in batch:
        p = item["prompt_ids"]
        pad_len = prompt_max_len - p.shape[0]
        if pad_len > 0:
            p = torch.cat(
                [p, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
        padded_prompts.append(p)
    prompt_ids = torch.stack(padded_prompts)  # (B, prompt_len)

    # Determine the minimum unpadded prompt length (for completion indexing)
    min_prompt_len = min(item["prompt_ids"].shape[0] for item in batch)

    # Pad rollout ids
    max_group = max(len(item["rollout_ids"]) for item in batch)
    rollout_max_len = max(
        r.shape[0]
        for item in batch
        for r in item["rollout_ids"]
    )

    padded_rollouts: list[torch.Tensor] = []
    padded_rewards: list[torch.Tensor] = []

    for item in batch:
        rollouts = item["rollout_ids"]
        rews = item["rewards"]

        # Pad each rollout to rollout_max_len
        padded_group: list[torch.Tensor] = []
        for r in rollouts:
            pad_len = rollout_max_len - r.shape[0]
            if pad_len > 0:
                r = torch.cat(
                    [r, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
                )
            padded_group.append(r)

        # Pad group size with zero sequences if needed
        while len(padded_group) < max_group:
            padded_group.append(
                torch.full((rollout_max_len,), pad_token_id, dtype=torch.long)
            )

        padded_rollouts.append(torch.stack(padded_group))  # (max_group, seq_len)

        # Pad rewards
        rew_pad = max_group - rews.shape[0]
        if rew_pad > 0:
            rews = torch.cat([rews, torch.zeros(rew_pad)])
        padded_rewards.append(rews)

    rollout_ids = torch.stack(padded_rollouts)  # (B, max_group, seq_len)
    rewards = torch.stack(padded_rewards)  # (B, max_group)

    return {
        "prompt_ids": prompt_ids,
        "rollout_ids": rollout_ids,
        "rewards": rewards,
        "prompt_len": min_prompt_len,
    }


def live_collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Pad and collate a batch of live RLVR examples (prompt + answer only).

    Args:
        batch: List of sample dicts with ``prompt_ids`` and ``answer``.
        pad_token_id: Token id used for padding.

    Returns:
        Collated dict with ``prompt_ids`` (B, T) and ``answers`` list of str.
    """
    max_len = max(item["prompt_ids"].shape[0] for item in batch)
    padded: list[torch.Tensor] = []
    answers: list[str] = []

    for item in batch:
        p = item["prompt_ids"]
        pad_len = max_len - p.shape[0]
        if pad_len > 0:
            p = torch.cat(
                [p, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
        padded.append(p)
        answers.append(item["answer"])

    return {
        "prompt_ids": torch.stack(padded),
        "answers": answers,
        "prompt_len": min(item["prompt_ids"].shape[0] for item in batch),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: RLVRConfig) -> None:
    """Run the full RLVR training loop.

    In live mode:
    1. Set LoRA enabled, run model.eval(), generate rollouts with no_grad.
    2. Score rollouts with the verifier.
    3. Set LoRA disabled, run reference log-probs with no_grad (GRPO only).
    4. Set LoRA enabled, model.train(), compute policy log-probs with grad.
    5. Compute GRPO or REINFORCE loss; backprop.

    In offline mode:
    1. Load pre-computed rollout_ids and rewards from the dataset.
    2. Set LoRA disabled, run reference log-probs with no_grad (GRPO only).
    3. Set LoRA enabled, compute policy log-probs with grad.
    4. Compute loss; backprop.

    Args:
        config: Populated RLVRConfig instance.

    Raises:
        ImportError: If accelerate is not installed.
        ValueError: If loss_type is not recognised.
    """
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required. Install with: pip install accelerate"
        )

    if config.loss_type not in ("grpo", "reinforce"):
        raise ValueError(
            f"Unknown loss_type '{config.loss_type}'. Choose from: grpo, reinforce"
        )

    is_grpo = config.loss_type == "grpo"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

    if accelerator.is_main_process:
        logger.info(f"RLVR config: {config}")
        logger.info(f"Loss type: {config.loss_type}")
        logger.info(f"Mode: {'offline' if config.offline else 'live'}")
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
    pad_token_id = 0
    eos_token_id = None

    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or 0
            pad_token_id = tokenizer.pad_token_id
            eos_token_id = tokenizer.eos_token_id
            if accelerator.is_main_process:
                logger.info(f"Loaded tokenizer: {config.tokenizer}")
                logger.info(f"Vocab size (tokenizer): {tokenizer.vocab_size}")
        except Exception as exc:
            logger.warning(f"Could not load tokenizer '{config.tokenizer}': {exc}")
            tokenizer = None

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------
    verifier_fn: Callable[[str, str], float] | None = None
    if not config.offline:
        if config.verifier in VERIFIERS:
            verifier_fn = VERIFIERS[config.verifier]
            if accelerator.is_main_process:
                logger.info(f"Using built-in verifier: {config.verifier}")
        else:
            # Custom verifier: path/to/module.py:function_name
            try:
                verifier_fn = load_custom_verifier(config.verifier)
                if accelerator.is_main_process:
                    logger.info(f"Loaded custom verifier from: {config.verifier}")
            except Exception as exc:
                raise ValueError(
                    f"Could not load verifier '{config.verifier}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    titans_config = build_titans_config(config)
    model = create_model(config.model_type, titans_config)
    num_params = sum(p.numel() for p in model.parameters())

    if accelerator.is_main_process:
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Model parameters: {num_params:,}")

    # Load pretrained / SFT / DPO weights
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
    # LoRA wrapping — always enabled for RLVR (LoRA-as-reference trick)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Dataset and dataloader
    # ------------------------------------------------------------------
    use_streaming = False

    if config.offline:
        # Offline mode: pre-computed rollouts
        if config.dataset is not None and HAS_DATASETS and tokenizer is not None:
            train_dataset_obj: torch.utils.data.Dataset = OfflineRLVRDataset(
                dataset_name=config.dataset,
                subset=config.dataset_subset,
                tokenizer=tokenizer,
                max_len=config.seq_len,
                prompt_field=config.prompt_field,
                rollout_field=config.rollout_field,
                reward_field=config.reward_field,
                min_rollouts=2,
                seed=config.seed,
            )
            use_streaming = True
            if accelerator.is_main_process:
                logger.info(f"Offline RLVR dataset: {config.dataset}")
        else:
            if accelerator.is_main_process:
                logger.info(
                    "No offline dataset — using synthetic data for demo."
                )
            train_dataset_obj = SyntheticRLVRDataset(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                group_size=config.group_size,
                num_samples=config.synthetic_samples,
                seed=config.seed,
            )

        collate_fn = lambda b: rlvr_collate_fn(b, pad_token_id=pad_token_id)
    else:
        # Live mode: generate rollouts on the fly
        if config.dataset is not None and HAS_DATASETS and tokenizer is not None:
            train_dataset_obj = LiveRLVRDataset(
                dataset_name=config.dataset,
                subset=config.dataset_subset,
                tokenizer=tokenizer,
                max_len=config.seq_len,
                prompt_field=config.prompt_field,
                answer_field=config.answer_field,
                seed=config.seed,
            )
            use_streaming = True
            if accelerator.is_main_process:
                logger.info(f"Live RLVR dataset: {config.dataset}")
        else:
            if accelerator.is_main_process:
                logger.info(
                    "No live dataset — using synthetic data for demo."
                )
            train_dataset_obj = SyntheticRLVRDataset(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                group_size=config.group_size,
                num_samples=config.synthetic_samples,
                seed=config.seed,
            )
            # Synthetic in live mode: wrap it to look like offline
            use_streaming = False

        collate_fn = (
            (lambda b: live_collate_fn(b, pad_token_id=pad_token_id))
            if use_streaming
            else (lambda b: rlvr_collate_fn(b, pad_token_id=pad_token_id))
        )

    is_iterable = isinstance(train_dataset_obj, IterableDataset)
    train_dataloader = DataLoader(
        train_dataset_obj,
        batch_size=config.batch_size,
        shuffle=not is_iterable,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Optimizer (LoRA-only params)
    # ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
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
            0.5 * (1.0 + math.cos(math.pi * progress)),
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
    # Resume
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    memory_states: list | None = None

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

        if not config.reset_memory_per_batch:
            mem_path = resume_path.parent / f"memory_step_{global_step}.npz"
            if not mem_path.exists():
                mem_path = resume_path.parent / "memory_final.npz"
            try:
                memory_states = load_memory_states(
                    mem_path, device=accelerator.device
                )
                if accelerator.is_main_process:
                    logger.info(f"Loaded memory states from {mem_path}")
            except Exception as exc:  # noqa: BLE001
                if accelerator.is_main_process:
                    logger.info(
                        f"No memory states found ({exc}), starting fresh"
                    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ema_baseline = 0.0  # For REINFORCE

    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_optimizer_steps = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
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
                unwrapped_model = accelerator.unwrap_model(model)

                # --------------------------------------------------------
                # Live mode: generate rollouts and score them
                # --------------------------------------------------------
                if not config.offline and "answers" in batch:
                    prompt_ids = batch["prompt_ids"]
                    answers = batch["answers"]
                    prompt_len = batch.get("prompt_len", prompt_ids.shape[1])

                    # Generate rollouts (LoRA enabled, eval mode, no grad)
                    set_lora_enabled(unwrapped_model, True)
                    model.eval()
                    with torch.no_grad():
                        generated_ids, gen_logps = generate_rollouts(
                            model,
                            prompt_ids,
                            max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature,
                            num_samples=config.group_size,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                        )

                    # Score with verifier
                    if tokenizer is not None and verifier_fn is not None:
                        rewards = verify_batch(
                            generated_ids,
                            answers,
                            tokenizer,
                            verifier_fn,
                            prompt_len=int(prompt_len) if not isinstance(prompt_len, int) else prompt_len,
                            num_samples=config.group_size,
                        )
                    else:
                        # No tokenizer/verifier: use uniform random rewards (demo)
                        rewards = torch.rand(
                            prompt_ids.shape[0],
                            config.group_size,
                            device=prompt_ids.device,
                        )

                    # Compute reference log-probs (LoRA disabled = base model)
                    if is_grpo:
                        set_lora_enabled(unwrapped_model, False)
                        with torch.no_grad():
                            ref_logps = compute_log_probs_for_generated(
                                model,
                                prompt_ids,
                                generated_ids,
                                config.vocab_size,
                            )
                        set_lora_enabled(unwrapped_model, True)

                    # Compute policy log-probs with grad (LoRA enabled, train mode)
                    model.train()
                    set_lora_enabled(unwrapped_model, True)
                    policy_logps = compute_log_probs_for_generated(
                        model,
                        prompt_ids,
                        generated_ids,
                        config.vocab_size,
                    )

                # --------------------------------------------------------
                # Offline mode OR synthetic: use pre-computed rollouts
                # --------------------------------------------------------
                else:
                    prompt_ids = batch["prompt_ids"]
                    rollout_ids = batch["rollout_ids"]  # (B, G, T)
                    rewards = batch["rewards"].float()   # (B, G)
                    prompt_len = batch.get("prompt_len", prompt_ids.shape[1])

                    # Compute reference log-probs (LoRA disabled)
                    if is_grpo:
                        set_lora_enabled(unwrapped_model, False)
                        with torch.no_grad():
                            ref_logps = compute_log_probs_for_generated(
                                model,
                                prompt_ids,
                                rollout_ids,
                                config.vocab_size,
                            )
                        set_lora_enabled(unwrapped_model, True)

                    # Compute policy log-probs with grad (LoRA enabled)
                    model.train()
                    set_lora_enabled(unwrapped_model, True)
                    policy_logps = compute_log_probs_for_generated(
                        model,
                        prompt_ids,
                        rollout_ids,
                        config.vocab_size,
                    )
                    # For offline/synthetic, generated_ids = rollout_ids
                    generated_ids = rollout_ids

                # --------------------------------------------------------
                # Compute loss
                # --------------------------------------------------------
                if is_grpo:
                    loss = grpo_loss(
                        policy_logps,
                        rewards,
                        ref_logps,
                        beta=config.beta,
                        clip_ratio=config.clip_ratio,
                    )
                else:
                    # REINFORCE: collapse group dimension by mean
                    flat_logps = policy_logps.mean(dim=-1)  # (B,)
                    flat_rewards = rewards.mean(dim=-1)      # (B,)
                    loss, ema_baseline = reinforce_loss(
                        flat_logps, flat_rewards, ema_baseline
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

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            loss_val = loss.item()
            reward_val = rewards.mean().item()
            epoch_loss += loss_val
            epoch_reward += reward_val

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % config.log_every == 0 and accelerator.is_main_process:
                n = num_optimizer_steps if num_optimizer_steps > 0 else 1
                avg_loss = epoch_loss / n
                avg_reward = epoch_reward / n
                lr_val = optimizer.param_groups[0]["lr"]

                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    avg=f"{avg_loss:.4f}",
                    reward=f"{reward_val:.3f}",
                    lr=f"{lr_val:.2e}",
                    step=global_step,
                )

                if config.wandb and HAS_WANDB:
                    log_dict: dict[str, float] = {
                        "train/loss": loss_val,
                        "train/avg_loss": avg_loss,
                        "train/reward_mean": reward_val,
                        "train/avg_reward": avg_reward,
                        "train/lr": lr_val,
                    }
                    if not is_grpo:
                        log_dict["train/ema_baseline"] = ema_baseline
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

                    if (
                        not config.reset_memory_per_batch
                        and memory_states is not None
                        and any(s is not None for s in memory_states)
                    ):
                        mem_path = (
                            checkpoint_dir
                            / f"memory_step_{global_step}.npz"
                        )
                        save_memory_states(memory_states, mem_path)
                        logger.info(
                            f"Saved memory states: step {global_step}"
                        )

        # End-of-epoch summary
        if accelerator.is_main_process:
            n = num_optimizer_steps if num_optimizer_steps > 0 else 1
            logger.info(
                f"Epoch {epoch + 1} complete — avg loss: {epoch_loss / n:.4f}, "
                f"avg reward: {epoch_reward / n:.4f}, "
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
        logger.info(f"RLVR training complete. Final checkpoint: {paths[0]}")

        # Adapter-only save
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
            "beta": config.beta,
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

        if (
            not config.reset_memory_per_batch
            and memory_states is not None
            and any(s is not None for s in memory_states)
        ):
            mem_path = checkpoint_dir / "memory_final.npz"
            save_memory_states(memory_states, mem_path)
            logger.info(f"Saved final memory states to {mem_path}")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> RLVRConfig:
    """Parse command-line arguments and return an RLVRConfig."""
    parser = argparse.ArgumentParser(
        description="RLVR (Reinforcement Learning with Verifiable Rewards) "
        "for Titans PyTorch models",
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
    mca.add_argument("--mca-gate-bias-init", type=float, default=-3.0)

    # RLVR-specific
    rlvr_g = parser.add_argument_group("RLVR")
    rlvr_g.add_argument(
        "--loss-type",
        type=str,
        default="grpo",
        choices=["grpo", "reinforce"],
        help="RL loss type",
    )
    rlvr_g.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty weight (GRPO)",
    )
    rlvr_g.add_argument(
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO-style importance ratio clipping range (GRPO)",
    )
    rlvr_g.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of rollout samples per prompt (GRPO group size)",
    )
    rlvr_g.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per rollout (live mode)",
    )
    rlvr_g.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollout generation (live mode)",
    )
    rlvr_g.add_argument(
        "--verifier",
        type=str,
        default="exact_match",
        help=(
            "Verifier: 'exact_match', 'numeric_match', or "
            "'path/to/file.py:function_name' for a custom verifier"
        ),
    )

    # LoRA
    lora_g = parser.add_argument_group("LoRA")
    lora_g.add_argument(
        "--lora-rank", type=int, default=8, help="LoRA rank r"
    )
    lora_g.add_argument(
        "--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale=alpha/rank)"
    )
    lora_g.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout on LoRA input path",
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
        "--prompt-field",
        type=str,
        default="prompt",
        help="Dataset field containing the prompt text",
    )
    data.add_argument(
        "--answer-field",
        type=str,
        default="answer",
        help="Dataset field containing the ground-truth answer (live mode)",
    )
    data.add_argument("--seq-len", type=int, default=2048)

    # Offline mode
    offline_g = parser.add_argument_group("Offline mode")
    offline_g.add_argument(
        "--offline",
        action="store_true",
        help="Use pre-computed rollouts and rewards from the dataset",
    )
    offline_g.add_argument(
        "--rollout-field",
        type=str,
        default="completions",
        help="Dataset field containing pre-computed completion strings (offline mode)",
    )
    offline_g.add_argument(
        "--reward-field",
        type=str,
        default="rewards",
        help="Dataset field containing pre-computed reward scores (offline mode)",
    )

    # Training
    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--epochs", type=int, default=1)
    train_g.add_argument("--max-steps", type=int, default=-1)
    train_g.add_argument("--batch-size", type=int, default=2)
    train_g.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train_g.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (lower than SFT; RLVR is sensitive to LR)",
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
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/rlvr")
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
        help="Resume RLVR from a previous RLVR checkpoint (.pt)",
    )
    ckpt.add_argument(
        "--init-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Load pretrained or SFT weights before RLVR",
    )
    ckpt.add_argument(
        "--reset-memory-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reset Titans memory at the start of each batch. "
            "Default True because rollouts are typically independent."
        ),
    )
    ckpt.add_argument(
        "--state-carry-warmup-steps",
        type=int,
        default=0,
        help=(
            "Steps to force memory reset at training start even when "
            "--no-reset-memory-per-batch is set."
        ),
    )

    # Logging
    log = parser.add_argument_group("Logging")
    log.add_argument("--log-every", type=int, default=10)
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="titans-rlvr")
    log.add_argument("--wandb-run-name", type=str, default=None)

    # Misc
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--synthetic-samples", type=int, default=500)

    args = parser.parse_args()

    return RLVRConfig(
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
        # RLVR
        loss_type=args.loss_type,
        beta=args.beta,
        clip_ratio=args.clip_ratio,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verifier=args.verifier,
        # LoRA
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        merge_and_save=args.merge_and_save,
        # Data
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        # Offline
        offline=args.offline,
        rollout_field=args.rollout_field,
        reward_field=args.reward_field,
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
        reset_memory_per_batch=args.reset_memory_per_batch,
        state_carry_warmup_steps=args.state_carry_warmup_steps,
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
