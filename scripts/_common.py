# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Shared helpers for training/eval scripts.

This module is the landing pad for DRY consolidation across the
sft / lora / dpo / rlvr scripts. It currently exposes only
``chunked_forward``; future plans will add tokenize_chat,
build_titans_config, and create_model helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_log = logging.getLogger(__name__)


def chunked_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    states: list | None = None,
    detach_between: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list | None]]:
    """Iterate chunks of ``input_ids`` through a Titans model.

    Mirrors the canonical chunked-training loop used in sft.py, lora.py,
    dpo.py::compute_log_probs, and rlvr.py::compute_token_log_probs.

    Args:
        model: Titans model returning ``(logits, states, *_)`` from its
            ``forward``. ``model`` may be a DDP/Accelerate wrapper — the
            helper does not peek at ``.module``.
        input_ids: Shape (B, T). Split along dim=1 into chunks of
            ``chunk_size`` tokens (final chunk may be shorter).
        chunk_size: Per-chunk token count along the sequence dimension.
        states: Optional initial memory states; ``None`` means start fresh.
        detach_between: If True (default), call ``.detach()`` on state
            tensors between chunks (truncated BPTT, matches the SFT/LoRA
            training loops). Set False when gradients must flow through
            the full fused sequence (e.g. DPO compute_log_probs, or
            RLVR log-prob recomputation that concatenates all chunk
            logits before backward).

    Yields:
        Per-chunk tuples ``(logits, chunk_input_ids, chunk_states)``:
        - ``logits`` has shape (B, chunk_len, V).
        - ``chunk_input_ids`` has shape (B, chunk_len).
        - ``chunk_states`` is the post-chunk memory state (detached when
          ``detach_between=True``).
    """
    id_chunks = input_ids.split(chunk_size, dim=1)
    for chunk_ids in id_chunks:
        logits, states, _ = model(chunk_ids, states=states)
        if detach_between and states is not None:
            states = [
                s.detach() if s is not None else None for s in states
            ]
        yield logits, chunk_ids, states


def maybe_compile(
    model: nn.Module,
    *,
    enabled: bool,
    device_type: str,
    use_attn_res: bool = False,
) -> nn.Module:
    """Conditionally wrap ``model`` with ``torch.compile``.

    Guardrails:
    - Disabled unless ``enabled and device_type == "cuda"``.

    Args:
        model: The module to (optionally) compile. Usually the post-
            ``accelerator.prepare`` model on its target device.
        enabled: Opt-in flag (e.g. parsed from ``COMPILE=1``). Callers
            should default to ``False``.
        device_type: ``accelerator.device.type`` (``"cuda"``, ``"cpu"``,
            ``"mps"``, ...). Only ``"cuda"`` gets compiled.
        use_attn_res: Deprecated, kept for call-site compatibility. The
            AttnRes sub-layer schedule is now compile-compatible
            (``src/titans/models.py::_build_attnres_schedule`` makes the
            loop a construction-time constant that Dynamo unrolls). The
            flag is ignored.

    Returns:
        Either the wrapped ``torch.compile`` model or the original model
        unchanged.
    """
    del use_attn_res  # retained in signature for back-compat; now a no-op
    if not enabled:
        return model
    if device_type != "cuda":
        _log.info("torch.compile requested but device=%s; skipping.", device_type)
        return model
    _log.info("Wrapping model with torch.compile(mode='default').")
    return torch.compile(model, mode="default")


def make_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    device_type: str,
    *,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    _force_fused_flag: bool = False,
) -> torch.optim.AdamW:
    """Return an AdamW optimizer with the fastest safe kernel.

    On CUDA, pass ``fused=True`` (5-10% step-time improvement). On CPU or
    MPS, fall back to the default (foreach) kernel.

    Args:
        params: Iterable of parameters to optimize.
        lr: Peak learning rate.
        weight_decay: Decoupled weight-decay coefficient.
        device_type: ``accelerator.device.type`` (``"cuda"``, ``"cpu"``, ...).
            Only ``"cuda"`` selects the fused kernel.
        betas: Adam moment decay rates.
        eps: Denominator epsilon.
        _force_fused_flag: Test hook — forces ``fused=True`` regardless of
            CUDA availability. Do not use in production code.

    Returns:
        A configured ``torch.optim.AdamW`` instance.
    """
    kwargs: dict = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
    }
    use_fused = device_type == "cuda" and (
        _force_fused_flag or torch.cuda.is_available()
    )
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(list(params), **kwargs)


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int = 4,
    device_type: str = "cpu",
    shuffle: bool = False,
    drop_last: bool = True,
    streaming: bool = False,
    collate_fn=None,
) -> DataLoader:
    """Construct a DataLoader with sensible throughput defaults.

    - ``pin_memory=True`` on CUDA host transfers.
    - ``persistent_workers=True`` when ``num_workers > 0``.
    - ``prefetch_factor=2`` when ``num_workers > 0``.
    - Streaming/iterable datasets force ``num_workers=0`` (forking iterable
      datasets duplicates the stream). ``shuffle`` is also omitted in the
      streaming path because ``DataLoader`` rejects it for ``IterableDataset``.

    Args:
        dataset: The source dataset. Map-style or iterable.
        batch_size: Per-batch sample count.
        num_workers: Requested worker count. Ignored when ``streaming=True``.
        device_type: ``accelerator.device.type`` (``"cuda"`` enables pinning).
        shuffle: Whether to shuffle (map-style only).
        drop_last: Drop the final partial batch.
        streaming: True for ``IterableDataset`` sources.
        collate_fn: Optional collate override.

    Returns:
        A configured ``torch.utils.data.DataLoader``.
    """
    effective_workers = 0 if streaming else num_workers
    pin_memory = device_type == "cuda"
    kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": effective_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if effective_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if not streaming:
        kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# ChatML constants and formatting
# ---------------------------------------------------------------------------

CHATML_IM_START = "<|im_start|>"
CHATML_IM_END = "<|im_end|>"


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Format a list of message dicts into a ChatML string.

    Args:
        messages: List of dicts with ``role`` and ``content`` keys. Missing
            ``role`` defaults to ``user``; missing ``content`` defaults to
            the empty string.

    Returns:
        A single string with all turns formatted in ChatML markup,
        including a trailing newline after each ``<|im_end|>``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{CHATML_IM_START}{role}\n{content}{CHATML_IM_END}\n")
    return "".join(parts)

# ---------------------------------------------------------------------------
# Loss-mask helpers
# ---------------------------------------------------------------------------


def build_loss_mask(
    seq_len: int,
    assistant_content_spans: list[tuple[int, int]],
    include_eos: bool = True,
    eos_positions: list[int] | None = None,
    train_on_all: bool = False,
) -> list[int]:
    """Build a per-token binary loss mask.

    Canonical consolidated form. Byte-identical behaviour to the pre-
    consolidation ``build_loss_mask`` functions in ``sft.py`` and ``dpo.py``.

    Args:
        seq_len: Total sequence length (after shifting for next-token prediction).
        assistant_content_spans: List of (start, end) token index pairs
            marking assistant-turn content in the shifted label sequence.
            End is exclusive; spans are clamped to ``seq_len``.
        include_eos: Whether to include EOS tokens that follow assistant turns.
        eos_positions: Positions of EOS tokens after assistant turns.
        train_on_all: If True, return all-ones regardless of spans.

    Returns:
        List of 0/1 ints of length ``seq_len``.
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


def loss_mask_to_zero_one(labels: list[int]) -> list[int]:
    """Convert a labels list with -100 sentinels into a 0/1 loss mask.

    Replaces lora.py's reduced ``build_loss_mask`` variant.

    Args:
        labels: Token labels where -100 means "masked, do not train".

    Returns:
        List of 0/1 ints: 0 iff label == -100, else 1.
    """
    return [0 if tok == -100 else 1 for tok in labels]


# ---------------------------------------------------------------------------
# tokenize_chat
# ---------------------------------------------------------------------------


def tokenize_chat(
    messages: list[dict],
    tokenizer,  # transformers.PreTrainedTokenizerBase when installed
    max_len: int,
    train_on_all: bool = False,
) -> dict[str, list[int]]:
    """Tokenize a ChatML conversation and produce input_ids + labels + mask.

    Uses ``tokenizer.apply_chat_template`` when the tokenizer provides a
    chat template; otherwise falls back to ChatML markup (``format_chatml``).
    Identifies assistant turns so non-assistant tokens can be masked out
    of the loss.

    Output is shifted for next-token prediction:
        input_ids = tokens[:-1]
        labels    = tokens[1:]
        loss_mask = mask[1:]

    Args:
        messages: List of role/content dicts.
        tokenizer: HuggingFace-style tokenizer.
        max_len: Sequences are truncated to at most ``max_len`` tokens
            before shifting.
        train_on_all: If True, every output position is supervised.

    Returns:
        Dict with keys ``input_ids``, ``labels``, ``loss_mask``.
        All lists of ints of equal length ``<= max_len - 1``.
    """
    use_native_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )

    if use_native_template:
        full_ids: list[int] = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )

        assistant_spans: list[tuple[int, int]] = []
        eos_after_assistant: list[int] = []

        if not train_on_all:
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                prefix_turns = messages[:i]
                if prefix_turns:
                    prefix_ids = tokenizer.apply_chat_template(
                        prefix_turns, tokenize=True, add_generation_prompt=True,
                    )
                else:
                    prefix_ids = tokenizer.encode(
                        f"{CHATML_IM_START}assistant\n", add_special_tokens=False,
                    )
                content_start = len(prefix_ids)
                through_ids = tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=True, add_generation_prompt=False,
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
                header = f"{CHATML_IM_START}{role}\n"
                footer = f"{CHATML_IM_END}\n"
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

    shifted_spans = [(max(0, s - 1), max(0, e - 1)) for s, e in assistant_spans]
    shifted_eos = [max(0, p - 1) for p in eos_after_assistant]

    loss_mask = build_loss_mask(
        seq_len=len(labels),
        assistant_content_spans=shifted_spans,
        include_eos=True,
        eos_positions=shifted_eos,
        train_on_all=train_on_all,
    )

    return {"input_ids": input_ids, "labels": labels, "loss_mask": loss_mask}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL  # noqa: E402

MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mac": TitansMAC,
    "mag": TitansMAG,
    "mal": TitansMAL,
    "lmm": TitansLMM,
}


def create_model(variant: str, config: TitansConfig) -> nn.Module:
    """Instantiate a Titans model by variant name.

    Args:
        variant: One of ``mac``, ``mag``, ``mal``, ``lmm``.
        config: Fully-populated ``TitansConfig``.

    Returns:
        Initialised (but untrained) model instance.

    Raises:
        ValueError: If ``variant`` is not a known key.
    """
    if variant not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown variant: {variant!r}. Options: {sorted(MODEL_CLASSES)}"
        )
    return MODEL_CLASSES[variant](config)
