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
