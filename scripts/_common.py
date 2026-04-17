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
from collections.abc import Iterator

import torch
import torch.nn as nn

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
    use_attn_res: bool,
) -> nn.Module:
    """Conditionally wrap ``model`` with ``torch.compile``.

    Guardrails:
    - Disabled unless ``enabled and device_type == "cuda"``.
    - Auto-disable + warn when ``use_attn_res`` is True (``process_chunk``
      contains Python control flow that Dynamo graph-breaks; see the
      source-level note in ``src/titans/models.py:process_chunk``).

    Args:
        model: The module to (optionally) compile. Usually the post-
            ``accelerator.prepare`` model on its target device.
        enabled: Opt-in flag (e.g. parsed from ``COMPILE=1``). Callers
            should default to ``False``.
        device_type: ``accelerator.device.type`` (``"cuda"``, ``"cpu"``,
            ``"mps"``, ...). Only ``"cuda"`` gets compiled.
        use_attn_res: ``config.use_attn_res``. When True we skip compile
            and log a warning instead of producing a noisy broken graph.

    Returns:
        Either the wrapped ``torch.compile`` model or the original model
        unchanged.
    """
    if not enabled:
        return model
    if device_type != "cuda":
        _log.info("torch.compile requested but device=%s; skipping.", device_type)
        return model
    if use_attn_res:
        _log.warning(
            "torch.compile skipped: use_attn_res=True is compile-hostile "
            "(process_chunk has Python control flow that graph-breaks "
            "Dynamo). Refactor tracked separately."
        )
        return model
    _log.info("Wrapping model with torch.compile(mode='default').")
    return torch.compile(model, mode="default")
