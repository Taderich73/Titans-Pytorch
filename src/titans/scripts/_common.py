# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Shared helpers for training / inference scripts.

This module is the single source of truth for code that would otherwise
drift across sft / lora / dpo / rlvr / pretrain / inference.

Forward (Plan 2):
    chunked_forward, compute_log_probs, compute_token_log_probs,
    and memory-plumbing helpers.

Optimiser / dataloader (Plan 8):
    make_optimizer, make_dataloader, maybe_compile.

DRY consolidation (Plan 3):
    - CHATML_IM_START, CHATML_IM_END: canonical ChatML markers.
    - format_chatml: list[dict] -> str in ChatML markup.
    - build_loss_mask: span-based 0/1 mask builder (sft/dpo canonical form).
    - loss_mask_to_zero_one: adapter for lora's -100-sentinel labels.
    - tokenize_chat: ChatML tokenisation + shift + loss-mask in one call.
    - MODEL_CLASSES / create_model: Titans variant registry.
    - build_titans_config: duck-typed config -> TitansConfig.
    - base_argparse_parser: argparse skeleton shared by all training scripts.
    - init_accelerator_and_logging: Accelerator + logging setup bundle.
    - setup_checkpoint_dir: output dir creation + resume-path resolution.

Do not add helpers here that belong in the library (``src/titans``); this
module is only for script-level orchestration glue.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Iterator
from typing import Any, cast

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
            states = [s.detach() if s is not None else None for s in states]
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
    # torch.compile returns an OptimizedModule (runtime subclass of nn.Module)
    # whose stub types as a callable; cast to the declared return type.
    return cast(nn.Module, torch.compile(model, mode="default"))


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


def move_optimizer_state_to_params(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Migrate optimizer state tensors onto each parameter's device + dtype.

    Why this exists:
        ``torch.optim.Optimizer.load_state_dict`` does not coerce state
        tensor device or dtype to match the live parameters — unlike
        ``nn.Module.load_state_dict``. On a resume path that calls
        ``optimizer.load_state_dict(...)`` *after* ``accelerator.prepare``
        has moved parameters onto the accelerator (e.g. CUDA) and
        ``load_checkpoint`` has surfaced the saved payload on CPU (the
        default), the optimizer ends up with stale ``exp_avg`` /
        ``exp_avg_sq`` tensors that don't match the live quartet of
        ``(params, grads, exp_avgs, exp_avg_sqs)``. The fused Adam /
        AdamW kernel enforces an identity constraint across all four and
        raises::

            RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must
            have same dtype, device, and layout

        This helper walks every registered param, looks up its state
        slot, and coerces each tensor-valued entry (``exp_avg``,
        ``exp_avg_sq``, ``max_exp_avg_sq``, and ``step`` when it is a
        tensor on some PyTorch builds) onto the param's device. For
        floating-point state entries it also coerces dtype to match the
        param — covering the case where a checkpoint was saved under a
        different precision config or where ``accelerate`` stores master
        parameters in bf16 rather than fp32.

        Integer state entries (e.g. ``step``) keep their original dtype
        to preserve counter semantics.

    Safe to call even when no state has been loaded: iterates registered
    params only and no-ops on any param whose state slot is missing or
    empty.

    Args:
        optimizer: A ``torch.optim.Optimizer`` (or the
            ``accelerate.optimizer.AcceleratedOptimizer`` wrapper, whose
            ``.state`` and ``.param_groups`` proxy through).

    Returns:
        ``(migrated, seen)`` — count of state tensors whose device or
        dtype changed, and the total tensor-valued state entries visited.
        Callers can log these to confirm the helper ran on the resume
        path, which is essential when diagnosing whether a downstream
        optimizer crash is caused by stale cached code on the runner.
    """
    migrated = 0
    seen = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state:
                continue
            for key, value in list(state.items()):
                if not torch.is_tensor(value):
                    continue
                seen += 1
                # For floating-point state tensors, coerce both device
                # and dtype to the param's. For integer state (e.g. the
                # AdamW ``step`` counter on newer PyTorch builds), only
                # coerce device to preserve int semantics.
                target_dtype = p.dtype if value.is_floating_point() else value.dtype
                if value.device != p.device or value.dtype != target_dtype:
                    state[key] = value.to(device=p.device, dtype=target_dtype)
                    migrated += 1
    return migrated, seen


def initialize_missing_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Eagerly populate Adam/AdamW state for any param missing exp_avg entries.

    Why this exists:
        ``optimizer.load_state_dict`` is a positional replay of the saved
        state dict. If the saved checkpoint contains state for only a
        subset of the current live params (e.g. a param that was frozen or
        never received gradients during the run that produced the
        checkpoint), those live params end up with an empty state slot
        after load. On the first sync-gradient step the base Adam step
        function synthesizes fresh ``exp_avg`` / ``exp_avg_sq`` inside
        ``_init_group`` and then hands the full param list to
        ``_fused_adam``. Mixing these freshly-synthesized state tensors
        with the restored-state tensors inside a single fused kernel call
        trips the kernel's identity check with::

            RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must
            have same dtype, device, and layout

        even though inspection shows uniform device / dtype / layout
        across all four — the fused kernel is stricter than the eager
        path about stride/memory-format alignment between freshly
        allocated and checkpoint-loaded tensors.

        This helper walks every param in the optimizer and, for any that
        lacks ``exp_avg`` / ``exp_avg_sq``, creates them eagerly using
        the same ``torch.zeros_like(p, memory_format=torch.preserve_format)``
        recipe that ``torch.optim.adam._init_group`` would use on first
        grad. After this runs, fused Adam sees uniform state across the
        whole param list from step one onward.

    The ``step`` counter is seeded to match the eager branch's choice of
    device-tensor (when ``fused`` or ``capturable``) vs CPU scalar,
    preserving the invariant the eager path relies on when
    differentiating capturable/fused Adam from the baseline.

    Args:
        optimizer: A ``torch.optim.Optimizer`` (or the
            ``accelerate.optimizer.AcceleratedOptimizer`` wrapper).
            Assumed to be Adam-family — the synthesized state keys match
            ``torch.optim.Adam`` and ``AdamW``.

    Returns:
        ``(initialized, seen)`` — count of params for which fresh state
        was created, and total params visited. On a clean resume from a
        compatible checkpoint ``initialized`` will be 0; any nonzero
        value on resume indicates the loaded checkpoint was missing
        state for some now-live params.
    """
    initialized = 0
    seen = 0
    for group in optimizer.param_groups:
        fused = bool(group.get("fused", False))
        capturable = bool(group.get("capturable", False))
        for p in group["params"]:
            seen += 1
            state = optimizer.state[p]
            if "exp_avg" in state and "exp_avg_sq" in state:
                continue
            # Match the eager-path _init_group recipe for Adam/AdamW.
            # ``step`` lives on-device as a 0-d tensor when fused or
            # capturable; otherwise it's a CPU scalar tensor. exp_avg
            # and exp_avg_sq mirror the param exactly.
            if fused or capturable:
                state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            else:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            initialized += 1
    return initialized, seen


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int = 4,
    device_type: str = "cpu",
    shuffle: bool = False,
    drop_last: bool = True,
    streaming: bool = False,
    collate_fn: Any = None,
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
    tokenizer: Any,  # transformers.PreTrainedTokenizerBase when installed
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
                    prefix_ids = tokenizer.apply_chat_template(
                        prefix_turns,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                else:
                    prefix_ids = tokenizer.encode(
                        f"{CHATML_IM_START}assistant\n",
                        add_special_tokens=False,
                    )
                content_start = len(prefix_ids)
                through_ids = tokenizer.apply_chat_template(
                    messages[: i + 1],
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

from titans import (  # noqa: E402
    TitansConfig,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)

MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "mac": TitansMAC,
    "mag": TitansMAG,
    "mal": TitansMAL,
    "lmm": TitansLMM,
}


def build_titans_config(cfg: Any) -> TitansConfig:
    """Translate any duck-typed training config into a ``TitansConfig``.

    This is the canonical builder used by sft / lora / dpo / rlvr / pretrain.
    Feature sub-groups (TNT / AttnRes / adaptive window / MCA) are only
    forwarded when their top-level toggle (e.g. ``cfg.use_tnt``) is truthy.

    Includes fields added in Plan 5 (``num_memory_inner_steps``,
    ``mac_per_position_memory_query``) when both the caller config and
    ``TitansConfig`` expose them.

    Args:
        cfg: Any object with the expected attribute names (a dataclass
            such as ``SFTConfig`` / ``DPOConfig`` / ``RLVRConfig`` /
            ``LoRATrainingConfig``, or an argparse Namespace).

    Returns:
        A fully populated ``TitansConfig`` ready to hand to ``create_model``.
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
    if getattr(cfg, "use_tnt", False):
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

    # Attention-residual
    if getattr(cfg, "use_attn_res", False):
        kwargs.update(
            use_attn_res=cfg.use_attn_res,
            num_attnres_blocks=cfg.num_attnres_blocks,
            attnres_warmup_steps=cfg.attnres_warmup_steps,
            attnres_modulate_global_memory=cfg.attnres_modulate_global_memory,
            attnres_modulate_local_memory=cfg.attnres_modulate_local_memory,
        )

    # Adaptive window
    if getattr(cfg, "adaptive_window", False):
        kwargs.update(
            adaptive_window=cfg.adaptive_window,
            adaptive_window_min=cfg.adaptive_window_min,
            adaptive_window_max=cfg.adaptive_window_max,
            adaptive_window_temperature=cfg.adaptive_window_temperature,
            adaptive_window_lambda=cfg.adaptive_window_lambda,
        )

    # MCA
    if getattr(cfg, "use_mca", False):
        kwargs.update(
            use_mca=cfg.use_mca,
            mca_num_heads=cfg.mca_num_heads,
            mca_gate_type=cfg.mca_gate_type,
            mca_gate_bias_init=cfg.mca_gate_bias_init,
        )
        if cfg.mca_insertion_layers:
            kwargs["mca_insertion_layers"] = cfg.mca_insertion_layers

    # Plan 5 additions (guarded by hasattr on TitansConfig)
    for extra in ("num_memory_inner_steps", "mac_per_position_memory_query"):
        if hasattr(cfg, extra) and extra in TitansConfig.__dataclass_fields__:
            kwargs[extra] = getattr(cfg, extra)

    return TitansConfig(**kwargs)


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


# ---------------------------------------------------------------------------
# base_argparse_parser
# ---------------------------------------------------------------------------


def base_argparse_parser(description: str) -> argparse.ArgumentParser:
    """Return an ``argparse.ArgumentParser`` pre-populated with the flags
    every Titans training/inference script shares.

    Calling scripts add script-specific flags by calling
    ``parser.add_argument(...)`` or
    ``parser.add_argument_group(...).add_argument(...)`` on the returned parser.

    Groups:
        - "Model architecture": --model, --dim, --num-heads, --num-layers,
          --vocab-size, --chunk-size, --window-size, --rope-proportion,
          --num-persistent-tokens, --num-memory-layers, --memory-objective,
          --huber-delta-init, --dropout, --use-conv
        - "TNT / hierarchical memory": --use-tnt, --global-chunk-size,
          --local-chunk-sizes, --local-shard-length, --use-qk-projection,
          --tnt-stage, --finetune-local-chunk-sizes
        - "Attention residual": --use-attn-res, --num-attnres-blocks,
          --attnres-warmup-steps, --attnres-modulate-global-memory,
          --no-attnres-modulate-global-memory,
          --attnres-modulate-local-memory
        - "Adaptive window": --adaptive-window, --adaptive-window-min,
          --adaptive-window-max, --adaptive-window-temperature,
          --adaptive-window-lambda
        - "MCA": --use-mca, --mca-insertion-layers, --mca-num-heads,
          --mca-gate-type, --mca-gate-bias-init
        - "Training": --epochs, --max-steps, --batch-size,
          --gradient-accumulation-steps, --lr, --weight-decay, --grad-clip,
          --warmup-ratio, --mixed-precision, --num-workers, --pin-memory,
          --persistent-workers
        - "Checkpointing": --checkpoint-dir, --save-every, --save-format,
          --resume, --init-weights
        - "Logging": --log-every, --wandb, --wandb-project, --wandb-run-name
        - "Misc": --seed, --deterministic

    Args:
        description: Passed through to ``ArgumentParser(description=...)``.

    Returns:
        Parser with shared flags. Callers set their own ``--dataset`` /
        script-specific flags on top.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    arch.add_argument("--rope-proportion", type=float, default=1.0)
    arch.add_argument("--num-persistent-tokens", type=int, default=16)
    arch.add_argument("--num-memory-layers", type=int, default=2)
    arch.add_argument(
        "--memory-objective",
        type=str,
        default="l2",
        choices=["l2", "huber"],
    )
    arch.add_argument("--huber-delta-init", type=float, default=0.0)
    arch.add_argument("--dropout", type=float, default=0.0)
    arch.add_argument("--use-conv", action="store_true")

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

    attn = parser.add_argument_group("Attention residual")
    attn.add_argument("--use-attn-res", action="store_true")
    attn.add_argument("--num-attnres-blocks", type=int, default=8)
    attn.add_argument("--attnres-warmup-steps", type=int, default=0)
    attn.add_argument(
        "--attnres-modulate-global-memory",
        action="store_true",
        default=True,
    )
    attn.add_argument(
        "--no-attnres-modulate-global-memory",
        dest="attnres_modulate_global_memory",
        action="store_false",
    )
    attn.add_argument("--attnres-modulate-local-memory", action="store_true")

    aw = parser.add_argument_group("Adaptive window")
    aw.add_argument("--adaptive-window", action="store_true")
    aw.add_argument("--adaptive-window-min", type=int, default=64)
    aw.add_argument("--adaptive-window-max", type=int, default=None)
    aw.add_argument("--adaptive-window-temperature", type=float, default=10.0)
    aw.add_argument("--adaptive-window-lambda", type=float, default=0.01)

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
    train_g.add_argument("--num-workers", type=int, default=0)
    train_g.add_argument("--pin-memory", action="store_true", default=False)
    train_g.add_argument(
        "--persistent-workers",
        action="store_true",
        default=False,
    )

    ckpt = parser.add_argument_group("Checkpointing")
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/run")
    ckpt.add_argument("--save-every", type=int, default=1000)
    ckpt.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
    )
    ckpt.add_argument("--resume", type=str, default=None, metavar="PATH")
    ckpt.add_argument("--init-weights", type=str, default=None, metavar="PATH")

    log = parser.add_argument_group("Logging")
    log.add_argument("--log-every", type=int, default=10)
    log.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "Root-logger level for structured (rich) console output. "
            "Scripts install a RichHandler on the main process only, so "
            "raising this to WARNING silences per-step info chatter on "
            "all ranks at once."
        ),
    )
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="titans")
    log.add_argument("--wandb-run-name", type=str, default=None)

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Enable fully deterministic kernels via "
            "torch.use_deterministic_algorithms(True) and "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU-only runs are already "
            "deterministic; this flag is mostly a belt-and-suspenders "
            "switch for CUDA at a moderate speed cost. See "
            "docs/reproducibility.md for the full contract."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Accelerator + logging setup
# ---------------------------------------------------------------------------

import importlib.util
from dataclasses import dataclass

try:
    from accelerate import Accelerator

    _HAS_ACCELERATE = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_ACCELERATE = False

_HAS_WANDB = importlib.util.find_spec("wandb") is not None


@dataclass
class AcceleratorBundle:
    """Tuple-ish return value of ``init_accelerator_and_logging``."""

    accelerator: Any
    logger: logging.Logger
    is_main_process: bool
    has_wandb: bool


def init_accelerator_and_logging(cfg: Any) -> AcceleratorBundle:
    """Initialize the Accelerate runtime and a stdlib logger.

    Installs a :class:`rich.logging.RichHandler` on the root logger of
    the **main process only** (avoids interleaved output from rank->0
    workers under multi-GPU). Non-main processes get the standard-
    library default (no rich formatting) but still have ``logger.info``
    etc. available for wandb.log / sync points.

    Args:
        cfg: Object with attributes ``gradient_accumulation_steps`` (int),
            ``mixed_precision`` (str in {"no","fp16","bf16"}), ``wandb``
            (bool). An optional ``log_level`` attribute (string or int)
            is honoured on the main process; defaults to ``"INFO"``.

    Returns:
        ``AcceleratorBundle`` with the accelerator instance (or a CPU
        stub), a module-level logger, ``is_main_process`` flag and a
        ``has_wandb`` flag indicating whether wandb logging is available.
    """
    if _HAS_ACCELERATE:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with="wandb" if getattr(cfg, "wandb", False) and _HAS_WANDB else None,
        )
        is_main = accelerator.is_main_process
    else:

        class _Stub:
            is_main_process = True
            device = "cpu"

            def prepare(self, *args):
                return args if len(args) > 1 else args[0]

            def print(self, *args, **kwargs):
                print(*args, **kwargs)

        accelerator = _Stub()
        is_main = True

    # Rich logging only on the main process so multi-rank runs do not
    # interleave per-step info output. Non-main ranks fall back to the
    # stdlib default and are effectively silent at WARNING+.
    log_level = getattr(cfg, "log_level", "INFO")
    if is_main:
        from titans._logging import setup_logging

        setup_logging(log_level)
    else:
        root = logging.getLogger()
        # Raise the bar so non-main ranks stay quiet by default.
        if not root.handlers:
            logging.basicConfig(level=logging.WARNING, format="%(message)s")
        else:
            root.setLevel(logging.WARNING)
    logger = logging.getLogger("scripts")

    return AcceleratorBundle(
        accelerator=accelerator,
        logger=logger,
        is_main_process=is_main,
        has_wandb=_HAS_WANDB,
    )


# ---------------------------------------------------------------------------
# setup_checkpoint_dir
# ---------------------------------------------------------------------------

import re
from pathlib import Path


@dataclass
class CheckpointSetup:
    """Return type for setup_checkpoint_dir."""

    output_dir: Path
    resume_path: Path | None
    resume_step: int


def setup_checkpoint_dir(
    output_dir: str,
    resume_path: str | None = None,
) -> CheckpointSetup:
    """Create the output directory (if missing) and resolve a resume path.

    Args:
        output_dir: Where future checkpoints will be written.
        resume_path: Optional explicit checkpoint file to resume from.
            Must exist if provided.

    Returns:
        ``CheckpointSetup`` with the resolved output path, the resume
        checkpoint (if any), and the step number parsed from the
        filename (``step_123.pt`` -> 123; unparseable -> 0).

    Raises:
        FileNotFoundError: If ``resume_path`` is given but the file does
            not exist.
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    resume: Path | None = None
    step = 0
    if resume_path is not None:
        resume = Path(resume_path).expanduser()
        if not resume.exists():
            raise FileNotFoundError(f"--resume file not found: {resume}")
        m = re.search(r"step[_-]?(\d+)", resume.stem)
        if m:
            step = int(m.group(1))

    return CheckpointSetup(output_dir=out, resume_path=resume, resume_step=step)
