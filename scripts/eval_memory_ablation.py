#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""Quantify the train-vs-eval gap by ablating Titans memory initialization.

The puzzle: a typical Titans pretraining run will report training loss
at ~0.03 while eval loss settles at ~1.3 — a ~45× gap. Two plausible
explanations:

  A. **In-context memorization (architectural).** Each block's
     ``NeuralLongTermMemory`` updates *during the forward pass* via
     test-time gradient descent on the keys/values it just saw. Train
     loss is therefore measured by a model that's already adapted to
     part of that same batch. The eval path in pretrain.py
     (``_chunked_loss_fn``) deliberately starts from
     ``eval_states=None``, so it reflects predictions WITHOUT this
     in-context advantage.
  B. **Classical overfit.** The model has memorized the training
     distribution and fails to generalize.

This script disambiguates by running inference under three memory
initializations on the same held-out corpus, then printing mean
cross-entropy for each:

  1. ``fresh``   — ``eval_states=None`` per batch (matches pretrain.py
                   eval path; the canonical "honest" eval signal).
  2. ``warm``    — pre-loaded from ``memory_final.npz``, reset to that
                   state at the start of every batch. Tests whether the
                   trained inner-loop memory carries information that
                   helps prediction on unseen text.
  3. ``stream``  — pre-loaded from ``memory_final.npz`` once, then
                   carried across batches (memory continues to update).
                   Simulates a streaming inference deployment.

Interpreting the numbers:
  * ``fresh`` ≈ ``warm``  → trained memory state has no transfer value;
    the train-eval gap is dominated by inside-batch in-context
    memorization (Hypothesis A).
  * ``warm`` ≪ ``fresh``   → trained memory state is genuinely a
    learned-prior that helps unseen-text prediction. Train-eval gap is
    partly the trained-memory contribution.
  * ``stream`` ≪ ``warm``  → online memory adaptation is working and
    helping prediction across batch boundaries.

Typical usage::

    uv run python scripts/eval_memory_ablation.py \\
        --checkpoint checkpoints/final.pt \\
        --memory-dump checkpoints/memory_final.npz \\
        --tokenizer gpt2 \\
        --dataset HuggingFaceFW/fineweb-edu --dataset-subset sample-10BT \\
        --num-batches 50 --batch-size 1 --seq-len 2048

Runs on CUDA when available, falls back to CPU (much slower). For a
quick smoke test, use ``--num-batches 5``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from titans._logging import setup_logging
from titans.checkpoint import load_checkpoint
from titans.memory_dump import load_memory_states
from titans.scripts import build_titans_config, create_model

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


def _load_model_and_config(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Load the model + reconstruct config from the checkpoint.

    Mirrors what ``scripts/inference.py`` does: pull the saved
    ``config`` dict, rebuild a TitansConfig namespace via
    ``build_titans_config``, instantiate the variant, then load the
    weights.
    """
    ck = load_checkpoint(str(checkpoint_path), weights_only=False)
    if "config" not in ck:
        raise RuntimeError(
            f"checkpoint at {checkpoint_path} has no 'config' entry; "
            "cannot reconstruct the model architecture"
        )
    cfg_dict = ck["config"]

    # build_titans_config takes a duck-typed namespace — wrap the dict
    # so getattr works.
    class _Cfg:
        pass

    ns = _Cfg()
    for k, v in cfg_dict.items():
        setattr(ns, k, v)
    ns.variant = cfg_dict.get("variant", "MAC")

    titans_cfg = build_titans_config(ns)
    model = create_model(ns.variant, titans_cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device)
    model.eval()
    return model, cfg_dict


def _make_eval_corpus(
    dataset_name: str,
    subset: str | None,
    tokenizer,
    seq_len: int,
    seed: int,
    n_batches: int,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Tokenize ``n_batches`` of streaming text into fixed-length chunks.

    Returns a list of (B, T) input_id tensors on ``device``. Each
    tensor is one batch; T == seq_len; B == batch_size.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10000)

    eos = tokenizer.eos_token_id or 0
    buffer: list[int] = []
    batches: list[torch.Tensor] = []
    needed_tokens = batch_size * seq_len * n_batches

    for ex in ds:
        text = ex["text"]
        ids = tokenizer.encode(text)
        ids.append(eos)
        buffer.extend(ids)
        if len(buffer) >= needed_tokens:
            break

    buffer = buffer[:needed_tokens]
    flat = torch.tensor(buffer, dtype=torch.long, device=device)
    flat = flat.view(n_batches, batch_size, seq_len)
    for i in range(n_batches):
        batches.append(flat[i])
    return batches


def _eval_one_condition(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    chunk_size: int,
    initial_states: list | None,
    *,
    carry_across_batches: bool,
    label: str,
) -> dict:
    """Forward all batches under one memory init regime; return loss stats.

    ``initial_states`` is the per-block state list to start each batch
    from. When ``carry_across_batches=True`` the state from the previous
    batch flows into the next; otherwise we restart from
    ``initial_states`` every batch.
    """
    losses: list[float] = []
    states = copy.deepcopy(initial_states) if initial_states is not None else None
    vocab_size = model.config.vocab_size

    with torch.no_grad():
        for bi, input_ids in enumerate(batches):
            labels = input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            chunks = inputs.split(chunk_size, dim=1)
            label_chunks = labels.split(chunk_size, dim=1)

            if not carry_across_batches:
                # Reset to the initial init for every batch — this is
                # the "warm" condition where each batch independently
                # starts from the trained memory state.
                states = (
                    copy.deepcopy(initial_states)
                    if initial_states is not None
                    else None
                )

            batch_loss = 0.0
            n_chunks = len(chunks)
            for chunk_in, chunk_lbl in zip(chunks, label_chunks):
                logits, states, _ = model(chunk_in, states=states)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), chunk_lbl.reshape(-1)
                )
                batch_loss += loss.item() / n_chunks
                # Detach so memory state doesn't accumulate the autograd
                # graph across chunks (important under no_grad too on
                # some torch versions).
                if states is not None:
                    states = [s.detach() if s is not None else None for s in states]

            losses.append(batch_loss)
            if (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                logger.info(
                    f"  [{label}] batch {bi + 1}/{len(batches)}: "
                    f"loss={batch_loss:.4f}  "
                    f"running_mean={np.mean(losses):.4f}"
                )

    return {
        "label": label,
        "n_batches": len(losses),
        "mean": float(np.mean(losses)),
        "std": float(np.std(losses)),
        "min": float(np.min(losses)),
        "max": float(np.max(losses)),
        "perplexity": float(np.exp(np.mean(losses))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--memory-dump",
        type=Path,
        required=True,
        help="memory_*.npz used for the warm and stream conditions.",
    )
    parser.add_argument("--tokenizer", default="gpt2", help="HF tokenizer id.")
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset id for the eval corpus.",
    )
    parser.add_argument(
        "--dataset-subset",
        default="sample-10BT",
        help="HF dataset subset / config name (None to omit).",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048, help="Tokens per batch row."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for the chunked forward (must match training).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of eval batches per condition (more = more stable).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda / cpu / auto (default: prefer cuda when available).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    logger.info(f"Using device: {device}")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers is required. Run: uv sync --all-extras")
        sys.exit(1)

    logger.info(f"Loading model from {args.checkpoint}")
    model, cfg_dict = _load_model_and_config(args.checkpoint, device)

    logger.info(f"Loading memory dump from {args.memory_dump}")
    initial_states = load_memory_states(
        str(args.memory_dump), device=device, reset_for_inference=True
    )

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(
        f"Building eval corpus: {args.num_batches} batches × "
        f"{args.batch_size} × {args.seq_len} tokens "
        f"({args.num_batches * args.batch_size * args.seq_len:,} tokens)"
    )
    batches = _make_eval_corpus(
        args.dataset,
        args.dataset_subset,
        tokenizer,
        args.seq_len,
        args.seed,
        args.num_batches,
        args.batch_size,
        device,
    )

    logger.info("=" * 78)
    logger.info("Running condition 1/3: fresh (eval_states=None per batch)")
    fresh = _eval_one_condition(
        model,
        batches,
        args.chunk_size,
        initial_states=None,
        carry_across_batches=False,
        label="fresh",
    )

    logger.info("=" * 78)
    logger.info(
        "Running condition 2/3: warm (preloaded memory_final.npz, reset per batch)"
    )
    warm = _eval_one_condition(
        model,
        batches,
        args.chunk_size,
        initial_states=initial_states,
        carry_across_batches=False,
        label="warm",
    )

    logger.info("=" * 78)
    logger.info(
        "Running condition 3/3: stream (preloaded memory, carry across batches)"
    )
    stream = _eval_one_condition(
        model,
        batches,
        args.chunk_size,
        initial_states=initial_states,
        carry_across_batches=True,
        label="stream",
    )

    # Final report
    print()
    print("=" * 78)
    print("MEMORY ABLATION RESULTS")
    print("=" * 78)
    print(
        f"\n  Eval corpus: {args.num_batches} batches × {args.batch_size} × "
        f"{args.seq_len} tokens"
    )
    print(f"  Dataset: {args.dataset}/{args.dataset_subset}")
    print(f"  Tokenizer: {args.tokenizer}\n")
    print(
        f"  {'condition':<10} {'mean':>10} {'std':>10} {'min':>10} "
        f"{'max':>10} {'pp':>10}"
    )
    for r in (fresh, warm, stream):
        print(
            f"  {r['label']:<10} {r['mean']:>10.4f} {r['std']:>10.4f} "
            f"{r['min']:>10.4f} {r['max']:>10.4f} {r['perplexity']:>10.3f}"
        )
    print()

    delta_warm = fresh["mean"] - warm["mean"]
    delta_stream = warm["mean"] - stream["mean"]
    print(f"  Δ fresh→warm   = {delta_warm:+.4f}")
    print(f"  Δ warm→stream  = {delta_stream:+.4f}")
    print()

    # Interpretation hint
    if abs(delta_warm) < 0.05:
        print(
            "  Verdict: trained memory state has no detectable transfer\n"
            "  value on this corpus → train-eval gap is dominated by\n"
            "  in-batch in-context memorization (Hypothesis A)."
        )
    elif delta_warm > 0.05:
        print(
            "  Verdict: trained memory state IS a useful prior for unseen\n"
            "  text. Some of the train-eval gap is the trained-memory\n"
            "  contribution; remainder is whatever fresh→warm doesn't\n"
            "  close."
        )
    else:
        print(
            "  Verdict: trained memory state HURTS prediction on unseen\n"
            "  text — likely overfit to training corpus statistics."
        )


if __name__ == "__main__":
    main()
