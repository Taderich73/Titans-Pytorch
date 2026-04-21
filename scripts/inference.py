#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans PyTorch models.

Usage:
    python scripts/inference.py --checkpoint checkpoints/final.pt \
        --tokenizer gpt2 --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from titans import TitansConfig, TitansMAC
from titans.checkpoint import load_checkpoint
from titans.memory_dump import load_memory_states, save_memory_states

# scripts/ is imported both as a namespace package (under pytest) and as a
# flat directory (when the script is invoked as ``python scripts/inference.py``).
try:
    from scripts._common import (  # type: ignore[import-not-found]
        MODEL_CLASSES,
        create_model,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (  # type: ignore[no-redef]
        MODEL_CLASSES,
        create_model,
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _save_final_memory(
    path: str,
    final_states,
    *,
    logger: logging.Logger,
) -> None:
    """Persist final_states to path, logging clearly when the list is empty.

    Args:
        path: Target .npz path.
        final_states: List of memory states returned from generate().
        logger: Module logger to report through.
    """
    if not final_states:
        logger.warning(
            "--save-memory=%s was set but generate() returned an empty list "
            "of final states (no memory to save). Check that your model is "
            "memory-bearing and that generate() threads states through.",
            path,
        )
        return

    save_memory_states(final_states, Path(path))
    logger.info("Saved memory state to %s", path)


def load_model(
    checkpoint_path: str, device: torch.device, variant: str = "mac",
) -> tuple[TitansMAC, TitansConfig]:
    """Load model from checkpoint (.pt or .safetensors, auto-detected).

    Uses the shared ``MODEL_CLASSES`` registry so inference stays in sync
    with the training scripts as new Titans variants land.

    Args:
        checkpoint_path: Path to a ``.pt`` or ``.safetensors`` checkpoint.
        device: Torch device to load the model onto.
        variant: Titans variant key (``mac``/``mag``/``mal``/``lmm``).

    Returns:
        Tuple of ``(model, config)`` with the model in ``eval()`` mode.
    """
    if variant not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown variant: {variant!r}. Options: {sorted(MODEL_CLASSES)}"
        )

    ckpt = load_checkpoint(checkpoint_path, device=device)

    if "config" in ckpt:
        config = TitansConfig.from_dict(ckpt["config"])
    elif "titans_config" in ckpt:
        config = TitansConfig.from_dict(ckpt["titans_config"])
    else:
        raise ValueError(
            "Checkpoint does not contain config. "
            "Pass model config via CLI args (--dim, --num-layers, etc.)"
        )

    model = create_model(variant, config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def generate(
    model: TitansMAC,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    memory_states: list | None = None,
    checkpointer=None,
) -> tuple[torch.Tensor, list]:
    """Generate tokens autoregressively, mirroring training chunking.

    Titans MAC uses segmented attention (no KV cache) with neural memory that
    updates per-chunk. During training, the model processes chunk_size tokens
    at a time — attention is local within each chunk, and memory states carry
    long-range context across chunks.

    To match this at inference:
    1. Prefill: process the prompt in chunk_size chunks (mirrors training)
    2. Decode: accumulate generated tokens into a buffer. Each step, feed the
       full buffer (up to chunk_size) so attention has local context. When the
       buffer reaches chunk_size, commit the memory update and start a new
       buffer. This avoids redundant memory updates on already-processed tokens.

    Args:
        model: Titans MAC model in eval mode.
        input_ids: Prompt token IDs, shape (1, prompt_len).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 to disable).
        memory_states: Optional initial memory states.
        checkpointer: Optional MemoryCheckpointer instance for novelty-triggered
            state checkpointing.

    Returns:
        Tuple of (generated_ids, final_states).
    """
    chunk_size = model.config.chunk_size
    generated = input_ids
    states = memory_states

    # Prefill: process full prompt, updating memory per-chunk (handled by model)
    logits, states, gate_snapshots = model(generated, states=states)
    if states is not None:
        states = [s.detach() if s is not None else None for s in states]

    # committed_states: memory after the last full chunk was processed
    # We restore from this each step so partial-buffer re-processing
    # doesn't compound memory updates
    committed_states = (
        [s.detach() if s is not None else None for s in states]
        if states
        else None
    )
    buffer_start = generated.shape[1]  # where the decode buffer begins
    chunk_idx = 0

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        # Current decode buffer: tokens generated since last committed chunk
        buffer = generated[:, buffer_start:]
        buffer_len = buffer.shape[1]

        if buffer_len >= chunk_size:
            # Full chunk ready — process and commit memory update
            chunk = buffer[:, :chunk_size]
            logits, states, gate_snapshots = model(chunk, states=committed_states)
            if states is not None:
                states = [s.detach() if s is not None else None for s in states]
            committed_states = (
                [s.detach() if s is not None else None for s in states]
                if states is not None
                else None
            )
            buffer_start += chunk_size

            if checkpointer is not None:
                checkpointer.on_chunk_commit(
                    committed_states, gate_snapshots, chunk_index=chunk_idx
                )
            chunk_idx += 1

            # If there are leftover tokens after the chunk boundary,
            # re-process them for the next logits
            if buffer_len > chunk_size:
                remainder = buffer[:, chunk_size:]
                logits, states, gate_snapshots = model(remainder, states=committed_states)
                if states is not None:
                    states = [s.detach() if s is not None else None for s in states]
        else:
            # Partial buffer — re-feed from committed state so memory
            # only sees these tokens once when the chunk completes
            logits, states, gate_snapshots = model(buffer, states=committed_states)
            if states is not None:
                states = [s.detach() if s is not None else None for s in states]

    if checkpointer is not None:
        checkpointer.flush()

    return generated, states


def main() -> None:
    parser = argparse.ArgumentParser(description="Titans inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="mac",
        choices=sorted(MODEL_CLASSES.keys()),
        help="Titans model variant to instantiate from the checkpoint",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--memory-state", type=str, default=None, help="Path to memory state .npz"
    )
    parser.add_argument(
        "--save-memory",
        type=str,
        default=None,
        help="Save memory state after inference",
    )
    parser.add_argument(
        "--auto-checkpoint", action="store_true", default=False,
        help="Enable novelty-triggered memory state checkpointing",
    )
    parser.add_argument(
        "--signal-log", action="store_true", default=False,
        help="Enable signal log (WAL) for post-hoc analysis (requires --auto-checkpoint)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")

    # Validate --checkpoint exists up front with a clear error message.
    # load_checkpoint supports extensionless paths (auto-resolves .safetensors
    # or .pt), so mirror that lookup here instead of requiring an exact file.
    ckpt_candidates = [Path(args.checkpoint)]
    if not ckpt_candidates[0].suffix:
        ckpt_candidates.extend(
            [Path(f"{args.checkpoint}.safetensors"), Path(f"{args.checkpoint}.pt")]
        )
    if not any(p.exists() for p in ckpt_candidates):
        raise FileNotFoundError(f"--checkpoint file not found: {args.checkpoint}")

    model, config = load_model(args.checkpoint, device, variant=args.model)
    logger.info(f"Model loaded: dim={config.dim}, layers={config.num_layers}")

    if args.signal_log and not args.auto_checkpoint:
        logger.warning("--signal-log has no effect without --auto-checkpoint. Ignoring.")

    checkpointer = None
    if args.auto_checkpoint:
        import hashlib
        import json as json_mod
        from titans.checkpoint_types import MemoryCheckpointConfig
        from titans.memory_checkpointer import MemoryCheckpointer
        ckpt_config = MemoryCheckpointConfig(signal_log_enabled=args.signal_log)
        config_hash = hashlib.sha256(
            json_mod.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]
        checkpointer = MemoryCheckpointer(ckpt_config, config_hash=config_hash)

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required for tokenization")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    memory_states = None
    if args.memory_state:
        memory_states = load_memory_states(
            args.memory_state, device=device, reset_for_inference=True
        )
        logger.info(f"Loaded memory state from {args.memory_state}")

    generated, final_states = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        memory_states=memory_states,
        checkpointer=checkpointer,
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output_text)

    if args.save_memory:
        _save_final_memory(args.save_memory, final_states, logger=logger)


if __name__ == "__main__":
    main()
