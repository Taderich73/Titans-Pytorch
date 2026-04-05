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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[TitansMAC, TitansConfig]:
    """Load model from checkpoint (.pt or .safetensors, auto-detected)."""
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

    model = TitansMAC(config)
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
    """
    chunk_size = model.config.chunk_size
    generated = input_ids
    states = memory_states

    # Prefill: process full prompt, updating memory per-chunk (handled by model)
    logits, states = model(generated, states=states)
    if states is not None:
        states = [s.detach() for s in states]

    # committed_states: memory after the last full chunk was processed
    # We restore from this each step so partial-buffer re-processing
    # doesn't compound memory updates
    committed_states = [s.detach() for s in states] if states else None
    buffer_start = generated.shape[1]  # where the decode buffer begins

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
            logits, states = model(chunk, states=committed_states)
            states = [s.detach() for s in states]
            committed_states = [s.detach() for s in states]
            buffer_start += chunk_size

            # If there are leftover tokens after the chunk boundary,
            # re-process them for the next logits
            if buffer_len > chunk_size:
                remainder = buffer[:, chunk_size:]
                logits, states = model(remainder, states=committed_states)
                states = [s.detach() for s in states]
        else:
            # Partial buffer — re-feed from committed state so memory
            # only sees these tokens once when the chunk completes
            logits, states = model(buffer, states=committed_states)
            states = [s.detach() for s in states]

    return generated, states


def main() -> None:
    parser = argparse.ArgumentParser(description="Titans inference")
    parser.add_argument("--checkpoint", type=str, required=True)
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

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    logger.info(f"Model loaded: dim={config.dim}, layers={config.num_layers}")

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required for tokenization")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    memory_states = None
    if args.memory_state:
        memory_states = load_memory_states(args.memory_state, device=device)
        logger.info(f"Loaded memory state from {args.memory_state}")

    generated, final_states = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        memory_states=memory_states,
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output_text)

    if args.save_memory and final_states:
        save_memory_states(final_states, Path(args.save_memory))
        logger.info(f"Saved memory state to {args.save_memory}")


if __name__ == "__main__":
    main()
