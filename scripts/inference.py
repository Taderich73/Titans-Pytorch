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
    """Generate tokens autoregressively."""
    generated = input_ids
    states = memory_states

    for _ in range(max_new_tokens):
        logits, states = model(generated, states=states)
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        if states is not None:
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
