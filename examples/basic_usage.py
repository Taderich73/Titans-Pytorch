#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Basic usage examples for Titans PyTorch models.

Demonstrates all four model variants (MAC, MAG, MAL, LMM),
configuration, and streaming inference with memory state persistence.

Run with:
    uv run python examples/basic_usage.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL


def example_basic_forward() -> None:
    """Basic forward pass with TitansMAC."""
    print("\n" + "=" * 60)
    print("Example: Basic Forward Pass (TitansMAC)")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=64,
        num_persistent_tokens=8,
        num_memory_layers=2,
    )

    model = TitansMAC(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Single chunk forward pass
    batch_size = 2
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.chunk_size))

    logits, states = model(input_ids)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Memory states: {len(states)} layers")


def example_all_variants() -> None:
    """Forward pass with all four model variants."""
    print("\n" + "=" * 60)
    print("Example: All Model Variants")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
    )

    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    variants: list[tuple[str, type]] = [
        ("MAC (Memory as Context)", TitansMAC),
        ("MAG (Memory as Gate)", TitansMAG),
        ("MAL (Memory as Layer)", TitansMAL),
        ("LMM (Memory Only)", TitansLMM),
    ]

    for name, model_cls in variants:
        model = model_cls(config)
        logits, states = model(input_ids)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: logits={logits.shape}, params={params:,}")


def example_streaming() -> None:
    """Process a long sequence in streaming chunks with memory persistence."""
    print("\n" + "=" * 60)
    print("Example: Streaming with Memory Persistence")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=64,
    )

    model = TitansMAC(config)
    model.eval()

    num_chunks = 4
    states = None

    with torch.no_grad():
        for i in range(num_chunks):
            chunk = torch.randint(0, config.vocab_size, (1, config.chunk_size))
            logits, states = model(chunk, states=states)
            print(
                f"  Chunk {i + 1}: processed {config.chunk_size} tokens, "
                f"memory weight norm = {states[0].weights[0].norm():.4f}"
            )

    print(f"Total tokens processed: {num_chunks * config.chunk_size}")
    print("Memory state threads across all chunks.")


def example_training_step() -> None:
    """A single training step with gradient computation."""
    print("\n" + "=" * 60)
    print("Example: Training Step")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=1,
        vocab_size=256,
        chunk_size=32,
    )

    model = TitansMAC(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create batch: predict next token
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    targets = torch.randint(0, config.vocab_size, (4, 32))

    # Forward
    logits, _ = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    # Backward + step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    import math

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {math.exp(loss.item()):.2f}")


def main() -> None:
    """Run all examples."""
    print("#" * 60)
    print("# Titans PyTorch: Basic Usage Examples")
    print("#" * 60)

    torch.manual_seed(42)

    example_basic_forward()
    example_all_variants()
    example_streaming()
    example_training_step()

    print("\n" + "=" * 60)
    print("All examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
