#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Memory persistence examples for Titans PyTorch.

Demonstrates saving and loading memory states across sessions.

Run with:
    uv run python examples/memory_persistence.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from titans import TitansConfig, TitansMAC
from titans.memory_dump import load_memory_states, save_memory_states


def example_save_load() -> None:
    """Save and load memory states to disk."""
    print("\n" + "=" * 60)
    print("Example: Save / Load Memory States")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
    )

    model = TitansMAC(config)
    model.eval()

    # Process some data to build up memory
    with torch.no_grad():
        input_ids = torch.randint(0, 256, (1, 32))
        _, states = model(input_ids)

        # Process a second chunk
        input_ids2 = torch.randint(0, 256, (1, 32))
        _, states = model(input_ids2, states=states)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory_state.npz"
        save_memory_states(states, path)
        print(f"  Saved {len(states)} layer states to {path.name}")

        # Load
        loaded = load_memory_states(path)
        print(f"  Loaded {len(loaded)} layer states")

        # Verify fidelity
        for i, (orig, load) in enumerate(zip(states, loaded)):
            diff = (orig.weights[0] - load.weights[0]).abs().max().item()
            print(f"  Layer {i} max weight diff: {diff:.1e}")


def example_memory_evolution() -> None:
    """Show how memory weights evolve across chunks."""
    print("\n" + "=" * 60)
    print("Example: Memory Evolution Across Chunks")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
    )

    model = TitansMAC(config)
    model.eval()

    states = None
    prev_norm = None

    with torch.no_grad():
        for i in range(5):
            chunk = torch.randint(0, 256, (1, 32))
            _, states = model(chunk, states=states)

            norm = states[0].weights[0].norm().item()
            delta = f" (delta: {norm - prev_norm:+.4f})" if prev_norm else ""
            print(f"  Chunk {i + 1}: weight norm = {norm:.4f}{delta}")
            prev_norm = norm


def main() -> None:
    """Run all memory persistence examples."""
    print("#" * 60)
    print("# Titans PyTorch: Memory Persistence Examples")
    print("#" * 60)

    torch.manual_seed(42)

    example_save_load()
    example_memory_evolution()

    print("\n" + "=" * 60)
    print("All memory persistence examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
