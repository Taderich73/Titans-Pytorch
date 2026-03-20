#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Example: Processing long sequences with Titans MLX.

This demonstrates how Titans handles long sequences by:
1. Chunking the sequence for MAC variant
2. Using sliding window attention for MAG/MAL
3. Maintaining memory state across chunks

Run with:
    PYTHONPATH=src python examples/long_sequence.py
"""

import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from titans_mlx import TitansConfig, TitansMAC, TitansMAG


def process_long_sequence_mac() -> None:
    """Process a long sequence with TitansMAC using chunking."""
    print("\n" + "=" * 60)
    print("Processing Long Sequence with TitansMAC")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=128,  # Each chunk is 128 tokens
        num_persistent_tokens=8,
        num_memory_layers=2,
    )

    model = TitansMAC(config)

    # Long sequence: 1024 tokens (8 chunks of 128)
    total_length = 1024
    batch_size = 2

    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, total_length))

    print(f"Input sequence length: {total_length}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Number of chunks: {total_length // config.chunk_size}")

    # Process entire sequence at once (internally chunked)
    start_time = time.time()
    logits, states = model(input_ids)
    mx.eval(logits)
    elapsed = time.time() - start_time

    print(f"Output shape: {logits.shape}")
    print(f"Processing time: {elapsed:.3f}s")
    print(f"Tokens per second: {total_length / elapsed:.1f}")


def process_streaming_mac() -> None:
    """Process sequence in streaming fashion with TitansMAC."""
    print("\n" + "=" * 60)
    print("Streaming Processing with TitansMAC")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=64,
        num_persistent_tokens=4,
    )

    model = TitansMAC(config)

    batch_size = 1
    chunk_size = config.chunk_size
    num_chunks = 8

    print(f"Processing {num_chunks} chunks of {chunk_size} tokens each")
    print(f"Total tokens: {num_chunks * chunk_size}")

    states = None
    all_logits = []

    for i in range(num_chunks):
        # Simulate receiving new chunk
        chunk = mx.random.randint(0, config.vocab_size, (batch_size, chunk_size))

        # Process chunk with previous state
        logits, states = model(chunk, states=states)
        mx.eval(logits)
        all_logits.append(logits)

        print(f"Chunk {i + 1}: processed {chunk_size} tokens, memory updated")

    # Concatenate all outputs
    final_logits = mx.concatenate(all_logits, axis=1)
    print(f"\nFinal output shape: {final_logits.shape}")


def compare_memory_usage() -> None:
    """Compare parameter counts across different configurations."""
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)

    configs = [
        ("Small", TitansConfig(dim=64, num_heads=2, num_layers=1, vocab_size=100)),
        ("Medium", TitansConfig(dim=128, num_heads=4, num_layers=2, vocab_size=256)),
        ("Large", TitansConfig(dim=256, num_heads=8, num_layers=4, vocab_size=1000)),
    ]

    for name, config in configs:
        model = TitansMAG(config)
        params = sum(v.size for _, v in tree_flatten(model.parameters()))
        param_mb = params * 4 / (1024 * 1024)  # Assuming float32

        print(f"{name}:")
        print(
            f"  dim={config.dim}, heads={config.num_heads}, layers={config.num_layers}"
        )
        print(f"  Parameters: {params:,} ({param_mb:.2f} MB)")


def demonstrate_memory_persistence() -> None:
    """Demonstrate how memory persists across chunks."""
    print("\n" + "=" * 60)
    print("Memory Persistence Demonstration")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=100,
        chunk_size=32,
    )

    model = TitansMAC(config)

    batch_size = 1
    seq_len = 32

    # First chunk
    chunk1 = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    _, state1 = model(chunk1)
    mx.eval(state1[0].weights[0])

    # Extract memory weights from first layer
    weights1 = mx.array(state1[0].weights[0])

    # Second chunk with same state
    chunk2 = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    _, state2 = model(chunk2, states=state1)
    mx.eval(state2[0].weights[0])

    weights2 = state2[0].weights[0]

    # Memory has changed
    weight_diff = float(mx.mean(mx.abs(weights2 - weights1)))
    print(f"Memory weight change after chunk 2: {weight_diff:.6f}")

    # Third chunk
    chunk3 = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    _, state3 = model(chunk3, states=state2)
    mx.eval(state3[0].weights[0])

    weights3 = state3[0].weights[0]
    weight_diff2 = float(mx.mean(mx.abs(weights3 - weights2)))
    print(f"Memory weight change after chunk 3: {weight_diff2:.6f}")

    print("\nMemory accumulates information across chunks!")


def main() -> None:
    """Run all long sequence examples."""
    print("\n" + "#" * 60)
    print("# TITANS: Long Sequence Processing Examples (MLX)")
    print("#" * 60)

    mx.random.seed(42)

    process_long_sequence_mac()
    process_streaming_mac()
    compare_memory_usage()
    demonstrate_memory_persistence()

    print("\n" + "=" * 60)
    print("All long sequence examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
