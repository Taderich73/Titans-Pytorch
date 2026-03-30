#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Basic usage examples for Titans MLX models.

Run with:
    PYTHONPATH=src python examples/basic_usage.py
"""

import mlx.core as mx
import mlx.nn as nn

from titans_mlx import (
    NeuralLongTermMemory,
    PersistentMemory,
    SegmentedAttention,
    SlidingWindowAttention,
    TitansConfig,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)


def example_config() -> None:
    """Example: Creating configurations."""
    print("\n" + "=" * 60)
    print("Example: TitansConfig")
    print("=" * 60)

    # Default config
    default_config = TitansConfig()
    print(f"Default dim: {default_config.dim}")
    print(f"Default num_heads: {default_config.num_heads}")
    print(f"Default num_layers: {default_config.num_layers}")

    # Custom config for small model
    small_config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        chunk_size=64,
        window_size=64,
        num_persistent_tokens=4,
        num_memory_layers=1,
    )
    print("\nSmall config:")
    print(f"  dim={small_config.dim}")
    print(f"  head_dim={small_config.head_dim}")
    print(f"  ffn_dim={small_config.ffn_dim}")


def example_neural_memory() -> None:
    """Example: Using NeuralLongTermMemory directly."""
    print("\n" + "=" * 60)
    print("Example: NeuralLongTermMemory")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
    )

    memory = NeuralLongTermMemory(config)

    # Process input
    batch_size, seq_len = 2, 32
    x = mx.random.normal((batch_size, seq_len, config.dim))

    print(f"Input shape: {x.shape}")

    # First forward pass (initializes memory)
    output1, state1 = memory(x)
    mx.eval(output1)
    print(f"Output 1 shape: {output1.shape}")
    print(f"Memory state has {len(state1.weights)} weight tensors")

    # Second forward pass (continues with state)
    x2 = mx.random.normal((batch_size, seq_len, config.dim))
    output2, state2 = memory(x2, state=state1)
    mx.eval(output2)
    print(f"Output 2 shape: {output2.shape}")

    # Retrieve without updating
    queries = mx.random.normal((batch_size, 8, config.dim))
    retrieved = memory.retrieve(queries, state2)
    mx.eval(retrieved)
    print(f"Retrieved shape: {retrieved.shape}")


def example_attention() -> None:
    """Example: Using attention modules."""
    print("\n" + "=" * 60)
    print("Example: Attention Modules")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        window_size=16,
    )

    batch_size, seq_len = 2, 32

    # Sliding Window Attention
    swa = SlidingWindowAttention(config)
    x = mx.random.normal((batch_size, seq_len, config.dim))
    out_swa = swa(x)
    mx.eval(out_swa)
    print(f"SlidingWindowAttention output: {out_swa.shape}")

    # With prefix (like persistent memory)
    prefix = mx.random.normal((batch_size, 8, config.dim))
    out_swa_prefix = swa(x, prefix=prefix)
    mx.eval(out_swa_prefix)
    print(f"SlidingWindowAttention with prefix: {out_swa_prefix.shape}")

    # Segmented Attention
    seg_attn = SegmentedAttention(config)
    out_seg = seg_attn(x)
    mx.eval(out_seg)
    print(f"SegmentedAttention output: {out_seg.shape}")

    # With persistent and memory
    persistent = mx.random.normal((batch_size, 4, config.dim))
    memory = mx.random.normal((batch_size, 8, config.dim))
    out_seg_full = seg_attn(x, persistent=persistent, memory=memory)
    mx.eval(out_seg_full)
    print(f"SegmentedAttention with context: {out_seg_full.shape}")


def example_persistent_memory() -> None:
    """Example: Using PersistentMemory."""
    print("\n" + "=" * 60)
    print("Example: PersistentMemory")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_persistent_tokens=8,
    )

    persistent = PersistentMemory(config)

    batch_size = 4
    tokens = persistent(batch_size)
    mx.eval(tokens)
    print(f"Persistent tokens shape: {tokens.shape}")
    print(
        f"  (batch={batch_size}, num_tokens={config.num_persistent_tokens}, dim={config.dim})"
    )

    # These are learnable parameters
    raw_tokens = persistent.get_tokens()
    mx.eval(raw_tokens)
    print(f"Raw tokens (no batch): {raw_tokens.shape}")


def example_titans_mac() -> None:
    """Example: Using TitansMAC model."""
    print("\n" + "=" * 60)
    print("Example: TitansMAC (Memory as Context)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=100,
        chunk_size=32,
        num_persistent_tokens=4,
    )

    model = TitansMAC(config)
    from mlx.utils import tree_flatten

    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {params:,}")

    # Process sequence
    batch_size, seq_len = 2, 64  # 2 chunks of 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, states = model(input_ids)
    mx.eval(logits)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of layer states: {len(states)}")

    # Continue with next chunk
    next_input = mx.random.randint(0, config.vocab_size, (batch_size, 32))
    logits2, states2 = model(next_input, states=states)
    mx.eval(logits2)
    print(f"Continuation logits shape: {logits2.shape}")


def example_titans_mag() -> None:
    """Example: Using TitansMAG model."""
    print("\n" + "=" * 60)
    print("Example: TitansMAG (Memory as Gate)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=100,
        window_size=16,
    )

    model = TitansMAG(config)

    batch_size, seq_len = 2, 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, states = model(input_ids)
    mx.eval(logits)
    print(f"MAG output shape: {logits.shape}")
    print("MAG uses sliding window attention + gated memory")


def example_titans_mal() -> None:
    """Example: Using TitansMAL model."""
    print("\n" + "=" * 60)
    print("Example: TitansMAL (Memory as Layer)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=100,
        window_size=16,
    )

    model = TitansMAL(config)

    batch_size, seq_len = 2, 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, states = model(input_ids)
    mx.eval(logits)
    print(f"MAL output shape: {logits.shape}")
    print("MAL processes: input -> memory -> attention -> ffn")


def example_titans_lmm() -> None:
    """Example: Using TitansLMM model (memory only)."""
    print("\n" + "=" * 60)
    print("Example: TitansLMM (Memory Only)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=100,
        num_memory_layers=2,
    )

    model = TitansLMM(config)

    batch_size, seq_len = 2, 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, states = model(input_ids)
    mx.eval(logits)
    print(f"LMM output shape: {logits.shape}")
    print("LMM uses only neural memory (no attention)")


def example_training_step() -> None:
    """Example: Simple training step."""
    print("\n" + "=" * 60)
    print("Example: Training Step")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=1,
        vocab_size=100,
        chunk_size=16,
    )

    model = TitansMAC(config)

    # Create batch
    batch_size, seq_len = 4, 16
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Define loss function
    def loss_fn(model):
        logits, _ = model(input_ids)
        logits_flat = logits.reshape(-1, config.vocab_size)
        targets_flat = targets.reshape(-1)
        return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")

    # Forward + backward
    loss, grads = nn.value_and_grad(model, loss_fn)(model)

    # Optimizer step
    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=1e-4)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    import math

    print(f"Loss: {float(loss):.4f}")
    print(f"Perplexity: {math.exp(float(loss)):.2f}")


def main() -> None:
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# TITANS: Learning to Memorize at Test Time")
    print("# MLX Implementation Examples")
    print("#" * 60)

    mx.random.seed(42)

    example_config()
    example_neural_memory()
    example_attention()
    example_persistent_memory()
    example_titans_mac()
    example_titans_mag()
    example_titans_mal()
    example_titans_lmm()
    example_training_step()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
