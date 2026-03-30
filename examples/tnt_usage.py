#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TNT (Test-Time Neural Turing) usage examples for Titans MLX.

Demonstrates the TNT hierarchical memory system from
"TNT: Improving Chunkwise Training for Test-Time Memorization"
(Li et al., 2025).

Run with:
    PYTHONPATH=src python examples/tnt_usage.py
"""

import mlx.core as mx
import mlx.nn as nn

from titans_mlx import (
    GlobalMemory,
    HierarchicalMemory,
    LocalMemory,
    TitansConfig,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TNTMemoryState,
)


def example_stage1_config() -> None:
    """Example: Creating a Stage 1 (pre-training) TNT configuration."""
    print("\n" + "=" * 60)
    print("TNT Stage 1 Configuration")
    print("=" * 60)

    config = TitansConfig.tnt_stage1(
        dim=256,
        num_heads=8,
        num_layers=4,
        vocab_size=32000,
    )

    print(f"  use_tnt:             {config.use_tnt}")
    print(f"  global_chunk_size:   {config.global_chunk_size}")
    print(f"  local_chunk_sizes:   {config.local_chunk_sizes}")
    print(f"  num_local_memories:  {config.num_local_memories}")
    print(f"  local_shard_length:  {config.local_shard_length}")
    print(f"  tnt_stage:           {config.tnt_stage}")
    print(f"  use_qk_projection:   {config.use_qk_projection}")


def example_stage2_config() -> None:
    """Example: Transitioning from Stage 1 to Stage 2 (fine-tuning)."""
    print("\n" + "=" * 60)
    print("TNT Stage 2 Configuration (from Stage 1)")
    print("=" * 60)

    stage1 = TitansConfig.tnt_stage1(
        dim=256,
        num_heads=8,
        num_layers=4,
        vocab_size=32000,
        local_chunk_sizes=[8, 16, 32],
    )
    stage2 = TitansConfig.tnt_stage2(stage1)

    print(f"  Stage 1 local_chunk_sizes: {stage1.local_chunk_sizes}")
    print(f"  Stage 2 finetune sizes:    {stage2.finetune_local_chunk_sizes}")
    print(f"  Stage 2 active sizes:      {stage2.active_local_chunk_sizes}")
    print(f"  tnt_stage:                 {stage2.tnt_stage}")


def example_hierarchical_memory() -> None:
    """Example: Using HierarchicalMemory directly."""
    print("\n" + "=" * 60)
    print("Hierarchical Memory (Global + Local)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_memory_layers=1,
        use_conv=False,
        use_tnt=True,
        local_chunk_sizes=[4, 8],
        local_shard_length=64,
    )

    mem = HierarchicalMemory(config)
    x = mx.random.normal((1, 16, 64))

    # Process input — updates both global and local memories
    output, state = mem(x)
    mx.eval(output)

    print(f"  Input shape:         {x.shape}")
    print(f"  Output shape:        {output.shape}")
    print(f"  Num local memories:  {len(state.local_states)}")
    print(f"  Step counters:       {state.local_step_counters}")

    # Process more input — state threads across calls
    x2 = mx.random.normal((1, 16, 64))
    output2, state2 = mem(x2, state=state)
    mx.eval(output2)
    print(f"  After 2nd call:      counters={state2.local_step_counters}")

    # Retrieve with different queries
    queries = mx.random.normal((1, 4, 64))
    retrieved = mem.retrieve(queries, state2)
    mx.eval(retrieved)
    print(f"  Retrieval shape:     {retrieved.shape}")


def example_multi_resolution() -> None:
    """Example: Multi-resolution local memories."""
    print("\n" + "=" * 60)
    print("Multi-Resolution Local Memories")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_memory_layers=1,
        use_conv=False,
        use_tnt=True,
        local_chunk_sizes=[4, 8, 16, 32],
        local_shard_length=128,
    )

    mem = HierarchicalMemory(config)
    print(f"  Num local memories: {len(mem.local_memories)}")
    for i, lm in enumerate(mem.local_memories):
        print(
            f"    Local {i}: chunk_size={lm.chunk_size}, shard_length={lm.shard_length}"
        )

    x = mx.random.normal((1, 32, 64))
    output, state = mem(x)
    mx.eval(output)
    print(f"  Output shape: {output.shape}")
    print(f"  Step counters: {state.local_step_counters}")


def example_tnt_model() -> None:
    """Example: Full TNT models with all three variants."""
    print("\n" + "=" * 60)
    print("TNT Models (TitansMAC, TitansMAG, TitansMAL with use_tnt=True)")
    print("=" * 60)

    config = TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=16,
        window_size=16,
        num_memory_layers=1,
        use_conv=False,
        use_rope=False,
        use_tnt=True,
        local_chunk_sizes=[4, 8],
        local_shard_length=64,
    )

    input_ids = mx.random.randint(0, 256, (1, 32))

    for name, model_cls in [("mac", TitansMAC), ("mag", TitansMAG), ("mal", TitansMAL)]:
        model = model_cls(config)
        logits, states = model(input_ids)
        mx.eval(logits)

        print(f"\n  Variant: {name}")
        print(f"    Logits shape: {logits.shape}")
        print(f"    Num layers:   {len(states)}")
        print(f"    Step counters (layer 0): {states[0].local_step_counters}")

        # Continue with more input
        input_ids2 = mx.random.randint(0, 256, (1, 16))
        logits2, states2 = model(input_ids2, states=states)
        mx.eval(logits2)
        print(f"    After 2nd call: counters={states2[0].local_step_counters}")


def example_state_persistence() -> None:
    """Example: Saving and loading TNT memory states."""
    print("\n" + "=" * 60)
    print("TNT State Persistence")
    print("=" * 60)

    from pathlib import Path
    from titans_mlx.memory import save_tnt_memory_states, load_tnt_memory_states

    config = TitansConfig(
        dim=64,
        num_memory_layers=1,
        use_conv=False,
        use_tnt=True,
        local_chunk_sizes=[4, 8],
    )

    mem = HierarchicalMemory(config)
    x = mx.random.normal((1, 16, 64))
    _, state = mem(x)
    mx.eval(state.global_state.weights[0])

    # Save
    path = Path("/tmp/tnt_example_state")
    save_tnt_memory_states([state], path)
    print(f"  Saved state to: {path}.npz")

    # Load
    loaded = load_tnt_memory_states(path)
    print(f"  Loaded {len(loaded)} layer state(s)")
    print(f"  Step counters: {loaded[0].local_step_counters}")
    print(f"  Q-K proj shape: {loaded[0].qk_projections[0].shape}")


if __name__ == "__main__":
    mx.random.seed(42)

    example_stage1_config()
    example_stage2_config()
    example_hierarchical_memory()
    example_multi_resolution()
    example_tnt_model()
    example_state_persistence()

    print("\n" + "=" * 60)
    print("All TNT examples completed successfully!")
    print("=" * 60)
