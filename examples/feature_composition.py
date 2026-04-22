#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Feature composition examples for Titans PyTorch.

Demonstrates enabling and combining independent features:
TNT, AttnRes, MCA, Yaad Huber, Adaptive Window, Proportional RoPE.

Run with:
    uv run python examples/feature_composition.py
"""

from __future__ import annotations

import torch

from titans import TitansConfig, TitansMAC, TitansMAG


def example_tnt() -> None:
    """TNT hierarchical memory with multi-resolution local memories."""
    print("\n" + "=" * 60)
    print("Feature: TNT Hierarchical Memory")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=64,
        use_tnt=True,
        local_chunk_sizes=[4, 8],
        local_shard_length=128,
    )

    model = TitansMAC(config)
    input_ids = torch.randint(0, 256, (1, 64))

    logits, states = model(input_ids)
    state = states[0]  # TNTMemoryState for layer 0

    print(f"  Global memory weight norm: {state.global_state.weights[0].norm():.4f}")
    print(f"  Local memories: {len(state.local_states)}")
    print(f"  Local step counters: {state.local_step_counters}")

    # Two-stage training configs
    stage1 = TitansConfig.tnt_stage1(dim=128, num_heads=4, num_layers=2, vocab_size=256)
    stage2 = TitansConfig.tnt_stage2(stage1)
    print(f"  Stage 1 chunk sizes: {stage1.local_chunk_sizes}")
    print(f"  Stage 2 chunk sizes: {stage2.active_local_chunk_sizes}")


def example_attn_res() -> None:
    """AttnRes: learned depth-wise residual attention."""
    print("\n" + "=" * 60)
    print("Feature: Attention Residuals (AttnRes)")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=4,
        vocab_size=256,
        chunk_size=32,
        use_attn_res=True,
        num_attnres_blocks=2,
    )

    model = TitansMAC(config)
    input_ids = torch.randint(0, 256, (1, 32))

    logits, states = model(input_ids)
    print(f"  Output shape: {logits.shape}")
    print(f"  AttnRes blocks: {config.num_attnres_blocks}")
    print(f"  Sub-layer block size: {config.attnres_sub_layer_block_size}")


def example_mca() -> None:
    """MCA: cross-attention to memory weight rows."""
    print("\n" + "=" * 60)
    print("Feature: Memory Cross-Attention (MCA)")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=4,
        vocab_size=256,
        chunk_size=32,
        use_mca=True,
        mca_num_heads=4,
    )

    model = TitansMAC(config)
    input_ids = torch.randint(0, 256, (1, 32))

    logits, states = model(input_ids)
    print(f"  MCA insertion layers: {config.mca_active_insertion_layers}")
    print(f"  MCA gate type: {config.mca_gate_type}")
    print(f"  Output shape: {logits.shape}")


def example_yaad_huber() -> None:
    """Yaad: Huber attentional bias for robust memory updates."""
    print("\n" + "=" * 60)
    print("Feature: Yaad Huber Attentional Bias")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        memory_objective="huber",
    )

    model = TitansMAC(config)
    input_ids = torch.randint(0, 256, (1, 32))

    logits, states = model(input_ids)
    print(f"  Memory objective: {config.memory_objective}")
    print(f"  Output shape: {logits.shape}")


def example_adaptive_window() -> None:
    """Adaptive window sizing for MAG/MAL."""
    print("\n" + "=" * 60)
    print("Feature: Adaptive Window Sizing")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        window_size=64,
        chunk_size=64,
        adaptive_window=True,
        adaptive_window_min=16,
        adaptive_window_temperature=10.0,
        adaptive_window_lambda=0.01,
    )

    model = TitansMAG(config)
    input_ids = torch.randint(0, 256, (1, 64))

    logits, states = model(input_ids)
    print(
        f"  Window range: [{config.adaptive_window_min}, "
        f"{config.effective_adaptive_window_max}]"
    )
    print(f"  Temperature: {config.adaptive_window_temperature}")
    print(f"  Output shape: {logits.shape}")


def example_proportional_rope() -> None:
    """Proportional RoPE: partial rotation for semantic preservation."""
    print("\n" + "=" * 60)
    print("Feature: Proportional RoPE (p-RoPE)")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        rope_proportion=0.25,
    )

    model = TitansMAC(config)
    input_ids = torch.randint(0, 256, (1, 32))

    logits, states = model(input_ids)
    head_dim = config.head_dim
    rotate_dim = 2 * (int(head_dim * config.rope_proportion) // 2)
    print(f"  head_dim: {head_dim}")
    print(f"  rotate_dim: {rotate_dim} ({config.rope_proportion:.0%} of pairs)")
    print(f"  semantic_dim: {head_dim - rotate_dim}")
    print(f"  Output shape: {logits.shape}")


def example_full_composition() -> None:
    """All features enabled simultaneously."""
    print("\n" + "=" * 60)
    print("Full Composition: TNT + AttnRes + MCA + Huber + p-RoPE")
    print("=" * 60)

    config = TitansConfig(
        dim=128,
        num_heads=4,
        num_layers=4,
        vocab_size=256,
        chunk_size=32,
        # TNT
        use_tnt=True,
        local_chunk_sizes=[4, 8],
        local_shard_length=64,
        # AttnRes
        use_attn_res=True,
        num_attnres_blocks=2,
        # MCA
        use_mca=True,
        mca_num_heads=4,
        # Yaad
        memory_objective="huber",
        # p-RoPE
        rope_proportion=0.25,
    )

    model = TitansMAC(config)
    total_params = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(0, 256, (1, 32))

    logits, states = model(input_ids)
    print(f"  Parameters: {total_params:,}")
    print(f"  Output shape: {logits.shape}")
    print(
        f"  Features: TNT={config.use_tnt}, AttnRes={config.use_attn_res}, "
        f"MCA={config.use_mca}, Huber={config.memory_objective}, "
        f"p-RoPE={config.rope_proportion}"
    )


def main() -> None:
    """Run all feature examples."""
    print("#" * 60)
    print("# Titans PyTorch: Feature Composition Examples")
    print("#" * 60)

    torch.manual_seed(42)

    example_tnt()
    example_attn_res()
    example_mca()
    example_yaad_huber()
    example_adaptive_window()
    example_proportional_rope()
    example_full_composition()

    print("\n" + "=" * 60)
    print("All feature examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
