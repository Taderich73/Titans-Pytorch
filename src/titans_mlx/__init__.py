# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans MLX Implementation - Optimized for Apple Silicon.

This module provides an MLX-native implementation of the Titans architecture,
optimized for Apple Silicon (M1/M2/M3/M4) GPUs.

MLX is Apple's array framework for machine learning on Apple silicon.
It provides:
- Unified memory architecture (no CPU/GPU transfers)
- Lazy evaluation for optimized computation graphs
- JIT compilation for maximum performance

Usage:
    import mlx.core as mx
    from titans_mlx import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    # Forward pass
    x = mx.random.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

from titans_mlx.attention import SegmentedAttention, SlidingWindowAttention
from titans_mlx.attn_res import AttnResMemoryGate, BlockAttnRes
from titans_mlx.config import TitansConfig
from titans_mlx.mca import MemoryCrossAttention
from titans_mlx.memory import (
    MemoryState,
    NeuralLongTermMemory,
    TNTMemoryState,
    load_memory_states,
    load_tnt_memory_states,
    save_memory_states,
    save_tnt_memory_states,
)
from titans_mlx.memory_dump import MemoryDumpManager
from titans_mlx.metal_kernels import (
    MetalFeedForward,
    MetalRMSNorm,
    MetalRotaryEmbedding,
    benchmark_metal_kernel,
    get_metal_kernel_info,
    metal_causal_attention,
    metal_memory_update,
    metal_rope,
    metal_silu_gate,
)
from titans_mlx.models import TitansLMM, TitansMAC, TitansMAG, TitansMAL, process_chunk
from titans_mlx.optimizations import (
    OptimizedFeedForward,
    OptimizedMemoryMLP,
    benchmark_function,
    chunked_attention,
    compile_function,
    compile_model,
    evaluate_all,
    fused_rmsnorm,
    fused_silu_gate,
    get_causal_mask,
    get_device_info,
    get_sliding_window_mask,
    rotary_embedding_optimized,
    scaled_dot_product_attention,
)
from titans_mlx.persistent import PersistentMemory
from titans_mlx.qk_projection import QKProjection, update_projection_state
from titans_mlx.quantize_state import (
    QuantizedMemoryState,
    QuantizedTensor,
    get_momentum,
    get_weights,
    quantize_memory_state,
    quantize_tensor,
)
from titans_mlx.tnt_memory import GlobalMemory, HierarchicalMemory, LocalMemory

__version__ = "0.1.0"
__all__ = [
    # Config
    "TitansConfig",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    "TNTMemoryState",
    "save_memory_states",
    "load_memory_states",
    "save_tnt_memory_states",
    "load_tnt_memory_states",
    # Attention
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Persistent Memory
    "PersistentMemory",
    # Q-K Projection
    "QKProjection",
    "update_projection_state",
    # AttnRes
    "BlockAttnRes",
    "AttnResMemoryGate",
    # TNT Hierarchical Memory
    "GlobalMemory",
    "LocalMemory",
    "HierarchicalMemory",
    # Models
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    "process_chunk",
    # Optimizations
    "benchmark_function",
    "chunked_attention",
    "compile_function",
    "compile_model",
    "evaluate_all",
    "fused_rmsnorm",
    "fused_silu_gate",
    "get_causal_mask",
    "get_device_info",
    "get_sliding_window_mask",
    "OptimizedFeedForward",
    "OptimizedMemoryMLP",
    "rotary_embedding_optimized",
    "scaled_dot_product_attention",
    # Metal Kernels
    "benchmark_metal_kernel",
    "get_metal_kernel_info",
    "metal_causal_attention",
    "metal_memory_update",
    "metal_rope",
    "metal_silu_gate",
    "MetalFeedForward",
    "MetalRMSNorm",
    "MetalRotaryEmbedding",
    # Memory State Quantization
    "QuantizedTensor",
    "QuantizedMemoryState",
    "quantize_tensor",
    "quantize_memory_state",
    "get_weights",
    "get_momentum",
    # Memory Cross-Attention
    "MemoryCrossAttention",
    # Memory Dump
    "MemoryDumpManager",
]
