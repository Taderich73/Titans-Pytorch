# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time — PyTorch Implementation.

Usage:
    import torch
    from titans import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    x = torch.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

from titans.adaptive_window import AdaptiveWindowPredictor, compute_window_regularization
from titans.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans.attn_res import AttnResMemoryGate, BlockAttnRes
from titans.config import TitansConfig
from titans.mca import MemoryCrossAttention
from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans.qk_projection import QKProjection
from titans.tnt_memory import GlobalMemory, HierarchicalMemory, LocalMemory
from titans.memory_dump import load_memory_states, save_memory_states
from titans.models import (
    FeedForward,
    LMMBlock,
    MAGBlock,
    MALBlock,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    process_chunk,
)
from titans.persistent import PersistentMemory

__version__ = "0.2.0"

__all__ = [
    # Config
    "TitansConfig",
    # Adaptive Window
    "AdaptiveWindowPredictor",
    "compute_window_regularization",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    "TNTMemoryState",
    "save_memory_states",
    "load_memory_states",
    # Q-K Projection
    "QKProjection",
    # TNT Memory
    "GlobalMemory",
    "LocalMemory",
    "HierarchicalMemory",
    # Attention
    "RotaryPositionEmbedding",
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Persistent Memory
    "PersistentMemory",
    # MCA
    "MemoryCrossAttention",
    # AttnRes
    "BlockAttnRes",
    "AttnResMemoryGate",
    # Models
    "RMSNorm",
    "FeedForward",
    "LMMBlock",
    "MAGBlock",
    "MALBlock",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    "process_chunk",
]
