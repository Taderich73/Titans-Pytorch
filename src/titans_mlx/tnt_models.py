# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TNT Model Architectures (MLX Implementation).

Implements Titans model variants using the TNT hierarchical memory system
from "TNT: Improving Chunkwise Training for Test-Time Memorization"
(Li et al., 2025).

Each TNT block replaces the single NeuralLongTermMemory with
HierarchicalMemory (global + N local memories). Three integration
strategies are supported:

- TNTMACBlock: Memory as Context — hierarchical memory retrieval
  concatenated with input before segmented attention
- TNTMAGBlock: Memory as Gate — hierarchical memory and sliding window
  attention combined via learned gating
- TNTMALBlock: Memory as Layer — hierarchical memory as a preprocessing
  layer before sliding window attention

TitansTNT wraps any of these block types into a full sequence model
with embedding, chunking, and output head.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.attention import SegmentedAttention, SlidingWindowAttention
from titans_mlx.config import TitansConfig
from titans_mlx.memory import TNTMemoryState
from titans_mlx.models import FeedForward, RMSNorm
from titans_mlx.persistent import PersistentMemory
from titans_mlx.attn_res import AttnResMemoryGate, BlockAttnRes
from titans_mlx.tnt_memory import HierarchicalMemory


# =============================================================================
# TNT MAC: Memory as Context with Hierarchical Memory
# =============================================================================


class TNTMACBlock(nn.Module):
    """TNT variant of MAC block.

    Same architecture as MACBlock but uses HierarchicalMemory instead of
    NeuralLongTermMemory. The global memory provides broad context via
    large chunks, while local memories provide fine-grained details via
    small chunks with periodic resets.

    Architecture (mirroring MACBlock, Eq. 21-25):
    1. Retrieve from hierarchical memory using a learned query
    2. Concatenate: [persistent] || [memory] || [input]
    3. Segmented attention (causal)
    4. Update hierarchical memory with attention output
    5. Gated output with learnable normalization
    6. Feed-forward network
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Hierarchical memory (replaces single NeuralLongTermMemory)
        self.hierarchical_memory = HierarchicalMemory(config)

        # Learned query for memory retrieval — data-independent to prevent
        # within-chunk causality violations (same rationale as MACBlock)
        self.memory_query = mx.random.normal((1, 1, config.dim)) * config.init_std

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Gating normalization (Section 4.2)
        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

        # AttnRes (optional)
        if config.use_attn_res:
            self.attn_res = BlockAttnRes(config.dim)
            self.attn_res_gate = AttnResMemoryGate()

    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
        memory_gate: mx.array | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
        """Forward pass for TNT MAC block.

        Args:
            x: Input tensor (batch, seq, dim) — single chunk/segment
            state: Hierarchical memory state from previous chunk
            memory_gate: Optional scalar importance weight from AttnRes

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Initialize state if needed
        if state is None:
            state = self.hierarchical_memory.init_state(batch_size)

        # Step 1: Retrieve from hierarchical memory using learned query
        query = mx.broadcast_to(self.memory_query, (batch_size, 1, self.config.dim))
        memory_retrieved = self.hierarchical_memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Step 2-3: Attention with [persistent || memory || input]
        persistent = self.persistent(batch_size)
        normed = self.norm1(x)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)

        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        y_t = x + attn_out

        # Step 4: Update hierarchical memory with attention output
        mem_out, new_state = self.hierarchical_memory(
            y_t, state=state, memory_gate=memory_gate
        )

        # Step 5: Gated output with learnable normalization
        gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
            self.gate_norm_mem(mem_out)
        )
        output = y_t + gated

        # Feed-forward
        normed = self.norm2(output)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        output = output + ffn_out

        return output, new_state


# =============================================================================
# TNT MAG: Memory as Gate with Hierarchical Memory
# =============================================================================


class TNTMAGBlock(nn.Module):
    """TNT variant of MAG block.

    Same architecture as MAGBlock but uses HierarchicalMemory. Sliding
    window attention handles precise local dependencies while hierarchical
    memory provides multi-scale long-range context via gating.

    Architecture (mirroring MAGBlock, Eq. 26-28):
    1. Sliding window attention on [persistent || input]
    2. Update hierarchical memory with persistent-augmented input
    3. Gated output combining attention and memory
    4. Feed-forward network
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory (prepended to input)
        self.persistent = PersistentMemory(config)

        # Sliding window attention
        self.attention = SlidingWindowAttention(config)

        # Hierarchical memory (replaces single NeuralLongTermMemory)
        self.hierarchical_memory = HierarchicalMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Gating normalization (Section 4.2)
        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
        """Forward pass for TNT MAG block.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Hierarchical memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory as prefix for attention
        persistent = self.persistent(batch_size)

        # Eq. 26: Attention branch
        normed = self.norm1(x)
        attn_out = self.attention(normed, prefix=persistent)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        y_t = x + attn_out

        # Eq. 27-28: Memory receives persistent-augmented input
        if persistent is not None:
            mem_input = mx.concatenate([persistent, normed], axis=1)
        else:
            mem_input = normed
        mem_out_full, new_state = self.hierarchical_memory(mem_input, state=state)
        # Slice off persistent prefix from output
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full

        # Gated output with learnable normalization
        gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
            self.gate_norm_mem(mem_out)
        )
        output = y_t + gated

        # Feed-forward
        normed = self.norm2(output)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        output = output + ffn_out

        return output, new_state


# =============================================================================
# TNT MAL: Memory as Layer with Hierarchical Memory
# =============================================================================


class TNTMALBlock(nn.Module):
    """TNT variant of MAL block.

    Same architecture as MALBlock but uses HierarchicalMemory. Memory
    acts as a preprocessing layer before sliding window attention,
    providing multi-scale context enrichment.

    Architecture (mirroring MALBlock):
    1. Hierarchical memory processes persistent-augmented input
    2. Sliding window attention on memory-enriched output
    3. Feed-forward network
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Hierarchical memory (first layer)
        self.hierarchical_memory = HierarchicalMemory(config)

        # Sliding window attention (second layer)
        self.attention = SlidingWindowAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm3 = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
        """Forward pass for TNT MAL block.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Hierarchical memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory
        persistent = self.persistent(batch_size)

        # Memory layer: h_t = M*_{t-1}([p; x_t])
        normed = self.norm1(x)
        if persistent is not None:
            mem_input = mx.concatenate([persistent, normed], axis=1)
        else:
            mem_input = normed
        mem_out_full, new_state = self.hierarchical_memory(mem_input, state=state)
        # Slice off persistent prefix from output
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full
        if self.dropout_p > 0:
            mem_out = nn.Dropout(self.dropout_p)(mem_out)
        x = x + mem_out

        # Attention layer with persistent prefix
        normed = self.norm2(x)
        attn_out = self.attention(normed, prefix=persistent)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        x = x + attn_out

        # Feed-forward
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        x = x + ffn_out

        return x, new_state


# =============================================================================
# Block type registry
# =============================================================================

_TNT_BLOCK_TYPES = {
    "mac": TNTMACBlock,
    "mag": TNTMAGBlock,
    "mal": TNTMALBlock,
}


# =============================================================================
# TitansTNT: Full model with hierarchical memory
# =============================================================================


class TitansTNT(nn.Module):
    """Titans with TNT hierarchical memory.

    Full sequence model using hierarchical memory (global + N local
    memories) with any of the three integration strategies: MAC, MAG, MAL.

    Segments the input into chunks and threads TNTMemoryState across them,
    following the same pattern as TitansMAC/MAG/MAL.

    Args:
        config: Model configuration with use_tnt=True
        variant: Integration strategy — "mac", "mag", or "mal" (default: "mac")
    """

    def __init__(self, config: TitansConfig, variant: str = "mac") -> None:
        super().__init__()
        self.config = config
        self.variant = variant

        if variant not in _TNT_BLOCK_TYPES:
            raise ValueError(
                f"Unknown TNT variant '{variant}'. "
                f"Choose from: {list(_TNT_BLOCK_TYPES.keys())}"
            )

        block_cls = _TNT_BLOCK_TYPES[variant]

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of TNT blocks
        self.blocks = [block_cls(config) for _ in range(config.num_layers)]

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialize embedding, then tie head weights
        self._init_weights()
        self.head.weight = self.embed.weight

        # AttnRes step counter for warmup
        self._step_count = 0

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def _process_single_chunk(
        self,
        chunk: mx.array,
        states: list[TNTMemoryState | None],
    ) -> tuple[mx.array, list[TNTMemoryState]]:
        """Process a single chunk through all blocks."""
        new_states = []

        if not self.config.use_attn_res:
            # Unchanged fast path
            for i, block in enumerate(self.blocks):
                chunk, new_state = block(chunk, state=states[i])
                new_states.append(new_state)
            return chunk, new_states

        # AttnRes path
        S = self.config.attnres_base_block_size
        completed_blocks: list[mx.array] = []
        partial_block: mx.array | None = chunk  # Token embedding = b_0

        for i, block in enumerate(self.blocks):
            # Compute AttnRes input
            h, attn_weights = block.attn_res(completed_blocks, partial_block)

            # Extract memory gate
            memory_gate = block.attn_res_gate(attn_weights)

            # Bypass gate during warmup
            if (
                self.config.attnres_warmup_steps > 0
                and self._step_count < self.config.attnres_warmup_steps
            ):
                memory_gate = None

            # Forward through block
            output, new_state = block(h, state=states[i], memory_gate=memory_gate)
            new_states.append(new_state)

            # Track block output for AttnRes.
            # (output - h) captures the total contribution of this block
            # across all its internal sub-layers (attention, gating, FFN).
            layer_output = output - h
            if partial_block is None:
                partial_block = layer_output
            else:
                partial_block = partial_block + layer_output

            # Block boundary check
            if (i + 1) % S == 0 or i == len(self.blocks) - 1:
                completed_blocks.append(partial_block)
                partial_block = None

            chunk = output

        return chunk, new_states

    def __call__(
        self,
        input_ids: mx.array,
        states: list[TNTMemoryState] | None = None,
    ) -> tuple[mx.array, list[TNTMemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of TNTMemoryState for each layer

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)

        if seq_len <= chunk_size:
            # Fast path: single chunk
            x, new_states = self._process_single_chunk(x, states)
        else:
            # Chunked path
            outputs = []
            new_states = list(states)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = x[:, chunk_start:chunk_end]
                chunk, new_states = self._process_single_chunk(chunk, new_states)
                outputs.append(chunk)
            x = mx.concatenate(outputs, axis=1)

        # Shared exit: output projection + step counter
        x = self.norm(x)
        logits = self.head(x)

        # Increment step counter for AttnRes warmup
        self._step_count += 1

        return logits, new_states
