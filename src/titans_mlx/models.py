# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans Model Architectures (MLX Implementation).

This module implements the three variants of Titans:
1. MAC (Memory as Context): Memory retrieval concatenated with input before attention
2. MAG (Memory as Gate): Memory and attention combined via gating
3. MAL (Memory as Layer): Memory used as a layer before attention

Plus the standalone LMM (Long-term Memory Module) without attention.

MLX-specific optimizations:
- Vectorized operations for Apple Silicon
- Unified memory architecture
- Lazy evaluation for optimal computation graphs
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.adaptive_window import AdaptiveWindowPredictor
from titans_mlx.attention import SegmentedAttention, SlidingWindowAttention
from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory
from titans_mlx.persistent import PersistentMemory


def _init_mca(block: nn.Module, config: TitansConfig, layer_idx: int) -> None:
    """Initialize MCA components on a block (shared across MAC/MAG/MAL)."""
    block.has_mca = layer_idx in config.mca_active_insertion_layers
    if block.has_mca:
        from titans_mlx.mca import MemoryCrossAttention

        block.mca = MemoryCrossAttention(config)
        if config.use_attn_res:
            from titans_mlx.attn_res import BlockAttnRes

            block.attn_res_mca = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )


def _mca_forward(block: nn.Module, h: mx.array, mem_state) -> mx.array:
    """MCA sub-layer: cross-attend to NeuralLTM weight rows (shared across MAC/MAG/MAL)."""
    weights = (
        mem_state.global_state.weights
        if hasattr(mem_state, "global_state")
        else mem_state.weights
    )
    if not weights:
        raise ValueError(
            "MCA requires non-empty memory weights. "
            "Check that NeuralLTM has num_memory_layers >= 1."
        )
    W = weights[0]
    # Internal invariant (not a caller API contract) — NeuralLTM always
    # produces 2D weight matrices; this catches corrupted state early.
    assert W.ndim == 2, f"Expected 2D weight matrix, got {W.ndim}D"
    W = mx.stop_gradient(W)
    return block.mca(h, W)


class FeedForward(nn.Module):
    """Feed-forward network with gating (following recent architectures) - MLX."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.ffn_dim

        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout_p = config.dropout

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SiLU gating."""
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout_p > 0:
            hidden = nn.Dropout(self.dropout_p)(hidden)
        return self.down_proj(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - MLX."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization."""
        orig_dtype = x.dtype
        x_f32 = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / rms * self.weight).astype(orig_dtype)


def process_chunk(
    blocks: list,
    chunk: mx.array,
    states: list,
    config: TitansConfig,
    step_count: int = 0,
) -> tuple[mx.array, list]:
    """Process a single chunk through all blocks.

    Handles both standard residuals and AttnRes paths.
    Used by TitansMAC, TitansMAG, TitansMAL.

    Args:
        blocks: List of block modules (MACBlock, MAGBlock, or MALBlock)
        chunk: Embedded input (batch, seq, dim)
        states: List of memory states, one per block
        config: Model configuration
        step_count: Current training step (for AttnRes warmup)

    Returns:
        Tuple of (output, new_states)
    """
    new_states = []

    if not config.use_attn_res:
        # Standard residual path
        x = chunk
        for i, block in enumerate(blocks):
            core_out, new_state = block.core_forward(x, state=states[i])
            x = x + core_out

            # MCA sub-layer (only at insertion layers)
            if hasattr(block, "has_mca") and block.has_mca:
                mca_out = block.mca_forward(x, new_state)
                x = x + mca_out

            ffn_out = block.ffn_forward(x)
            x = x + ffn_out
            new_states.append(new_state)
        return x, new_states

    # AttnRes path — replaces residual connections per paper
    S = config.attnres_sub_layer_block_size
    completed_blocks: list[mx.array] = [chunk]  # b_0 = embedding
    partial_block: mx.array | None = None
    sub_idx = 0
    warmup = (
        config.attnres_warmup_steps > 0 and step_count < config.attnres_warmup_steps
    )

    for i, block in enumerate(blocks):
        # --- Core sub-layer ---
        h, attn_weights = block.attn_res_core(completed_blocks, partial_block)

        # Memory gate (bypassed during warmup)
        memory_gate = None
        if not warmup:
            memory_gate = block.attn_res_gate(attn_weights)

        core_out, new_state = block.core_forward(
            h, state=states[i], memory_gate=memory_gate
        )
        new_states.append(new_state)

        if partial_block is None:
            partial_block = core_out
        else:
            partial_block = partial_block + core_out
        sub_idx += 1

        # Block boundary check
        if sub_idx % S == 0:
            completed_blocks.append(partial_block)
            partial_block = None

        # --- MCA sub-layer (only at insertion layers) ---
        if hasattr(block, "has_mca") and block.has_mca:
            h_mca, _ = block.attn_res_mca(completed_blocks, partial_block)
            mca_out = block.mca_forward(h_mca, new_state)

            if partial_block is None:
                partial_block = mca_out
            else:
                partial_block = partial_block + mca_out
            sub_idx += 1
            if sub_idx % S == 0:
                completed_blocks.append(partial_block)
                partial_block = None

        # --- FFN sub-layer ---
        h, _ = block.attn_res_ffn(completed_blocks, partial_block)
        ffn_out = block.ffn_forward(h)

        if partial_block is None:
            partial_block = ffn_out
        else:
            partial_block = partial_block + ffn_out
        sub_idx += 1

        # Block boundary check
        if sub_idx % S == 0 or i == len(blocks) - 1:
            completed_blocks.append(partial_block)
            partial_block = None

        # Track model output (last hidden state)
        chunk = h + ffn_out

    return chunk, new_states


# =============================================================================
# MAC: Memory as Context
# =============================================================================


class MACBlock(nn.Module):
    """Memory as Context Block - MLX.

    Architecture:
    1. Retrieve from long-term memory using a learned query (1 token)
    2. Concatenate: [persistent] || [memory] || [input]
    3. Apply segmented attention (causal)
    4. Feed-forward network

    The memory query is a learned parameter (not data-dependent) to prevent
    within-chunk causality violations: if the query were derived from current-
    chunk tokens, per-token retrievals placed before the input in attention
    would let early positions attend to memory tokens encoding future tokens.

    At test time:
    - Persistent memory parameters are fixed
    - Attention performs in-context learning
    - Long-term memory continues learning (weight updates)
    """

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        # Config-driven memory selection
        if config.use_tnt:
            from titans_mlx.tnt_memory import HierarchicalMemory

            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        # Learned query for memory retrieval — data-independent so it cannot
        # leak current-chunk token information into the attention context.
        self.memory_query = mx.random.normal((1, 1, config.dim)) * config.init_std

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention (Core module)
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Gating normalization (Section 4.2: "learnable vector-valued weights
        # followed by a non-linearity σ(·)")
        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

        # AttnRes (optional)
        if config.use_attn_res:
            from titans_mlx.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

        # MCA (optional, only at insertion layers)
        _init_mca(self, config, layer_idx)

    def mca_forward(self, h: mx.array, mem_state) -> mx.array:
        """MCA sub-layer: cross-attend to NeuralLTM weight rows."""
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: mx.array,
        state=None,
        memory_gate=None,
    ):
        """Core sub-layer: retrieve + attention + memory update + gating.

        Returns (core_out, new_state). core_out is the net contribution
        (attn_out + gated), excluding h.
        """
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        # Retrieve from memory using learned query
        query = mx.broadcast_to(self.memory_query, (batch_size, 1, self.config.dim))
        memory_retrieved = self.memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Attention on [persistent || memory || norm(h)]
        persistent = self.persistent(batch_size)
        normed = self.norm1(h)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)

        # Internal residual: memory and gating see full representation
        y_t = h + attn_out

        # Memory update
        mem_out, new_state = self.memory(y_t, state=state, memory_gate=memory_gate)

        # Gating
        gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        core_out = attn_out + gated
        return core_out, new_state

    def ffn_forward(self, h: mx.array) -> mx.array:
        """FFN sub-layer. Returns net contribution, excludes h."""
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        return ffn_out

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Backward-compatible wrapper: standard residuals over sub-layers.

        Args:
            x: Input tensor (batch, seq, dim) - single chunk/segment
            state: Memory state from previous chunk

        Returns:
            Tuple of (output, new_state)
        """
        core_out, new_state = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context - MLX.

    Segments the sequence into chunks and processes each with MAC blocks.
    Long-term memory persists across chunks within a sequence.

    Optimized for Apple Silicon with:
    - JIT compiled block processing
    - Minimized intermediate evaluations
    - Efficient chunk concatenation
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAC blocks
        self.blocks = [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialize embedding, then tie head weights to embedding
        self._init_weights()
        self.head.weight = self.embed.weight

        # Step counter for AttnRes warmup tracking
        self._step_count = 0

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states for each layer

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        if seq_len <= chunk_size:
            x, new_states = process_chunk(
                self.blocks, x, states, self.config, self._step_count
            )
        else:
            outputs = []
            new_states = list(states)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = x[:, chunk_start:chunk_end]
                chunk, new_states = process_chunk(
                    self.blocks, chunk, new_states, self.config, self._step_count
                )
                outputs.append(chunk)
            x = mx.concatenate(outputs, axis=1)

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states


# =============================================================================
# MAG: Memory as Gate
# =============================================================================


class MAGBlock(nn.Module):
    """Memory as Gate Block - MLX.

    Architecture (Section 4.2, Eq. 26-28):
    1. y_t = Attn(x) - Sliding window attention (Eq. 26)
    2. M_t = M_{t-1}(x_t) - Update memory with input (Eq. 27)
    3. o_t = y_t ⊗ M*_t(x_t) - Element-wise product (Eq. 28)

    The attention handles precise local dependencies,
    while memory provides fading long-range context.
    """

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        # Persistent memory (prepended to input)
        self.persistent = PersistentMemory(config)

        # Sliding window attention
        self.attention = SlidingWindowAttention(config)

        # Config-driven memory selection
        if config.use_tnt:
            from titans_mlx.tnt_memory import HierarchicalMemory

            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Gating normalization (Section 4.2: "learnable vector-valued weights
        # followed by a non-linearity σ(·)")
        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

        # Adaptive window sizing (optional)
        self._last_falloff_centers: mx.array | None = None
        if config.adaptive_window:
            self.window_predictor = AdaptiveWindowPredictor(config)

        # AttnRes (optional)
        if config.use_attn_res:
            from titans_mlx.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

        # MCA (optional, only at insertion layers)
        _init_mca(self, config, layer_idx)

    def mca_forward(self, h: mx.array, mem_state) -> mx.array:
        """MCA sub-layer: cross-attend to NeuralLTM weight rows."""
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: mx.array,
        state=None,
        memory_gate=None,
    ):
        """Core sub-layer: attention + memory update (on normed input) + gating.

        IMPORTANT: Memory receives `normed` (pre-attention normalized input),
        NOT y_t (the post-attention representation). This follows the MAG paper
        equations where memory is updated with the input x, not the attention output.

        Returns (core_out, new_state). core_out is the net contribution
        (attn_out + gated), excluding h.
        """
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        # Get persistent memory as prefix for attention
        persistent = self.persistent(batch_size)

        # Eq. 26: y_t = Attn(x) - Attention branch
        normed = self.norm1(h)

        # Adaptive window: predict soft mask from hidden state
        adaptive_mask = None
        if hasattr(self, "window_predictor"):
            adaptive_mask, self._last_falloff_centers = self.window_predictor(normed)

        attn_out = self.attention(normed, prefix=persistent, adaptive_mask=adaptive_mask)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)

        # Internal residual: y_t = h + attn_out
        y_t = h + attn_out

        # Eq. 27-28: Memory receives persistent-augmented normalized input
        # (normed = norm1(h), NOT y_t — this is the key MAG difference from MAC)
        if persistent is not None:
            mem_input = mx.concatenate([persistent, normed], axis=1)
        else:
            mem_input = normed
        mem_out_full, new_state = self.memory(
            mem_input, state=state, memory_gate=memory_gate
        )
        # Slice off persistent prefix from output
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full

        # Eq. 28: Gated output with learnable normalization
        gated = mx.sigmoid(self.gate_norm_attn(y_t)) * mx.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        core_out = attn_out + gated
        return core_out, new_state

    def ffn_forward(self, h: mx.array) -> mx.array:
        """FFN sub-layer. Returns net contribution, excludes h."""
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        return ffn_out

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Backward-compatible wrapper: standard residuals over sub-layers.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        core_out, new_state = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAG(nn.Module):
    """Titans with Memory as Gate - MLX.

    Uses sliding window attention and long-term memory in parallel,
    combined via a gating mechanism.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAG blocks
        self.blocks = [MAGBlock(config, layer_idx=i) for i in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight

        # Step counter for AttnRes warmup tracking
        self._step_count = 0

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        if seq_len <= chunk_size:
            x, new_states = process_chunk(
                self.blocks, x, states, self.config, self._step_count
            )
        else:
            outputs = []
            new_states = list(states)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = x[:, chunk_start:chunk_end]
                chunk, new_states = process_chunk(
                    self.blocks, chunk, new_states, self.config, self._step_count
                )
                outputs.append(chunk)
            x = mx.concatenate(outputs, axis=1)

        # Output
        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states


# =============================================================================
# MAL: Memory as Layer
# =============================================================================


class MALBlock(nn.Module):
    """Memory as Layer Block - MLX.

    Architecture:
    1. Long-term memory processes input (norm1)
    2. Sliding window attention on memory output (norm2)
    3. Feed-forward network (norm3)

    Memory acts as a preprocessing layer before attention.
    """

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Config-driven memory selection
        if config.use_tnt:
            from titans_mlx.tnt_memory import HierarchicalMemory

            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        # Sliding window attention (second layer)
        self.attention = SlidingWindowAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms: norm1=memory, norm2=attention, norm3=FFN
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm3 = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

        # AttnRes (optional)
        if config.use_attn_res:
            from titans_mlx.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

        # MCA (optional, only at insertion layers)
        _init_mca(self, config, layer_idx)

    def mca_forward(self, h: mx.array, mem_state) -> mx.array:
        """MCA sub-layer: cross-attend to NeuralLTM weight rows."""
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: mx.array,
        state=None,
        memory_gate=None,
    ):
        """Core sub-layer: memory → h_mid → attention.

        MAL-specific order: memory first, then attention.
        norm1 is for memory; norm2 is for attention.

        Returns (core_out, new_state). core_out is the net contribution
        (mem_out + attn_out), excluding h.
        """
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        # Get persistent memory
        persistent = self.persistent(batch_size)

        # Memory layer (Eq. 29-30: h_t = M*_{t-1}([p; x_t]))
        normed = self.norm1(h)
        if persistent is not None:
            mem_input = mx.concatenate([persistent, normed], axis=1)
        else:
            mem_input = normed
        mem_out_full, new_state = self.memory(
            mem_input, state=state, memory_gate=memory_gate
        )
        # Slice off persistent prefix from output
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full
        if self.dropout_p > 0:
            mem_out = nn.Dropout(self.dropout_p)(mem_out)

        # Internal residual: attention sees h + mem contribution
        h_mid = h + mem_out

        # Attention layer with persistent prefix (uses norm2)
        normed_mid = self.norm2(h_mid)
        attn_out = self.attention(normed_mid, prefix=persistent)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)

        # Net contribution excludes original h
        core_out = mem_out + attn_out
        return core_out, new_state

    def ffn_forward(self, h: mx.array) -> mx.array:
        """FFN sub-layer. Uses norm3. Returns net contribution, excludes h."""
        normed = self.norm3(h)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        return ffn_out

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Backward-compatible wrapper: standard residuals over sub-layers.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        core_out, new_state = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAL(nn.Module):
    """Titans with Memory as Layer - MLX.

    Memory processes input before attention in a sequential manner.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAL blocks
        self.blocks = [MALBlock(config, layer_idx=i) for i in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight

        # Step counter for AttnRes warmup tracking
        self._step_count = 0

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        if seq_len <= chunk_size:
            x, new_states = process_chunk(
                self.blocks, x, states, self.config, self._step_count
            )
        else:
            outputs = []
            new_states = list(states)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = x[:, chunk_start:chunk_end]
                chunk, new_states = process_chunk(
                    self.blocks, chunk, new_states, self.config, self._step_count
                )
                outputs.append(chunk)
            x = mx.concatenate(outputs, axis=1)

        # Output
        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states


# =============================================================================
# LMM: Long-term Memory Module (standalone)
# =============================================================================


class LMMBlock(nn.Module):
    """Standalone Long-term Memory Block (no attention) - MLX.

    Uses only the neural memory module as a sequence model.
    This tests the memory's ability to work independently.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Dropout
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        # Memory
        normed = self.norm1(x)
        mem_out, new_state = self.memory(normed, state=state)
        if self.dropout_p > 0:
            mem_out = nn.Dropout(self.dropout_p)(mem_out)
        x = x + mem_out

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        x = x + ffn_out

        return x, new_state


class TitansLMM(nn.Module):
    """Titans with only Long-term Memory (no attention) - MLX.

    A sequence model using only the neural memory module.
    Tests memory's standalone capability.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of LMM blocks
        self.blocks = [LMMBlock(config) for _ in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states
