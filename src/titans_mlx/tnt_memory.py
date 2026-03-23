# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TNT Hierarchical Memory System (MLX Implementation).

Implements the core TNT innovation from "TNT: Improving Chunkwise Training
for Test-Time Memorization" (Li et al., 2025):

- GlobalMemory: Long-range context via large chunk sizes (C_G)
- LocalMemory: Fine-grained detail via small chunks (C_L) with periodic
  resets to learnable initial weights (W_init) at shard boundaries (S_L)
- HierarchicalMemory: Combines global + N local memories with Q-K
  projection for local retrieval (TNT Eq. 5-7, 15)

Key equations:
    Global update (Eq. 5):
        V_{(n+1)C_G} ← V_{nC_G} - Σ η_t ∇_V L(f(V, k_t), v_t)

    Local update (Eq. 6):
        W_t ← W_init                               if t ≡ 0 (mod S_L)
               W_{t-1} - Σ η_τ ∇_W L(f(W, k_τ), v_τ)   otherwise

    Hierarchical retrieval (Eq. 15):
        o_t = f(V, q_t) + Σ_{i=1}^{N} f(W^(i), M_t^(i) · q_t)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans_mlx.qk_projection import QKProjection


class GlobalMemory(nn.Module):
    """Global memory module for TNT (Eq. 5).

    Processes the input with large chunk sizes (C_G) to capture
    long-range dependencies. The state evolves sequentially between
    global chunks. Uses raw query (no Q-K projection) for retrieval.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.memory = NeuralLongTermMemory(config)

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        lr_scale: float | mx.array = 1.0,
    ) -> tuple[mx.array, MemoryState]:
        """Update global memory with input chunk.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous global memory state
            lr_scale: Multiplicative scale factor for learning rate

        Returns:
            Tuple of (output, new_state)
        """
        return self.memory(x, state=state, lr_scale=lr_scale)

    def retrieve(self, queries: mx.array, state: MemoryState) -> mx.array:
        """Retrieve from global memory using raw queries (no Q-K projection).

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Global memory state

        Returns:
            Retrieved values (batch, seq, dim)
        """
        return self.memory.retrieve(queries, state)

    def init_state(self, batch_size: int) -> MemoryState:
        """Initialize global memory state."""
        return self.memory.init_state(batch_size)


class LocalMemory(nn.Module):
    """Local memory module for TNT with periodic state reset (Eq. 6).

    Operates on small chunks (C_L) for fine-grained memorization.
    State periodically resets to a learnable W_init at shard boundaries
    (every S_L tokens), breaking sequential dependencies for parallelism.

    Each local memory has its own Q-K projection that aligns the query
    space to the key space used during memory compression.
    """

    def __init__(
        self,
        config: TitansConfig,
        chunk_size: int,
        shard_length: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size
        self.shard_length = shard_length

        # Core memory module
        self.memory = NeuralLongTermMemory(config)

        # Learnable initial state W_init — distinct from memory.memory weights.
        # This is what we reset to at shard boundaries. Wrapped in a list
        # to match MemoryState.weights structure (one array per MLP layer).
        self._w_init = [
            mx.random.normal(layer.weight.shape) * config.init_std
            for layer in self.memory.memory.layers
        ]

        # Q-K Projection for this local memory
        if config.use_qk_projection:
            self.qk_proj = QKProjection(config.dim)
        else:
            self.qk_proj = None

    @property
    def w_init(self) -> list[mx.array]:
        """Learnable initial weights for periodic reset."""
        return self._w_init

    def init_state(self, batch_size: int) -> MemoryState:
        """Initialize local memory with learnable W_init."""
        weights = [mx.array(w) for w in self._w_init]
        momentum = [mx.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def maybe_reset(
        self,
        state: MemoryState,
        step_counter: int,
    ) -> tuple[MemoryState, int]:
        """Reset state to W_init if at shard boundary.

        Args:
            state: Current local memory state
            step_counter: Current position within shard

        Returns:
            Tuple of (possibly reset state, updated counter)
        """
        if step_counter > 0 and step_counter % self.shard_length == 0:
            return self.init_state(batch_size=1), 0
        return state, step_counter

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        lr_scale: float | mx.array = 1.0,
    ) -> tuple[mx.array, MemoryState]:
        """Update local memory with input chunk.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous local memory state
            lr_scale: Multiplicative scale factor for learning rate

        Returns:
            Tuple of (output, new_state)
        """
        if state is None:
            state = self.init_state(x.shape[0])
        return self.memory(x, state=state, lr_scale=lr_scale)

    def retrieve(
        self,
        queries: mx.array,
        state: MemoryState,
    ) -> mx.array:
        """Retrieve from local memory.

        Args:
            queries: Query vectors (batch, seq, dim) — should already be
                Q-K projected by the caller if use_qk_projection is enabled
            state: Local memory state

        Returns:
            Retrieved values (batch, seq, dim)
        """
        return self.memory.retrieve(queries, state)


class HierarchicalMemory(nn.Module):
    """TNT Hierarchical Memory System (Eq. 15).

    Combines one global memory with N local memories at different resolutions.
    Global memory captures long-range context with large chunks (C_G).
    Local memories capture fine-grained details with small chunks (C_L).
    Local memories periodically reset to learnable initial weights,
    enabling context parallelism across shards.

    Retrieval (Eq. 15):
        o_t = f(V, q_t) + Σ_{i=1}^{N} f(W^(i), M_t^(i) · q_t)
              └─ global ─┘   └──── local with Q-K projection ────┘
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # One global memory (large chunk size C_G)
        self.global_memory = GlobalMemory(config)

        # N local memories (each with its own chunk size C_L_i)
        self.local_memories = [
            LocalMemory(
                config,
                chunk_size=cs,
                shard_length=config.local_shard_length,
            )
            for cs in config.active_local_chunk_sizes
        ]

        # Output projection (combines global + local outputs)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out.weight = (
            mx.random.normal(self.proj_out.weight.shape) * config.init_std
        )

    def init_state(self, batch_size: int) -> TNTMemoryState:
        """Initialize the full hierarchical memory state.

        Args:
            batch_size: Batch size

        Returns:
            Initialized TNTMemoryState
        """
        global_state = self.global_memory.init_state(batch_size)
        local_states = [lm.init_state(batch_size) for lm in self.local_memories]
        local_inits = [
            [mx.array(w) for w in lm.w_init] for lm in self.local_memories
        ]
        qk_projections = [
            mx.zeros((self.config.dim, self.config.dim))
            for _ in self.local_memories
        ]
        local_step_counters = [0] * len(self.local_memories)

        return TNTMemoryState(
            global_state=global_state,
            local_states=local_states,
            local_inits=local_inits,
            qk_projections=qk_projections,
            local_step_counters=local_step_counters,
        )

    def __call__(
        self,
        x: mx.array,
        state: TNTMemoryState | None = None,
        memory_gate: mx.array | None = None,
    ) -> tuple[mx.array, TNTMemoryState]:
        """Process input through hierarchical memory.

        Updates global memory and each local memory. Retrieval is done
        via the retrieve() method using the updated state.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous hierarchical memory state
            memory_gate: Optional scalar importance weight from AttnRes;
                translated to lr_scale based on config flags

        Returns:
            Tuple of (retrieved_output, new_state)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if state is None:
            state = self.init_state(batch_size)

        # Determine lr_scale from memory_gate
        global_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_global_memory
            else 1.0
        )
        local_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_local_memory
            else 1.0
        )

        # 1. Update global memory
        global_out, new_global_state = self.global_memory(
            x, state=state.global_state, lr_scale=global_lr_scale
        )

        # 2. Update each local memory (with periodic reset check)
        new_local_states = []
        new_qk_projections = []
        new_step_counters = []

        for i, local_mem in enumerate(self.local_memories):
            # Check for shard boundary reset
            local_state, counter = local_mem.maybe_reset(
                state.local_states[i],
                state.local_step_counters[i],
            )

            # Reset Q-K projection if local memory was reset
            if counter == 0 and state.local_step_counters[i] > 0:
                qk_carry = mx.zeros((self.config.dim, self.config.dim))
            else:
                qk_carry = state.qk_projections[i]

            # Update local memory
            local_out, new_local_state = local_mem(x, state=local_state, lr_scale=local_lr_scale)
            new_local_states.append(new_local_state)

            # Update Q-K projection state with keys from this chunk
            if local_mem.qk_proj is not None:
                # Extract normalized keys from the memory's projection
                k = local_mem.memory.proj_k(x)
                if local_mem.memory.use_conv:
                    k = local_mem.memory.conv_k(k)[:, :seq_len, :]
                k = nn.silu(k)
                # L2-norm in float32 to avoid bfloat16 underflow
                k_f32 = k.astype(mx.float32)
                k = (
                    k_f32 / (mx.sqrt(mx.sum(k_f32 * k_f32, axis=-1, keepdims=True)) + 1e-8)
                ).astype(k.dtype)

                # Update projection matrix
                _, new_carry = local_mem.qk_proj(
                    mx.zeros_like(x),  # queries not needed for carry update
                    k,
                    qk_carry,
                )
                new_qk_projections.append(new_carry)
            else:
                new_qk_projections.append(qk_carry)

            new_step_counters.append(counter + seq_len)

        new_state = TNTMemoryState(
            global_state=new_global_state,
            local_states=new_local_states,
            local_inits=state.local_inits,
            qk_projections=new_qk_projections,
            local_step_counters=new_step_counters,
        )

        # 3. Retrieve from hierarchical memory using updated state
        output = self.retrieve(x, new_state)

        return output, new_state

    def retrieve(
        self,
        queries: mx.array,
        state: TNTMemoryState,
    ) -> mx.array:
        """Hierarchical retrieval per Eq. 15.

        o_t = f(V, q_t) + Σ_{i=1}^{N} f(W^(i), M_t^(i) · q_t)

        Global retrieval uses raw queries. Local retrieval projects
        queries through Q-K projection matrices when enabled.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Current hierarchical memory state

        Returns:
            Combined retrieval output (batch, seq, dim)
        """
        # Global: raw query retrieval
        global_out = self.global_memory.retrieve(queries, state.global_state)

        # Local: Q-K projected queries
        output = global_out
        for i, local_mem in enumerate(self.local_memories):
            if local_mem.qk_proj is not None:
                # Build per-position projection from carry-over state
                # For retrieval, we use the accumulated projection matrix
                # as a single transform (not per-position cumsum)
                proj_matrix = state.qk_projections[i]  # (D, D)
                # Apply projection: q' = M_t · q
                projected_q = queries @ proj_matrix.T  # (B, S, D)
            else:
                projected_q = queries

            local_out = local_mem.retrieve(projected_q, state.local_states[i])
            output = output + local_out

        return self.proj_out(output)
