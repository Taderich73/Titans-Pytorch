# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""TNT Hierarchical Memory System (PyTorch Implementation).

Implements the TNT paper's hierarchical memory:
- GlobalMemory: Large chunks (C_G) for long-range context (Eq. 5)
- LocalMemory: Small chunks (C_L) with periodic resets (Eq. 6)
- HierarchicalMemory: Combined retrieval with Q-K projection (Eq. 7, 15)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans.qk_projection import QKProjection


class GlobalMemory(nn.Module):
    """Global memory module for TNT (Eq. 5)."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.memory = NeuralLongTermMemory(config)

    def forward(
        self, x: torch.Tensor, state: MemoryState | None = None,
        lr_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, MemoryState]:
        return self.memory(x, state=state, lr_scale=lr_scale)

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        return self.memory.retrieve(queries, state)

    def init_state(self, batch_size: int) -> MemoryState:
        return self.memory.init_state(batch_size)


class LocalMemory(nn.Module):
    """Local memory module for TNT with periodic state reset (Eq. 6)."""

    def __init__(self, config: TitansConfig, chunk_size: int, shard_length: int) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size
        self.shard_length = shard_length

        self.memory = NeuralLongTermMemory(config)

        # Learnable initial state W_init
        self._w_init = nn.ParameterList([
            nn.Parameter(torch.randn_like(layer.weight) * config.init_std)
            for layer in self.memory.memory.layers
        ])

        if config.use_qk_projection:
            self.qk_proj = QKProjection(config.dim)
        else:
            self.qk_proj = None

    @property
    def w_init(self) -> list[torch.Tensor]:
        return [w.data for w in self._w_init]

    def init_state(self, batch_size: int) -> MemoryState:  # noqa: ARG002
        weights = [w.data.clone() for w in self._w_init]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def maybe_reset(
        self, state: MemoryState, step_counter: int, batch_size: int,
    ) -> tuple[MemoryState, int]:
        """Reset local memory state at shard boundaries.

        Args:
            state: Current local memory state.
            step_counter: Cumulative tokens processed since last reset.
            batch_size: Batch size to use when reinitializing the state.
                Forwarded to init_state for API consistency; LocalMemory
                weights are currently shape [out_dim, in_dim] with no batch
                axis (see init_state).

        Returns:
            Tuple of (state, counter). On reset, returns a freshly
            initialized state and counter=0. Otherwise returns the inputs
            unchanged.
        """
        if step_counter > 0 and step_counter % self.shard_length == 0:
            return self.init_state(batch_size=batch_size), 0
        return state, step_counter

    def forward(
        self, x: torch.Tensor, state: MemoryState | None = None,
        lr_scale: float | torch.Tensor = 1.0, return_keys: bool = False,
    ) -> tuple[torch.Tensor, MemoryState] | tuple[torch.Tensor, MemoryState, torch.Tensor]:
        if state is None:
            state = self.init_state(x.shape[0])
        return self.memory(x, state=state, lr_scale=lr_scale, return_keys=return_keys)

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        return self.memory.retrieve(queries, state)


class HierarchicalMemory(nn.Module):
    """TNT Hierarchical Memory System (Eq. 15).

    Retrieval: o_t = f(V, q_t) + Σ f(W^(i), M_t^(i) · q_t)
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.global_memory = GlobalMemory(config)
        self.local_memories = nn.ModuleList([
            LocalMemory(config, chunk_size=cs, shard_length=config.local_shard_length)
            for cs in config.active_local_chunk_sizes
        ])

        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        nn.init.normal_(self.proj_out.weight, std=config.init_std)

    def init_state(self, batch_size: int) -> TNTMemoryState:
        global_state = self.global_memory.init_state(batch_size)
        local_states = [lm.init_state(batch_size) for lm in self.local_memories]
        local_inits = [[w.clone() for w in lm.w_init] for lm in self.local_memories]
        qk_projections = [
            torch.zeros(self.config.dim, self.config.dim,
                        device=next(self.parameters()).device)
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

    def forward(
        self, x: torch.Tensor, state: TNTMemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TNTMemoryState]:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if state is None:
            state = self.init_state(batch_size)

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
        _, new_global_state = self.global_memory(
            x, state=state.global_state, lr_scale=global_lr_scale
        )

        # 2. Update each local memory
        new_local_states = []
        new_qk_projections = []
        new_step_counters = []

        for i, local_mem in enumerate(self.local_memories):
            local_state, counter = local_mem.maybe_reset(
                state.local_states[i],
                state.local_step_counters[i],
                batch_size=batch_size,
            )

            if counter == 0 and state.local_step_counters[i] > 0:
                qk_carry = torch.zeros(self.config.dim, self.config.dim, device=x.device)
            else:
                qk_carry = state.qk_projections[i]

            needs_keys = local_mem.qk_proj is not None
            if needs_keys:
                local_out, new_local_state, normed_keys = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale, return_keys=True,
                )
            else:
                local_out, new_local_state = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale,
                )
            new_local_states.append(new_local_state)

            if needs_keys:
                new_carry = local_mem.qk_proj.update_carry(normed_keys, qk_carry)
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

        # 3. Retrieve
        output = self.retrieve(x, new_state)
        return output, new_state

    def retrieve(self, queries: torch.Tensor, state: TNTMemoryState) -> torch.Tensor:
        """Hierarchical retrieval per Eq. 15."""
        global_out = self.global_memory.retrieve(queries, state.global_state)

        output = global_out
        for i, local_mem in enumerate(self.local_memories):
            if local_mem.qk_proj is not None:
                proj_matrix = state.qk_projections[i]
                projected_q = queries @ proj_matrix.T
            else:
                projected_q = queries

            local_out = local_mem.retrieve(projected_q, state.local_states[i])
            output = output + local_out

        return self.proj_out(output)
