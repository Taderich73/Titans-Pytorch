# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""TNT Hierarchical Memory System (PyTorch Implementation).

Paper alignment: TNT (Li et al., 2025) — Faithful, with three Plan-6 fixes:
    * W_init is a proper nn.Parameter (was accidentally frozen via .data).
    * Per-position causal Q-K projection via prefix-sum scan (was chunk-mean,
      which leaked future context within a chunk — violated causality).
    * Reset fires at every token t with t ≡ 0 (mod S_L) (was chunk-boundary only).

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
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        lr_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, MemoryState]:
        output, new_state, _gate_snapshot = self.memory(
            x, state=state, lr_scale=lr_scale
        )
        return output, new_state

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        return self.memory.retrieve(queries, state)

    def init_state(self, batch_size: int) -> MemoryState:
        return self.memory.init_state(batch_size)


class LocalMemory(nn.Module):
    """Local memory module for TNT with periodic state reset (Eq. 6)."""

    def __init__(
        self, config: TitansConfig, chunk_size: int, shard_length: int
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size
        self.shard_length = shard_length

        self.memory = NeuralLongTermMemory(config)

        # Learnable initial state W_init
        self._w_init = nn.ParameterList(
            [
                nn.Parameter(torch.randn_like(layer.weight) * config.init_std)
                for layer in self.memory.memory.layers
            ]
        )

        if config.use_qk_projection:
            self.qk_proj = QKProjection(config.dim)
        else:
            self.qk_proj = None

    @property
    def w_init(self) -> list[torch.Tensor]:
        """Return the learnable initial-state parameters (gradient-connected).

        Returns the live ``nn.Parameter`` tensors, not ``.data``, so autograd
        from downstream retrievals flows back into ``_w_init`` (TNT §4.1.1,
        Eq. 6).
        """
        return list(self._w_init)

    def init_state(self, batch_size: int) -> MemoryState:  # noqa: ARG002
        """Initialise a fresh ``MemoryState`` whose weights are clones of the
        learnable ``_w_init`` parameters.

        ``clone()`` (no ``.detach()``) preserves the autograd edge from the
        state's weights back to ``_w_init`` while guaranteeing that in-place
        mutations on the returned state do not alias the parameter.
        """
        weights = [w.clone() for w in self._w_init]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def maybe_reset(
        self,
        state: MemoryState,
        step_counter: int,
        batch_size: int,
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

    @staticmethod
    def _reset_segments(
        start_counter: int,
        seq_len: int,
        shard_length: int,
    ) -> list[tuple[int, int, bool]]:
        """Enumerate ``(local_begin, local_end, reset_at_begin)`` segments.

        Given a chunk of length ``seq_len`` whose first token has global
        step index ``start_counter`` and a shard length ``shard_length``,
        returns contiguous segments over local indices ``[0, seq_len)``
        such that within each segment no reset fires. The
        ``reset_at_begin`` flag indicates whether the segment should
        re-initialise the memory state at its local begin.

        Reset rule (paper Eq. 6): a reset fires AT every local index ``i``
        whose global step ``start_counter + i`` is a positive multiple
        of ``shard_length``.

        Fast path: when no reset falls inside ``[0, seq_len)``, exactly
        one segment ``(0, seq_len, False)`` is returned (assuming
        ``start_counter`` itself is not a positive multiple of
        ``shard_length``).

        Args:
            start_counter: Global step index of the first token in the chunk.
            seq_len: Length of the chunk.
            shard_length: Reset period (S_L).

        Returns:
            List of ``(begin, end, reset_at_begin)`` tuples covering
            ``[0, seq_len)`` contiguously.
        """
        segments: list[tuple[int, int, bool]] = []
        current_start = 0
        for i in range(1, seq_len):
            global_step = start_counter + i
            if global_step > 0 and global_step % shard_length == 0:
                segments.append((current_start, i, current_start != 0))
                current_start = i
        segments.append((current_start, seq_len, current_start != 0))

        # Resolve whether the FIRST segment resets at begin. It does iff
        # ``start_counter`` itself is a positive multiple of
        # ``shard_length``.
        first_begin, first_end, _ = segments[0]
        first_reset = start_counter > 0 and start_counter % shard_length == 0
        segments[0] = (first_begin, first_end, first_reset)
        return segments

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        lr_scale: float | torch.Tensor = 1.0,
        return_keys: bool = False,
        return_q: bool = False,
    ) -> tuple:
        """Forward a chunk through local memory.

        When both ``return_keys`` and ``return_q`` are True, returns
        ``(output, new_state, keys, q)`` so :class:`HierarchicalMemory` can
        run the efficient per-position QK projection. ``q`` is the
        L2-normalised, post-conv, post-SiLU query tensor that
        :class:`NeuralLongTermMemory` already computes internally; exposing
        it avoids a second projection pass through ``proj_q``.
        """
        if state is None:
            state = self.init_state(x.shape[0])
        if return_keys and return_q:
            output, new_state, _gate, k, q = self.memory(
                x,
                state=state,
                lr_scale=lr_scale,
                return_keys=True,
                return_q=True,
            )
            return output, new_state, k, q
        if return_keys:
            output, new_state, _gate_snapshot, keys = self.memory(
                x, state=state, lr_scale=lr_scale, return_keys=True
            )
            return output, new_state, keys
        output, new_state, _gate_snapshot = self.memory(
            x, state=state, lr_scale=lr_scale, return_keys=False
        )
        return output, new_state

    def forward_with_resets(
        self,
        x: torch.Tensor,
        state: MemoryState,
        start_counter: int,
        lr_scale: float | torch.Tensor = 1.0,
        return_keys: bool = False,
        return_q: bool = False,
    ) -> tuple:
        """Forward a chunk while honouring per-token shard resets (Eq. 6).

        Splits the chunk at every position whose global step is a
        positive multiple of ``shard_length``, re-initialises the
        memory state at each boundary, and concatenates the outputs.
        The fast path (no in-chunk reset and no reset at start) runs
        a single ``forward`` call identical to today's hot path.

        Args:
            x: Input chunk of shape ``(B, C, D)``.
            state: Local memory state entering this chunk.
            start_counter: Global step index of the first token in ``x``
                (equivalently, position-within-shard on entry, since
                the counter tracks ``global_step mod shard_length``).
            lr_scale: Memory gate LR scale, forwarded to NLTM.
            return_keys: If True, return concatenated L2-normalised keys.
            return_q: If True, return concatenated L2-normalised queries.
                Requires ``return_keys`` to also be True.

        Returns:
            Up to 5-tuple ``(output, new_state, end_counter[, keys[, q]])``.
            ``output`` has shape ``(B, C, D)``. ``end_counter`` equals
            ``(start_counter + C) mod shard_length`` and lies in
            ``[0, shard_length)``.
        """
        seq_len = x.shape[1]
        segments = self._reset_segments(
            start_counter=start_counter,
            seq_len=seq_len,
            shard_length=self.shard_length,
        )

        cur_state = state
        out_parts: list[torch.Tensor] = []
        k_parts: list[torch.Tensor] = []
        q_parts: list[torch.Tensor] = []

        for begin, end, reset_at_begin in segments:
            if reset_at_begin:
                cur_state = self.init_state(batch_size=x.shape[0])
            sub_x = x[:, begin:end, :]
            if return_keys and return_q:
                out, cur_state, k, q = self(
                    sub_x,
                    state=cur_state,
                    lr_scale=lr_scale,
                    return_keys=True,
                    return_q=True,
                )
                k_parts.append(k)
                q_parts.append(q)
            elif return_keys:
                out, cur_state, k = self(
                    sub_x,
                    state=cur_state,
                    lr_scale=lr_scale,
                    return_keys=True,
                )
                k_parts.append(k)
            else:
                out, cur_state = self(
                    sub_x,
                    state=cur_state,
                    lr_scale=lr_scale,
                )
            out_parts.append(out)

        output = out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=1)
        end_counter = (start_counter + seq_len) % self.shard_length

        if return_keys and return_q:
            keys = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=1)
            qs = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=1)
            return output, cur_state, end_counter, keys, qs
        if return_keys:
            keys = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=1)
            return output, cur_state, end_counter, keys
        return output, cur_state, end_counter

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
        self.local_memories = nn.ModuleList(
            [
                LocalMemory(
                    config, chunk_size=cs, shard_length=config.local_shard_length
                )
                for cs in config.active_local_chunk_sizes
            ]
        )

        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        nn.init.normal_(self.proj_out.weight, std=config.init_std)

    def init_state(self, batch_size: int) -> TNTMemoryState:
        global_state = self.global_memory.init_state(batch_size)
        local_states = [lm.init_state(batch_size) for lm in self.local_memories]
        qk_projections = [
            torch.zeros(
                self.config.dim, self.config.dim, device=next(self.parameters()).device
            )
            for _ in self.local_memories
        ]
        local_step_counters = [0] * len(self.local_memories)

        return TNTMemoryState(
            global_state=global_state,
            local_states=local_states,
            qk_projections=qk_projections,
            local_step_counters=local_step_counters,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: TNTMemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TNTMemoryState, None]:
        batch_size = x.shape[0]

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

        # 1. Update global memory and capture its output
        global_out, new_global_state = self.global_memory(
            x, state=state.global_state, lr_scale=global_lr_scale
        )

        # 2. Update each local memory and collect outputs
        new_local_states = []
        new_qk_projections = []
        new_step_counters = []
        local_outs = []

        use_per_position = self.config.tnt_qk_projection == "per_position"

        for i, local_mem in enumerate(self.local_memories):
            start_counter = state.local_step_counters[i]

            # Decide whether any reset at the chunk's *start* zeroes the
            # QK-projection carry. Per-token mid-chunk resets do not
            # currently zero the carry (it accumulates across the chunk
            # in a single call, matching pre-fix behaviour for the
            # common case where at most one boundary falls at the start).
            will_reset_at_start = (
                start_counter > 0 and start_counter % local_mem.shard_length == 0
            )
            if will_reset_at_start:
                # Preserve both device *and* dtype from the existing carry
                # (plain torch.zeros with device=x.device silently drops
                # any non-default dtype).
                qk_carry = state.qk_projections[i].new_zeros(
                    self.config.dim, self.config.dim
                )
            else:
                qk_carry = state.qk_projections[i]

            has_qk = local_mem.qk_proj is not None

            if has_qk and use_per_position:
                # Paper Eq. 7: per-position projection inside the chunk.
                # Pull (k, q) out of NLTM; project queries causally; then
                # re-retrieve local memory with the projected queries.
                (
                    _local_out_unprojected,
                    new_local_state,
                    end_counter,
                    normed_keys,
                    normed_q,
                ) = local_mem.forward_with_resets(
                    x,
                    state=state.local_states[i],
                    start_counter=start_counter,
                    lr_scale=local_lr_scale,
                    return_keys=True,
                    return_q=True,
                )
                projected_q, new_carry = local_mem.qk_proj(
                    normed_q,
                    normed_keys,
                    qk_carry,
                )
                # Retrieve with projected queries using the UPDATED memory
                # weights (matches paper Eq. 15: M_t^(i) · q_t applied to W^(i)_t).
                effective = local_mem.memory._get_effective_weights(
                    new_local_state.weights,
                    detach_base=False,
                )
                retrieved = local_mem.memory.memory.forward_with_weights(
                    projected_q,
                    effective,
                )
                local_out = local_mem.memory.proj_out(retrieved)
                new_qk_projections.append(new_carry)
            elif has_qk:
                # Legacy chunk-mean path: single carry applied uniformly.
                (
                    local_out,
                    new_local_state,
                    end_counter,
                    normed_keys,
                ) = local_mem.forward_with_resets(
                    x,
                    state=state.local_states[i],
                    start_counter=start_counter,
                    lr_scale=local_lr_scale,
                    return_keys=True,
                )
                new_carry = local_mem.qk_proj.update_carry(normed_keys, qk_carry)
                new_qk_projections.append(new_carry)
            else:
                local_out, new_local_state, end_counter = local_mem.forward_with_resets(
                    x,
                    state=state.local_states[i],
                    start_counter=start_counter,
                    lr_scale=local_lr_scale,
                )
                new_qk_projections.append(qk_carry)

            local_outs.append(local_out)
            new_local_states.append(new_local_state)
            new_step_counters.append(end_counter)

        new_state = TNTMemoryState(
            global_state=new_global_state,
            local_states=new_local_states,
            qk_projections=new_qk_projections,
            local_step_counters=new_step_counters,
        )

        # 3. Combine NLTM outputs directly — no redundant retrieve pass needed
        combined = global_out
        for local_out in local_outs:
            combined = combined + local_out
        output = self.proj_out(combined)
        # HierarchicalMemory doesn't produce GateSnapshot directly (its
        # sub-memories' snapshots are handled internally). Return None as
        # the third element to match the NeuralLongTermMemory 3-tuple contract.
        return output, new_state, None

    def retrieve(self, queries: torch.Tensor, state: TNTMemoryState) -> torch.Tensor:
        """Hierarchical retrieval per Eq. 15.

        For ``tnt_qk_projection == "per_position"``, the forward-time
        retrieval is already paper-exact; this ad-hoc ``retrieve`` is a
        best-effort projection using the committed per-chunk carry as a
        single operator (useful for out-of-trajectory lookups such as
        inference-time memory probing).

        For ``tnt_qk_projection == "chunk_mean"``, the same operator is
        applied — this is the legacy behaviour that the config flag
        preserves for checkpoints trained under the old semantics.
        """
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
