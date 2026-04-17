# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Titans Model Architectures (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.attention import SegmentedAttention, SlidingWindowAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState
from titans.persistent import PersistentMemory


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight).to(orig_dtype)


class FeedForward(nn.Module):
    """Feed-forward network with SiLU gating."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.down_proj(hidden)


def _init_mca(block: nn.Module, config: TitansConfig, layer_idx: int) -> None:
    """Initialize MCA components on a block (shared across MAC/MAG/MAL)."""
    block.has_mca = layer_idx in config.mca_active_insertion_layers
    if block.has_mca:
        from titans.mca import MemoryCrossAttention

        block.mca = MemoryCrossAttention(config)
        if config.use_attn_res:
            from titans.attn_res import BlockAttnRes

            block.attn_res_mca = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )


def _mca_forward(block: nn.Module, h: torch.Tensor, mem_state) -> torch.Tensor:
    """MCA sub-layer: cross-attend to NeuralLTM weight rows."""
    weights = (
        mem_state.global_state.weights
        if hasattr(mem_state, "global_state")
        else mem_state.weights
    )
    if not weights:
        raise ValueError("MCA requires non-empty memory weights.")
    W = weights[0].detach()
    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {W.ndim}D")
    return block.mca(h, W)


def process_chunk(
    blocks: nn.ModuleList,
    chunk: torch.Tensor,
    states: list[MemoryState | TNTMemoryState],
    config: TitansConfig,
    _step_count: int = 0,
) -> tuple[torch.Tensor, list[MemoryState | TNTMemoryState], list]:
    """Process a single chunk through all blocks."""
    new_states = []
    gate_snapshots = []

    if not config.use_attn_res:
        # Standard residual path
        x = chunk
        for i, block in enumerate(blocks):
            core_out, new_state, gate_snapshot = block.core_forward(x, state=states[i])
            x = x + core_out

            if hasattr(block, "has_mca") and block.has_mca:
                mca_out = block.mca_forward(x, new_state)
                x = x + mca_out

            ffn_out = block.ffn_forward(x)
            x = x + ffn_out
            new_states.append(new_state)
            gate_snapshots.append(gate_snapshot)
        return x, new_states, gate_snapshots

    # AttnRes path — replaces residual connections per AttnRes paper
    S = config.attnres_sub_layer_block_size
    completed_blocks: list[torch.Tensor] = [chunk]  # b_0 = embedding
    partial_block: torch.Tensor | None = None
    sub_idx = 0
    warmup = (
        config.attnres_warmup_steps > 0
        and _step_count < config.attnres_warmup_steps
    )

    for i, block in enumerate(blocks):
        # --- Core sub-layer ---
        h, attn_weights = block.attn_res_core(completed_blocks, partial_block)

        memory_gate = None
        if not warmup:
            memory_gate = block.attn_res_gate(attn_weights)

        core_out, new_state, gate_snapshot = block.core_forward(
            h, state=states[i], memory_gate=memory_gate
        )
        new_states.append(new_state)
        gate_snapshots.append(gate_snapshot)

        if partial_block is None:
            partial_block = core_out
        else:
            partial_block = partial_block + core_out
        sub_idx += 1

        if sub_idx % S == 0:
            completed_blocks.append(partial_block)
            partial_block = None

        # --- MCA sub-layer ---
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

        if sub_idx % S == 0 or i == len(blocks) - 1:
            completed_blocks.append(partial_block)
            partial_block = None

    return completed_blocks[-1], new_states, gate_snapshots


class MACBlock(nn.Module):
    """Memory as Context Block."""

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        if config.use_tnt:
            from titans.tnt_memory import HierarchicalMemory
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)
        if config.mac_per_position_memory_query:
            # Paper Eq. 21: q_t = S^(t) W_Q (per-position linear projection).
            self.memory_query_proj = nn.Linear(config.dim, config.dim, bias=False)
            nn.init.normal_(self.memory_query_proj.weight, std=config.init_std)
        else:
            # Legacy: single learned query broadcast across batch and positions.
            self.memory_query = nn.Parameter(
                torch.randn(1, 1, config.dim) * config.init_std
            )
        self.persistent = PersistentMemory(config)
        self.attention = SegmentedAttention(config)
        self.ffn = FeedForward(config)

        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        _init_mca(self, config, layer_idx)

        if config.use_attn_res:
            from titans.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

    def mca_forward(self, h: torch.Tensor, mem_state) -> torch.Tensor:
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: torch.Tensor,
        state: MemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState, object]:
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        normed = self.norm1(h)
        if self.config.mac_per_position_memory_query:
            # Paper Eq. 21: per-position query q_t = S^(t) W_Q.  The query
            # lives in the same normalized space as the attention input.
            query = self.memory_query_proj(normed)
        else:
            # Legacy: single learned query broadcast across batch and positions.
            query = self.memory_query.expand(batch_size, -1, -1)
        memory_retrieved = self.memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        persistent = self.persistent(batch_size)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        y_t = h + attn_out

        # Paper Eq. 24: retrieve from M_{t-1} (pre-update), not post-update.
        # The memory is still UPDATED with y_t; only the output path reads
        # from the incoming state. TNT's HierarchicalMemory does not expose
        # this flag, so we only pass it when using NeuralLongTermMemory.
        if self.config.use_tnt:
            mem_out, new_state, gate_snapshot = self.memory(
                y_t, state=state, memory_gate=memory_gate
            )
        else:
            mem_out, new_state, gate_snapshot = self.memory(
                y_t,
                state=state,
                memory_gate=memory_gate,
                retrieve_after_update=False,
            )

        # Paper Eq. 25: o_t = y_t ⊗ M_t*(y_t) (element-wise multiply).
        # After Task 5, mem_out is per-position with shape matching y_t.
        core_out = y_t * mem_out
        return core_out, new_state, gate_snapshot

    def ffn_forward(self, h: torch.Tensor) -> torch.Tensor:
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        return ffn_out

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        core_out, new_state, _gate_snapshot = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight
        self._step_count = 0

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState | TNTMemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState | TNTMemoryState], list]:
        """Process a single chunk. Returns (logits, new_states, gate_snapshots).

        Args:
            input_ids: Token IDs, shape (B, seq_len) where seq_len <= chunk_size.
            states: Per-block memory states from a previous chunk, or None.

        Raises:
            ValueError: If seq_len > chunk_size. Callers must chunk externally.
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if seq_len > chunk_size:
            raise ValueError(
                f"seq_len ({seq_len}) > chunk_size ({chunk_size}). "
                f"Multi-chunk input is no longer supported in forward(). "
                f"Split input_ids into chunks and call forward() per chunk."
            )

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)
        x, new_states, gate_snapshots = process_chunk(
            self.blocks, x, states, self.config, self._step_count
        )

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states, gate_snapshots


class MAGBlock(nn.Module):
    """Memory as Gate Block (Titans Eq. 26-28).

    Architecture:
    1. y_t = SW-Attn([persistent || norm(h)]) — sliding window attention
    2. mem_out = M([persistent || normed]) — memory on normed input (NOT y_t)
    3. gated = sigmoid(gate_attn(y_t)) * sigmoid(gate_mem(mem_out))
    4. core_out = attn_out + gated
    """

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        if config.use_tnt:
            from titans.tnt_memory import HierarchicalMemory
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        self.persistent = PersistentMemory(config)
        self.attention = SlidingWindowAttention(config)
        self.ffn = FeedForward(config)

        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._last_falloff_centers: torch.Tensor | None = None
        if config.adaptive_window:
            from titans.adaptive_window import AdaptiveWindowPredictor

            self.window_predictor = AdaptiveWindowPredictor(config)
        _init_mca(self, config, layer_idx)

        if config.use_attn_res:
            from titans.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

    def mca_forward(self, h: torch.Tensor, mem_state) -> torch.Tensor:
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: torch.Tensor,
        state: MemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState, object]:
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        persistent = self.persistent(batch_size)
        normed = self.norm1(h)

        # Eq. 26: y = SW-Attn([p || norm(h)])
        adaptive_mask = None
        if hasattr(self, "window_predictor"):
            adaptive_mask, self._last_falloff_centers = self.window_predictor(normed)

        attn_out = self.attention(normed, prefix=persistent, adaptive_mask=adaptive_mask)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        y_t = h + attn_out

        # Eq. 27: Memory receives [persistent || normed], NOT y_t
        if persistent is not None:
            mem_input = torch.cat([persistent, normed], dim=1)
        else:
            mem_input = normed
        mem_out_full, new_state, gate_snapshot = self.memory(
            mem_input, state=state, memory_gate=memory_gate
        )
        # Slice off persistent prefix
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full

        # Eq. 28: Gated output
        gated = torch.sigmoid(self.gate_norm_attn(y_t)) * torch.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        core_out = attn_out + gated
        return core_out, new_state, gate_snapshot

    def ffn_forward(self, h: torch.Tensor) -> torch.Tensor:
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        return ffn_out

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        core_out, new_state, _gate_snapshot = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAG(nn.Module):
    """Titans with Memory as Gate (Titans §4.2)."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [MAGBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight
        self._step_count = 0

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState | TNTMemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState | TNTMemoryState], list]:
        """Process a single chunk. Returns (logits, new_states, gate_snapshots).

        Args:
            input_ids: Token IDs, shape (B, seq_len) where seq_len <= chunk_size.
            states: Per-block memory states from a previous chunk, or None.

        Raises:
            ValueError: If seq_len > chunk_size. Callers must chunk externally.
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if seq_len > chunk_size:
            raise ValueError(
                f"seq_len ({seq_len}) > chunk_size ({chunk_size}). "
                f"Multi-chunk input is no longer supported in forward(). "
                f"Split input_ids into chunks and call forward() per chunk."
            )

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)
        x, new_states, gate_snapshots = process_chunk(
            self.blocks, x, states, self.config, self._step_count
        )

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states, gate_snapshots


class MALBlock(nn.Module):
    """Memory as Layer Block (Titans Eq. 29-31).

    Architecture:
    1. mem_out = M([persistent || norm1(h)]) — memory first
    2. h_mid = h + mem_out — internal residual
    3. attn_out = SW-Attn(norm2(h_mid), prefix=persistent) — attention second
    4. core_out = mem_out + attn_out
    """

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        if config.use_tnt:
            from titans.tnt_memory import HierarchicalMemory
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = NeuralLongTermMemory(config)

        self.persistent = PersistentMemory(config)
        self.attention = SlidingWindowAttention(config)
        self.ffn = FeedForward(config)

        # norm1=memory, norm2=attention, norm3=FFN
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm3 = RMSNorm(config.dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._last_falloff_centers: torch.Tensor | None = None
        if config.adaptive_window:
            from titans.adaptive_window import AdaptiveWindowPredictor

            self.window_predictor = AdaptiveWindowPredictor(config)
        _init_mca(self, config, layer_idx)

        if config.use_attn_res:
            from titans.attn_res import AttnResMemoryGate, BlockAttnRes

            self.attn_res_core = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_ffn = BlockAttnRes(
                config.dim, logit_clip=config.attnres_logit_clip
            )
            self.attn_res_gate = AttnResMemoryGate()

    def mca_forward(self, h: torch.Tensor, mem_state) -> torch.Tensor:
        return _mca_forward(self, h, mem_state)

    def core_forward(
        self,
        h: torch.Tensor,
        state: MemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState, object]:
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        persistent = self.persistent(batch_size)

        # Eq. 29-30: Memory layer first
        normed = self.norm1(h)
        if persistent is not None:
            mem_input = torch.cat([persistent, normed], dim=1)
        else:
            mem_input = normed
        mem_out_full, new_state, gate_snapshot = self.memory(
            mem_input, state=state, memory_gate=memory_gate
        )
        if persistent is not None:
            mem_out = mem_out_full[:, persistent.shape[1] :, :]
        else:
            mem_out = mem_out_full
        if self.dropout is not None:
            mem_out = self.dropout(mem_out)

        # Internal residual: attention sees h + mem contribution
        h_mid = h + mem_out

        # Eq. 31: Attention on memory-enriched representation
        normed_mid = self.norm2(h_mid)
        adaptive_mask = None
        if hasattr(self, "window_predictor"):
            adaptive_mask, self._last_falloff_centers = self.window_predictor(normed_mid)

        attn_out = self.attention(normed_mid, prefix=persistent, adaptive_mask=adaptive_mask)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        core_out = mem_out + attn_out
        return core_out, new_state, gate_snapshot

    def ffn_forward(self, h: torch.Tensor) -> torch.Tensor:
        normed = self.norm3(h)
        ffn_out = self.ffn(normed)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        return ffn_out

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        core_out, new_state, _gate_snapshot = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAL(nn.Module):
    """Titans with Memory as Layer (Titans §4.3)."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [MALBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight
        self._step_count = 0

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState | TNTMemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState | TNTMemoryState], list]:
        """Process a single chunk. Returns (logits, new_states, gate_snapshots).

        Args:
            input_ids: Token IDs, shape (B, seq_len) where seq_len <= chunk_size.
            states: Per-block memory states from a previous chunk, or None.

        Raises:
            ValueError: If seq_len > chunk_size. Callers must chunk externally.
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if seq_len > chunk_size:
            raise ValueError(
                f"seq_len ({seq_len}) > chunk_size ({chunk_size}). "
                f"Multi-chunk input is no longer supported in forward(). "
                f"Split input_ids into chunks and call forward() per chunk."
            )

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)
        x, new_states, gate_snapshots = process_chunk(
            self.blocks, x, states, self.config, self._step_count
        )

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states, gate_snapshots


class LMMBlock(nn.Module):
    """Standalone Long-term Memory Block (no attention).

    Architecture: norm -> memory -> residual -> norm -> FFN -> residual.
    Tests memory's ability to work independently as a sequence model.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.memory = NeuralLongTermMemory(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        normed = self.norm1(x)
        mem_out, new_state, _gate_snapshot = self.memory(normed, state=state)
        if self.dropout is not None:
            mem_out = self.dropout(mem_out)
        x = x + mem_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        x = x + ffn_out

        return x, new_state


class TitansLMM(nn.Module):
    """Titans Long-term Memory Module (standalone, no attention).

    Uses only the neural memory module as a sequence model (Titans §4.3).
    Direct block iteration — does not use process_chunk.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [LMMBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState], list]:
        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_states, []
