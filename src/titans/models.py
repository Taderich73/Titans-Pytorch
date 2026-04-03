# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Titans Model Architectures (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.attention import SegmentedAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
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


def process_chunk(
    blocks: nn.ModuleList,
    chunk: torch.Tensor,
    states: list,
    config: TitansConfig,
    _step_count: int = 0,
) -> tuple[torch.Tensor, list]:
    """Process a single chunk through all blocks with standard residuals."""
    if config.use_attn_res:
        raise NotImplementedError(
            "AttnRes not yet ported. See archive/titans_mlx/attn_res.py"
        )

    new_states = []
    x = chunk
    for i, block in enumerate(blocks):
        core_out, new_state = block.core_forward(x, state=states[i])
        x = x + core_out

        if hasattr(block, "has_mca") and block.has_mca:
            raise NotImplementedError(
                "MCA not yet ported. See archive/titans_mlx/mca.py"
            )

        ffn_out = block.ffn_forward(x)
        x = x + ffn_out
        new_states.append(new_state)
    return x, new_states


class MACBlock(nn.Module):
    """Memory as Context Block."""

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        if config.use_tnt:
            raise NotImplementedError(
                "TNT hierarchical memory not yet ported. "
                "See archive/titans_mlx/tnt_memory.py"
            )
        if config.use_attn_res:
            raise NotImplementedError(
                "AttnRes not yet ported. See archive/titans_mlx/attn_res.py"
            )
        if config.use_mca and layer_idx in config.mca_active_insertion_layers:
            raise NotImplementedError(
                "MCA not yet ported. See archive/titans_mlx/mca.py"
            )

        self.memory = NeuralLongTermMemory(config)
        self.memory_query = nn.Parameter(
            torch.randn(1, 1, config.dim) * config.init_std
        )
        self.persistent = PersistentMemory(config)
        self.attention = SegmentedAttention(config)
        self.ffn = FeedForward(config)

        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.has_mca = False

    def core_forward(
        self,
        h: torch.Tensor,
        state: MemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        query = self.memory_query.expand(batch_size, -1, -1)
        memory_retrieved = self.memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        persistent = self.persistent(batch_size)
        normed = self.norm1(h)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        y_t = h + attn_out

        mem_out, new_state = self.memory(y_t, state=state, memory_gate=memory_gate)

        gated = torch.sigmoid(self.gate_norm_attn(y_t)) * torch.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        core_out = attn_out + gated
        return core_out, new_state

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
        core_out, new_state = self.core_forward(x, state=state)
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
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if states is None:
            states = [None] * len(self.blocks)

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
            x = torch.cat(outputs, dim=1)

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states


class TitansMAG(nn.Module):
    """Titans with Memory as Gate — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansMAG not yet ported. See archive/titans_mlx/models.py"
        )


class TitansMAL(nn.Module):
    """Titans with Memory as Layer — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansMAL not yet ported. See archive/titans_mlx/models.py"
        )


class TitansLMM(nn.Module):
    """Titans Long-term Memory Module (standalone) — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansLMM not yet ported. See archive/titans_mlx/models.py"
        )
