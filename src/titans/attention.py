# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Attention modules for Titans architecture (PyTorch Implementation)."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig


def log_sdpa_backend() -> str:
    """Log which scaled_dot_product_attention backend is active.

    Returns one of: 'flash', 'mem_efficient', 'math', 'cudnn', 'unavailable'.
    """
    if not torch.cuda.is_available():
        return "cpu_only"
    backends = []
    if torch.backends.cuda.flash_sdp_enabled():
        backends.append("flash")
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        backends.append("mem_efficient")
    if torch.backends.cuda.math_sdp_enabled():
        backends.append("math")
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") and torch.backends.cuda.cudnn_sdp_enabled():
        backends.append("cudnn")
    return ",".join(backends) if backends else "none_enabled"


@lru_cache(maxsize=32)
def _cached_sliding_window_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """LRU-cached sliding window mask to avoid rebuilding every forward pass."""
    positions = torch.arange(seq_len, device=device)
    row_idx = positions.unsqueeze(1)
    col_idx = positions.unsqueeze(0)
    causal_mask = col_idx <= row_idx
    window_mask = (row_idx - col_idx) < window_size
    bool_mask = causal_mask & window_mask
    return torch.where(bool_mask, 0.0, float("-inf"))


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with proportional dimension support.

    When rope_proportion < 1.0, only the first fraction of dimension pairs
    receive rotary embeddings. The remaining pairs pass through unchanged,
    preserving them for semantic content (p-RoPE, as used in Gemma 4).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        rope_proportion: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.rotate_dim = 2 * (int(dim * rope_proportion) // 2)
        if self.rotate_dim > 0:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.rotate_dim, 2).float() / self.rotate_dim)
            )
            self.register_buffer("inv_freq", inv_freq)
            self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))
        self._max_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: (batch, heads, seq, head_dim)
            k: (batch, heads, seq, head_dim)
            seq_offset: Offset for position indices
        """
        if self.rotate_dim == 0:
            return q, k

        seq_len = q.shape[2]
        if seq_offset + seq_len > self._max_seq_len:
            self._build_cache(seq_offset + seq_len)

        cos = self.cos_cached[seq_offset : seq_offset + seq_len]
        sin = self.sin_cached[seq_offset : seq_offset + seq_len]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        return q_rotated, k_rotated

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x_rotate = x[..., : self.rotate_dim]
        x_pass = x[..., self.rotate_dim :]

        x1 = x_rotate[..., ::2]
        x2 = x_rotate[..., 1::2]

        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, rotate_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_even = x1 * cos - x2 * sin
        rotated_odd = x1 * sin + x2 * cos

        batch, heads, seq, half_dim = rotated_even.shape
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated = rotated.reshape(batch, heads, seq, half_dim * 2)

        return torch.cat([rotated, x_pass], dim=-1)


def _rearrange_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, seq, dim) -> (batch, heads, seq, head_dim)"""
    batch, seq, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)


def _rearrange_from_heads(x: torch.Tensor) -> torch.Tensor:
    """(batch, heads, seq, head_dim) -> (batch, seq, dim)"""
    batch, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention for MAG/MAL variants."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.scale = self.head_dim**-0.5

        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                rope_proportion=config.rope_proportion,
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=std)

    def _create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        return _cached_sliding_window_mask(seq_len, self.window_size, device)

    def forward(
        self,
        x: torch.Tensor,
        prefix: torch.Tensor | None = None,
        seq_offset: int = 0,
        adaptive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if prefix is not None:
            full_x = torch.cat([prefix, x], dim=1)
            prefix_len = prefix.shape[1]
        else:
            full_x = x
            prefix_len = 0

        q = self.proj_q(x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        if self.rope is not None:
            q, _ = self.rope(q, q, seq_offset=prefix_len + seq_offset)
            k, _ = self.rope(k, k, seq_offset=seq_offset)

        if adaptive_mask is not None:
            # adaptive_mask: (batch, 1, seq_len, seq_len) with values in [0, 1]
            # Convert to additive: log(mask + eps) so 1->0.0, 0->-inf
            if prefix_len > 0:
                prefix_attn = torch.zeros(
                    (batch_size, 1, seq_len, prefix_len), device=x.device
                )
                additive = torch.log(adaptive_mask + 1e-8)
                mask = torch.cat([prefix_attn, additive], dim=-1)
            else:
                mask = torch.log(adaptive_mask + 1e-8)
        else:
            mask = self._create_extended_mask(
                seq_len, full_x.shape[1], prefix_len, x.device
            )
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scale
        )

        output = _rearrange_from_heads(output)
        output = self.proj_out(output)
        return output

    def _create_extended_mask(
        self,
        query_len: int,
        key_len: int,
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        prefix_mask = torch.zeros((query_len, prefix_len), device=device)

        if key_len > prefix_len:
            main_mask = self._create_sliding_window_mask(query_len, device)
        else:
            main_mask = torch.zeros((query_len, 0), device=device)

        mask = torch.cat([prefix_mask, main_mask], dim=1)
        return mask.unsqueeze(0).unsqueeze(0)


class SegmentedAttention(nn.Module):
    """Segmented/Chunked Attention for MAC variant."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                rope_proportion=config.rope_proportion,
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        x: torch.Tensor,
        persistent: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        components = []
        if persistent is not None:
            components.append(persistent)
        if memory is not None:
            components.append(memory)
        components.append(x)

        full_x = torch.cat(components, dim=1)
        prefix_len = full_x.shape[1] - seq_len

        q = self.proj_q(full_x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        if self.rope is not None:
            q, k = self.rope(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )

        output = _rearrange_from_heads(output)
        output = self.proj_out(output)

        return output[:, prefix_len:]
