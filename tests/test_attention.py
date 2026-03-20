# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for attention modules (MLX)."""

import mlx.core as mx
import numpy as np

from titans_mlx.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans_mlx.config import TitansConfig


class TestRotaryPositionEmbedding:
    """Tests for RoPE."""

    def test_forward(self) -> None:
        """Test RoPE forward pass."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)

        q = mx.random.normal((2, 4, 16, 64))
        k = mx.random.normal((2, 4, 16, 64))

        q_rot, k_rot = rope(q, k)
        mx.eval(q_rot, k_rot)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_with_offset(self) -> None:
        """Test RoPE with sequence offset."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)

        q = mx.random.normal((2, 4, 16, 64))
        k = mx.random.normal((2, 4, 16, 64))

        q_rot, k_rot = rope(q, k, seq_offset=10)
        mx.eval(q_rot, k_rot)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_cache_rebuild(self) -> None:
        """Test cache rebuild for long sequences."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=32)

        q = mx.random.normal((2, 4, 64, 64))
        k = mx.random.normal((2, 4, 64, 64))

        q_rot, k_rot = rope(q, k)
        mx.eval(q_rot, k_rot)

        assert q_rot.shape == q.shape
        assert rope._cos_cached.shape[0] >= 64

    def test_apply_rotary_preserves_shape(self) -> None:
        """Test _apply_rotary preserves tensor shape."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)
        rope._build_cache(16)

        x = mx.random.normal((2, 4, 16, 64))
        cos = rope._cos_cached[:16]
        sin = rope._sin_cached[:16]

        rotated = rope._apply_rotary(x, cos, sin)
        mx.eval(rotated)

        assert rotated.shape == x.shape


class TestSlidingWindowAttention:
    """Tests for Sliding Window Attention."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test SWA forward pass."""
        attn = SlidingWindowAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))

        output = attn(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_forward_with_prefix(self, default_config: TitansConfig) -> None:
        """Test SWA with prefix tokens."""
        attn = SlidingWindowAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))
        prefix = mx.random.normal((2, 8, default_config.dim))

        output = attn(x, prefix=prefix)
        mx.eval(output)

        assert output.shape == x.shape

    def test_forward_with_offset(self, default_config: TitansConfig) -> None:
        """Test SWA with sequence offset."""
        attn = SlidingWindowAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))

        output = attn(x, seq_offset=16)
        mx.eval(output)

        assert output.shape == x.shape

    def test_sliding_window_mask(self, default_config: TitansConfig) -> None:
        """Test sliding window mask creation."""
        attn = SlidingWindowAttention(default_config)

        mask = attn._create_sliding_window_mask(16)
        mx.eval(mask)

        assert mask.shape == (16, 16)
        assert mask.dtype == mx.bool_

        mask_np = np.array(mask)
        assert mask_np[0, 0]
        assert not mask_np[0, 1]

        if default_config.window_size < 16:
            assert not mask_np[15, 0]

    def test_extended_mask(self, default_config: TitansConfig) -> None:
        """Test extended mask for prefix attention."""
        attn = SlidingWindowAttention(default_config)

        mask = attn._create_extended_mask(
            query_len=8, key_len=16, prefix_len=8
        )
        mx.eval(mask)

        assert mask.shape == (1, 1, 8, 16)

        mask_np = np.array(mask)
        assert mask_np[0, 0, 0, :8].all()

    def test_without_rope(self) -> None:
        """Test SWA without RoPE."""
        config = TitansConfig(dim=64, num_heads=4, use_rope=False)
        attn = SlidingWindowAttention(config)
        x = mx.random.normal((2, 16, config.dim))

        output = attn(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_different_window_sizes(self) -> None:
        """Test different window sizes."""
        for window_size in [4, 8, 16]:
            config = TitansConfig(dim=64, num_heads=4, window_size=window_size)
            attn = SlidingWindowAttention(config)
            x = mx.random.normal((2, 32, config.dim))

            output = attn(x)
            mx.eval(output)

            assert output.shape == x.shape


class TestSegmentedAttention:
    """Tests for Segmented Attention (MAC Core)."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test segmented attention forward pass."""
        attn = SegmentedAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))

        output = attn(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_forward_with_persistent(self, default_config: TitansConfig) -> None:
        """Test with persistent memory tokens."""
        attn = SegmentedAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))
        persistent = mx.random.normal((2, 8, default_config.dim))

        output = attn(x, persistent=persistent)
        mx.eval(output)

        assert output.shape == x.shape

    def test_forward_with_memory(self, default_config: TitansConfig) -> None:
        """Test with memory tokens."""
        attn = SegmentedAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))
        memory = mx.random.normal((2, 16, default_config.dim))

        output = attn(x, memory=memory)
        mx.eval(output)

        assert output.shape == x.shape

    def test_forward_with_all_components(
        self, default_config: TitansConfig
    ) -> None:
        """Test with persistent and memory tokens."""
        attn = SegmentedAttention(default_config)
        x = mx.random.normal((2, 32, default_config.dim))
        persistent = mx.random.normal((2, 8, default_config.dim))
        memory = mx.random.normal((2, 16, default_config.dim))

        output = attn(x, persistent=persistent, memory=memory)
        mx.eval(output)

        assert output.shape == x.shape

    def test_without_rope(self) -> None:
        """Test without RoPE."""
        config = TitansConfig(dim=64, num_heads=4, use_rope=False)
        attn = SegmentedAttention(config)
        x = mx.random.normal((2, 16, config.dim))

        output = attn(x)
        mx.eval(output)

        assert output.shape == x.shape
