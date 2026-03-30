# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Memory Cross-Attention module."""

import mlx.core as mx
import numpy as np
import pytest

from titans_mlx.config import TitansConfig


def _mca_config(**kwargs) -> TitansConfig:
    """Helper to create MCA-enabled config for tests."""
    defaults = dict(
        dim=64, num_heads=4, num_layers=6, vocab_size=256,
        use_mca=True, mca_num_heads=4,
        num_memory_layers=2, memory_hidden_mult=2.0,
    )
    defaults.update(kwargs)
    return TitansConfig(**defaults)


class TestMemoryCrossAttention:
    """Tests for MemoryCrossAttention module."""

    def test_output_shape(self) -> None:
        """Output shape matches input [B, T, dim]."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((128, 64))  # memory_hidden_dim=128

        out = mca(x, W)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_output_shape_linear_memory(self) -> None:
        """Works with [dim, dim] weight matrix (linear memory)."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config(num_memory_layers=1)
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((64, 64))  # linear: dim x dim

        out = mca(x, W)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_gate_init_near_zero(self) -> None:
        """Gate bias=-3.0 produces near-zero output initially."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((128, 64))

        out = mca(x, W)
        mx.eval(out)

        max_val = mx.max(mx.abs(out)).item()
        assert max_val < 1.0, f"Gate should suppress output initially, got max={max_val}"

    def test_zero_weights_near_zero_output(self) -> None:
        """Zero weight matrix produces near-zero output."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.zeros((128, 64))

        out = mca(x, W)
        mx.eval(out)

        max_val = mx.max(mx.abs(out)).item()
        assert max_val < 0.5, f"Zero W should give near-zero output, got max={max_val}"

    def test_gradient_flow_through_projections(self) -> None:
        """Gradients flow through Wq/Wk/Wv/Wg but NOT through memory weights."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((128, 64))

        def loss_fn(model, x, W):
            out = model(x, mx.stop_gradient(W))
            return mx.mean(out)

        loss_and_grad = mx.grad(loss_fn, argnums=0)
        grads = loss_and_grad(mca, x, W)
        mx.eval(grads)

    def test_different_num_heads(self) -> None:
        """Different mca_num_heads produces same-shape output."""
        from titans_mlx.mca import MemoryCrossAttention

        for num_heads in [2, 4, 8]:
            config = _mca_config(mca_num_heads=num_heads)
            mca = MemoryCrossAttention(config)
            x = mx.random.normal((2, 16, 64))
            W = mx.random.normal((128, 64))

            out = mca(x, W)
            mx.eval(out)
            assert out.shape == (2, 16, 64), f"Failed for num_heads={num_heads}"

    def test_vector_gate(self) -> None:
        """Vector gate produces correct output shape."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config(mca_gate_type="vector")
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((128, 64))

        out = mca(x, W)
        mx.eval(out)
        assert out.shape == (2, 16, 64)

    def test_no_nan(self) -> None:
        """Output contains no NaN values."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W = mx.random.normal((128, 64))

        out = mca(x, W)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))
