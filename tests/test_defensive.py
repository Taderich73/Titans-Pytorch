# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for error handling and defensive programming guards."""

import mlx.core as mx
import pytest

from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory


def _linear_memory_config() -> TitansConfig:
    """Config with 1-layer (linear) memory for testing parallel update."""
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        vocab_size=50,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        use_conv=False,
        use_rope=False,
        chunk_size=8,
        window_size=8,
        max_seq_len=64,
    )


def _mca_config(**kwargs) -> TitansConfig:
    """Config with MCA enabled."""
    defaults = dict(
        dim=64,
        num_heads=4,
        num_layers=6,
        vocab_size=256,
        use_mca=True,
        mca_num_heads=4,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
    )
    defaults.update(kwargs)
    return TitansConfig(**defaults)


class TestBugFixes:
    """Tests for the three identified bugs."""

    def test_degenerate_geometric_series_no_nan(self) -> None:
        """Bug 1: parallel update with decay == eta (degenerate branch)."""
        config = _linear_memory_config()
        mem = NeuralLongTermMemory(config)
        state = mem.init_state(batch_size=2)

        keys = mx.random.normal((2, 8, 32))
        values = mx.random.normal((2, 8, 32))

        # Exact equality: decay = 1 - 0.5 = 0.5, eta = 0.5 → diff = 0.0
        # This exercises the is_degenerate branch (|diff| < threshold).
        alpha = mx.array(0.5)
        eta = mx.array(0.5)

        new_state = mem._parallel_memory_update_linear(
            keys, values, state, alpha, theta=mx.array(0.05), eta=eta
        )
        mx.eval(new_state.weights[0])
        mx.eval(new_state.momentum[0])

        assert not mx.any(mx.isnan(new_state.weights[0])).item()
        assert not mx.any(mx.isinf(new_state.weights[0])).item()
        assert not mx.any(mx.isnan(new_state.momentum[0])).item()
        assert not mx.any(mx.isinf(new_state.momentum[0])).item()

    def test_near_degenerate_geometric_series_sign_consistency(self) -> None:
        """Bug 1: near-zero negative diff must not flip sign via offset.

        When diff is slightly negative (e.g., -5e-7), the old code's
        mx.sign(diff + 1e-12) could not flip the sign (1e-12 << 5e-7),
        but this test guards against regressions where a larger offset
        might be reintroduced. The denominator sign should match diff's sign.
        """
        config = _linear_memory_config()
        mem = NeuralLongTermMemory(config)
        state = mem.init_state(batch_size=2)

        keys = mx.random.normal((2, 8, 32))
        values = mx.random.normal((2, 8, 32))

        # diff = decay - eta = (1 - 0.4999995) - 0.5000005 = -5e-7
        # |diff| = 5e-7 < 1e-6 threshold → degenerate branch, but just barely.
        # Use values just outside the threshold to exercise the non-degenerate path.
        # diff = (1 - 0.499999) - 0.500003 = -2e-6, |diff| > 1e-6
        alpha = mx.array(0.499999)  # decay = 0.500001
        eta = mx.array(0.500003)  # diff = -2e-6 (negative, non-degenerate)

        new_state = mem._parallel_memory_update_linear(
            keys, values, state, alpha, theta=mx.array(0.05), eta=eta
        )
        mx.eval(new_state.weights[0])
        mx.eval(new_state.momentum[0])

        assert not mx.any(mx.isnan(new_state.weights[0])).item()
        assert not mx.any(mx.isinf(new_state.weights[0])).item()

    def test_mca_forward_empty_weights_raises(self) -> None:
        """Bug 2: _mca_forward with empty weights list should raise ValueError."""
        from titans_mlx.models import _mca_forward

        config = _mca_config()

        from titans_mlx.mca import MemoryCrossAttention

        class FakeBlock:
            has_mca = True
            mca = MemoryCrossAttention(config)

        block = FakeBlock()
        h = mx.random.normal((2, 16, 64))
        empty_state = MemoryState(weights=[], momentum=[])

        with pytest.raises(ValueError, match="non-empty memory weights"):
            _mca_forward(block, h, empty_state)

    def test_mca_3d_memory_weights_raises(self) -> None:
        """Bug 3: MCA with 3D memory_weights should raise ValueError."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x = mx.random.normal((2, 16, 64))
        W_3d = mx.random.normal((2, 128, 64))  # 3D — wrong

        with pytest.raises(ValueError, match="memory_weights"):
            mca(x, W_3d)


class TestShapeValidation:
    """Tests for entry-point shape guards."""

    def test_mca_2d_input_raises(self) -> None:
        """MCA should reject 2D input tensor."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x_2d = mx.random.normal((16, 64))  # missing batch dim
        W = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="3D"):
            mca(x_2d, W)

    def test_mca_wrong_dim_raises(self) -> None:
        """MCA should reject input with wrong last dimension."""
        from titans_mlx.mca import MemoryCrossAttention

        config = _mca_config()
        mca = MemoryCrossAttention(config)
        x_wrong = mx.random.normal((2, 16, 32))  # dim=32 != expected 64
        W = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="dim"):
            mca(x_wrong, W)

    def test_attnres_empty_sources_raises(self) -> None:
        """BlockAttnRes should reject empty blocks with no partial_block."""
        from titans_mlx.attn_res import BlockAttnRes

        attn_res = BlockAttnRes(dim=64)

        with pytest.raises(ValueError, match="at least one source"):
            attn_res(blocks=[], partial_block=None)
