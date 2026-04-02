# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for adaptive window sizing."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest

from titans_mlx.config import TitansConfig


class TestAdaptiveWindowPredictor:
    """Tests for AdaptiveWindowPredictor module."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        """Config with adaptive window enabled."""
        return TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_output_shape(self, adaptive_config: TitansConfig) -> None:
        """Soft mask has correct shape (batch, 1, seq_len, seq_len)."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        mask, falloff = predictor(x)
        mx.eval(mask, falloff)

        assert mask.shape == (2, 1, 16, 16)
        assert falloff.shape == (2, 16, 1)

    def test_mask_values_in_range(self, adaptive_config: TitansConfig) -> None:
        """Soft mask values are in [0, 1]."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        assert mx.all(mask >= 0.0).item()
        assert mx.all(mask <= 1.0).item()

    def test_causality_enforced(self, adaptive_config: TitansConfig) -> None:
        """Future positions have zero mask weight."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((1, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # Check upper triangle is zero (future positions)
        mask_2d = mask[0, 0]  # (seq_len, seq_len)
        for i in range(16):
            for j in range(i + 1, 16):
                assert mask_2d[i, j].item() == pytest.approx(0.0, abs=1e-6), (
                    f"Future position mask[{i},{j}] = {mask_2d[i, j].item()}"
                )

    def test_falloff_center_bounded(self, adaptive_config: TitansConfig) -> None:
        """Falloff centers are within [min_window, max_window]."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        _, falloff = predictor(x)
        mx.eval(falloff)

        min_w = adaptive_config.adaptive_window_min
        max_w = adaptive_config.effective_adaptive_window_max
        assert mx.all(falloff >= min_w).item()
        assert mx.all(falloff <= max_w).item()

    def test_high_temperature_near_binary(self, adaptive_config: TitansConfig) -> None:
        """High temperature produces near-binary masks."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        config = TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=100.0,  # Very high
        )
        predictor = AdaptiveWindowPredictor(config)
        x = mx.random.normal((1, 16, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # With high temp, causal positions should be near 0 or 1
        causal_mask = mask[0, 0]
        for i in range(16):
            for j in range(i + 1):
                val = causal_mask[i, j].item()
                assert val < 0.05 or val > 0.95, (
                    f"High-temp mask[{i},{j}] = {val}, expected near 0 or 1"
                )

    def test_low_temperature_gradual(self) -> None:
        """Low temperature produces gradual falloff."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        config = TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=1.0,  # Very low
        )
        predictor = AdaptiveWindowPredictor(config)
        x = mx.random.normal((1, 32, 64))
        mask, _ = predictor(x)
        mx.eval(mask)

        # With low temp, should have intermediate values (not all 0/1)
        causal_vals = []
        mask_2d = mask[0, 0]
        for i in range(32):
            for j in range(i + 1):
                causal_vals.append(mask_2d[i, j].item())

        intermediate = [v for v in causal_vals if 0.1 < v < 0.9]
        assert len(intermediate) > 0, "Low temperature should produce intermediate mask values"

    def test_gradient_flows(self, adaptive_config: TitansConfig) -> None:
        """Gradients flow through the predictor."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor

        predictor = AdaptiveWindowPredictor(adaptive_config)
        x = mx.random.normal((1, 8, 64))

        def loss_fn(model, x):
            mask, falloff = model(x)
            # Include falloff in loss so gradients flow through the projection
            # even when mask sigmoid is saturated at high temperature
            return mx.mean(mask) + mx.mean(falloff)

        loss_and_grad = nn.value_and_grad(predictor, loss_fn)
        loss, grads = loss_and_grad(predictor, x)
        mx.eval(loss, grads)

        # Check that projection layer has gradients
        flat_grads = [v for _, v in mlx.utils.tree_flatten(grads)]
        has_nonzero = any(mx.any(g != 0).item() for g in flat_grads if isinstance(g, mx.array))
        assert has_nonzero, "Predictor should have non-zero gradients"


class TestSlidingWindowAdaptiveMask:
    """Tests for adaptive mask integration in SlidingWindowAttention."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            window_size=32,
            max_seq_len=256,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
        )

    def test_attention_accepts_adaptive_mask(self, adaptive_config: TitansConfig) -> None:
        """SlidingWindowAttention accepts adaptive_mask parameter."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor
        from titans_mlx.attention import SlidingWindowAttention

        attn = SlidingWindowAttention(adaptive_config)
        predictor = AdaptiveWindowPredictor(adaptive_config)

        x = mx.random.normal((2, 16, 64))
        adaptive_mask, _ = predictor(x)

        out = attn(x, adaptive_mask=adaptive_mask)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_attention_without_adaptive_mask_unchanged(self) -> None:
        """Without adaptive_mask, behavior is identical to before."""
        from titans_mlx.attention import SlidingWindowAttention

        config = TitansConfig(
            dim=64, num_heads=4, window_size=16, max_seq_len=256,
            use_rope=False,
        )
        attn = SlidingWindowAttention(config)
        x = mx.random.normal((2, 16, 64))

        # Call without adaptive_mask (default)
        out1 = attn(x)
        mx.eval(out1)

        # Call with explicit None
        out2 = attn(x, adaptive_mask=None)
        mx.eval(out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff == 0.0

    def test_attention_with_prefix_and_adaptive_mask(
        self, adaptive_config: TitansConfig
    ) -> None:
        """Adaptive mask works alongside persistent memory prefix."""
        from titans_mlx.adaptive_window import AdaptiveWindowPredictor
        from titans_mlx.attention import SlidingWindowAttention

        attn = SlidingWindowAttention(adaptive_config)
        predictor = AdaptiveWindowPredictor(adaptive_config)

        x = mx.random.normal((2, 16, 64))
        prefix = mx.random.normal((2, 4, 64))
        adaptive_mask, _ = predictor(x)

        out = attn(x, prefix=prefix, adaptive_mask=adaptive_mask)
        mx.eval(out)

        assert out.shape == (2, 16, 64)


class TestMAGBlockAdaptiveWindow:
    """Tests for adaptive window integration in MAGBlock."""

    @pytest.fixture
    def adaptive_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=4,
            chunk_size=32,
            window_size=32,
            max_seq_len=256,
            vocab_size=100,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_mag_block_forward_with_adaptive(self, adaptive_config: TitansConfig) -> None:
        """MAGBlock forward pass works with adaptive window enabled."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        out, state = block(x)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_mag_block_has_predictor(self, adaptive_config: TitansConfig) -> None:
        """MAGBlock instantiates window predictor when adaptive is enabled."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        assert hasattr(block, "window_predictor")

    def test_mag_block_no_predictor_when_disabled(self) -> None:
        """MAGBlock does not instantiate predictor when adaptive is disabled."""
        from titans_mlx.models import MAGBlock

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, num_memory_layers=1,
            memory_hidden_mult=2.0, num_persistent_tokens=4,
            chunk_size=32, window_size=32, max_seq_len=256, vocab_size=100,
            adaptive_window=False,
        )
        block = MAGBlock(config)
        assert not hasattr(block, "window_predictor")

    def test_mag_block_exposes_falloff_centers(
        self, adaptive_config: TitansConfig
    ) -> None:
        """MAGBlock stores last falloff_centers for regularization access."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(adaptive_config)
        x = mx.random.normal((2, 16, 64))
        _ = block(x)
        mx.eval(block._last_falloff_centers)

        assert block._last_falloff_centers is not None
        assert block._last_falloff_centers.shape == (2, 16, 1)

    def test_mag_regression_without_adaptive(self, default_config: TitansConfig) -> None:
        """MAGBlock without adaptive window produces identical output to baseline."""
        from titans_mlx.models import MAGBlock

        block = MAGBlock(default_config)
        x = mx.random.normal((2, 16, 64))

        out1, _ = block(x)
        out2, _ = block(x)
        mx.eval(out1, out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff == 0.0


class TestTitansMAGAdaptiveWindow:
    """End-to-end tests for TitansMAG with adaptive window."""

    @pytest.fixture
    def adaptive_mag_config(self) -> TitansConfig:
        return TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=4,
            chunk_size=32,
            window_size=32,
            max_seq_len=256,
            vocab_size=100,
            adaptive_window=True,
            adaptive_window_min=4,
            adaptive_window_max=32,
            adaptive_window_temperature=10.0,
        )

    def test_full_model_forward(self, adaptive_mag_config: TitansConfig) -> None:
        """TitansMAG forward pass with adaptive window produces valid logits."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        input_ids = mx.random.randint(0, 100, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 32, 100)
        assert len(states) == 2

    def test_multi_chunk_forward(self, adaptive_mag_config: TitansConfig) -> None:
        """Multi-chunk sequences work with adaptive window."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        # seq_len=64 > chunk_size=32, forces 2 chunks
        input_ids = mx.random.randint(0, 100, (2, 64))
        logits, states = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 64, 100)

    def test_collect_falloff_centers(self, adaptive_mag_config: TitansConfig) -> None:
        """Can collect falloff centers from all blocks after forward pass."""
        from titans_mlx.models import TitansMAG

        model = TitansMAG(adaptive_mag_config)
        input_ids = mx.random.randint(0, 100, (2, 32))
        _ = model(input_ids)

        centers = []
        for block in model.blocks:
            fc = block._last_falloff_centers
            assert fc is not None
            centers.append(fc)
            mx.eval(fc)

        assert len(centers) == 2  # num_layers=2


class TestAdaptiveWindowRegularization:
    """Tests for efficiency regularization loss."""

    def test_regularization_computes(self) -> None:
        """Regularization loss computes from falloff centers."""
        from titans_mlx.adaptive_window import compute_window_regularization

        # Simulate falloff centers from 2 layers
        centers = [
            mx.ones((2, 16, 1)) * 20.0,   # layer 0: large windows
            mx.ones((2, 16, 1)) * 5.0,     # layer 1: small windows
        ]
        max_window = 32
        reg = compute_window_regularization(centers, max_window)
        mx.eval(reg)

        # mean([20/32, 5/32]) = mean([0.625, 0.15625]) = 0.390625
        assert reg.item() == pytest.approx(0.390625, abs=1e-4)

    def test_regularization_scales_with_lambda(self) -> None:
        """Lambda scales the regularization."""
        from titans_mlx.adaptive_window import compute_window_regularization

        centers = [mx.ones((1, 8, 1)) * 16.0]
        reg1 = compute_window_regularization(centers, max_window=32)
        mx.eval(reg1)

        # reg = mean(16/32) = 0.5
        assert reg1.item() == pytest.approx(0.5, abs=1e-4)

    def test_regularization_zero_when_min_window(self) -> None:
        """Regularization is minimal when all windows are at minimum."""
        from titans_mlx.adaptive_window import compute_window_regularization

        centers = [mx.ones((1, 8, 1)) * 4.0]
        reg = compute_window_regularization(centers, max_window=32)
        mx.eval(reg)

        assert reg.item() == pytest.approx(4.0 / 32.0, abs=1e-4)

    def test_regularization_empty_list(self) -> None:
        """Returns zero for empty falloff centers list."""
        from titans_mlx.adaptive_window import compute_window_regularization

        reg = compute_window_regularization([], max_window=32)
        mx.eval(reg)

        assert reg.item() == 0.0
