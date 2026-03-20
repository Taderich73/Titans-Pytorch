# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
MLX-specific tests for the Titans architecture.

Tests that MLX models instantiate correctly, produce valid outputs,
persist memory state, and support lazy evaluation and batching.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

SMALL_CONFIG = {
    "dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "vocab_size": 1000,
    "num_memory_layers": 1,
    "memory_hidden_mult": 2.0,
    "num_persistent_tokens": 4,
    "chunk_size": 32,
    "window_size": 32,
    "max_seq_len": 128,
    "dropout": 0.0,
    "use_conv": False,
    "use_rope": True,
}


class TestModelInstantiation:
    """Test that models can be instantiated."""

    def test_mac_instantiation(self) -> None:
        """Test TitansMAC instantiation."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAC(config)
        assert model is not None

    def test_mag_instantiation(self) -> None:
        """Test TitansMAG instantiation."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAG(config)
        assert model is not None

    def test_mal_instantiation(self) -> None:
        """Test TitansMAL instantiation."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAL(config)
        assert model is not None

    def test_lmm_instantiation(self) -> None:
        """Test TitansLMM instantiation."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansLMM(config)
        assert model is not None


class TestOutputShapes:
    """Test that output shapes are correct for each model variant."""

    def _make_input(self) -> mx.array:
        np.random.seed(42)
        return mx.array(np.random.randint(0, 1000, (2, 32)).astype(np.int64))

    def test_mac_output_shape(self) -> None:
        """Test TitansMAC output shape."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAC(config)
        input_mlx = self._make_input()

        logits, _ = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (2, 32, config.vocab_size)

    def test_mag_output_shape(self) -> None:
        """Test TitansMAG output shape."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAG(config)
        input_mlx = self._make_input()

        logits, _ = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (2, 32, config.vocab_size)

    def test_mal_output_shape(self) -> None:
        """Test TitansMAL output shape."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAL(config)
        input_mlx = self._make_input()

        logits, _ = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (2, 32, config.vocab_size)

    def test_lmm_output_shape(self) -> None:
        """Test TitansLMM output shape."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansLMM(config)
        input_mlx = self._make_input()

        logits, _ = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (2, 32, config.vocab_size)


class TestMemoryStatePersistence:
    """Test that memory states persist correctly."""

    def test_mac_state_persistence(self) -> None:
        """Test MAC memory state persists across forward passes."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAC(config)
        input_mlx = mx.zeros((2, 32), dtype=mx.int32)

        _, states1 = model(input_mlx)
        mx.eval([s.weights[0] for s in states1 if s is not None])

        _, states2 = model(input_mlx, states=states1)
        mx.eval([s.weights[0] for s in states2 if s is not None])

        for s1, s2 in zip(states1, states2, strict=True):
            if s1 is not None and s2 is not None:
                w1 = np.array(s1.weights[0])
                w2 = np.array(s2.weights[0])
                assert not np.allclose(w1, w2, rtol=1e-3)

    def test_mag_state_persistence(self) -> None:
        """Test MAG memory state persists across forward passes."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAG(config)
        input_mlx = mx.zeros((2, 32), dtype=mx.int32)

        _, states1 = model(input_mlx)
        mx.eval([s.weights[0] for s in states1 if s is not None])

        _, states2 = model(input_mlx, states=states1)
        mx.eval([s.weights[0] for s in states2 if s is not None])

        for s1, s2 in zip(states1, states2, strict=True):
            if s1 is not None and s2 is not None:
                w1 = np.array(s1.weights[0])
                w2 = np.array(s2.weights[0])
                assert not np.allclose(w1, w2, rtol=1e-3)


class TestMLXFeatures:
    """Test MLX-specific features work correctly."""

    def test_lazy_evaluation(self) -> None:
        """Test that MLX lazy evaluation works."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansLMM(config)
        input_mlx = mx.zeros((1, 16), dtype=mx.int32)

        logits, _ = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (1, 16, config.vocab_size)

    def test_batched_processing(self) -> None:
        """Test batched processing works correctly."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAG(config)

        input_single = mx.zeros((1, 16), dtype=mx.int32)
        logits_single, _ = model(input_single)
        mx.eval(logits_single)

        input_batch = mx.zeros((4, 16), dtype=mx.int32)
        logits_batch, _ = model(input_batch)
        mx.eval(logits_batch)

        assert logits_single.shape == (1, 16, config.vocab_size)
        assert logits_batch.shape == (4, 16, config.vocab_size)


class TestGradientComputation:
    """Test that gradient computation works for memory updates."""

    def test_memory_gradient_flow(self) -> None:
        """Test memory gradients are computed correctly."""
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(**SMALL_CONFIG)
        memory = NeuralLongTermMemory(config)
        batch_size = 2
        seq_len = 8

        x = mx.random.normal((batch_size, seq_len, config.dim))
        state = memory.init_state(batch_size)

        output, new_state = memory(x, state=state)
        mx.eval(output)

        assert new_state is not None
        assert len(new_state.weights) > 0

        for w_old, w_new in zip(state.weights, new_state.weights, strict=True):
            mx.eval(w_old, w_new)
            assert not np.allclose(
                np.array(w_old), np.array(w_new), rtol=1e-3
            )


class TestChunkedProcessing:
    """Test chunked processing for MAC model."""

    def test_mac_chunking(self) -> None:
        """Test MAC processes long sequences in chunks correctly."""
        config = TitansConfig(**SMALL_CONFIG)
        model = TitansMAC(config)

        seq_len = config.chunk_size * 2 + 10
        input_mlx = mx.zeros((1, seq_len), dtype=mx.int32)

        logits, states = model(input_mlx)
        mx.eval(logits)

        assert logits.shape == (1, seq_len, config.vocab_size)
        assert all(s is not None for s in states)
