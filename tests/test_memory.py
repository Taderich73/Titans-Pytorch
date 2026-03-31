# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Neural Long-term Memory module (MLX)."""

import mlx.core as mx
import numpy as np
import pytest

from titans_mlx.config import TitansConfig
from titans_mlx.memory import (
    MemoryMLP,
    MemoryState,
    NeuralLongTermMemory,
    get_activation,
)


class TestGetActivation:
    """Tests for get_activation utility."""

    def test_valid_activations(self) -> None:
        """Test valid activation functions return callables."""
        for name in ("silu", "gelu", "relu"):
            fn = get_activation(name)
            assert callable(fn)
            result = fn(mx.array([0.0, 1.0, -1.0]))
            mx.eval(result)
            assert result.shape == (3,)

    def test_invalid_activation(self) -> None:
        """Test invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("invalid")


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_creation(self) -> None:
        """Test MemoryState creation."""
        weights = [mx.random.normal((64, 64))]
        momentum = [mx.zeros((64, 64))]
        state = MemoryState(weights=weights, momentum=momentum)

        assert len(state.weights) == 1
        assert len(state.momentum) == 1

    def test_detach(self) -> None:
        """Test detach creates stopped-gradient copies."""
        weights = [mx.random.normal((64, 64))]
        momentum = [mx.random.normal((64, 64))]
        state = MemoryState(weights=weights, momentum=momentum)

        detached = state.detach()

        assert len(detached.weights) == 1
        assert len(detached.momentum) == 1
        assert detached.weights[0].shape == (64, 64)

    def test_clone(self) -> None:
        """Test clone creates independent copy."""
        weights = [mx.random.normal((64, 64))]
        momentum = [mx.random.normal((64, 64))]
        mx.eval(weights[0], momentum[0])
        state = MemoryState(weights=weights, momentum=momentum)

        cloned = state.clone()
        mx.eval(cloned.weights[0])

        original_val = np.array(state.weights[0])[0, 0]
        cloned_val = np.array(cloned.weights[0])[0, 0]
        assert np.isclose(original_val, cloned_val)


class TestMemoryMLP:
    """Tests for MemoryMLP module."""

    def test_linear_memory(self, default_config: TitansConfig) -> None:
        """Test linear memory (single layer)."""
        config = TitansConfig(
            dim=default_config.dim,
            num_memory_layers=1,
        )
        mlp = MemoryMLP(config)

        assert len(mlp.layers) == 1

        x = mx.random.normal((2, 16, config.dim))
        y = mlp(x)
        mx.eval(y)

        assert y.shape == x.shape

    def test_deep_memory(self, default_config: TitansConfig) -> None:
        """Test deep memory (multiple layers)."""
        config = TitansConfig(
            dim=default_config.dim,
            num_memory_layers=3,
            memory_hidden_mult=2.0,
        )
        mlp = MemoryMLP(config)

        assert len(mlp.layers) == 3

        x = mx.random.normal((2, 16, config.dim))
        y = mlp(x)
        mx.eval(y)

        assert y.shape == x.shape

    def test_get_weights(self, default_config: TitansConfig) -> None:
        """Test get_weights returns weight matrices."""
        mlp = MemoryMLP(default_config)
        weights = mlp.get_weights()

        assert len(weights) == default_config.num_memory_layers

    def test_forward_with_weights(self, default_config: TitansConfig) -> None:
        """Test forward_with_weights uses explicit weights."""
        mlp = MemoryMLP(default_config)
        weights = mlp.get_weights()

        x = mx.random.normal((2, 8, default_config.dim))
        y = mlp.forward_with_weights(x, weights)
        mx.eval(y)

        assert y.shape == x.shape

    def test_compute_loss(self, default_config: TitansConfig) -> None:
        """Test associative memory loss computation."""
        mlp = MemoryMLP(default_config)

        keys = mx.random.normal((2, 8, default_config.dim))
        values = mx.random.normal((2, 8, default_config.dim))

        loss = mlp.compute_loss(keys, values)
        mx.eval(loss)

        assert loss.ndim == 0
        assert float(loss) >= 0


class TestNeuralLongTermMemory:
    """Tests for NeuralLongTermMemory module."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward pass without initial state."""
        memory = NeuralLongTermMemory(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = memory(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None
        assert len(state.weights) == default_config.num_memory_layers

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward pass with existing state."""
        memory = NeuralLongTermMemory(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = memory(x)
        mx.eval(state1.weights[0])

        output2, state2 = memory(x, state=state1)
        mx.eval(output2)

        assert output2.shape == x.shape
        assert state2 is not None

    def test_forward_no_return_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without returning state."""
        memory = NeuralLongTermMemory(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        output, state = memory(x, return_state=False)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is None

    def test_init_state(self, default_config: TitansConfig, batch_size: int) -> None:
        """Test memory state initialization."""
        memory = NeuralLongTermMemory(default_config)

        state = memory.init_state(batch_size)

        assert len(state.weights) == default_config.num_memory_layers
        assert len(state.momentum) == default_config.num_memory_layers

        for m in state.momentum:
            mx.eval(m)
            assert np.allclose(np.array(m), 0.0)

    def test_retrieve(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test memory retrieval without update."""
        memory = NeuralLongTermMemory(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        state = memory.init_state(batch_size)
        retrieved = memory.retrieve(x, state)
        mx.eval(retrieved)

        assert retrieved.shape == x.shape

    def test_with_conv(self, batch_size: int, seq_len: int) -> None:
        """Test memory with convolution enabled."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            use_conv=True,
            conv_kernel_size=4,
            num_memory_layers=2,
        )
        memory = NeuralLongTermMemory(config)
        x = mx.random.normal((batch_size, seq_len, config.dim))

        output, state = memory(x)
        mx.eval(output)

        assert output.shape == x.shape
        assert state is not None

    def test_without_conv(self, batch_size: int, seq_len: int) -> None:
        """Test memory without convolution."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            use_conv=False,
            num_memory_layers=2,
        )
        memory = NeuralLongTermMemory(config)
        x = mx.random.normal((batch_size, seq_len, config.dim))

        output, state = memory(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_memory_update_changes_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test that memory updates change the state."""
        memory = NeuralLongTermMemory(default_config)
        x = mx.random.normal((batch_size, seq_len, default_config.dim))

        _, state1 = memory(x)
        mx.eval(state1.weights[0])

        _, state2 = memory(x, state=state1)
        mx.eval(state2.weights[0])

        w1 = np.array(state1.weights[0])
        w2 = np.array(state2.weights[0])
        assert not np.allclose(w1, w2)

    def test_gradient_computation(self, small_config: TitansConfig) -> None:
        """Test gradient computation for memory update."""
        memory = NeuralLongTermMemory(small_config)

        keys = mx.random.normal((2, 8, small_config.dim))
        values = mx.random.normal((2, 8, small_config.dim))
        weights = memory.memory.get_weights()

        grads = memory._compute_gradients(keys, values, weights)

        assert len(grads) == small_config.num_memory_layers
        for g in grads:
            mx.eval(g)
            assert g is not None
            assert not np.any(np.isnan(np.array(g)))


class TestHuberGradients:
    """Tests for Huber loss attentional bias (Yaad)."""

    def test_huber_memory_creates_delta_gate(self, huber_config: TitansConfig) -> None:
        """Huber memory should have a delta gate projection."""
        mem = NeuralLongTermMemory(huber_config)
        assert hasattr(mem, "gate_delta_proj")

    def test_huber_small_error_matches_l2(self, huber_config: TitansConfig) -> None:
        """When error is small relative to delta, Huber gradient should approximate L2."""
        mem_huber = NeuralLongTermMemory(huber_config)
        l2_config = TitansConfig(**{**huber_config.to_dict(), "memory_objective": "l2"})
        mem_l2 = NeuralLongTermMemory(l2_config)

        # Copy weights
        for i in range(len(mem_l2.memory.layers)):
            mem_huber.memory.layers[i].weight = mem_l2.memory.layers[i].weight

        # Small input = small errors = L2 regime
        keys = mx.random.normal((1, 4, 64)) * 0.001
        values = mx.random.normal((1, 4, 64)) * 0.001
        weights = mem_l2.memory.get_weights()

        grads_l2 = mem_l2._compute_gradients(keys, values, weights)
        grads_huber = mem_huber._compute_gradients(keys, values, weights, delta=mx.array(100.0))

        mx.eval(*grads_l2, *grads_huber)
        for g_l2, g_h in zip(grads_l2, grads_huber):
            np.testing.assert_allclose(np.array(g_l2), np.array(g_h), atol=1e-5)

    def test_huber_large_error_clips_gradient(self, huber_config: TitansConfig) -> None:
        """When error > delta, Huber gradient magnitude should be bounded."""
        mem = NeuralLongTermMemory(huber_config)

        keys = mx.random.normal((1, 4, 64)) * 10.0
        values = mx.random.normal((1, 4, 64)) * 10.0
        weights = mem.memory.get_weights()

        grads_small_delta = mem._compute_gradients(keys, values, weights, delta=mx.array(0.001))
        grads_large_delta = mem._compute_gradients(keys, values, weights, delta=mx.array(1000.0))

        mx.eval(*grads_small_delta, *grads_large_delta)

        for g_l1, g_l2 in zip(grads_small_delta, grads_large_delta):
            norm_l1 = float(mx.sqrt(mx.sum(g_l1 * g_l1)))
            norm_l2 = float(mx.sqrt(mx.sum(g_l2 * g_l2)))
            assert norm_l1 < norm_l2, (
                f"Huber L1 regime grad norm ({norm_l1:.4f}) should be < "
                f"L2 regime ({norm_l2:.4f}) for large errors"
            )

    def test_huber_forward_pass_runs(self, huber_config: TitansConfig) -> None:
        """Full forward pass with Huber memory should produce valid output."""
        mem = NeuralLongTermMemory(huber_config)
        x = mx.random.normal((2, 32, 64))
        state = mem.init_state(2)
        output, new_state = mem(x, state)
        mx.eval(output)
        assert output.shape == (2, 32, 64)
        assert not mx.any(mx.isnan(output))


class TestHuberLinearMemory:
    """Tests for Huber loss with linear (1-layer) memory."""

    @pytest.fixture
    def huber_linear_config(self) -> TitansConfig:
        """Huber config with linear (1-layer) memory."""
        return TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            ffn_mult=2.0,
            num_memory_layers=1,
            memory_hidden_mult=2.0,
            num_persistent_tokens=4,
            chunk_size=32,
            window_size=16,
            dropout=0.0,
            use_conv=True,
            conv_kernel_size=4,
            use_rope=True,
            max_seq_len=256,
            vocab_size=100,
            memory_lr=0.1,
            memory_momentum=0.9,
            memory_objective="huber",
        )

    def test_linear_huber_forward(self, huber_linear_config: TitansConfig) -> None:
        """Linear memory with Huber should produce valid output."""
        mem = NeuralLongTermMemory(huber_linear_config)
        x = mx.random.normal((2, 32, 64))
        state = mem.init_state(2)
        output, new_state = mem(x, state)
        mx.eval(output)
        assert output.shape == (2, 32, 64)
        assert not mx.any(mx.isnan(output))

    def test_linear_huber_state_changes(self, huber_linear_config: TitansConfig) -> None:
        """Memory state should change after Huber update."""
        mem = NeuralLongTermMemory(huber_linear_config)
        x = mx.random.normal((2, 32, 64))
        state = mem.init_state(2)
        _, new_state = mem(x, state)
        mx.eval(state.weights[0], new_state.weights[0])
        assert not np.allclose(
            np.array(state.weights[0]),
            np.array(new_state.weights[0]),
            atol=1e-7,
        ), "Memory weights should change after update"


def test_memory_gate_parameter():
    """memory_gate should modulate lr_scale."""
    config = TitansConfig(dim=32, num_heads=4, num_layers=2, vocab_size=100)
    mem = NeuralLongTermMemory(config)
    x = mx.random.normal((1, 8, 32))
    state = mem.init_state(1)

    # With memory_gate=None, should behave like lr_scale=1.0
    out_none, st_none = mem(x, state=state, memory_gate=None)
    out_default, st_default = mem(x, state=state)
    assert mx.allclose(out_none, out_default, atol=1e-5)

    # With memory_gate=scalar, should modulate learning rate
    gate = mx.array(0.5)
    out_gated, _ = mem(x, state=state, memory_gate=gate)
    out_lr, _ = mem(x, state=state, lr_scale=0.5)
    assert mx.allclose(out_gated, out_lr, atol=1e-5)
