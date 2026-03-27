# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for memory state quantization."""

import mlx.core as mx
import pytest


class TestQuantizedTensor:
    """Tests for QuantizedTensor quantize/dequantize."""

    def test_quantize_8bit_round_trip(self) -> None:
        """8-bit quantize -> dequantize preserves values within tolerance."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=8)
        restored = qt.dequantize()

        assert restored.shape == original.shape
        assert restored.dtype == mx.float32

        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.001, f"8-bit round-trip MSE too high: {mse}"

    def test_quantize_8bit_metadata(self) -> None:
        """8-bit QuantizedTensor stores correct metadata."""
        mx.random.seed(42)
        original = mx.random.normal((32, 128))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=8)

        assert qt.bits == 8
        assert qt.original_shape == (32, 128)
        assert qt.data.dtype == mx.uint8
        assert qt.data.shape == (32, 128)
        assert qt.scale.dtype == mx.float32
        assert qt.zero_point.dtype == mx.float32

    def test_quantize_4bit_round_trip(self) -> None:
        """4-bit quantize -> dequantize preserves values within tolerance."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)
        restored = qt.dequantize()

        assert restored.shape == original.shape
        assert restored.dtype == mx.float32

        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.05, f"4-bit round-trip MSE too high: {mse}"

    def test_quantize_4bit_packing(self) -> None:
        """4-bit packing stores two values per byte."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)

        assert qt.bits == 4
        assert qt.data.dtype == mx.uint8
        # 64*64 = 4096 values, packed into 2048 bytes
        assert qt.data.shape == (2048,)

    def test_quantize_4bit_odd_elements(self) -> None:
        """4-bit packing handles odd number of elements."""
        mx.random.seed(42)
        original = mx.random.normal((3, 5))  # 15 elements (odd)
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)
        restored = qt.dequantize()

        assert restored.shape == (3, 5)
        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.05

    def test_quantize_constant_tensor(self) -> None:
        """Quantization handles constant tensors (zero range) without error."""
        original = mx.ones((16, 16)) * 3.14
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        for bits in (4, 8):
            qt = quantize_tensor(original, bits=bits)
            restored = qt.dequantize()
            mse = mx.mean((original - restored) ** 2).item()
            assert mse < 1e-6, f"Constant tensor MSE too high at {bits}-bit: {mse}"

    def test_quantize_invalid_bits(self) -> None:
        """Quantization rejects invalid bit-widths."""
        original = mx.random.normal((8, 8))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        with pytest.raises(ValueError, match="Unsupported bit-width"):
            quantize_tensor(original, bits=3)


class TestQuantizedMemoryState:
    """Tests for QuantizedMemoryState quantize/dequantize."""

    def test_quantize_linear_memory_state(self) -> None:
        """Linear memory (1 layer): weights quantized, momentum cast to float16."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 64))],
            momentum=[mx.random.normal((64, 64))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import quantize_memory_state, QuantizedMemoryState, QuantizedTensor

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)

        assert isinstance(qstate, QuantizedMemoryState)
        assert isinstance(qstate.weights[0], QuantizedTensor)
        assert qstate.weights[0].bits == 4
        # Linear path: momentum is float16, not QuantizedTensor
        assert isinstance(qstate.momentum[0], mx.array)
        assert qstate.momentum[0].dtype == mx.float16

    def test_quantize_deep_memory_state(self) -> None:
        """Deep memory (2 layers): weights 4-bit, momentum 8-bit."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 128)), mx.random.normal((128, 64))],
            momentum=[mx.random.normal((64, 128)), mx.random.normal((128, 64))],
        )
        mx.eval(*state.weights, *state.momentum)

        from titans_mlx.quantize_state import quantize_memory_state, QuantizedMemoryState, QuantizedTensor

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=8)

        assert isinstance(qstate, QuantizedMemoryState)
        assert len(qstate.weights) == 2
        assert all(isinstance(w, QuantizedTensor) for w in qstate.weights)
        assert all(w.bits == 4 for w in qstate.weights)
        assert len(qstate.momentum) == 2
        assert all(isinstance(m, QuantizedTensor) for m in qstate.momentum)
        assert all(m.bits == 8 for m in qstate.momentum)

    def test_dequantize_round_trip(self) -> None:
        """QuantizedMemoryState.dequantize() returns MemoryState with correct shapes."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 64))],
            momentum=[mx.random.normal((64, 64))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)
        restored = qstate.dequantize()

        assert isinstance(restored, MemoryState)
        assert restored.weights[0].shape == (64, 64)
        assert restored.weights[0].dtype == mx.float32
        assert restored.momentum[0].shape == (64, 64)
        assert restored.momentum[0].dtype == mx.float32

    def test_detach_interface(self) -> None:
        """QuantizedMemoryState.detach() returns QuantizedMemoryState."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import quantize_memory_state, QuantizedMemoryState

        qstate = quantize_memory_state(state, weight_bits=8, momentum_bits=None)
        detached = qstate.detach()

        assert isinstance(detached, QuantizedMemoryState)


class TestStateAccessors:
    """Tests for get_weights / get_momentum helpers."""

    def test_get_weights_from_memory_state(self) -> None:
        """get_weights passes through MemoryState weights unchanged."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        w = mx.random.normal((32, 32))
        mx.eval(w)
        state = MemoryState(weights=[w], momentum=[mx.zeros_like(w)])

        from titans_mlx.quantize_state import get_weights

        result = get_weights(state)
        assert len(result) == 1
        assert mx.array_equal(result[0], w)

    def test_get_weights_from_quantized_state(self) -> None:
        """get_weights dequantizes QuantizedMemoryState weights."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        w = mx.random.normal((32, 32))
        mx.eval(w)
        state = MemoryState(weights=[w], momentum=[mx.zeros_like(w)])

        from titans_mlx.quantize_state import get_weights, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=8, momentum_bits=None)
        result = get_weights(qstate)

        assert len(result) == 1
        assert result[0].shape == (32, 32)
        assert result[0].dtype == mx.float32

    def test_get_momentum_from_memory_state(self) -> None:
        """get_momentum passes through MemoryState momentum unchanged."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        m = mx.random.normal((32, 32))
        mx.eval(m)
        state = MemoryState(weights=[mx.zeros_like(m)], momentum=[m])

        from titans_mlx.quantize_state import get_momentum

        result = get_momentum(state)
        assert len(result) == 1
        assert mx.array_equal(result[0], m)

    def test_get_momentum_from_quantized_float16(self) -> None:
        """get_momentum casts float16 momentum back to float32."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import get_momentum, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)
        result = get_momentum(qstate)

        assert len(result) == 1
        assert result[0].dtype == mx.float32
        assert result[0].shape == (32, 32)

    def test_get_momentum_from_quantized_8bit(self) -> None:
        """get_momentum dequantizes 8-bit QuantizedTensor momentum."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 64)), mx.random.normal((64, 32))],
            momentum=[mx.random.normal((32, 64)), mx.random.normal((64, 32))],
        )
        mx.eval(*state.weights, *state.momentum)

        from titans_mlx.quantize_state import get_momentum, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=8)
        result = get_momentum(qstate)

        assert len(result) == 2
        assert all(r.dtype == mx.float32 for r in result)
