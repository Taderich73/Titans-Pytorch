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
