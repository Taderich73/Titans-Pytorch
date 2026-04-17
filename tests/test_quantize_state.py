"""Tests for baseline int4/int8 memory-state quantization."""
from __future__ import annotations

import torch

from titans.quantize_state import QuantizedTensor, quantize_tensor


def test_dequantize_default_returns_fp32():
    """Regression guard: the default dtype is fp32 for backwards compatibility."""
    x = torch.randn(4, 4)
    q = quantize_tensor(x, bits=8)
    out = q.dequantize()
    assert out.dtype == torch.float32


def test_dequantize_accepts_dtype_kwarg_bf16():
    """Dequantize should be able to return bf16 when asked — avoids autocast
    upcast hazard in memory.forward."""
    x = torch.randn(4, 4)
    q = quantize_tensor(x, bits=8)
    out = q.dequantize(dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16


def test_dequantize_accepts_dtype_kwarg_fp16():
    x = torch.randn(4, 4)
    q = quantize_tensor(x, bits=8)
    out = q.dequantize(dtype=torch.float16)
    assert out.dtype == torch.float16


def test_dequantize_numerical_similarity_across_dtypes():
    """Within each dtype's precision, dequantize values should match modulo
    dtype rounding."""
    x = torch.randn(4, 4)
    q = quantize_tensor(x, bits=8)
    out_fp32 = q.dequantize()
    out_bf16 = q.dequantize(dtype=torch.bfloat16)
    # bf16 has ~3 decimal digits of precision; fp32 cast to bf16 and back
    # is the right comparison.
    assert torch.allclose(out_fp32.to(torch.bfloat16).float(), out_bf16.float(), atol=1e-3)
