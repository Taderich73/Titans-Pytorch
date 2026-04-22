"""Tests for baseline int4/int8 memory-state quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from titans.quantize_state import quantize_memory_state, quantize_tensor

if TYPE_CHECKING:
    from titans.memory import MemoryState


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
    assert torch.allclose(
        out_fp32.to(torch.bfloat16).float(), out_bf16.float(), atol=1e-3
    )


# ---------------------------------------------------------------------------
# flatten / unflatten round-trip for safetensors
# ---------------------------------------------------------------------------


def _make_memory_state_fixture() -> MemoryState:
    from titans.memory import MemoryState

    weights = [torch.randn(8, 16), torch.randn(16, 8)]
    momentum = [torch.randn(8, 16), torch.randn(16, 8)]
    return MemoryState(weights=weights, momentum=momentum)


def test_flatten_quantized_state_produces_only_tensors() -> None:
    from titans.quantize_state import flatten_quantized_state

    state = _make_memory_state_fixture()
    q_state = quantize_memory_state(state, weight_bits=8, momentum_bits=4)

    flat = flatten_quantized_state(q_state, prefix="mem")

    assert isinstance(flat, dict)
    assert all(isinstance(v, torch.Tensor) for v in flat.values())
    assert all(k.startswith("mem.") for k in flat)


def test_flatten_then_unflatten_round_trip() -> None:
    from titans.quantize_state import flatten_quantized_state, unflatten_quantized_state

    state = _make_memory_state_fixture()
    q_state = quantize_memory_state(state, weight_bits=8, momentum_bits=4)

    flat = flatten_quantized_state(q_state, prefix="mem")
    restored = unflatten_quantized_state(flat, prefix="mem")

    # Shapes and bits agree on each quantized tensor.
    assert len(restored.weights) == len(q_state.weights)
    for w_orig, w_rest in zip(q_state.weights, restored.weights, strict=True):
        assert w_rest.bits == w_orig.bits
        assert w_rest.shape == w_orig.shape
        torch.testing.assert_close(w_rest.data, w_orig.data)
        torch.testing.assert_close(w_rest.scale, w_orig.scale)
        torch.testing.assert_close(w_rest.zero_point, w_orig.zero_point)

    # Dequantized values match within the int8 tolerance.
    dq_orig = q_state.dequantize()
    dq_rest = restored.dequantize()
    for a, b in zip(dq_orig.weights, dq_rest.weights, strict=True):
        torch.testing.assert_close(a, b)


def test_flatten_handles_unquantized_momentum() -> None:
    from titans.quantize_state import flatten_quantized_state, unflatten_quantized_state

    state = _make_memory_state_fixture()
    # momentum_bits=None leaves momentum as plain float tensors
    q_state = quantize_memory_state(state, weight_bits=8, momentum_bits=None)

    flat = flatten_quantized_state(q_state, prefix="mem")
    restored = unflatten_quantized_state(flat, prefix="mem")

    assert len(restored.momentum) == len(q_state.momentum)
    for m_orig, m_rest in zip(q_state.momentum, restored.momentum, strict=True):
        assert isinstance(m_orig, torch.Tensor)
        assert isinstance(m_rest, torch.Tensor)
        torch.testing.assert_close(m_rest, m_orig)


# ---------------------------------------------------------------------------
# Round-trip via titans.checkpoint
# ---------------------------------------------------------------------------


def test_quantized_state_round_trips_via_safetensors(tmp_path) -> None:
    pytest.importorskip("safetensors")

    from titans.checkpoint import load_checkpoint, save_checkpoint
    from titans.quantize_state import flatten_quantized_state, unflatten_quantized_state

    state = _make_memory_state_fixture()
    q_state = quantize_memory_state(state, weight_bits=8, momentum_bits=4)

    flat = flatten_quantized_state(q_state, prefix="mem")
    save_checkpoint(flat, tmp_path / "ckpt", format="safetensors")

    loaded = load_checkpoint(tmp_path / "ckpt.safetensors")
    restored = unflatten_quantized_state(loaded["model"], prefix="mem")

    dq_before = q_state.dequantize()
    dq_after = restored.dequantize()
    for a, b in zip(dq_before.weights, dq_after.weights, strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(dq_before.momentum, dq_after.momentum, strict=True):
        torch.testing.assert_close(a, b)


def test_save_safetensors_accepts_quantized_memory_state_directly(tmp_path) -> None:
    pytest.importorskip("safetensors")

    from titans.checkpoint import load_checkpoint, save_checkpoint
    from titans.quantize_state import unflatten_quantized_state

    state = _make_memory_state_fixture()
    q_state = quantize_memory_state(state, weight_bits=8, momentum_bits=4)

    # Wrap the QuantizedMemoryState under a string key alongside a normal
    # tensor to confirm mixed payloads work.
    payload = {"mem": q_state, "step": torch.tensor([42], dtype=torch.int64)}

    save_checkpoint(payload, tmp_path / "ckpt", format="safetensors")

    loaded = load_checkpoint(tmp_path / "ckpt.safetensors")
    restored = unflatten_quantized_state(loaded["model"], prefix="mem")

    assert loaded["model"]["step"].tolist() == [42]

    dq_before = q_state.dequantize()
    dq_after = restored.dequantize()
    for a, b in zip(dq_before.weights, dq_after.weights, strict=True):
        torch.testing.assert_close(a, b)
