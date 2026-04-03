# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state quantization for Titans inference.

Provides QuantizedTensor and QuantizedMemoryState for reducing memory
footprint of persistent state between chunks. Quantization happens at
the .detach() boundary after each memory update; dequantization is
lazy, triggered when retrieve() or gradient computation needs values.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from titans_mlx.memory import MemoryState


@dataclass
class QuantizedTensor:
    """A scalar-quantized tensor with metadata for dequantization.

    Uses per-tensor asymmetric quantization:
        quantized = round(clamp((x - zero_point) / scale, 0, 2^b - 1))
        dequantized = quantized * scale + zero_point

    For 4-bit: two values packed per uint8 (high nibble / low nibble).
    For 8-bit: stored as uint8 directly.
    """

    data: mx.array  # uint8 storage (packed for 4-bit)
    scale: mx.array  # float32 scalar
    zero_point: mx.array  # float32 scalar
    original_shape: tuple[int, ...]
    bits: int  # 4 or 8

    def dequantize(self) -> mx.array:
        """Restore to float32 tensor."""
        if self.bits == 4:
            return _unpack_4bit(
                self.data, self.original_shape, self.scale, self.zero_point
            )
        # 8-bit: direct conversion
        return self.data.astype(mx.float32) * self.scale + self.zero_point


def quantize_tensor(x: mx.array, bits: int) -> QuantizedTensor:
    """Quantize a float tensor to the specified bit-width.

    Args:
        x: Float tensor to quantize.
        bits: Target bit-width (4 or 8).

    Returns:
        QuantizedTensor with packed data and dequantization metadata.
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bit-width: {bits}. Must be 4 or 8.")

    x_f32 = x.astype(mx.float32)
    mx.eval(x_f32)

    x_min = mx.min(x_f32)
    x_max = mx.max(x_f32)
    mx.eval(x_min, x_max)

    max_val = (1 << bits) - 1  # 255 for 8-bit, 15 for 4-bit
    rng = x_max - x_min
    # Avoid division by zero for constant tensors
    scale = mx.where(rng > 0, rng / max_val, mx.array(1.0, dtype=mx.float32))
    mx.eval(scale)

    quantized = mx.round((x_f32 - x_min) / scale)
    quantized = mx.clip(quantized, 0, max_val).astype(mx.uint8)
    mx.eval(quantized)

    if bits == 4:
        quantized = _pack_4bit(quantized)

    return QuantizedTensor(
        data=quantized,
        scale=scale,
        zero_point=x_min,
        original_shape=tuple(x.shape),
        bits=bits,
    )


def _pack_4bit(tensor: mx.array) -> mx.array:
    """Pack 4-bit values into uint8 pairs (high nibble / low nibble)."""
    flat = tensor.reshape(-1)
    # Pad to even length if needed
    size = flat.shape[0]
    if size % 2 != 0:
        flat = mx.concatenate([flat, mx.array([0], dtype=mx.uint8)])
    high = flat[::2] << 4
    low = flat[1::2] & 0x0F
    packed = (high | low).astype(mx.uint8)
    mx.eval(packed)
    return packed


def _unpack_4bit(
    packed: mx.array,
    original_shape: tuple[int, ...],
    scale: mx.array,
    zero_point: mx.array,
) -> mx.array:
    """Unpack 4-bit values from uint8 pairs and dequantize."""
    high = (packed >> 4).astype(mx.float32)
    low = (packed & 0x0F).astype(mx.float32)
    # Stack and interleave: (N,) + (N,) -> (N, 2) -> (2N,)
    interleaved = mx.stack([high, low], axis=-1).reshape(-1)
    # Trim padding and reshape
    numel = 1
    for s in original_shape:
        numel *= s
    interleaved = interleaved[:numel]
    return (interleaved * scale + zero_point).reshape(original_shape)


@dataclass
class QuantizedMemoryState:
    """Memory state with quantized tensors.

    Mirrors MemoryState but stores weights as QuantizedTensors and
    momentum as either QuantizedTensor (deep memory) or float16 mx.array
    (linear memory, where momentum math is numerically sensitive).
    """

    weights: list[QuantizedTensor]
    momentum: list[QuantizedTensor | mx.array]

    def dequantize(self) -> MemoryState:
        """Full dequantization back to MemoryState."""
        dq_weights = [w.dequantize() for w in self.weights]
        dq_momentum = [
            m.dequantize() if isinstance(m, QuantizedTensor) else m.astype(mx.float32)
            for m in self.momentum
        ]
        return MemoryState(weights=dq_weights, momentum=dq_momentum)

    def detach(self) -> QuantizedMemoryState:
        """Interface compatibility with MemoryState.detach()."""
        new_weights = [
            QuantizedTensor(
                data=mx.stop_gradient(w.data),
                scale=mx.stop_gradient(w.scale),
                zero_point=mx.stop_gradient(w.zero_point),
                original_shape=w.original_shape,
                bits=w.bits,
            )
            for w in self.weights
        ]
        new_momentum = [
            QuantizedTensor(
                data=mx.stop_gradient(m.data),
                scale=mx.stop_gradient(m.scale),
                zero_point=mx.stop_gradient(m.zero_point),
                original_shape=m.original_shape,
                bits=m.bits,
            )
            if isinstance(m, QuantizedTensor)
            else mx.stop_gradient(m)
            for m in self.momentum
        ]
        return QuantizedMemoryState(weights=new_weights, momentum=new_momentum)


def quantize_memory_state(
    state: MemoryState,
    weight_bits: int,
    momentum_bits: int | None,
) -> QuantizedMemoryState:
    """Quantize a MemoryState.

    Args:
        state: Full-precision memory state to quantize.
        weight_bits: Bit-width for weight matrices (4 or 8).
        momentum_bits: Bit-width for momentum tensors (4 or 8), or None
            to cast momentum to float16 (used for linear memory path
            where momentum math is numerically sensitive).

    Returns:
        QuantizedMemoryState with quantized weights and momentum.
    """
    q_weights = [quantize_tensor(w, bits=weight_bits) for w in state.weights]

    if momentum_bits is None:
        q_momentum: list[QuantizedTensor | mx.array] = [
            m.astype(mx.float16) for m in state.momentum
        ]
    else:
        q_momentum = [quantize_tensor(m, bits=momentum_bits) for m in state.momentum]

    return QuantizedMemoryState(weights=q_weights, momentum=q_momentum)


def get_weights(state: MemoryState | QuantizedMemoryState) -> list[mx.array]:
    """Extract dequantized weight matrices from either state type."""
    if isinstance(state, QuantizedMemoryState):
        return [w.dequantize() for w in state.weights]
    return state.weights


def get_momentum(state: MemoryState | QuantizedMemoryState) -> list[mx.array]:
    """Extract dequantized momentum tensors from either state type."""
    if isinstance(state, QuantizedMemoryState):
        return [
            m.dequantize() if isinstance(m, QuantizedTensor) else m.astype(mx.float32)
            for m in state.momentum
        ]
    return state.momentum
