# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state quantization for Titans inference.

Provides QuantizedTensor and QuantizedMemoryState for reducing memory
footprint of persistent state between chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from titans.memory import MemoryState


# ---------------------------------------------------------------------------
# Low-level packing helpers
# ---------------------------------------------------------------------------

def _pack_4bit(x: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor of 4-bit values (values 0–15) into half-size uint8.

    Args:
        x: 1-D uint8 tensor whose length must be even. Each element holds a
           4-bit value stored in the low nibble.

    Returns:
        Packed uint8 tensor of length ``len(x) // 2``. The even-indexed
        source values are stored in the high nibble, odd-indexed values in the
        low nibble.
    """
    flat = x.reshape(-1)
    if flat.numel() % 2 != 0:
        raise ValueError(
            f"_pack_4bit requires an even number of elements, got {flat.numel()}"
        )
    packed = (flat[::2] << 4) | (flat[1::2] & 0x0F)
    return packed.to(torch.uint8)


def _unpack_4bit(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    """Unpack a uint8 tensor produced by ``_pack_4bit``.

    Args:
        packed: 1-D uint8 tensor of packed 4-bit values.
        original_numel: Number of elements in the original (pre-packing) tensor.

    Returns:
        1-D uint8 tensor of length ``original_numel`` with values in [0, 15].
    """
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    # Interleave: [high[0], low[0], high[1], low[1], ...]
    unpacked = torch.stack([high, low], dim=1).reshape(-1)
    return unpacked[:original_numel].to(torch.uint8)


# ---------------------------------------------------------------------------
# QuantizedTensor
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Quantized representation of a float tensor.

    Supports per-tensor asymmetric quantization at 4 or 8 bits.

    Attributes:
        data: Quantized uint8 storage. For 4-bit mode this is the *packed*
              representation produced by ``_pack_4bit``; for 8-bit mode it
              holds one byte per original element.
        scale: Scalar float32 scale factor used during quantization.
        zero_point: Scalar float32 zero-point used during quantization.
        shape: Original tensor shape, needed to restore the tensor.
        bits: Quantization bit-width (4 or 8).
    """

    data: torch.Tensor       # uint8, packed for 4-bit
    scale: torch.Tensor      # scalar float32
    zero_point: torch.Tensor # scalar float32
    shape: torch.Size
    bits: int

    def dequantize(self) -> torch.Tensor:
        """Reconstruct the approximate float32 tensor.

        Returns:
            Float32 tensor with the same shape as the original.
        """
        numel = 1
        for s in self.shape:
            numel *= s

        if self.bits == 4:
            int_vals = _unpack_4bit(self.data, numel).float()
        else:
            int_vals = self.data.float()

        return (int_vals * self.scale + self.zero_point).reshape(self.shape)


# ---------------------------------------------------------------------------
# quantize_tensor
# ---------------------------------------------------------------------------

def quantize_tensor(x: torch.Tensor, bits: int) -> QuantizedTensor:
    """Quantize a float tensor to 4 or 8 bits (per-tensor asymmetric).

    Args:
        x: Input float tensor of any shape.
        bits: Quantization bit-width. Must be 4 or 8.

    Returns:
        A ``QuantizedTensor`` holding the quantized representation.

    Raises:
        ValueError: If ``bits`` is not 4 or 8.
    """
    if bits not in (4, 8):
        raise ValueError(f"quantize_tensor: bits must be 4 or 8, got {bits}")

    x_f32 = x.detach().float()
    x_min = x_f32.min()
    x_max = x_f32.max()

    max_val = float(2**bits - 1)  # 15.0 for 4-bit, 255.0 for 8-bit

    val_range = x_max - x_min
    # Avoid division by zero for constant tensors
    if val_range.item() == 0.0:
        val_range = torch.tensor(1.0, dtype=torch.float32, device=x.device)

    scale = val_range / max_val
    zero_point = x_min

    # Quantize: q = round((x - zero_point) / scale), clamped to [0, max_val]
    q = torch.round((x_f32 - zero_point) / scale).clamp(0, max_val).to(torch.uint8)

    if bits == 4:
        data = _pack_4bit(q.reshape(-1))
    else:
        data = q.reshape(-1)

    return QuantizedTensor(
        data=data,
        scale=scale.detach(),
        zero_point=zero_point.detach(),
        shape=x.shape,
        bits=bits,
    )


# ---------------------------------------------------------------------------
# QuantizedMemoryState
# ---------------------------------------------------------------------------

@dataclass
class QuantizedMemoryState:
    """Memory-efficient version of ``MemoryState`` using quantized tensors.

    Mirrors the ``MemoryState`` interface so it can be used as a drop-in
    for inference pipelines that do not need full float32 precision.

    Attributes:
        weights: List of ``QuantizedTensor`` for the MLP weight matrices.
        momentum: List of tensors for the optimizer momentum. These are stored
                  as ``QuantizedTensor`` when ``momentum_bits`` was specified,
                  or as plain float32 ``torch.Tensor`` when momentum was kept
                  unquantized (``momentum_bits=None``).
    """

    weights: list[QuantizedTensor]
    momentum: list[QuantizedTensor | torch.Tensor]

    def dequantize(self) -> MemoryState:
        """Reconstruct the full-precision ``MemoryState``.

        Returns:
            A ``MemoryState`` with float32 weights and momentum tensors.
        """
        dq_weights = [w.dequantize() for w in self.weights]
        dq_momentum = [
            m.dequantize() if isinstance(m, QuantizedTensor) else m.float()
            for m in self.momentum
        ]
        return MemoryState(weights=dq_weights, momentum=dq_momentum)

    def detach(self) -> QuantizedMemoryState:
        """Return a detached copy — mirrors the ``MemoryState.detach()`` API.

        For ``QuantizedTensor`` fields the underlying uint8 data has no
        gradient, so this is effectively a no-op beyond detaching any plain
        float momentum tensors.

        Returns:
            A new ``QuantizedMemoryState`` with detached momentum tensors.
        """
        detached_momentum = [
            m if isinstance(m, QuantizedTensor) else m.detach()
            for m in self.momentum
        ]
        return QuantizedMemoryState(
            weights=list(self.weights),
            momentum=detached_momentum,
        )


# ---------------------------------------------------------------------------
# quantize_memory_state
# ---------------------------------------------------------------------------

def quantize_memory_state(
    state: MemoryState,
    weight_bits: int = 8,
    momentum_bits: Optional[int] = None,
) -> QuantizedMemoryState:
    """Quantize a ``MemoryState`` into a ``QuantizedMemoryState``.

    Args:
        state: The full-precision ``MemoryState`` to compress.
        weight_bits: Bit-width for quantizing weight tensors (4 or 8).
        momentum_bits: Bit-width for quantizing momentum tensors (4 or 8).
                       If ``None``, momentum tensors are kept as float32 and
                       not quantized.

    Returns:
        A ``QuantizedMemoryState`` with compressed weight (and optionally
        momentum) tensors.
    """
    q_weights = [quantize_tensor(w, weight_bits) for w in state.weights]

    if momentum_bits is not None:
        q_momentum: list[QuantizedTensor | torch.Tensor] = [
            quantize_tensor(m, momentum_bits) for m in state.momentum
        ]
    else:
        q_momentum = [m.detach().float() for m in state.momentum]

    return QuantizedMemoryState(weights=q_weights, momentum=q_momentum)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_weights(state: MemoryState | QuantizedMemoryState) -> list[torch.Tensor]:
    """Extract dequantized weight tensors from either state type.

    Args:
        state: A ``MemoryState`` or ``QuantizedMemoryState``.

    Returns:
        List of float32 weight tensors.
    """
    if isinstance(state, QuantizedMemoryState):
        return [w.dequantize() for w in state.weights]
    return [w.float() for w in state.weights]


def get_momentum(state: MemoryState | QuantizedMemoryState) -> list[torch.Tensor]:
    """Extract dequantized momentum tensors from either state type.

    Args:
        state: A ``MemoryState`` or ``QuantizedMemoryState``.

    Returns:
        List of float32 momentum tensors.
    """
    if isinstance(state, QuantizedMemoryState):
        return [
            m.dequantize() if isinstance(m, QuantizedTensor) else m.float()
            for m in state.momentum
        ]
    return [m.float() for m in state.momentum]
