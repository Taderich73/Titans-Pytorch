# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Baseline asymmetric min-max integer quantization (int4/int8) for memory state.

This is NOT TurboQuant. See ``todos/papers/TurboQuant.pdf`` for the rotation +
Max-Lloyd codebook + QJL residual scheme that would achieve the paper's
distortion bounds. The scheme implemented here is a simple per-tensor asymmetric
min-max quantizer: it has no random rotation matrix, no Max-Lloyd codebook, and
no QJL residual sign. It is suitable for compressing persistent Titans memory
state between chunks during long inference runs, but it should not be confused
with a paper-quality weight quantizer with tight distortion guarantees.

Provides ``QuantizedTensor`` and ``QuantizedMemoryState`` for reducing the
memory footprint of persistent state between chunks.

Only bit-widths 4 and 8 are supported. 1-, 2-, and 3-bit modes (as explored in
the TurboQuant paper) are not implemented in this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    data: torch.Tensor  # uint8, packed for 4-bit
    scale: torch.Tensor  # scalar float32
    zero_point: torch.Tensor  # scalar float32
    shape: torch.Size
    bits: int

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Reconstruct the approximate floating-point tensor.

        Args:
            dtype: Optional target dtype. Default is ``torch.float32`` (backwards-
                compatible). Pass ``torch.bfloat16`` or ``torch.float16`` to avoid
                upcasting inside autocast regions (where the caller is already
                operating in a reduced-precision dtype).

        Returns:
            Floating-point tensor with the same shape as the original. The dtype
            is ``torch.float32`` by default, or ``dtype`` if provided.
        """
        numel = 1
        for s in self.shape:
            numel *= s

        if self.bits == 4:
            int_vals = _unpack_4bit(self.data, numel).float()
        else:
            int_vals = self.data.float()

        out = (int_vals * self.scale + self.zero_point).reshape(self.shape)
        if dtype is not None and dtype != out.dtype:
            out = out.to(dtype)
        return out


# ---------------------------------------------------------------------------
# quantize_tensor — baseline per-tensor asymmetric min-max
#
# Per-tensor scale + zero-point; no random rotation; no codebook. Good enough
# for memory-state compression during long runs; NOT a substitute for
# weight-quantization schemes with tight distortion bounds.
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
    # Constant tensor (x_max == x_min): fall back to scale=1/max_val so the
    # round+clamp below produces zeros. Correctness for this branch comes
    # from adding zero_point (= x_min) back in dequantize, not from the
    # divide guard itself -- without the guard we'd get NaN from 0/0.
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
# QuantizedMemoryState — baseline container for quantized memory tensors
#
# Mirrors MemoryState; each field holds a QuantizedTensor (or plain float
# tensor for momentum when momentum_bits=None). No TurboQuant primitives.
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

    def dequantize(self, dtype: torch.dtype | None = None) -> MemoryState:
        """Reconstruct the full-precision ``MemoryState``.

        Args:
            dtype: Optional target dtype for the dequantized weight and momentum
                tensors. Default is ``torch.float32`` (backwards-compatible).
                Pass the caller's compute dtype (e.g. ``torch.bfloat16``) to
                avoid a silent upcast inside autocast regions.

        Returns:
            A ``MemoryState`` with weight and momentum tensors at the requested
            dtype (fp32 by default).
        """
        dq_weights = [w.dequantize(dtype=dtype) for w in self.weights]
        dq_momentum = [
            m.dequantize(dtype=dtype)
            if isinstance(m, QuantizedTensor)
            else (m.to(dtype) if dtype is not None else m.float())
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
            m if isinstance(m, QuantizedTensor) else m.detach() for m in self.momentum
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
    momentum_bits: int | None = None,
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


# ---------------------------------------------------------------------------
# Flatten / unflatten for safetensors round-trip
# ---------------------------------------------------------------------------

_QT_FIELDS: tuple[str, ...] = ("data", "scale", "zero_point", "shape", "bits")


def _flatten_qt(qt: QuantizedTensor, prefix: str) -> dict[str, torch.Tensor]:
    """Flatten a single :class:`QuantizedTensor` into a dict of plain tensors.

    ``shape`` and ``bits`` are encoded as int64 tensors so the caller ends up
    with a dict whose values are all ``torch.Tensor`` instances — a hard
    requirement of the safetensors file format.
    """
    shape_t = torch.tensor(list(qt.shape), dtype=torch.int64)
    bits_t = torch.tensor(qt.bits, dtype=torch.int64)
    return {
        f"{prefix}.data": qt.data,
        f"{prefix}.scale": qt.scale,
        f"{prefix}.zero_point": qt.zero_point,
        f"{prefix}.shape": shape_t,
        f"{prefix}.bits": bits_t,
    }


def _unflatten_qt(flat: dict[str, torch.Tensor], prefix: str) -> QuantizedTensor:
    """Reconstruct a :class:`QuantizedTensor` from the flat dict produced by
    :func:`_flatten_qt`."""
    shape = torch.Size(flat[f"{prefix}.shape"].tolist())
    bits = int(flat[f"{prefix}.bits"].item())
    return QuantizedTensor(
        data=flat[f"{prefix}.data"],
        scale=flat[f"{prefix}.scale"],
        zero_point=flat[f"{prefix}.zero_point"],
        shape=shape,
        bits=bits,
    )


def flatten_quantized_state(
    state: QuantizedMemoryState,
    *,
    prefix: str = "mem",
) -> dict[str, torch.Tensor]:
    """Flatten a :class:`QuantizedMemoryState` into a dict of plain tensors.

    Keys follow the pattern ``{prefix}.weights.{i}.{field}`` and
    ``{prefix}.momentum.{i}.{field}``, where ``field`` is one of
    ``data``, ``scale``, ``zero_point``, ``shape``, ``bits`` for a
    quantized entry, or a single ``tensor`` key for an unquantized
    float momentum entry. An extra ``{prefix}.meta.sizes`` int64 tensor
    records ``[len(weights), len(momentum)]`` so the flat dict is
    self-describing.

    Args:
        state: The quantized state to flatten.
        prefix: Key prefix applied to every entry in the returned dict.

    Returns:
        Dict mapping string keys to :class:`torch.Tensor` values. All values
        are plain ``torch.Tensor`` (never dataclasses, lists, or scalars).
    """
    out: dict[str, torch.Tensor] = {}
    out[f"{prefix}.meta.sizes"] = torch.tensor(
        [len(state.weights), len(state.momentum)], dtype=torch.int64
    )

    for i, w in enumerate(state.weights):
        out.update(_flatten_qt(w, f"{prefix}.weights.{i}"))

    for i, m in enumerate(state.momentum):
        if isinstance(m, QuantizedTensor):
            out.update(_flatten_qt(m, f"{prefix}.momentum.{i}"))
        else:
            # Plain float tensor (momentum_bits=None path).
            out[f"{prefix}.momentum.{i}.tensor"] = m

    return out


def unflatten_quantized_state(
    flat: dict[str, torch.Tensor],
    *,
    prefix: str = "mem",
) -> QuantizedMemoryState:
    """Inverse of :func:`flatten_quantized_state`.

    Args:
        flat: Dict produced by :func:`flatten_quantized_state`.
        prefix: Key prefix used when the dict was built.

    Returns:
        Reconstructed :class:`QuantizedMemoryState`.

    Raises:
        KeyError: If the meta entry is missing (indicates the dict wasn't
            produced by :func:`flatten_quantized_state` with the given prefix).
    """
    sizes = flat[f"{prefix}.meta.sizes"].tolist()
    num_weights, num_momentum = int(sizes[0]), int(sizes[1])

    weights: list[QuantizedTensor] = [
        _unflatten_qt(flat, f"{prefix}.weights.{i}") for i in range(num_weights)
    ]

    momentum: list[QuantizedTensor | torch.Tensor] = []
    for i in range(num_momentum):
        qt_key = f"{prefix}.momentum.{i}.data"
        tensor_key = f"{prefix}.momentum.{i}.tensor"
        if qt_key in flat:
            momentum.append(_unflatten_qt(flat, f"{prefix}.momentum.{i}"))
        elif tensor_key in flat:
            momentum.append(flat[tensor_key])
        else:
            raise KeyError(
                f"unflatten_quantized_state: no entry for momentum index {i} "
                f"under prefix {prefix!r}"
            )

    return QuantizedMemoryState(weights=weights, momentum=momentum)
