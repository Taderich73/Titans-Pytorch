# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Pure signal-extraction functions for the memory auto-checkpointing system.

Computes per-layer Frobenius-norm signals (weight delta, momentum shift,
absolute weight / momentum norms) from :class:`~titans.memory.MemoryState`
and :class:`~titans.memory.TNTMemoryState` snapshots, and assembles them
into a :class:`~titans.checkpoint_types.SignalFrame` for downstream novelty
detection and optional signal logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from titans.checkpoint_types import GateSnapshot, SignalFrame
from titans.memory import MemoryState, TNTMemoryState

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_weights(state: MemoryState | TNTMemoryState) -> list[torch.Tensor]:
    """Return the weight tensor list from either state type.

    For :class:`TNTMemoryState`, delegates to ``global_state.weights``.

    Args:
        state: A :class:`MemoryState` or :class:`TNTMemoryState` instance.

    Returns:
        List of weight tensors, one per memory layer.
    """
    if isinstance(state, TNTMemoryState):
        return state.global_state.weights
    return state.weights


def _extract_momentum(state: MemoryState | TNTMemoryState) -> list[torch.Tensor]:
    """Return the momentum tensor list from either state type.

    For :class:`TNTMemoryState`, delegates to ``global_state.momentum``.

    Args:
        state: A :class:`MemoryState` or :class:`TNTMemoryState` instance.

    Returns:
        List of momentum tensors, one per memory layer.
    """
    if isinstance(state, TNTMemoryState):
        return state.global_state.momentum
    return state.momentum


def _frobenius(tensor: torch.Tensor) -> float:
    """Return the Frobenius norm of *tensor* as a Python float.

    Computation is performed in float32 to avoid precision issues with
    half-precision tensors.

    Args:
        tensor: Arbitrary-rank tensor.

    Returns:
        Frobenius norm as a Python float.
    """
    return float(torch.linalg.norm(tensor.float(), ord="fro").item())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_weight_delta(
    old_state: MemoryState | TNTMemoryState,
    new_state: MemoryState | TNTMemoryState,
) -> list[float]:
    """Compute the per-layer Frobenius norm of weight changes.

    For each layer *i*, computes ``||W_new[i] - W_old[i]||_F``.

    For :class:`TNTMemoryState` inputs the global-state weights are used;
    local-state weights are not considered here.

    Args:
        old_state: Memory state before the update.
        new_state: Memory state after the update.

    Returns:
        List of Frobenius norms, one per memory layer.
    """
    old_weights = _extract_weights(old_state)
    new_weights = _extract_weights(new_state)
    return [_frobenius(w_new - w_old) for w_old, w_new in zip(old_weights, new_weights)]


def compute_momentum_shift(
    old_state: MemoryState | TNTMemoryState,
    new_state: MemoryState | TNTMemoryState,
) -> list[float]:
    """Compute the per-layer Frobenius norm of momentum changes.

    For each layer *i*, computes ``||m_new[i] - m_old[i]||_F``.

    For :class:`TNTMemoryState` inputs the global-state momentum is used.

    Args:
        old_state: Memory state before the update.
        new_state: Memory state after the update.

    Returns:
        List of Frobenius norms, one per memory layer.
    """
    old_mom = _extract_momentum(old_state)
    new_mom = _extract_momentum(new_state)
    return [_frobenius(m_new - m_old) for m_old, m_new in zip(old_mom, new_mom)]


def compute_weight_norms(state: MemoryState | TNTMemoryState) -> list[float]:
    """Compute the absolute per-layer Frobenius norms of weight tensors.

    For :class:`TNTMemoryState` inputs the global-state weights are used.

    Args:
        state: A memory state instance.

    Returns:
        List of Frobenius norms, one per memory layer.
    """
    return [_frobenius(w) for w in _extract_weights(state)]


def compute_momentum_norms(state: MemoryState | TNTMemoryState) -> list[float]:
    """Compute the absolute per-layer Frobenius norms of momentum tensors.

    For :class:`TNTMemoryState` inputs the global-state momentum is used.

    Args:
        state: A memory state instance.

    Returns:
        List of Frobenius norms, one per memory layer.
    """
    return [_frobenius(m) for m in _extract_momentum(state)]


def build_signal_frame(
    old_state: MemoryState | TNTMemoryState,
    new_state: MemoryState | TNTMemoryState,
    gates: GateSnapshot,
    chunk_index: int,
    prediction_error_norms: list[float] | None = None,
    gradient_norms: list[float] | None = None,
    batch_variance: float | None = None,
) -> SignalFrame:
    """Build a complete :class:`SignalFrame` from two consecutive memory states.

    Computes all derivable signals (weight delta, momentum shift, weight norms,
    momentum norms, gate means) and combines them with any externally supplied
    signals (prediction errors, gradients) that require access to model
    internals unavailable in this module.

    For :class:`TNTMemoryState` inputs, the TNT-specific field
    ``local_signal_norms`` is populated; for plain :class:`MemoryState`
    inputs it is set to ``None``.

    Args:
        old_state: Memory state before the update.
        new_state: Memory state after the update.
        gates: Gate snapshot captured during the corresponding forward pass.
        chunk_index: Index of the chunk this frame represents.
        prediction_error_norms: Per-layer L2 norm of prediction errors.
            Defaults to a list of zeros when not provided.
        gradient_norms: Per-layer gradient norms.  Defaults to zeros.
        batch_variance: Variance across the batch dimension, or ``None``.

    Returns:
        A fully populated :class:`SignalFrame`.
    """
    n_layers = len(_extract_weights(new_state))

    # Signals computed from state tensors.
    weight_delta_norms = compute_weight_delta(old_state, new_state)
    momentum_shift_norms = compute_momentum_shift(old_state, new_state)
    weight_norms = compute_weight_norms(new_state)
    momentum_norms = compute_momentum_norms(new_state)

    # Gate means — one scalar per layer from the GateSnapshot.
    gate_alpha_means = [float(t.float().mean().item()) for t in gates.alpha]
    gate_theta_means = [float(t.float().mean().item()) for t in gates.theta]
    gate_eta_means = [float(t.float().mean().item()) for t in gates.eta]

    # Optional signals that require model internals — default to zeros.
    if prediction_error_norms is None:
        prediction_error_norms = [0.0] * n_layers
    if gradient_norms is None:
        gradient_norms = [0.0] * n_layers

    # TNT-specific fields.
    local_signal_norms: list[list[float]] | None = None

    if isinstance(new_state, TNTMemoryState):
        local_signal_norms = [
            [_frobenius(w) for w in local.weights] for local in new_state.local_states
        ]

    return SignalFrame(
        chunk_index=chunk_index,
        prediction_error_norms=prediction_error_norms,
        weight_delta_norms=weight_delta_norms,
        momentum_shift_norms=momentum_shift_norms,
        gradient_norms=gradient_norms,
        weight_norms=weight_norms,
        momentum_norms=momentum_norms,
        gate_alpha_means=gate_alpha_means,
        gate_theta_means=gate_theta_means,
        gate_eta_means=gate_eta_means,
        batch_variance=batch_variance,
        local_signal_norms=local_signal_norms,
    )
