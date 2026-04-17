# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans (PyTorch Implementation).

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig

_L2_NORM_EPS: float = 1e-8
_DEGENERATE_THRESHOLD: float = 1e-6


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable] = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


@dataclass
class MemoryState:
    """State of the neural long-term memory."""

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach().clone() for w in self.weights],
            momentum=[m.detach().clone() for m in self.momentum],
        )


@dataclass
class TNTMemoryState:
    """State for TNT hierarchical memory system.

    Attributes:
        global_state: MemoryState for the global memory (V)
        local_states: List of MemoryState, one per local memory (W^(i))
        qk_projections: Accumulated Q-K projection matrices (M_t^(i))
        local_step_counters: Position within shard for each local memory
    """

    global_state: MemoryState
    local_states: list[MemoryState]
    qk_projections: list[torch.Tensor]
    local_step_counters: list[int]

    def detach(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.detach(),
            local_states=[s.detach() for s in self.local_states],
            qk_projections=[qk.detach() for qk in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )

    def clone(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.clone(),
            local_states=[s.clone() for s in self.local_states],
            qk_projections=[qk.detach().clone() for qk in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )


class MemoryMLP(nn.Module):
    """MLP that stores information in its weights."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim
        self.activation = get_activation(config.activation)

        layers: list[nn.Linear] = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        self.layers = nn.ModuleList(layers)
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=std)

    def forward_with_weights(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]

    def get_base_weights(self) -> list[torch.Tensor]:
        """Return live weight parameter references (with autograd graph)."""
        return [layer.weight for layer in self.layers]


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module (PyTorch Implementation).

    Learns to memorize at test time using gradient descent with momentum
    and weight decay. Gradient computation is analytical (not autograd).
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        self.memory = MemoryMLP(config)

        self.gate_decay_proj = nn.Linear(config.dim, 1)
        self.gate_lr_proj = nn.Linear(config.dim, 1)
        self.gate_momentum_proj = nn.Linear(config.dim, 1)

        self.memory_objective = config.memory_objective
        if self.memory_objective == "huber":
            self.gate_delta_proj = nn.Linear(config.dim, 1)

        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

        nn.init.constant_(self.gate_decay_proj.bias, self.config.gate_decay_bias_init)

        if self.memory_objective == "huber":
            nn.init.constant_(self.gate_delta_proj.bias, self.config.huber_delta_init)

    def _apply_conv(
        self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_conv:
            return k, v, q
        seq_len = k.shape[1]
        # PyTorch Conv1d: (B, C, L) — transpose from (B, L, C)
        k = self.conv_k(k.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        v = self.conv_v(v.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        q = self.conv_q(q.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        return k, v, q

    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        num_layers = len(weights)
        if num_layers == 1:
            return self._compute_gradients_linear(keys, values, weights[0], delta=delta)
        return self._compute_gradients_deep(keys, values, weights, delta=delta)

    def _compute_gradients_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weight: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        pred_W = self._get_effective_weights([weight], detach_base=True)[0]
        predictions = F.linear(keys, pred_W)
        err_clip = self.config.memory_error_clip
        raw_error = torch.clamp(predictions - values, -err_clip, err_clip)

        if self.memory_objective == "huber" and delta is not None:
            abs_error = torch.abs(raw_error)
            error = torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))
        else:
            error = raw_error

        # Per paper §3.1: loss averages over sequence length S only.
        # error.shape = (B, S, D_out); we want 2/S.
        seq_len = error.shape[1]
        scale = 2.0 / float(seq_len)
        batch_seq = error.shape[0] * error.shape[1]
        error_flat = error.reshape(batch_seq, -1)
        keys_flat = keys.reshape(batch_seq, -1)
        grad_w = scale * (error_flat.T @ keys_flat)

        grad_clip = self.config.memory_grad_clip
        return [torch.clamp(grad_w, -grad_clip, grad_clip)]

    def _compute_gradients_deep(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        num_layers = len(weights)
        effective = self._get_effective_weights(weights, detach_base=True)
        batch_size, seq_len = keys.shape[0], keys.shape[1]
        batch_seq = batch_size * seq_len

        activations = [keys]
        pre_activations = []
        h = keys

        for i in range(num_layers):
            h_pre = F.linear(h, effective[i])
            pre_activations.append(h_pre)
            if i < num_layers - 1:
                h = self.memory.activation(h_pre)
                activations.append(h)
            else:
                h = h_pre

        err_clip = self.config.memory_error_clip
        raw_error = torch.clamp(h - values, -err_clip, err_clip)

        if self.memory_objective == "huber" and delta is not None:
            abs_error = torch.abs(raw_error)
            error = torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))
        else:
            error = raw_error

        # Per paper §3.1: outer scale is 2/S, not 2/(B*S*D_out).
        # This multiplier is absorbed by learnable theta = sigmoid(.)*memory_lr;
        # no behavioral regression expected, just paper-correct calibration.
        scale = 2.0 / float(seq_len)
        delta_bp = scale * error

        grad_clip = self.config.memory_grad_clip
        grads: list[torch.Tensor | None] = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            act = activations[i]
            delta_bp_flat = delta_bp.reshape(batch_seq, -1)
            act_flat = act.reshape(batch_seq, -1)
            grad_w = delta_bp_flat.T @ act_flat
            grads[i] = torch.clamp(grad_w, -grad_clip, grad_clip)

            if i > 0:
                delta_bp = F.linear(delta_bp, effective[i].T)
                x = pre_activations[i - 1]
                delta_bp = delta_bp * self._activation_derivative(x)

        return grads  # type: ignore[return-value]

    def _activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.activation == "silu":
            sig = torch.sigmoid(x)
            return sig * (1.0 + x * (1.0 - sig))
        elif self.config.activation == "gelu":
            x_f32 = x.float()
            sqrt2 = 1.4142135623730951
            cdf = 0.5 * (1.0 + torch.erf(x_f32 / sqrt2))
            pdf = torch.exp(-0.5 * x_f32 * x_f32) * 0.3989422804014327
            return (cdf + x_f32 * pdf).to(x.dtype)
        elif self.config.activation == "relu":
            return (x > 0).to(x.dtype)
        else:
            raise ValueError(f"No derivative for activation: {self.config.activation}")

    def init_state(self, batch_size: int) -> MemoryState:  # noqa: ARG002
        if self.config.delta_memory_param:
            # Deltas start at zero — no corrections from base initially
            base_weights = self.memory.get_weights()  # for shape/device only
            weights = [torch.zeros_like(w) for w in base_weights]
        else:
            weights = [w.detach().clone() for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def _get_effective_weights(
        self, deltas: list[torch.Tensor], detach_base: bool = False,
    ) -> list[torch.Tensor]:
        """Compose effective weights from base + delta.

        Args:
            deltas: State weight tensors (deltas when delta_memory_param=True,
                absolute weights when False).
            detach_base: If True, detach base weights to prevent outer-loop
                gradients from flowing through the inner-loop gradient
                computation. Use True for inner-loop predictions, False
                for retrieval (where outer-loop gradient to base is desired).

        Returns:
            Effective weight tensors for use in predictions/retrieval.
        """
        if not self.config.delta_memory_param:
            return deltas
        bases = self.memory.get_base_weights()
        if detach_base:
            return [b.detach() + d for b, d in zip(bases, deltas)]
        return [b + d for b, d in zip(bases, deltas)]

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        return_state: bool = True,
        lr_scale: float | torch.Tensor = 1.0,
        memory_gate: torch.Tensor | None = None,
        return_keys: bool = False,
        retrieve_after_update: bool = True,
    ) -> tuple:
        """Forward with optional retrieval source.

        Args:
            retrieve_after_update: When True (default), retrieve with the
                POST-update effective weights (paper-consistent for
                MAG/MAL/LMM where ``x̃`` is the same input fed to memory and
                consumed by the gate). When False (paper Eq. 24 for MAC),
                retrieve with the INCOMING state — the returned ``output`` is
                computed from ``state`` before the update.
        """
        batch_size = x.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        if self.config.quantize_memory_state:
            from titans.quantize_state import QuantizedMemoryState

            if isinstance(state, QuantizedMemoryState):
                # Pass the caller's compute dtype so the dequantized tensors
                # don't silently upcast inside bf16/fp16 autocast regions.
                state = state.dequantize(dtype=x.dtype)

        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        k, v, q = self._apply_conv(k, v, q)

        # Per paper Eq. 11-13: only Q and K pass through SiLU + L2-norm.
        # V is kept as the raw linear projection so the MLP target is
        # not forced onto the unit sphere (which would impose an error floor).
        k = F.silu(k)
        q = F.silu(q)

        # L2-normalize Q and K in float32 (V is left untouched)
        q_f32 = q.float()
        k_f32 = k.float()
        q = (q_f32 / torch.sqrt(
            torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(q.dtype)
        k = (k_f32 / torch.sqrt(
            torch.sum(k_f32 * k_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(k.dtype)

        # Data-dependent gates — chunk-mean over sequence axis is retained
        # (Titans Revisited endorses chunk-level gates), but the per-sample
        # dimension is preserved: resulting shape is (B, 1, 1) so each sample
        # in the batch gets its own gate.  Broadcast works against (B, S, D).
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.sigmoid(self.gate_decay_proj(x_mean))
        theta = torch.sigmoid(self.gate_lr_proj(x_mean)) * self.config.memory_lr
        eta = torch.sigmoid(self.gate_momentum_proj(x_mean)) * self.config.memory_momentum

        delta_val: torch.Tensor | None = None
        if self.memory_objective == "huber":
            # Per-sample delta: shape (B, 1, 1), no batch-mean.
            delta_val = (
                torch.sigmoid(self.gate_delta_proj(x_mean)) * self.config.memory_error_clip
            )

        gate_snapshot = None
        if self.config.auto_checkpoint:
            from titans.checkpoint_types import GateSnapshot
            gate_snapshot = GateSnapshot(
                alpha=[alpha.detach()],
                theta=[theta.detach()],
                eta=[eta.detach()],
                delta=(
                    [delta_val.detach()]
                    if self.memory_objective == "huber"
                    else None
                ),
                input_activation_norm=float(x_mean.detach().float().norm().item()),
                chunk_index=0,  # Set by caller
            )

        if memory_gate is not None:
            lr_scale = memory_gate

        theta = theta * lr_scale

        if len(state.weights) == 1:
            if self.config.per_chunk_decay:
                # Sigmoid output = per-chunk decay fraction.  Convert to the
                # per-token alpha the parallel formula needs so that
                # (1 - token_alpha)^S = 1 - chunk_alpha.
                chunk_alpha = alpha
                seq_len = float(x.shape[1])
                alpha = 1.0 - torch.pow(1.0 - chunk_alpha, 1.0 / seq_len)
            new_state = self._parallel_memory_update_linear(
                k, v, state, alpha, theta, eta, delta=delta_val
            )
        else:
            # Shared deep-memory weights require batch-scalar gate values for
            # the weight update.  Per-sample gates preserve their shape during
            # the loss/gradient computation (they enter via error scaling); for
            # the update itself we mean-reduce over the batch so the shared
            # weight tensor remains 2D.
            alpha_scalar = alpha.mean()
            theta_scalar = theta.mean()
            eta_scalar = eta.mean()

            K = int(self.config.num_memory_inner_steps)
            if K == 1:
                # Legacy single-step path (exact backward-compat for K=1).
                grads = self._compute_gradients(
                    k, v, state.weights, delta=delta_val
                )
                new_momentum = [
                    eta_scalar * m - theta_scalar * g
                    for m, g in zip(state.momentum, grads)
                ]
                new_weights = [
                    (1 - alpha_scalar) * w + s
                    for w, s in zip(state.weights, new_momentum)
                ]
            else:
                # K-step inner loop approximating per-token online updates
                # (paper §3.2).  Split the chunk into K roughly-equal
                # sub-chunks and run one analytical gradient + momentum
                # update per sub-chunk.  The first K-1 steps run under
                # no_grad (analytical grads; no retained autograd graph).
                # The final step is re-done WITH autograd so the outer-loop
                # gate projections (alpha, theta, eta) receive gradient
                # from the LM loss via the retained effective-weights path.
                seq_len = k.shape[1]
                bounds = [
                    int(round(seq_len * i / K)) for i in range(K + 1)
                ]
                # Rewind the first K-1 updates under no_grad with detached
                # gates so no autograd graph is retained across steps.
                prev_w = [w for w in state.weights]
                prev_m = [m for m in state.momentum]
                alpha_d = alpha_scalar.detach()
                theta_d = theta_scalar.detach()
                eta_d = eta_scalar.detach()
                with torch.no_grad():
                    for i in range(K - 1):
                        lo_i, hi_i = bounds[i], bounds[i + 1]
                        if hi_i <= lo_i:
                            continue
                        k_sub = k[:, lo_i:hi_i, :]
                        v_sub = v[:, lo_i:hi_i, :]
                        grads = self._compute_gradients(
                            k_sub, v_sub, prev_w, delta=delta_val,
                        )
                        prev_m = [
                            eta_d * m - theta_d * g
                            for m, g in zip(prev_m, grads)
                        ]
                        prev_w = [
                            (1 - alpha_d) * w + s
                            for w, s in zip(prev_w, prev_m)
                        ]
                # Final step WITH autograd on the gates.  prev_w / prev_m
                # carry no graph (detached), but alpha_scalar, theta_scalar,
                # eta_scalar do — so gate projections receive gradient from
                # the returned new_weights / new_momentum.
                lo, hi = bounds[K - 1], bounds[K]
                if hi > lo:
                    k_final = k[:, lo:hi, :]
                    v_final = v[:, lo:hi, :]
                    grads = self._compute_gradients(
                        k_final, v_final, prev_w, delta=delta_val,
                    )
                    new_momentum = [
                        eta_scalar * m - theta_scalar * g
                        for m, g in zip(prev_m, grads)
                    ]
                    new_weights = [
                        (1 - alpha_scalar) * w + s
                        for w, s in zip(prev_w, new_momentum)
                    ]
                else:
                    # Degenerate final sub-chunk: still route through gates
                    # so their autograd graph is attached to new_state.
                    new_momentum = [eta_scalar * m for m in prev_m]
                    new_weights = [
                        (1 - alpha_scalar) * w + s
                        for w, s in zip(prev_w, new_momentum)
                    ]
            new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        if retrieve_after_update:
            # Paper Eq. 3-4 for MAG/MAL/LMM: retrieve from the updated state.
            # Places alpha/theta/eta in the output's graph so gate projections
            # receive gradient from the LM loss.
            effective = self._get_effective_weights(
                new_state.weights, detach_base=False
            )
        else:
            # Paper Eq. 24 for MAC: retrieve from the INCOMING state
            # (M_{t-1}). Gate projections still receive gradient via the
            # returned new_state through the caller's retain-state flow.
            effective = self._get_effective_weights(
                state.weights, detach_base=False
            )
        retrieved = self.memory.forward_with_weights(q, effective)

        output = self.proj_out(retrieved)

        if return_state:
            # Detach controls whether the RETURNED state has its autograd graph.
            # After the retrieval reorder, output already depends on new_state
            # (via forward_with_weights), so gates receive gradients within the
            # current chunk regardless of this flag. The flag now only controls
            # whether cross-chunk gradient flow is possible through the returned
            # state. Kept for config serialization compatibility.
            must_detach = (
                self.config.detach_memory_state_in_forward
                or self.config.quantize_memory_state
            )
            returned_state: MemoryState = (
                new_state.detach() if must_detach else new_state
            )
            if self.config.quantize_memory_state:
                from titans.quantize_state import quantize_memory_state

                returned_state = quantize_memory_state(
                    returned_state,
                    weight_bits=self.config.memory_state_weight_bits,
                    momentum_bits=self.config.memory_state_momentum_bits,
                )
            if return_keys:
                return output, returned_state, gate_snapshot, k
            return output, returned_state, gate_snapshot

        if return_keys:
            return output, None, gate_snapshot, k
        return output, None, gate_snapshot

    def _parallel_memory_update_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        state: MemoryState,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> MemoryState:
        """Tensorized parallel memory update for linear memory (Section 3.2)."""
        B, S, D = keys.shape
        W_0 = state.weights[0]
        S_prev = state.momentum[0]

        # Shared (non-per-sample) weights: reduce per-sample gates to batch
        # scalars so new W / S remain (hidden_dim, dim).  The gate gradient
        # signal still flows per-sample through errors_scaled below — gates
        # only collapse at the final weight-aggregation boundary.  .mean() on
        # a singleton (B=1) is a no-op mathematically but rank-reduces to 0-D.
        alpha = alpha.mean()
        eta = eta.mean()
        theta = theta.mean()

        err_clip = self.config.memory_error_clip
        grad_clip = self.config.memory_grad_clip
        decay = 1.0 - alpha
        S_f = float(S)

        # Predict from effective weights (base + delta when delta_memory_param)
        pred_W = self._get_effective_weights([W_0], detach_base=True)[0]
        preds = F.linear(keys, pred_W)
        errors = torch.clamp(preds - values, -err_clip, err_clip)

        if self.memory_objective == "huber":
            hub_delta = delta
            if hub_delta is not None:
                abs_errors = torch.abs(errors)
                errors = torch.where(abs_errors <= hub_delta, errors, hub_delta * torch.sign(errors))

        # Per paper §3.1: outer scale is 2/S (absorbed into learnable theta).
        scale = 2.0 / float(S)
        errors_scaled = errors * scale

        positions = torch.arange(S, dtype=torch.float32, device=keys.device)

        diff = decay - eta
        abs_diff = torch.abs(diff)
        is_degenerate = abs_diff < _DEGENERATE_THRESHOLD
        safe_diff = torch.where(
            is_degenerate,
            torch.tensor(1.0, device=keys.device),
            torch.maximum(abs_diff, torch.tensor(_L2_NORM_EPS, device=keys.device)) * torch.sign(diff),
        )

        # New momentum
        eta_powers = torch.pow(eta, S_f - 1.0 - positions)
        eta_w = eta_powers.reshape(1, S, 1)

        weighted_eta = errors_scaled * eta_w
        grad_eta_sum = torch.mean(
            weighted_eta.permute(0, 2, 1) @ keys,
            dim=0,
        )
        grad_eta_sum = torch.clamp(grad_eta_sum, -grad_clip, grad_clip)
        new_momentum = torch.pow(eta, S_f) * S_prev - theta * grad_eta_sum

        # New weights
        decay_S = torch.pow(decay, S_f)
        eta_S = torch.pow(eta, S_f)
        c_S0_general = eta * (decay_S - eta_S) / safe_diff
        c_S0_degen = S_f * eta_S
        c_S0 = torch.where(is_degenerate, c_S0_degen, c_S0_general)

        n_vals = S_f - 1.0 - positions
        w_general = (
            torch.pow(decay, n_vals + 1.0) - torch.pow(eta, n_vals + 1.0)
        ) / safe_diff
        w_degen = (n_vals + 1.0) * torch.pow((decay + eta) / 2.0, n_vals)
        w_weights = torch.where(is_degenerate, w_degen, w_general)
        w_w = w_weights.reshape(1, S, 1)

        weighted_w = errors_scaled * w_w
        grad_combined = torch.mean(
            weighted_w.permute(0, 2, 1) @ keys,
            dim=0,
        )
        grad_combined = torch.clamp(grad_combined, -grad_clip, grad_clip)

        new_weights = decay_S * W_0 + c_S0 * S_prev - theta * grad_combined

        return MemoryState(weights=[new_weights], momentum=[new_momentum])

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        q = self.proj_q(queries)

        if self.use_conv:
            seq_len = q.shape[1]
            q = self.conv_q(q.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]

        q = F.silu(q)
        q_f32 = q.float()
        q = (q_f32 / torch.sqrt(
            torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(q.dtype)

        effective = self._get_effective_weights(state.weights, detach_base=False)
        retrieved = self.memory.forward_with_weights(q, effective)
        return self.proj_out(retrieved)
