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
        local_inits: Initial weight snapshots per local memory
        qk_projections: Accumulated Q-K projection matrices (M_t^(i))
        local_step_counters: Position within shard for each local memory
    """

    global_state: MemoryState
    local_states: list[MemoryState]
    local_inits: list[list[torch.Tensor]]
    qk_projections: list[torch.Tensor]
    local_step_counters: list[int]

    def detach(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.detach(),
            local_states=[s.detach() for s in self.local_states],
            local_inits=[
                [w.detach() for w in init_list] for init_list in self.local_inits
            ],
            qk_projections=[qk.detach() for qk in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )

    def clone(self) -> TNTMemoryState:
        return TNTMemoryState(
            global_state=self.global_state.clone(),
            local_states=[s.clone() for s in self.local_states],
            local_inits=[
                [w.detach().clone() for w in init_list]
                for init_list in self.local_inits
            ],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def forward_with_weights(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]


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
        predictions = F.linear(keys, weight)
        err_clip = self.config.memory_error_clip
        raw_error = torch.clamp(predictions - values, -err_clip, err_clip)

        if self.memory_objective == "huber" and delta is not None:
            abs_error = torch.abs(raw_error)
            error = torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))
        else:
            error = raw_error

        scale = 2.0 / float(error.numel())
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
        batch_size, seq_len = keys.shape[0], keys.shape[1]
        batch_seq = batch_size * seq_len

        activations = [keys]
        pre_activations = []
        h = keys

        for i in range(num_layers):
            h_pre = F.linear(h, weights[i])
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

        scale = 2.0 / float(error.numel())
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
                delta_bp = F.linear(delta_bp, weights[i].T)
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
        weights = [w.detach().clone() for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        return_state: bool = True,
        lr_scale: float | torch.Tensor = 1.0,
        memory_gate: torch.Tensor | None = None,
        return_keys: bool = False,
    ) -> (
        tuple[torch.Tensor, MemoryState | None]
        | tuple[torch.Tensor, MemoryState | None, torch.Tensor]
    ):
        batch_size = x.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        if self.config.quantize_memory_state:
            raise NotImplementedError(
                "Quantized memory state not yet ported. "
                "See archive/titans_mlx/quantize_state.py"
            )

        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        k, v, q = self._apply_conv(k, v, q)

        k = F.silu(k)
        v = F.silu(v)
        q = F.silu(q)

        # L2-normalize in float32
        q_f32 = q.float()
        k_f32 = k.float()
        q = (q_f32 / torch.sqrt(
            torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(q.dtype)
        k = (k_f32 / torch.sqrt(
            torch.sum(k_f32 * k_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(k.dtype)

        retrieved = self.memory.forward_with_weights(q, state.weights)

        # Data-dependent gates
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.sigmoid(self.gate_decay_proj(x_mean))
        theta = torch.sigmoid(self.gate_lr_proj(x_mean)) * self.config.memory_lr
        eta = torch.sigmoid(self.gate_momentum_proj(x_mean)) * self.config.memory_momentum
        alpha = torch.mean(alpha)
        theta = torch.mean(theta)
        eta = torch.mean(eta)

        if self.memory_objective == "huber":
            delta_val = torch.sigmoid(self.gate_delta_proj(x_mean))
            delta_val = torch.mean(delta_val) * self.config.memory_error_clip
            self._current_delta = delta_val

        if memory_gate is not None:
            lr_scale = memory_gate

        theta = theta * lr_scale

        if len(state.weights) == 1:
            new_state = self._parallel_memory_update_linear(
                k, v, state, alpha, theta, eta
            )
        else:
            delta_val = getattr(self, "_current_delta", None)
            grads = self._compute_gradients(k, v, state.weights, delta=delta_val)
            new_momentum = [eta * m - theta * g for m, g in zip(state.momentum, grads)]
            new_weights = [
                (1 - alpha) * w + s for w, s in zip(state.weights, new_momentum)
            ]
            new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        output = self.proj_out(retrieved)

        if return_state:
            detached = new_state.detach()
            if return_keys:
                return output, detached, k
            return output, detached

        if return_keys:
            return output, None, k
        return output, None

    def _parallel_memory_update_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        state: MemoryState,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
    ) -> MemoryState:
        """Tensorized parallel memory update for linear memory (Section 3.2)."""
        B, S, D = keys.shape
        W_0 = state.weights[0]
        S_prev = state.momentum[0]

        err_clip = self.config.memory_error_clip
        grad_clip = self.config.memory_grad_clip
        decay = 1.0 - alpha
        S_f = float(S)

        preds = F.linear(keys, W_0)
        errors = torch.clamp(preds - values, -err_clip, err_clip)

        if self.memory_objective == "huber":
            hub_delta = getattr(self, "_current_delta", None)
            if hub_delta is not None:
                abs_errors = torch.abs(errors)
                errors = torch.where(abs_errors <= hub_delta, errors, hub_delta * torch.sign(errors))

        scale = 2.0 / float(B * S * D)
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

        retrieved = self.memory.forward_with_weights(q, state.weights)
        return self.proj_out(retrieved)
