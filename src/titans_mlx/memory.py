# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans (MLX Implementation).

This module implements the core innovation of Titans: a neural memory that
learns to memorize at test time using gradient descent with momentum and
weight decay.

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2

MLX-specific optimizations:
- Lazy evaluation for efficient computation graphs
- Unified memory (no CPU/GPU transfers)
- Vectorized operations for Apple Silicon
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx.config import TitansConfig


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Get activation function by name."""
    activations = {
        "silu": nn.silu,
        "gelu": nn.gelu,
        "relu": nn.relu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


@dataclass
class MemoryState:
    """State of the neural long-term memory.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """

    weights: list[mx.array]
    momentum: list[mx.array]

    def detach(self) -> MemoryState:
        """Detach state (stop gradients)."""
        return MemoryState(
            weights=[mx.stop_gradient(w) for w in self.weights],
            momentum=[mx.stop_gradient(m) for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        """Clone the memory state."""
        return MemoryState(
            weights=[mx.array(w) for w in self.weights],
            momentum=[mx.array(m) for m in self.momentum],
        )


@dataclass
class TNTMemoryState:
    """State for TNT hierarchical memory system.

    The TNT architecture uses a global memory (V) for long-range context
    and N local memories (W^(i)) with different chunk sizes for fine-grained
    recall. Local memories are periodically reset to learned initial weights.

    Attributes:
        global_state: MemoryState for the global memory (V)
        local_states: List of MemoryState, one per local memory (W^(i))
        local_inits: Initial weight snapshots per local memory (W_init^(i)),
            used to reset local memories at shard boundaries
        qk_projections: Q-K projection matrices (M_t^(i)), one per local
            memory — accumulated outer products that align query space to
            the key space used during memory compression (TNT Eq. 7)
        local_step_counters: Position within shard for each local memory,
            used to trigger resets when reaching local_shard_length
    """

    global_state: MemoryState
    local_states: list[MemoryState]
    local_inits: list[list[mx.array]]
    qk_projections: list[mx.array]
    local_step_counters: list[int]

    def detach(self) -> TNTMemoryState:
        """Detach state (stop gradients)."""
        return TNTMemoryState(
            global_state=self.global_state.detach(),
            local_states=[s.detach() for s in self.local_states],
            local_inits=[
                [mx.stop_gradient(w) for w in init]
                for init in self.local_inits
            ],
            qk_projections=[mx.stop_gradient(m) for m in self.qk_projections],
            local_step_counters=list(self.local_step_counters),
        )

    def reset_local(self, index: int) -> TNTMemoryState:
        """Reset a local memory to its initial weights.

        Called at shard boundaries when local_step_counters[index] reaches
        local_shard_length.

        Args:
            index: Which local memory to reset

        Returns:
            New TNTMemoryState with the specified local memory reset
        """
        new_local_states = list(self.local_states)
        init_weights = self.local_inits[index]
        new_local_states[index] = MemoryState(
            weights=[mx.array(w) for w in init_weights],
            momentum=[mx.zeros_like(w) for w in init_weights],
        )
        new_qk = list(self.qk_projections)
        new_qk[index] = mx.zeros_like(self.qk_projections[index])
        new_counters = list(self.local_step_counters)
        new_counters[index] = 0
        return TNTMemoryState(
            global_state=self.global_state,
            local_states=new_local_states,
            local_inits=self.local_inits,
            qk_projections=new_qk,
            local_step_counters=new_counters,
        )


def save_memory_states(states: list[MemoryState], path: Path) -> None:
    """Serialize memory states from all layers to a single .npz file.

    Naming scheme: layer_{i}_weight_{j}, layer_{i}_momentum_{j}.
    Includes metadata for validation on load.

    Args:
        states: List of MemoryState, one per model layer.
        path: Output file path (will be saved as .npz).
    """
    arrays: dict[str, np.ndarray] = {}
    arrays["num_layers"] = np.array([len(states)])

    for i, state in enumerate(states):
        arrays[f"num_memory_layers_{i}"] = np.array([len(state.weights)])
        for j, w in enumerate(state.weights):
            arrays[f"layer_{i}_weight_{j}"] = np.array(w)
        for j, m in enumerate(state.momentum):
            arrays[f"layer_{i}_momentum_{j}"] = np.array(m)

    path = Path(path)
    np.savez(path, **arrays)


def load_memory_states(path: Path) -> list[MemoryState]:
    """Deserialize memory states from a .npz file.

    Validates structure and returns list of MemoryState ready to pass to model.

    Args:
        path: Path to .npz file saved by save_memory_states.

    Returns:
        List of MemoryState, one per model layer.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        # numpy appends .npz if missing
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Memory state file not found: {path}")
        path = path.with_suffix(".npz")

    data = np.load(str(path))

    if "num_layers" not in data:
        raise ValueError(f"Invalid memory state file: missing 'num_layers' metadata")

    num_layers = int(data["num_layers"][0])
    states: list[MemoryState] = []

    for i in range(num_layers):
        key = f"num_memory_layers_{i}"
        if key not in data:
            raise ValueError(f"Invalid memory state file: missing '{key}'")
        num_memory_layers = int(data[key][0])

        weights: list[mx.array] = []
        momentum: list[mx.array] = []
        for j in range(num_memory_layers):
            wk = f"layer_{i}_weight_{j}"
            mk = f"layer_{i}_momentum_{j}"
            if wk not in data:
                raise ValueError(f"Invalid memory state file: missing '{wk}'")
            if mk not in data:
                raise ValueError(f"Invalid memory state file: missing '{mk}'")
            weights.append(mx.array(data[wk]))
            momentum.append(mx.array(data[mk]))

        states.append(MemoryState(weights=weights, momentum=momentum))

    return states


def save_tnt_memory_states(states: list[TNTMemoryState], path: Path) -> None:
    """Serialize TNT hierarchical memory states to a single .npz file.

    Stores global state, N local states, local init weights, Q-K projection
    matrices, and step counters for each model layer.

    Args:
        states: List of TNTMemoryState, one per model layer.
        path: Output file path (will be saved as .npz).
    """
    arrays: dict[str, np.ndarray] = {}
    arrays["num_layers"] = np.array([len(states)])
    arrays["state_type"] = np.array([1])  # 1 = TNT

    for i, state in enumerate(states):
        # Global state
        n_global = len(state.global_state.weights)
        arrays[f"layer_{i}_global_num_weights"] = np.array([n_global])
        for j, w in enumerate(state.global_state.weights):
            arrays[f"layer_{i}_global_weight_{j}"] = np.array(w)
        for j, m in enumerate(state.global_state.momentum):
            arrays[f"layer_{i}_global_momentum_{j}"] = np.array(m)

        # Local states
        n_locals = len(state.local_states)
        arrays[f"layer_{i}_num_locals"] = np.array([n_locals])

        for li in range(n_locals):
            local = state.local_states[li]
            n_local_weights = len(local.weights)
            arrays[f"layer_{i}_local_{li}_num_weights"] = np.array([n_local_weights])
            for j, w in enumerate(local.weights):
                arrays[f"layer_{i}_local_{li}_weight_{j}"] = np.array(w)
            for j, m in enumerate(local.momentum):
                arrays[f"layer_{i}_local_{li}_momentum_{j}"] = np.array(m)

            # Local init weights
            for j, w in enumerate(state.local_inits[li]):
                arrays[f"layer_{i}_local_{li}_init_{j}"] = np.array(w)

            # Q-K projection matrix
            arrays[f"layer_{i}_qk_proj_{li}"] = np.array(state.qk_projections[li])

        # Step counters
        arrays[f"layer_{i}_step_counters"] = np.array(state.local_step_counters)

    path = Path(path)
    np.savez(path, **arrays)


def load_tnt_memory_states(path: Path) -> list[TNTMemoryState]:
    """Deserialize TNT hierarchical memory states from a .npz file.

    Args:
        path: Path to .npz file saved by save_tnt_memory_states.

    Returns:
        List of TNTMemoryState, one per model layer.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"TNT memory state file not found: {path}")
        path = path.with_suffix(".npz")

    data = np.load(str(path))

    if "num_layers" not in data:
        raise ValueError("Invalid TNT memory state file: missing 'num_layers'")

    num_layers = int(data["num_layers"][0])
    states: list[TNTMemoryState] = []

    for i in range(num_layers):
        # Global state
        n_global = int(data[f"layer_{i}_global_num_weights"][0])
        global_weights = [
            mx.array(data[f"layer_{i}_global_weight_{j}"])
            for j in range(n_global)
        ]
        global_momentum = [
            mx.array(data[f"layer_{i}_global_momentum_{j}"])
            for j in range(n_global)
        ]
        global_state = MemoryState(weights=global_weights, momentum=global_momentum)

        # Local states
        n_locals = int(data[f"layer_{i}_num_locals"][0])
        local_states = []
        local_inits = []
        qk_projections = []

        for li in range(n_locals):
            n_lw = int(data[f"layer_{i}_local_{li}_num_weights"][0])
            lw = [
                mx.array(data[f"layer_{i}_local_{li}_weight_{j}"])
                for j in range(n_lw)
            ]
            lm = [
                mx.array(data[f"layer_{i}_local_{li}_momentum_{j}"])
                for j in range(n_lw)
            ]
            local_states.append(MemoryState(weights=lw, momentum=lm))

            # Local inits
            li_inits = [
                mx.array(data[f"layer_{i}_local_{li}_init_{j}"])
                for j in range(n_lw)
            ]
            local_inits.append(li_inits)

            # Q-K projection
            qk_projections.append(mx.array(data[f"layer_{i}_qk_proj_{li}"]))

        # Step counters
        step_counters = list(int(x) for x in data[f"layer_{i}_step_counters"])

        states.append(TNTMemoryState(
            global_state=global_state,
            local_states=local_states,
            local_inits=local_inits,
            qk_projections=qk_projections,
            local_step_counters=step_counters,
        ))

    return states


class MemoryMLP(nn.Module):
    """MLP architecture for the neural memory.

    This is the actual memory module that stores information in its weights.
    For L_M = 1 (linear memory), this is equivalent to a matrix-valued memory.
    For L_M >= 2 (deep memory), this provides more expressive power.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim
        self.activation = get_activation(config.activation)

        # Build MLP layers
        self.layers: list[nn.Linear] = []

        if self.num_layers == 1:
            # Linear memory: single linear layer
            self.layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            # Deep memory: MLP with hidden layers
            self.layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))

            for _ in range(self.num_layers - 2):
                self.layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                )

            self.layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for layer in self.layers:
            # MLX uses different initialization
            layer.weight = mx.random.normal(layer.weight.shape) * self.config.init_std

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through memory MLP.

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply activation for all but last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def forward_with_weights(self, x: mx.array, weights: list[mx.array]) -> mx.array:
        """Forward pass using explicit weights (no module mutation).

        Args:
            x: Input tensor (batch, seq, dim)
            weights: List of weight matrices, one per layer

        Returns:
            Output tensor (batch, seq, dim)
        """
        h = x
        for i, w in enumerate(weights):
            h = h @ w.T
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[mx.array]:
        """Get current weight matrices."""
        return [mx.array(layer.weight) for layer in self.layers]

    def compute_loss(self, keys: mx.array, values: mx.array) -> mx.array:
        """Compute associative memory loss.

        Loss: ||M(k) - v||^2

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            Scalar loss value
        """
        predictions = self(keys)
        diff = predictions - values
        return mx.mean(diff * diff)


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module (MLX Implementation).

    This is the main memory component of Titans. It learns to memorize
    at test time by treating training as an online learning problem.

    The memory is updated using gradient descent with:
    - Momentum (for past surprise)
    - Weight decay (for forgetting)

    MLX-specific optimizations:
    - Uses mx.grad for efficient gradient computation
    - Lazy evaluation for optimal memory usage
    - Vectorized operations for Apple Silicon
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Projections for keys, values, and queries
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        # Optional 1D convolution
        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        # The actual memory module
        self.memory = MemoryMLP(config)

        # Data-dependent gates (Section 3.1: α_t, θ_t, η_t are functions of x_t)
        # Per Section 3.2: gates are chunk-constant scalars. Use dedicated
        # scalar projections (dim -> 1) instead of projecting to dim and
        # averaging over features, which loses per-dimension information
        # before collapsing to a scalar anyway.
        self.gate_decay_proj = nn.Linear(config.dim, 1)
        self.gate_lr_proj = nn.Linear(config.dim, 1)
        self.gate_momentum_proj = nn.Linear(config.dim, 1)

        # Output projection
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            module.weight = mx.random.normal(module.weight.shape) * self.config.init_std

        # Gate initialization: these gates don't receive backprop gradients (the
        # memory state is detached), so initialization determines their permanent
        # behavior.  Choose values that give reasonable memory dynamics.
        #
        # Decay gate (α): controls per-token forgetting.  Retention after S tokens
        # in a chunk = (1-α)^S.  For chunk_size=512:
        #   sigmoid(-6) ≈ 0.0025 → retention ≈ 0.28 per chunk (current default)
        #   sigmoid( 0) ≈ 0.5    → retention ≈ 0     (catastrophic forgetting)
        self.gate_decay_proj.bias = mx.array([-6.0])
        # LR gate (θ): sigmoid(0)*memory_lr = 0.5*0.1 = 0.05  — reasonable, no change
        # Momentum gate (η): sigmoid(0)*momentum = 0.5*0.9 = 0.45 — reasonable, no change

    def _apply_conv(
        self, k: mx.array, v: mx.array, q: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Apply 1D convolution to K, V, Q.

        MLX Conv1d uses channels-last format (N, L, C), so input (B, S, D)
        maps directly without transposition. Truncate on axis 1 (sequence)
        to remove causal padding.
        """
        if not self.use_conv:
            return k, v, q

        seq_len = k.shape[1]
        k = self.conv_k(k)[:, :seq_len, :]
        v = self.conv_v(v)[:, :seq_len, :]
        q = self.conv_q(q)[:, :seq_len, :]

        return k, v, q

    def _compute_gradients(
        self,
        keys: mx.array,
        values: mx.array,
        weights: list[mx.array],
    ) -> list[mx.array]:
        """Compute gradients for memory update analytically.

        Optimized implementation that minimizes Python overhead.

        For the loss ||M(k) - v||^2, we compute gradients analytically
        to avoid nested mx.grad calls which cause VJP issues.

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)
            weights: Current memory weights

        Returns:
            List of gradient tensors for each memory layer
        """
        num_layers = len(weights)

        # Fast path for linear memory (1 layer) - most common case
        if num_layers == 1:
            return self._compute_gradients_linear(keys, values, weights[0])

        # Multi-layer case: use optimized computation
        return self._compute_gradients_deep(keys, values, weights)

    def _compute_gradients_linear(
        self,
        keys: mx.array,
        values: mx.array,
        weight: mx.array,
    ) -> list[mx.array]:
        """Optimized gradient computation for linear (1-layer) memory.

        For M(k) = W @ k, the gradient is:
            dL/dW = 2/n * sum((W @ k - v) @ k^T)

        Uses matmul instead of expand_dims for efficiency.
        """
        # Forward pass: predictions = keys @ W^T
        predictions = keys @ weight.T

        # Error and gradient scale
        err_clip = self.config.memory_error_clip
        error = mx.clip(predictions - values, -err_clip, err_clip)
        scale = 2.0 / float(error.size)

        # Efficient gradient via matmul: (D_out, B*S) @ (B*S, D_in) -> (D_out, D_in)
        # Flatten batch and seq dims, then use matmul instead of outer product
        batch_seq = error.shape[0] * error.shape[1]
        error_flat = error.reshape(batch_seq, -1)  # (B*S, D_out)
        keys_flat = keys.reshape(batch_seq, -1)    # (B*S, D_in)
        grad_w = scale * (error_flat.T @ keys_flat)  # (D_out, D_in)

        grad_clip = self.config.memory_grad_clip
        return [mx.clip(grad_w, -grad_clip, grad_clip)]

    def _compute_gradients_deep(
        self,
        keys: mx.array,
        values: mx.array,
        weights: list[mx.array],
    ) -> list[mx.array]:
        """Optimized gradient computation for deep (multi-layer) memory.

        Uses matmul instead of expand_dims for efficient gradient computation.
        """
        num_layers = len(weights)
        batch_size, seq_len = keys.shape[0], keys.shape[1]
        batch_seq = batch_size * seq_len

        # Forward pass - collect activations
        activations = [keys]
        pre_activations = []
        h = keys

        for i in range(num_layers):
            h_pre = h @ weights[i].T
            pre_activations.append(h_pre)
            if i < num_layers - 1:
                h = self.memory.activation(h_pre)
                activations.append(h)
            else:
                h = h_pre

        # Error computation
        err_clip = self.config.memory_error_clip
        error = mx.clip(h - values, -err_clip, err_clip)
        scale = 2.0 / float(error.size)
        delta = scale * error

        # Backward pass - compute gradients using efficient matmul
        grad_clip = self.config.memory_grad_clip
        grads = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            act = activations[i]

            # Efficient gradient via matmul: (D_out, B*S) @ (B*S, D_in) -> (D_out, D_in)
            delta_flat = delta.reshape(batch_seq, -1)  # (B*S, D_out)
            act_flat = act.reshape(batch_seq, -1)      # (B*S, D_in)
            grad_w = delta_flat.T @ act_flat           # (D_out, D_in)
            grads[i] = mx.clip(grad_w, -grad_clip, grad_clip)

            # Propagate gradient to previous layer
            if i > 0:
                delta = delta @ weights[i]
                x = pre_activations[i - 1]
                delta = delta * self._activation_derivative(x)

        return grads

    def init_state(self, batch_size: int) -> MemoryState:
        """Initialize memory state.

        Args:
            batch_size: Batch size (reserved for future use)

        Returns:
            Initial memory state
        """
        # Get initial weights from memory module (detached snapshot —
        # the memory learns through its internal gradient mechanism,
        # not through the main optimizer's backprop)
        weights = [mx.stop_gradient(w) for w in self.memory.get_weights()]
        momentum = [mx.zeros_like(w) for w in weights]

        return MemoryState(weights=weights, momentum=momentum)

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        return_state: bool = True,
        lr_scale: float | mx.array = 1.0,
        memory_gate: mx.array | None = None,
        return_keys: bool = False,
    ) -> tuple[mx.array, MemoryState | None] | tuple[mx.array, MemoryState | None, mx.array]:
        """Forward pass with memory update.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous memory state (optional)
            return_state: Whether to return updated state
            lr_scale: Multiplicative scale factor for learning rate
            memory_gate: Optional scalar importance weight from AttnRes
            return_keys: If True, also return the L2-normalized keys
                (avoids redundant recomputation in TNT)

        Returns:
            Tuple of (output, state) or (output, state, normed_keys)
        """
        batch_size = x.shape[0]

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size)

        # Project to keys, values, queries
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        # Apply convolution
        k, v, q = self._apply_conv(k, v, q)

        # Apply SiLU activation
        k = nn.silu(k)
        v = nn.silu(v)
        q = nn.silu(q)

        # Normalize using L2-norm (compute in float32 to avoid bfloat16 underflow)
        q_f32 = q.astype(mx.float32)
        k_f32 = k.astype(mx.float32)
        q = (q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-8)).astype(
            q.dtype
        )
        k = (k_f32 / mx.sqrt(mx.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-8)).astype(
            k.dtype
        )

        # Retrieve from memory using explicit state weights (no module mutation)
        retrieved = self.memory.forward_with_weights(q, state.weights)

        # Compute data-dependent gates (Section 3.1, Eq 13-14)
        # Per Section 3.2: gates are chunk-constant scalars. We average over
        # the sequence dimension to get a chunk-level representation, then
        # project to scalar via dedicated (dim -> 1) projections.
        x_mean = mx.mean(x, axis=1, keepdims=True)  # (B, 1, D)
        alpha = mx.sigmoid(self.gate_decay_proj(x_mean))  # (B, 1, 1)
        theta = (
            mx.sigmoid(self.gate_lr_proj(x_mean)) * self.config.memory_lr
        )  # (B, 1, 1)
        eta = (
            mx.sigmoid(self.gate_momentum_proj(x_mean)) * self.config.memory_momentum
        )  # (B, 1, 1)
        # Average across batch — weights are shared, so per-sample gates must be
        # collapsed. This is a practical tradeoff vs per-sample weight copies.
        alpha = mx.mean(alpha)
        theta = mx.mean(theta)
        eta = mx.mean(eta)

        # memory_gate overrides lr_scale when provided (interface alignment
        # with HierarchicalMemory which also accepts memory_gate)
        if memory_gate is not None:
            lr_scale = memory_gate

        # Apply AttnRes modulation to learning rate
        theta = theta * lr_scale

        # Update memory weights and momentum
        if len(state.weights) == 1:
            # Linear memory: use tensorized parallel update (Section 3.2, Eq 16-18)
            new_state = self._parallel_memory_update_linear(
                k, v, state, alpha, theta, eta
            )
        else:
            # Deep memory: batch-level gradient update (no closed-form parallel)
            grads = self._compute_gradients(k, v, state.weights)
            new_momentum = [eta * m - theta * g for m, g in zip(state.momentum, grads)]
            new_weights = [
                (1 - alpha) * w + s for w, s in zip(state.weights, new_momentum)
            ]
            new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        # Output projection
        output = self.proj_out(retrieved)

        if return_keys:
            if return_state:
                return output, new_state.detach(), k
            return output, None, k

        if return_state:
            return output, new_state.detach()
        return output, None

    def _parallel_memory_update_linear(
        self,
        keys: mx.array,
        values: mx.array,
        state: MemoryState,
        alpha: mx.array,
        theta: mx.array,
        eta: mx.array,
    ) -> MemoryState:
        """Tensorized parallel memory update for linear memory (Section 3.2).

        For the linear case (M = W, 1-layer memory), computes per-token
        gradients w.r.t. initial weights W_0, then solves the momentum
        recurrence (Eq 18) and weight decay (Eq 13) in closed form using
        chunk-constant gates.

        Derivation for chunk-constant α, θ, η over S tokens:
          S_S = η^S · S_0 - θ · Σ_{i=0}^{S-1} η^{S-1-i} · u_i
          M_S = (1-α)^S · W_0 + c_{S0} · S_0 - θ · Σ_{j=0}^{S-1} w_j · u_j
        where:
          u_i = per-token gradient ∇ℓ(W_0; x_i) = 2·(W_0·k_i - v_i)·k_i^T
          c_{S0} = η·[(1-α)^S - η^S] / [(1-α) - η]
          w_j = [(1-α)^{n+1} - η^{n+1}] / [(1-α) - η], n = S-1-j

        The weighted gradient sums are computed via batched matmul,
        avoiding explicit per-token outer products.

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)
            state: Current memory state
            alpha: Decay gate (scalar)
            theta: Learning rate gate (scalar)
            eta: Momentum gate (scalar)

        Returns:
            Updated MemoryState with new weights and momentum
        """
        B, S, D = keys.shape
        W_0 = state.weights[0]
        S_prev = state.momentum[0]

        err_clip = self.config.memory_error_clip
        grad_clip = self.config.memory_grad_clip
        decay = 1.0 - alpha
        S_f = float(S)

        # Per-token errors with initial weights (Eq 17: ∇ℓ(W_0; x_i))
        preds = keys @ W_0.T  # (B, S, D)
        errors = mx.clip(preds - values, -err_clip, err_clip)

        # Scale matches current aggregate normalization for hyperparameter compat
        scale = 2.0 / float(B * S * D)
        errors_scaled = errors * scale  # (B, S, D)

        # Token positions (0-indexed)
        positions = mx.arange(S).astype(mx.float32)  # (S,)

        # Safe denominator for geometric series (decay - η)
        diff = decay - eta
        abs_diff = mx.abs(diff)
        is_degenerate = abs_diff < 1e-6
        safe_diff = mx.where(is_degenerate, mx.array(1.0), mx.maximum(abs_diff, mx.array(1e-8)) * mx.sign(diff + 1e-12))

        # --- New momentum: S_S = η^S · S_prev - θ · Σ η^{S-1-i} · u_i ---
        eta_powers = mx.power(eta, S_f - 1.0 - positions)  # (S,)
        eta_w = mx.reshape(eta_powers, (1, S, 1))

        weighted_eta = errors_scaled * eta_w  # (B, S, D)
        # Batched matmul: (B, D, S) @ (B, S, D) -> (B, D, D), mean over B
        grad_eta_sum = mx.mean(
            mx.transpose(weighted_eta, (0, 2, 1)) @ keys,
            axis=0,
        )
        grad_eta_sum = mx.clip(grad_eta_sum, -grad_clip, grad_clip)

        new_momentum = mx.power(eta, S_f) * S_prev - theta * grad_eta_sum

        # --- New weights: M_S = decay^S · W_0 + c_S0 · S_prev - θ · Σ w_j · u_j ---

        # c_S0: coefficient for S_prev in weight update
        # General: η · (decay^S - η^S) / (decay - η)
        # Degenerate (decay ≈ η): S · η^S
        decay_S = mx.power(decay, S_f)
        eta_S = mx.power(eta, S_f)
        c_S0_general = eta * (decay_S - eta_S) / safe_diff
        c_S0_degen = S_f * eta_S
        c_S0 = mx.where(is_degenerate, c_S0_degen, c_S0_general)

        # w_j: combined decay-momentum weight for each token position
        # n = S-1-j; w_j = (decay^{n+1} - η^{n+1}) / (decay - η)
        n_vals = S_f - 1.0 - positions  # (S,)
        w_general = (
            mx.power(decay, n_vals + 1.0) - mx.power(eta, n_vals + 1.0)
        ) / safe_diff
        w_degen = (n_vals + 1.0) * mx.power((decay + eta) / 2.0, n_vals)
        w_weights = mx.where(is_degenerate, w_degen, w_general)
        w_w = mx.reshape(w_weights, (1, S, 1))

        weighted_w = errors_scaled * w_w  # (B, S, D)
        grad_combined = mx.mean(
            mx.transpose(weighted_w, (0, 2, 1)) @ keys,
            axis=0,
        )
        grad_combined = mx.clip(grad_combined, -grad_clip, grad_clip)

        new_weights = decay_S * W_0 + c_S0 * S_prev - theta * grad_combined

        return MemoryState(weights=[new_weights], momentum=[new_momentum])

    def _activation_derivative(self, x: mx.array) -> mx.array:
        """Compute activation function derivative for backprop in deep memory.

        Args:
            x: Pre-activation values

        Returns:
            Element-wise derivative of the activation function
        """
        if self.config.activation == "silu":
            sig = mx.sigmoid(x)
            return sig * (1.0 + x * (1.0 - sig))
        elif self.config.activation == "gelu":
            # GELU derivative: 0.5*(1 + erf(x/sqrt(2))) + x * pdf(x)
            # Compute in float32 to prevent exp overflow/underflow in bfloat16
            x_f32 = x.astype(mx.float32)
            sqrt2 = 1.4142135623730951
            cdf = 0.5 * (1.0 + mx.erf(x_f32 / sqrt2))
            pdf = mx.exp(-0.5 * x_f32 * x_f32) * 0.3989422804014327  # 1/sqrt(2*pi)
            return (cdf + x_f32 * pdf).astype(x.dtype)
        elif self.config.activation == "relu":
            return (x > 0).astype(x.dtype)
        else:
            raise ValueError(f"No derivative for activation: {self.config.activation}")

    def retrieve(
        self,
        queries: mx.array,
        state: MemoryState,
    ) -> mx.array:
        """Retrieve from memory without updating.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Memory state to query

        Returns:
            Retrieved values (batch, seq, dim)
        """
        # Project queries
        q = self.proj_q(queries)

        if self.use_conv:
            seq_len = q.shape[1]
            q = self.conv_q(q)[:, :seq_len, :]

        q = nn.silu(q)
        # L2-norm in float32 to avoid bfloat16 underflow
        q_f32 = q.astype(mx.float32)
        q = (q_f32 / mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-8)).astype(
            q.dtype
        )

        # Retrieve using explicit state weights (no module mutation)
        retrieved = self.memory.forward_with_weights(q, state.weights)
        return self.proj_out(retrieved)
