# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state serialization for Titans (PyTorch Implementation).

Uses .npz format (NumPy) for cross-framework compatibility.
Memory dumps saved by the MLX version can be loaded here and vice versa.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from titans.checkpoint_types import CheckpointEntry, GateSnapshot
from titans.memory import MemoryState, TNTMemoryState

logger = logging.getLogger(__name__)

# Tensors below this Frobenius norm are considered degenerate (e.g. zeroed
# global memory caused by long-horizon decay collapse). load_memory_states
# warns when it sees them so silent inference corruption is impossible.
_DEGENERATE_NORM_THRESHOLD: float = 1e-6


def _save_memory_state(arrays: dict, prefix: str, state: MemoryState) -> None:
    """Save a single MemoryState into the arrays dict."""
    arrays[f"{prefix}_num_memory_layers"] = np.array([len(state.weights)])
    for j, w in enumerate(state.weights):
        arrays[f"{prefix}_weight_{j}"] = w.detach().cpu().numpy()
    for j, m in enumerate(state.momentum):
        arrays[f"{prefix}_momentum_{j}"] = m.detach().cpu().numpy()


def _load_memory_state(
    data: np.lib.npyio.NpzFile, prefix: str, device: torch.device
) -> MemoryState:
    """Load a single MemoryState from the npz data."""
    num_memory_layers = int(data[f"{prefix}_num_memory_layers"][0])
    weights: list[torch.Tensor] = []
    momentum: list[torch.Tensor] = []
    for j in range(num_memory_layers):
        weights.append(torch.from_numpy(data[f"{prefix}_weight_{j}"].copy()).to(device))
        momentum.append(torch.from_numpy(data[f"{prefix}_momentum_{j}"].copy()).to(device))
    return MemoryState(weights=weights, momentum=momentum)


def save_memory_states(
    states: list[MemoryState | TNTMemoryState], path: Path
) -> None:
    """Serialize memory states to a single .npz file.

    Handles both MemoryState and TNTMemoryState transparently.
    """
    arrays: dict[str, np.ndarray] = {}
    arrays["num_layers"] = np.array([len(states)])

    for i, state in enumerate(states):
        if isinstance(state, TNTMemoryState):
            arrays[f"layer_{i}_type"] = np.array([1])  # 1 = TNT
            _save_memory_state(arrays, f"layer_{i}_global", state.global_state)
            arrays[f"layer_{i}_num_locals"] = np.array([len(state.local_states)])
            for k, local_s in enumerate(state.local_states):
                _save_memory_state(arrays, f"layer_{i}_local_{k}", local_s)
            # Save local_inits
            for k, inits in enumerate(state.local_inits):
                arrays[f"layer_{i}_local_init_{k}_count"] = np.array([len(inits)])
                for j, t in enumerate(inits):
                    arrays[f"layer_{i}_local_init_{k}_{j}"] = t.detach().cpu().numpy()
            # Save qk_projections
            arrays[f"layer_{i}_num_qk"] = np.array([len(state.qk_projections)])
            for k, qk in enumerate(state.qk_projections):
                arrays[f"layer_{i}_qk_{k}"] = qk.detach().cpu().numpy()
            # Save step counters
            arrays[f"layer_{i}_step_counters"] = np.array(
                state.local_step_counters, dtype=np.int64
            )
        else:
            arrays[f"layer_{i}_type"] = np.array([0])  # 0 = plain MemoryState
            _save_memory_state(arrays, f"layer_{i}", state)

    path = Path(path)
    np.savez(path, **arrays)


def load_memory_states(
    path: Path,
    device: torch.device | None = None,
    *,
    reset_for_inference: bool = False,
) -> list[MemoryState | TNTMemoryState]:
    """Deserialize memory states from a .npz file.

    Handles both MemoryState and TNTMemoryState transparently.
    Also loads legacy files that lack the layer_type marker.

    Args:
        path: Path to the .npz file produced by save_memory_states.
        device: Torch device for the loaded tensors. Defaults to CPU.
        reset_for_inference: When True, zero ``local_step_counters`` and
            ``qk_projections`` for any TNTMemoryState before returning. Use this
            only for inference warm-start. Default is False so training resume
            callers get exact state continuity. The previous default silently
            corrupted TNT training state — see the reliability plan for
            background.

    Logs a warning for any loaded tensor whose Frobenius norm falls below
    ``_DEGENERATE_NORM_THRESHOLD``. The most common cause of this is the
    long-horizon decay-to-zero pathology in TNT global memory when state is
    threaded across many training batches without periodic reset.
    """
    if device is None:
        device = torch.device("cpu")

    path = Path(path)
    if not path.exists():
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Memory state file not found: {path}")
        path = path.with_suffix(".npz")

    data = np.load(str(path))

    if "num_layers" not in data:
        raise ValueError("Invalid memory state file: missing 'num_layers' metadata")

    num_layers = int(data["num_layers"][0])
    states: list[MemoryState | TNTMemoryState] = []

    for i in range(num_layers):
        # Check type marker (default 0 for backwards compat with old files)
        type_key = f"layer_{i}_type"
        layer_type = int(data[type_key][0]) if type_key in data else 0

        if layer_type == 1:
            # TNTMemoryState
            global_state = _load_memory_state(data, f"layer_{i}_global", device)
            num_locals = int(data[f"layer_{i}_num_locals"][0])
            local_states = [
                _load_memory_state(data, f"layer_{i}_local_{k}", device)
                for k in range(num_locals)
            ]
            # Load local_inits
            local_inits: list[list[torch.Tensor]] = []
            for k in range(num_locals):
                count = int(data[f"layer_{i}_local_init_{k}_count"][0])
                inits = [
                    torch.from_numpy(data[f"layer_{i}_local_init_{k}_{j}"].copy()).to(device)
                    for j in range(count)
                ]
                local_inits.append(inits)
            # Load qk_projections
            num_qk = int(data[f"layer_{i}_num_qk"][0])
            qk_projections = [
                torch.from_numpy(data[f"layer_{i}_qk_{k}"].copy()).to(device)
                for k in range(num_qk)
            ]
            # Load step counters
            step_counters = data[f"layer_{i}_step_counters"].tolist()
            states.append(
                TNTMemoryState(
                    global_state=global_state,
                    local_states=local_states,
                    local_inits=local_inits,
                    qk_projections=qk_projections,
                    local_step_counters=step_counters,
                )
            )
        else:
            # Legacy / plain MemoryState
            # Support both new prefix format and legacy format
            prefix = f"layer_{i}"
            if f"{prefix}_num_memory_layers" in data:
                states.append(_load_memory_state(data, prefix, device))
            else:
                # Legacy format: num_memory_layers_{i}
                num_memory_layers = int(data[f"num_memory_layers_{i}"][0])
                weights = [
                    torch.from_numpy(data[f"layer_{i}_weight_{j}"].copy()).to(device)
                    for j in range(num_memory_layers)
                ]
                momentum = [
                    torch.from_numpy(data[f"layer_{i}_momentum_{j}"].copy()).to(device)
                    for j in range(num_memory_layers)
                ]
                states.append(MemoryState(weights=weights, momentum=momentum))

    if reset_for_inference:
        for s in states:
            if isinstance(s, TNTMemoryState):
                s.local_step_counters = [0] * len(s.local_states)
                s.qk_projections = [torch.zeros_like(qk) for qk in s.qk_projections]

    _warn_on_degenerate_states(states, source=str(path))

    return states


def _warn_on_degenerate_states(
    states: list[MemoryState | TNTMemoryState],
    source: str,
) -> None:
    """Log a warning for any state whose weight tensors have collapsed to zero.

    The most common cause is the long-horizon decay-to-zero pathology in TNT
    global memory when memory state is threaded across many training batches
    without periodic reset. Loading such a state into inference produces
    degenerate retrieval (zero output from the global branch) and is strictly
    worse than starting from a fresh init_state. We warn loudly here so the
    failure mode is visible at load time rather than silently corrupting
    generation downstream.
    """
    bad: list[str] = []
    for layer_idx, s in enumerate(states):
        if isinstance(s, TNTMemoryState):
            for j, w in enumerate(s.global_state.weights):
                if w.float().norm().item() < _DEGENERATE_NORM_THRESHOLD:
                    bad.append(f"layer {layer_idx} global_weight[{j}]")
            for k, local in enumerate(s.local_states):
                for j, w in enumerate(local.weights):
                    if w.float().norm().item() < _DEGENERATE_NORM_THRESHOLD:
                        bad.append(f"layer {layer_idx} local[{k}].weight[{j}]")
        else:
            for j, w in enumerate(s.weights):
                if w.float().norm().item() < _DEGENERATE_NORM_THRESHOLD:
                    bad.append(f"layer {layer_idx} weight[{j}]")

    if bad:
        preview = ", ".join(bad[:5])
        more = f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""
        logger.warning(
            "Loaded memory state from %s contains %d near-zero weight tensor(s): "
            "%s%s. This usually indicates global-memory decay collapse from "
            "long-horizon training. Loading this state into inference may "
            "produce degenerate output — consider running without "
            "--memory-state to use the model's learned init weights instead.",
            source, len(bad), preview, more,
        )


# ---------------------------------------------------------------------------
# CheckpointEntry serialization
# ---------------------------------------------------------------------------


def save_checkpoint_entry(entry: CheckpointEntry, path: Path) -> None:
    """Serialize a CheckpointEntry to a .npz file.

    The memory state portion uses the same key format as
    :func:`save_memory_states` so that :func:`load_memory_states` can still
    read the resulting file (backward compatibility). Gate and metadata keys
    use ``gates_`` and ``meta_`` namespace prefixes respectively.

    Args:
        entry: The CheckpointEntry to serialise.
        path: Destination path. A ``.npz`` extension is added if absent.
    """
    arrays: dict[str, np.ndarray] = {}

    # --- State portion (mirrors save_memory_states for a single state) ---
    state = entry.state
    arrays["num_layers"] = np.array([1])
    if isinstance(state, TNTMemoryState):
        arrays["layer_0_type"] = np.array([1])  # 1 = TNT
        _save_memory_state(arrays, "layer_0_global", state.global_state)
        arrays["layer_0_num_locals"] = np.array([len(state.local_states)])
        for k, local_s in enumerate(state.local_states):
            _save_memory_state(arrays, f"layer_0_local_{k}", local_s)
        for k, inits in enumerate(state.local_inits):
            arrays[f"layer_0_local_init_{k}_count"] = np.array([len(inits)])
            for j, t in enumerate(inits):
                arrays[f"layer_0_local_init_{k}_{j}"] = t.detach().cpu().numpy()
        arrays["layer_0_num_qk"] = np.array([len(state.qk_projections)])
        for k, qk in enumerate(state.qk_projections):
            arrays[f"layer_0_qk_{k}"] = qk.detach().cpu().numpy()
        arrays["layer_0_step_counters"] = np.array(
            state.local_step_counters, dtype=np.int64
        )
    else:
        arrays["layer_0_type"] = np.array([0])  # 0 = plain MemoryState
        _save_memory_state(arrays, "layer_0", state)

    # --- Gate portion ---
    if entry.gates is not None:
        gates = entry.gates
        arrays["gates_has_data"] = np.array([1], dtype=np.uint8)
        num_gate_layers = len(gates.alpha)
        arrays["gates_num_layers"] = np.array([num_gate_layers])
        for k in range(num_gate_layers):
            arrays[f"gates_layer_{k}_alpha_0"] = gates.alpha[k].detach().cpu().numpy()
            arrays[f"gates_layer_{k}_theta_0"] = gates.theta[k].detach().cpu().numpy()
            arrays[f"gates_layer_{k}_eta_0"] = gates.eta[k].detach().cpu().numpy()
        arrays["gates_has_delta"] = np.array(
            [1 if gates.delta is not None else 0], dtype=np.uint8
        )
        if gates.delta is not None:
            for k in range(num_gate_layers):
                arrays[f"gates_layer_{k}_delta_0"] = (
                    gates.delta[k].detach().cpu().numpy()
                )
        arrays["gates_input_activation_norm"] = np.array(
            [gates.input_activation_norm], dtype=np.float32
        )
        arrays["gates_chunk_index"] = np.array([gates.chunk_index], dtype=np.int64)
    else:
        arrays["gates_has_data"] = np.array([0], dtype=np.uint8)

    # --- Metadata portion ---
    arrays["meta_trigger_phase"] = np.frombuffer(
        entry.trigger_phase.encode("utf-8"), dtype=np.uint8
    )
    arrays["meta_config_hash"] = np.frombuffer(
        entry.config_hash.encode("utf-8"), dtype=np.uint8
    )
    arrays["meta_weight_norms"] = np.array(entry.weight_norms, dtype=np.float32)
    arrays["meta_momentum_norms"] = np.array(entry.momentum_norms, dtype=np.float32)

    np.savez(Path(path), **arrays)


def load_checkpoint_entry(
    path: Path,
    device: torch.device | None = None,
) -> CheckpointEntry:
    """Deserialize a CheckpointEntry from a .npz file.

    Uses :func:`load_memory_states` for the state portion and then reads the
    ``gates_`` and ``meta_`` namespaced keys.

    Args:
        path: Path to the .npz file produced by :func:`save_checkpoint_entry`.
        device: Torch device for loaded tensors. Defaults to CPU.

    Returns:
        The reconstructed :class:`~titans.checkpoint_types.CheckpointEntry`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if device is None:
        device = torch.device("cpu")

    states = load_memory_states(path, device=device, reset_for_inference=False)
    if len(states) != 1:
        raise ValueError(
            f"Expected exactly 1 state in checkpoint entry, got {len(states)}"
        )
    state = states[0]

    path = Path(path)
    if not path.exists():
        path = path.with_suffix(".npz")
    data = np.load(str(path))

    # --- Gate portion ---
    gates: GateSnapshot | None = None
    has_data = int(data["gates_has_data"][0]) if "gates_has_data" in data else 0
    if has_data:
        num_gate_layers = int(data["gates_num_layers"][0])
        alpha = [
            torch.from_numpy(data[f"gates_layer_{k}_alpha_0"].copy()).to(device)
            for k in range(num_gate_layers)
        ]
        theta = [
            torch.from_numpy(data[f"gates_layer_{k}_theta_0"].copy()).to(device)
            for k in range(num_gate_layers)
        ]
        eta = [
            torch.from_numpy(data[f"gates_layer_{k}_eta_0"].copy()).to(device)
            for k in range(num_gate_layers)
        ]
        has_delta = int(data["gates_has_delta"][0])
        delta: list[torch.Tensor] | None = None
        if has_delta:
            delta = [
                torch.from_numpy(data[f"gates_layer_{k}_delta_0"].copy()).to(device)
                for k in range(num_gate_layers)
            ]
        input_activation_norm = float(data["gates_input_activation_norm"][0])
        chunk_index = int(data["gates_chunk_index"][0])
        gates = GateSnapshot(
            alpha=alpha,
            theta=theta,
            eta=eta,
            delta=delta,
            input_activation_norm=input_activation_norm,
            chunk_index=chunk_index,
        )

    # --- Metadata portion ---
    trigger_phase = bytes(data["meta_trigger_phase"]).decode("utf-8")
    config_hash = bytes(data["meta_config_hash"]).decode("utf-8")
    weight_norms = data["meta_weight_norms"].tolist()
    momentum_norms = data["meta_momentum_norms"].tolist()

    return CheckpointEntry(
        state=state,
        gates=gates,
        metadata={},
        trigger_phase=trigger_phase,
        weight_norms=weight_norms,
        momentum_norms=momentum_norms,
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------------
# MemoryDumpManager
# ---------------------------------------------------------------------------

import copy
import json
import shutil
from datetime import UTC, datetime


class MemoryDumpManager:
    """Manages memory state dumps with auto-dump triggers and retention policy.

    Wraps save_memory_states / load_memory_states with timestamped dump
    directories, per-layer inspection, state diffing, merging, and forking.

    Args:
        dump_dir: Root directory where dump subdirectories are stored.
        keep_last_n: Maximum number of dumps to retain. Older dumps are pruned
            automatically after each save.

    Example::

        manager = MemoryDumpManager("./memory_dumps", keep_last_n=5)
        path = manager.save(states, tag="step_1000")
        loaded = manager.load_latest()
        info = manager.inspect(states)
    """

    def __init__(self, dump_dir: str | Path, keep_last_n: int = 10) -> None:
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, states: list[MemoryState], tag: str = "auto") -> Path:
        """Save memory states to a timestamped dump directory.

        Args:
            states: List of MemoryState objects (one per memory-carrying block).
            tag: Human-readable label appended to the dump directory name.

        Returns:
            Path to the new dump directory.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        dump_dir = self.dump_dir / f"dump_{timestamp}_{safe_tag}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        state_file = dump_dir / "state.npz"
        save_memory_states(states, state_file)

        # Compute per-layer stats for metadata
        per_layer_stats: dict[str, dict] = {}
        for i, state in enumerate(states):
            weights, momentum = self._extract_tensors(state)
            w_norms = [float(w.float().norm().item()) for w in weights]
            m_norms = [float(m.float().norm().item()) for m in momentum]
            per_layer_stats[str(i)] = {
                "weight_norm": sum(w_norms) / len(w_norms) if w_norms else 0.0,
                "momentum_norm": sum(m_norms) / len(m_norms) if m_norms else 0.0,
                "num_memory_layers": len(weights),
            }

        metadata = {
            "version": "1.0",
            "tag": tag,
            "created_at": datetime.now(UTC).isoformat(),
            "num_layers": len(states),
            "per_layer_stats": per_layer_stats,
        }
        (dump_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        self._enforce_retention()
        return dump_dir

    def load_latest(self, device: torch.device | None = None) -> list[MemoryState] | None:
        """Load memory states from the most recent dump.

        Args:
            device: Torch device to load tensors onto. Defaults to CPU.

        Returns:
            List of MemoryState objects, or None if no dumps exist.
        """
        dumps = self.list_dumps()
        if not dumps:
            return None
        return load_memory_states(dumps[-1] / "state.npz", device=device)

    def list_dumps(self) -> list[Path]:
        """Return all dump directories sorted chronologically (oldest first).

        Returns:
            Sorted list of Path objects pointing to dump directories.
        """
        if not self.dump_dir.exists():
            return []
        dirs = [d for d in self.dump_dir.iterdir() if d.is_dir() and d.name.startswith("dump_")]
        return sorted(dirs, key=lambda d: d.name)

    def inspect(self, states: list[MemoryState]) -> dict:
        """Compute per-layer weight and momentum norms for live states.

        Args:
            states: List of MemoryState objects to inspect.

        Returns:
            Dict with 'num_layers' and 'layers' (list of per-layer dicts
            containing weight_norm, momentum_norm, and per-sublayer norms).
        """
        layers_info: list[dict] = []
        for i, state in enumerate(states):
            weights, momentum = self._extract_tensors(state)
            w_norms = [float(w.float().norm().item()) for w in weights]
            m_norms = [float(m.float().norm().item()) for m in momentum]
            layers_info.append({
                "layer_idx": i,
                "num_memory_layers": len(weights),
                "weight_norm_mean": sum(w_norms) / len(w_norms) if w_norms else 0.0,
                "momentum_norm_mean": sum(m_norms) / len(m_norms) if m_norms else 0.0,
                "weight_norms_per_sublayer": w_norms,
                "momentum_norms_per_sublayer": m_norms,
            })
        return {
            "num_layers": len(states),
            "layers": layers_info,
        }

    def diff(self, states_a: list[MemoryState], states_b: list[MemoryState]) -> dict:
        """Compute Frobenius distance between two sets of memory states.

        Args:
            states_a: First list of MemoryState objects.
            states_b: Second list of MemoryState objects (must match length).

        Returns:
            Dict with 'per_layer' mapping layer index to Frobenius distance
            and 'total_distance'.

        Raises:
            ValueError: If states_a and states_b have different lengths.
        """
        if len(states_a) != len(states_b):
            raise ValueError(
                f"State lists have different lengths: {len(states_a)} vs {len(states_b)}"
            )

        per_layer: dict[str, dict] = {}
        total_distance = 0.0

        for i, (sa, sb) in enumerate(zip(states_a, states_b)):
            weights_a, _ = self._extract_tensors(sa)
            weights_b, _ = self._extract_tensors(sb)

            if len(weights_a) != len(weights_b):
                raise ValueError(
                    f"Layer {i} has mismatched num_memory_layers: "
                    f"{len(weights_a)} vs {len(weights_b)}"
                )

            layer_dist = 0.0
            for wa, wb in zip(weights_a, weights_b):
                delta = (wa.float() - wb.float())
                layer_dist += float(torch.norm(delta, p="fro").item())

            avg_dist = layer_dist / max(len(weights_a), 1)
            per_layer[str(i)] = {"frobenius_distance": avg_dist}
            total_distance += avg_dist

        return {
            "per_layer": per_layer,
            "total_distance": total_distance,
            "num_layers": len(states_a),
        }

    def merge(
        self,
        states_list: list[list[MemoryState]],
        strategy: str = "weighted_mean",
        weights: list[float] | None = None,
    ) -> list[MemoryState]:
        """Combine multiple memory state sets into one.

        Args:
            states_list: List of state lists to merge. All lists must have the
                same structure (num_layers and num_memory_layers).
            strategy: Merge strategy. Supported values:
                - 'weighted_mean': Weighted average of tensors. If *weights*
                  is None, uses uniform weights.
                - 'max_norm': For each sublayer position, keep the tensor from
                  whichever state has the highest weight norm.
                - 'recency': Linearly increasing weights so more recent states
                  (later in states_list) are weighted higher.
            weights: Per-state blending weights for 'weighted_mean'. Must sum
                to 1.0 (or will be normalised). Ignored for other strategies.

        Returns:
            Merged list of MemoryState objects.

        Raises:
            ValueError: If an unknown strategy is given or state structures differ.
        """
        if not states_list:
            raise ValueError("states_list is empty.")

        n = len(states_list)
        num_layers = len(states_list[0])

        if strategy == "weighted_mean":
            if weights is None:
                blend = [1.0 / n] * n
            else:
                total = sum(weights)
                blend = [w / total for w in weights]
            return self._merge_weighted(states_list, blend, num_layers)

        elif strategy == "recency":
            # Linearly increasing: oldest weight = 1, newest weight = n
            raw = list(range(1, n + 1))
            total = sum(raw)
            blend = [r / total for r in raw]
            return self._merge_weighted(states_list, blend, num_layers)

        elif strategy == "max_norm":
            return self._merge_max_norm(states_list, num_layers)

        else:
            raise ValueError(
                f"Unknown merge strategy: {strategy!r}. "
                "Choose from 'weighted_mean', 'max_norm', 'recency'."
            )

    def fork(self, states: list[MemoryState]) -> list[MemoryState]:
        """Deep-copy memory states without modifying the originals.

        This is a pure in-memory operation (no disk I/O).

        Args:
            states: List of MemoryState objects to copy.

        Returns:
            New list of MemoryState objects with cloned tensors.
        """
        return [self._clone_state(s) for s in states]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enforce_retention(self) -> None:
        """Delete oldest dumps beyond keep_last_n."""
        dumps = self.list_dumps()
        while len(dumps) > self.keep_last_n:
            oldest = dumps.pop(0)
            shutil.rmtree(oldest)

    @staticmethod
    def _extract_tensors(
        state: MemoryState,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Unpack weight and momentum tensors from a MemoryState or TNTMemoryState.

        Args:
            state: A MemoryState or TNTMemoryState instance.

        Returns:
            Tuple of (weights, momentum) lists.
        """
        if hasattr(state, "global_state"):
            return state.global_state.weights, state.global_state.momentum
        return state.weights, state.momentum

    @staticmethod
    def _clone_state(state: MemoryState) -> MemoryState:
        """Deep-clone a single MemoryState (or TNTMemoryState).

        Args:
            state: State to clone.

        Returns:
            A new MemoryState with cloned tensors.
        """
        if hasattr(state, "global_state"):
            # TNTMemoryState — use copy.deepcopy for full fidelity
            return copy.deepcopy(state)
        return MemoryState(
            weights=[w.detach().clone() for w in state.weights],
            momentum=[m.detach().clone() for m in state.momentum],
        )

    @staticmethod
    def _merge_weighted(
        states_list: list[list[MemoryState]],
        blend: list[float],
        num_layers: int,
    ) -> list[MemoryState]:
        """Weighted average merge across all state sets.

        Args:
            states_list: List of state lists.
            blend: Per-state blend weights (must sum to 1.0).
            num_layers: Number of layers expected per state list.

        Returns:
            Merged list of MemoryState objects.
        """
        merged: list[MemoryState] = []

        for layer_idx in range(num_layers):
            # Collect all (weight, momentum) tensors at this layer
            all_weights: list[list[torch.Tensor]] = []
            all_momentum: list[list[torch.Tensor]] = []

            for states in states_list:
                if hasattr(states[layer_idx], "global_state"):
                    all_weights.append(states[layer_idx].global_state.weights)
                    all_momentum.append(states[layer_idx].global_state.momentum)
                else:
                    all_weights.append(states[layer_idx].weights)
                    all_momentum.append(states[layer_idx].momentum)

            num_sub = len(all_weights[0])
            blended_w: list[torch.Tensor] = []
            blended_m: list[torch.Tensor] = []

            for sub_idx in range(num_sub):
                w_blend = sum(
                    b * all_weights[d][sub_idx].float()
                    for d, b in enumerate(blend)
                )
                m_blend = sum(
                    b * all_momentum[d][sub_idx].float()
                    for d, b in enumerate(blend)
                )
                blended_w.append(w_blend)
                blended_m.append(m_blend)

            merged.append(MemoryState(weights=blended_w, momentum=blended_m))

        return merged

    @staticmethod
    def _merge_max_norm(
        states_list: list[list[MemoryState]],
        num_layers: int,
    ) -> list[MemoryState]:
        """Select the state with the highest weight norm per sublayer.

        Args:
            states_list: List of state lists.
            num_layers: Number of layers expected per state list.

        Returns:
            Merged list of MemoryState objects.
        """
        merged: list[MemoryState] = []

        for layer_idx in range(num_layers):
            all_weights: list[list[torch.Tensor]] = []
            all_momentum: list[list[torch.Tensor]] = []

            for states in states_list:
                if hasattr(states[layer_idx], "global_state"):
                    all_weights.append(states[layer_idx].global_state.weights)
                    all_momentum.append(states[layer_idx].global_state.momentum)
                else:
                    all_weights.append(states[layer_idx].weights)
                    all_momentum.append(states[layer_idx].momentum)

            num_sub = len(all_weights[0])
            selected_w: list[torch.Tensor] = []
            selected_m: list[torch.Tensor] = []

            for sub_idx in range(num_sub):
                # Pick the state whose weight has the highest Frobenius norm
                best_d = max(
                    range(len(states_list)),
                    key=lambda d: float(all_weights[d][sub_idx].float().norm().item()),
                )
                selected_w.append(all_weights[best_d][sub_idx].clone())
                selected_m.append(all_momentum[best_d][sub_idx].clone())

            merged.append(MemoryState(weights=selected_w, momentum=selected_m))

        return merged
