# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state serialization for Titans (PyTorch Implementation).

Uses .npz format (NumPy) for cross-framework compatibility.
Memory dumps saved by the MLX version can be loaded here and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from titans.memory import MemoryState


def save_memory_states(states: list[MemoryState], path: Path) -> None:
    """Serialize memory states to a single .npz file."""
    arrays: dict[str, np.ndarray] = {}
    arrays["num_layers"] = np.array([len(states)])

    for i, state in enumerate(states):
        arrays[f"num_memory_layers_{i}"] = np.array([len(state.weights)])
        for j, w in enumerate(state.weights):
            arrays[f"layer_{i}_weight_{j}"] = w.detach().cpu().numpy()
        for j, m in enumerate(state.momentum):
            arrays[f"layer_{i}_momentum_{j}"] = m.detach().cpu().numpy()

    path = Path(path)
    np.savez(path, **arrays)


def load_memory_states(
    path: Path, device: torch.device | None = None
) -> list[MemoryState]:
    """Deserialize memory states from a .npz file."""
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
    states: list[MemoryState] = []

    for i in range(num_layers):
        key = f"num_memory_layers_{i}"
        if key not in data:
            raise ValueError(f"Invalid memory state file: missing '{key}'")
        num_memory_layers = int(data[key][0])

        weights: list[torch.Tensor] = []
        momentum: list[torch.Tensor] = []
        for j in range(num_memory_layers):
            wk = f"layer_{i}_weight_{j}"
            mk = f"layer_{i}_momentum_{j}"
            if wk not in data:
                raise ValueError(f"Invalid memory state file: missing '{wk}'")
            if mk not in data:
                raise ValueError(f"Invalid memory state file: missing '{mk}'")
            weights.append(torch.from_numpy(data[wk].copy()).to(device))
            momentum.append(torch.from_numpy(data[mk].copy()).to(device))

        states.append(MemoryState(weights=weights, momentum=momentum))

    return states


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
