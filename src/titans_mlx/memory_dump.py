# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state persistence for Titans (MLX Implementation).

Provides dump/load/inspect/diff/merge/fork operations for NeuralLTM
memory states, enabling inference-time continual learning across sessions.

Wraps existing save_memory_states/load_memory_states for core I/O,
adding metadata, inspection, diffing, merging, and lifecycle management.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from titans_mlx.config import TitansConfig
from titans_mlx.memory import (
    MemoryState,
    TNTMemoryState,
    save_memory_states,
    load_memory_states,
    save_tnt_memory_states,
    load_tnt_memory_states,
)


class MemoryDumpManager:
    """Serialization and inspection for NeuralLTM memory state."""

    def __init__(self, config: TitansConfig) -> None:
        self.config = config
        self.dump_path = Path(config.mca_dump_path)

    def _is_tnt(self, states: list) -> bool:
        return len(states) > 0 and isinstance(states[0], TNTMemoryState)

    def dump(
        self,
        states: list,
        step_count: int = 0,
        description: str | None = None,
    ) -> Path:
        """Serialize all layer memory states to disk."""
        self.dump_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        dump_dir = self.dump_path / f"dump_{timestamp}"
        dump_dir.mkdir()

        # Save state using existing infrastructure
        state_file = dump_dir / "state.npz"
        is_tnt = self._is_tnt(states)
        if is_tnt:
            save_tnt_memory_states(states, state_file)
        else:
            save_memory_states(states, state_file)

        # Compute per-layer stats
        per_layer_stats = {}
        for i, state in enumerate(states):
            weights = state.global_state.weights if is_tnt else state.weights
            momentum = state.global_state.momentum if is_tnt else state.momentum
            w_norms = [float(mx.sqrt(mx.sum(w ** 2)).item()) for w in weights]
            m_norms = [float(mx.sqrt(mx.sum(m ** 2)).item()) for m in momentum]
            per_layer_stats[str(i)] = {
                "weight_norm": sum(w_norms) / len(w_norms),
                "momentum_norm": sum(m_norms) / len(m_norms),
            }

        # Write metadata
        metadata = {
            "version": "1.0",
            "model_dim": self.config.dim,
            "num_layers": len(states),
            "num_memory_layers": self.config.num_memory_layers,
            "memory_hidden_dim": self.config.memory_hidden_dim,
            "use_tnt": is_tnt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step_count": step_count,
            "description": description,
            "per_layer_stats": per_layer_stats,
        }
        (dump_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        self._prune()
        return dump_dir

    def load(self, path: str | Path, strict: bool = True) -> list:
        """Restore memory states from dump."""
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())

        if strict:
            if meta["model_dim"] != self.config.dim:
                raise ValueError(
                    f"dimension mismatch: dump has dim={meta['model_dim']}, "
                    f"config has dim={self.config.dim}"
                )

        state_file = path / "state.npz"
        if meta.get("use_tnt", False):
            return load_tnt_memory_states(state_file)
        return load_memory_states(state_file)

    def inspect(self, path: str | Path) -> dict:
        """Human-readable summary of memory state."""
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        return {
            "metadata": meta,
            "per_layer_stats": meta.get("per_layer_stats", {}),
        }

    def diff(self, path_a: str | Path, path_b: str | Path) -> dict:
        """Weight-level diff between two dumps."""
        states_a = self.load(path_a, strict=False)
        states_b = self.load(path_b, strict=False)

        per_layer = {}
        for i, (sa, sb) in enumerate(zip(states_a, states_b)):
            weights_a = sa.global_state.weights if isinstance(sa, TNTMemoryState) else sa.weights
            weights_b = sb.global_state.weights if isinstance(sb, TNTMemoryState) else sb.weights

            total_dist = 0.0
            for wa, wb in zip(weights_a, weights_b):
                diff_val = np.array(wa) - np.array(wb)
                total_dist += float(np.sqrt(np.sum(diff_val ** 2)))

            per_layer[str(i)] = {
                "frobenius_distance": total_dist / len(weights_a),
            }

        return {"per_layer": per_layer}

    def merge(
        self,
        paths: list[str | Path],
        strategy: str = "weighted_mean",
    ) -> list:
        """Combine multiple dumps."""
        all_states = []
        all_steps = []
        for p in paths:
            p = Path(p)
            meta = json.loads((p / "metadata.json").read_text())
            all_steps.append(meta.get("step_count", 1))
            all_states.append(self.load(p, strict=False))

        if strategy == "weighted_mean":
            total_steps = sum(all_steps)
            if total_steps == 0:
                weights_per_dump = [1.0 / len(paths)] * len(paths)
            else:
                weights_per_dump = [s / total_steps for s in all_steps]

            num_layers = len(all_states[0])
            merged = []
            for layer_idx in range(num_layers):
                layer_weights = []
                layer_momentum = []

                for w_idx in range(len(all_states[0][layer_idx].weights)):
                    blended_w = sum(
                        wt * mx.array(np.array(all_states[d][layer_idx].weights[w_idx]))
                        for d, wt in enumerate(weights_per_dump)
                    )
                    blended_m = sum(
                        wt * mx.array(np.array(all_states[d][layer_idx].momentum[w_idx]))
                        for d, wt in enumerate(weights_per_dump)
                    )
                    layer_weights.append(blended_w)
                    layer_momentum.append(blended_m)

                merged.append(MemoryState(weights=layer_weights, momentum=layer_momentum))
            return merged

        raise ValueError(f"Unknown merge strategy: {strategy}")

    def reset(self, states: list, layers: list[int] | None = None) -> list:
        """Reset memory weights and momentum to zeros."""
        result = []
        for i, state in enumerate(states):
            if layers is not None and i not in layers:
                result.append(state)
                continue
            result.append(MemoryState(
                weights=[mx.zeros_like(w) for w in state.weights],
                momentum=[mx.zeros_like(m) for m in state.momentum],
            ))
        return result

    def fork(self, states: list, description: str | None = None) -> Path:
        """Snapshot current state without altering live state."""
        return self.dump(states, description=description or "fork")

    def _prune(self) -> None:
        """Remove old dumps beyond keep_last_n."""
        if not self.dump_path.exists():
            return
        dumps = sorted(
            [d for d in self.dump_path.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        keep = self.config.mca_dump_keep_last_n
        while len(dumps) > keep:
            shutil.rmtree(dumps.pop(0))
