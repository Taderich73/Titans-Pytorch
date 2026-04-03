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
