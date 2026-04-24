# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state serialization for Titans (PyTorch Implementation).

Uses .npz format (NumPy) for cross-framework compatibility.
Memory dumps saved by the MLX version can be loaded here and vice versa.

Schema versioning (Task P5)
---------------------------
Starting with 0.7.0 every .npz written by :func:`save_memory_states`
carries a top-level ``titans_schema_version`` entry. See
``MIGRATIONS.md`` at the repo root for the change log.

The load path dispatches on the version:

* **missing** — unversioned (pre-0.7) file; we warn once with
  :class:`DeprecationWarning` and best-effort load the legacy layout.
* **equal** to :data:`titans.TITANS_SCHEMA_VERSION` — load normally.
* **newer** than the code's current schema — raise
  :class:`RuntimeError`; the user needs to upgrade ``titans``.
* **older** with a migration entry in :data:`_MIGRATIONS` — migrate the
  array dict in-memory, then load.
* **older** without a migration — raise :class:`RuntimeError` with a
  clear "no migration available" message.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from titans.memory import MemoryState, TNTMemoryState

logger = logging.getLogger(__name__)

# Tensors below this Frobenius norm are considered degenerate (e.g. zeroed
# global memory caused by long-horizon decay collapse). load_memory_states
# warns when it sees them so silent inference corruption is impossible.
_DEGENERATE_NORM_THRESHOLD: float = 1e-6

# Top-level npz key used to carry the schema version. Kept module-private
# because callers should never read it directly — use the version metadata
# returned by the load path instead.
_SCHEMA_VERSION_KEY: str = "titans_schema_version"


# ---------------------------------------------------------------------------
# Migration protocol (Task P5)
# ---------------------------------------------------------------------------
# ``_MIGRATIONS`` maps ``(from_version, to_version)`` -> a function that
# takes a *mutable* dict of numpy arrays keyed by npz entry name and
# returns a new dict in the ``to_version`` layout. The dispatcher in
# :func:`_migrate_arrays_to_current` composes a chain of migrations when
# the checkpoint is more than one version behind.
#
# For the very first versioned release (schema 1) there are no migrations
# registered — unversioned files take a separate legacy codepath. When
# schema 2 lands, register ``(1, 2)`` here AND add a row to
# ``MIGRATIONS.md``.
#
# The walker lives in :mod:`titans._schema_migrations` and is shared
# with :mod:`titans.checkpoint` so the two dispatchers stay symmetric.
_MIGRATIONS: dict[
    tuple[int, int], Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
] = {}


def _migrate_arrays_to_current(
    arrays: dict[str, np.ndarray],
    from_version: int,
    current_version: int,
) -> dict[str, np.ndarray]:
    """Apply registered migrations to bring ``arrays`` up to ``current_version``.

    Thin wrapper around :func:`titans._schema_migrations.walk_migrations`
    that pins the registry to ``_MIGRATIONS`` and tags the error messages
    with ``kind="memory_dump"``. Kept as a named entry point so existing
    tests and external callers that import the symbol keep working.
    """
    from titans._schema_migrations import walk_migrations

    return walk_migrations(
        arrays,
        from_version=from_version,
        to_version=current_version,
        migrations=_MIGRATIONS,
        kind="memory_dump",
    )


def _save_memory_state(arrays: dict, prefix: str, state: MemoryState) -> None:
    """Save a single MemoryState into the arrays dict."""
    arrays[f"{prefix}_num_memory_layers"] = np.array([len(state.weights)])
    for j, w in enumerate(state.weights):
        arrays[f"{prefix}_weight_{j}"] = w.detach().cpu().numpy()
    for j, m in enumerate(state.momentum):
        arrays[f"{prefix}_momentum_{j}"] = m.detach().cpu().numpy()


def _load_memory_state(
    data: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    prefix: str,
    device: torch.device,
) -> MemoryState:
    """Load a single MemoryState from the npz data.

    Accepts either a raw :class:`numpy.lib.npyio.NpzFile` or a plain
    ``dict`` keyed by npz entry names — the latter is what the version
    migration path returns after rewriting keys in-memory.
    """
    num_memory_layers = int(data[f"{prefix}_num_memory_layers"][0])
    weights: list[torch.Tensor] = []
    momentum: list[torch.Tensor] = []
    for j in range(num_memory_layers):
        weights.append(torch.from_numpy(data[f"{prefix}_weight_{j}"].copy()).to(device))
        momentum.append(
            torch.from_numpy(data[f"{prefix}_momentum_{j}"].copy()).to(device)
        )
    return MemoryState(weights=weights, momentum=momentum)


def save_memory_states(states: list[MemoryState | TNTMemoryState], path: Path) -> None:
    """Serialize memory states to a single .npz file.

    Handles both :class:`MemoryState` and :class:`TNTMemoryState`
    transparently. The resulting file carries a top-level
    ``titans_schema_version`` scalar array set to
    :data:`titans.TITANS_SCHEMA_VERSION` at write time. See the module
    docstring for the migration protocol.
    """
    # Local import to avoid a circular import at module load time
    # (``titans/__init__.py`` imports ``memory_dump`` during package init).
    from titans import TITANS_SCHEMA_VERSION

    arrays: dict[str, np.ndarray] = {}
    arrays[_SCHEMA_VERSION_KEY] = np.array([TITANS_SCHEMA_VERSION], dtype=np.int64)
    arrays["num_layers"] = np.array([len(states)])

    for i, state in enumerate(states):
        if isinstance(state, TNTMemoryState):
            arrays[f"layer_{i}_type"] = np.array([1])  # 1 = TNT
            _save_memory_state(arrays, f"layer_{i}_global", state.global_state)
            arrays[f"layer_{i}_num_locals"] = np.array([len(state.local_states)])
            for k, local_s in enumerate(state.local_states):
                _save_memory_state(arrays, f"layer_{i}_local_{k}", local_s)
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
    # numpy's savez stub types the first kw-slot as ``allow_pickle: bool`` which
    # clashes with the ``**arrays`` spread (each value is an ndarray); suppress
    # since the call is the documented public API for saving many named arrays.
    np.savez(path, **arrays)  # type: ignore[arg-type]


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
            ``qk_projections`` on every returned ``TNTMemoryState``. Use this
            only for inference warm-start where a clean starting point is
            desired. Default is False so training-resume callers get exact
            state continuity — the QK carry and shard-boundary counters that
            make TNT's per-token reset cadence correct are preserved across
            checkpoints.

    Logs a warning for any loaded tensor whose Frobenius norm falls below
    ``_DEGENERATE_NORM_THRESHOLD``. The most common cause of this is the
    long-horizon decay-to-zero pathology in TNT global memory when state is
    threaded across many training batches without periodic reset.
    """
    # Local import to avoid circular import: titans/__init__.py imports
    # this module eagerly.
    from titans import TITANS_SCHEMA_VERSION

    if device is None:
        device = torch.device("cpu")

    path = Path(path)
    if not path.exists():
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Memory state file not found: {path}")
        path = path.with_suffix(".npz")

    # Read the full npz into an in-memory dict so migrations (which may
    # rewrite keys) can operate on it without going back to disk.
    with np.load(str(path)) as npz:
        data: dict[str, np.ndarray] = {k: npz[k] for k in npz.files}

    # -------------------- Schema version dispatch (P5) --------------------
    if _SCHEMA_VERSION_KEY in data:
        file_version = int(data[_SCHEMA_VERSION_KEY][0])
        if file_version > TITANS_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint schema {file_version} > code schema "
                f"{TITANS_SCHEMA_VERSION}; upgrade titans. See "
                "MIGRATIONS.md for the per-version change log."
            )
        if file_version < TITANS_SCHEMA_VERSION:
            # _migrate_arrays_to_current raises a clear RuntimeError when
            # no migration path is registered.
            data = _migrate_arrays_to_current(
                data,
                from_version=file_version,
                current_version=TITANS_SCHEMA_VERSION,
            )
    else:
        # Pre-0.7 files were written before schema versioning shipped,
        # but the on-disk layout matches v1 exactly. Stamp the in-memory
        # dict as current so downstream consumers see a self-describing
        # payload, and continue silently. If a future breaking change
        # lands, bump TITANS_SCHEMA_VERSION and register a migration --
        # the older-than-current branch above will then apply it.
        data[_SCHEMA_VERSION_KEY] = np.array([TITANS_SCHEMA_VERSION], dtype=np.int64)

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
            # Legacy files may contain layer_{i}_local_init_{k}_* keys from the
            # removed ``local_inits`` field. These are ignored silently here;
            # the reset path reads live module parameters instead.
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
            source,
            len(bad),
            preview,
            more,
        )
