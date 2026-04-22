# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Centralized checkpoint save/load with safetensors support.

Provides format-agnostic save and load functions that handle both PyTorch
native (``.pt``) and ``safetensors`` formats, including automatic format
detection on load.

Schema versioning (Task P5)
---------------------------
Every checkpoint written via :func:`save_checkpoint` carries a
``titans_schema_version`` metadata key (top-level in ``.pt`` files, in the
``.meta.pt`` sidecar for ``.safetensors``). :func:`load_checkpoint`
dispatches on that key: missing -> warn once + best-effort load, equal ->
silent, newer -> :class:`RuntimeError`, older -> in-file migration when
registered (see :mod:`titans.memory_dump` for the migration protocol) or
:class:`RuntimeError` when no migration is available.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Top-level metadata key carrying the schema version. Kept in sync with
# the constant in ``titans/__init__.py`` — see MIGRATIONS.md.
_SCHEMA_VERSION_KEY: str = "titans_schema_version"


# ---------------------------------------------------------------------------
# Migration protocol (Task P5) — whole-checkpoint payloads
# ---------------------------------------------------------------------------
# ``_CHECKPOINT_MIGRATIONS`` maps ``(from_version, to_version)`` -> a
# function that takes the loaded payload dict (tensors plus metadata
# keyed by top-level names like ``"model"``, ``"titans_schema_version"``)
# and returns a new dict in the ``to_version`` layout. The walker lives
# in :mod:`titans._schema_migrations` and is shared with
# :mod:`titans.memory_dump` so the two dispatchers stay symmetric.
#
# For the first versioned release (schema 1) the registry is empty. When
# a whole-checkpoint breaking change lands, register ``(from, to)`` here
# AND add a row to ``MIGRATIONS.md``. Unversioned files take a separate
# legacy codepath (see ``_check_schema_version``).
_CHECKPOINT_MIGRATIONS: dict[
    tuple[int, int], Callable[[dict[str, Any]], dict[str, Any]]
] = {}


def _migrate_payload_to_current(
    payload: dict[str, Any],
    from_version: int,
    current_version: int,
) -> dict[str, Any]:
    """Apply registered migrations to bring ``payload`` up to ``current_version``.

    Thin wrapper around :func:`titans._schema_migrations.walk_migrations`
    that pins the registry to ``_CHECKPOINT_MIGRATIONS`` and tags the
    error messages with ``kind="checkpoint"``. Kept as a named entry
    point for symmetry with ``memory_dump._migrate_arrays_to_current``.
    """
    from titans._schema_migrations import walk_migrations

    return walk_migrations(
        payload,
        from_version=from_version,
        to_version=current_version,
        migrations=_CHECKPOINT_MIGRATIONS,
        kind="checkpoint",
    )


def _inject_schema_version(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a metadata dict that always carries the current schema version.

    If *metadata* is ``None`` we still emit a dict so the saved payload
    is self-describing; callers that previously passed ``None`` see their
    output gain a single extra key, which is the point of P5.
    """
    # Local import: titans/__init__.py imports this module eagerly, so
    # reaching back at module scope would create a circular import.
    from titans import TITANS_SCHEMA_VERSION

    if metadata is None:
        return {_SCHEMA_VERSION_KEY: TITANS_SCHEMA_VERSION}
    # Respect an explicit caller-supplied version (e.g. convert scripts
    # re-emitting an older file). Otherwise stamp the current version.
    merged = dict(metadata)
    merged.setdefault(_SCHEMA_VERSION_KEY, TITANS_SCHEMA_VERSION)
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_pt(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, Any] | None,
) -> list[Path]:
    """Save a checkpoint in PyTorch native format.

    Args:
        state_dict: Model state dict to save.
        path: Stem path (without extension).
        metadata: Optional metadata dict merged at top level.

    Returns:
        List containing the single ``.pt`` file written.
    """
    pt_path = path.with_suffix(".pt")
    tmp_path = pt_path.with_name(pt_path.name + ".tmp")
    payload: dict[str, Any] = {"model": state_dict}
    # Always emit the schema version so the saved file is self-describing.
    payload.update(_inject_schema_version(metadata))
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, pt_path)
    except (
        BaseException
    ):  # includes KeyboardInterrupt/SystemExit — we want cleanup on any exit
        # Clean up partial tmp file so subsequent runs don't stumble on it.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    logger.info("Saved pt checkpoint: %s", pt_path)
    return [pt_path]


def _save_safetensors(
    state_dict: dict[str, Any],
    path: Path,
    metadata: dict[str, Any] | None,
) -> list[Path]:
    """Save a checkpoint in safetensors format with optional metadata sidecar.

    Values may be :class:`torch.Tensor` instances, or
    :class:`titans.quantize_state.QuantizedMemoryState` instances which are
    flattened into per-field tensors using the supplied key as the prefix.

    Args:
        state_dict: Mapping of keys to tensors or ``QuantizedMemoryState``.
        path: Stem path (without extension).
        metadata: Optional metadata dict saved as a ``.meta.pt`` sidecar.

    Returns:
        List of files written (safetensors file, and optionally sidecar).
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for format='safetensors'. "
            "Install with: pip install safetensors"
        ) from exc

    from titans.quantize_state import QuantizedMemoryState, flatten_quantized_state

    sf_path = path.with_suffix(".safetensors")

    # Expand QuantizedMemoryState values first so the seen/contiguous pass
    # below only has to deal with torch.Tensor instances.
    expanded: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if isinstance(v, QuantizedMemoryState):
            expanded.update(flatten_quantized_state(v, prefix=k))
        elif isinstance(v, torch.Tensor):
            expanded[k] = v
        else:
            raise TypeError(
                f"_save_safetensors: entry {k!r} is a "
                f"{type(v).__name__}, not a torch.Tensor. "
                "safetensors format only supports flat tensor state "
                "dicts. If this is a QuantizedMemoryState or similar "
                "composite, call .dequantize() first or save in 'pt' "
                "format."
            )

    # safetensors requires contiguous tensors; clone shared tensors to avoid
    # duplicate-memory errors (e.g. tied embed/head weights)
    seen: dict[int, str] = {}
    prepared: dict[str, torch.Tensor] = {}
    for k, v in expanded.items():
        data_ptr = v.data_ptr()
        if data_ptr in seen:
            prepared[k] = v.clone().contiguous()
        else:
            seen[data_ptr] = k
            prepared[k] = v.contiguous()
    sf_tmp = sf_path.with_name(sf_path.name + ".tmp")
    try:
        save_file(prepared, sf_tmp)
        os.replace(sf_tmp, sf_path)
    except (
        BaseException
    ):  # includes KeyboardInterrupt/SystemExit — we want cleanup on any exit
        # Clean up partial tmp file so subsequent runs don't stumble on it.
        try:
            if sf_tmp.exists():
                sf_tmp.unlink()
        except OSError:
            pass
        raise
    written: list[Path] = [sf_path]
    logger.info("Saved safetensors checkpoint: %s", sf_path)

    # Always write a sidecar carrying at least the schema version so
    # load_checkpoint can verify compatibility without guessing.
    metadata_with_version = _inject_schema_version(metadata)
    if metadata_with_version:
        sidecar = path.with_suffix(".meta.pt")
        sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
        try:
            torch.save(metadata_with_version, sidecar_tmp)
            os.replace(sidecar_tmp, sidecar)
        except (
            BaseException
        ):  # includes KeyboardInterrupt/SystemExit — we want cleanup on any exit
            # Clean up partial tmp file so subsequent runs don't stumble on it.
            try:
                if sidecar_tmp.exists():
                    sidecar_tmp.unlink()
            except OSError:
                pass
            raise
        written.append(sidecar)
        logger.info("Saved metadata sidecar: %s", sidecar)

    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    state_dict: dict[str, Any],
    path: str | Path,
    *,
    format: str = "pt",
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Save a model checkpoint in the specified format.

    Args:
        state_dict: Mapping of keys to ``torch.Tensor`` values. When
            ``format='safetensors'``, values may additionally be
            ``titans.quantize_state.QuantizedMemoryState`` instances, which
            are flattened into per-field tensors using the key as prefix.
        path: Stem path **without** extension. The appropriate extension is
            appended based on *format*.
        format: ``"pt"`` for PyTorch native or ``"safetensors"``.
        metadata: Optional extra keys (e.g. step, loss, config). For pt
            format these are merged at the top level; for safetensors they
            are written to a ``.meta.pt`` sidecar file.

    Returns:
        List of :class:`~pathlib.Path` objects for every file written.

    Raises:
        ValueError: If *format* is not ``"pt"`` or ``"safetensors"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pt":
        return _save_pt(state_dict, path, metadata)
    elif format == "safetensors":
        return _save_safetensors(state_dict, path, metadata)
    else:
        raise ValueError(
            f"Unsupported checkpoint format {format!r}. Must be 'pt' or 'safetensors'."
        )


def _check_schema_version(payload: dict[str, Any], source: Path) -> None:
    """Dispatch on the schema version carried by a loaded checkpoint.

    * Missing key -> emit a ``DeprecationWarning`` and best-effort load
      (pre-0.7 layout).
    * Equal to current -> silent.
    * Newer -> :class:`RuntimeError` (tell the user to upgrade titans).
    * Older -> route through :func:`_migrate_payload_to_current` which
      applies any registered migrations from
      :data:`_CHECKPOINT_MIGRATIONS`. For v1 the registry is empty so
      this raises :class:`RuntimeError` with "no migration available" —
      but the infrastructure now mirrors ``memory_dump._MIGRATIONS``.
    """
    from titans import TITANS_SCHEMA_VERSION

    if _SCHEMA_VERSION_KEY not in payload:
        warnings.warn(
            (
                f"Loading unversioned checkpoint {source!s}: assuming "
                "pre-0.7 layout. Re-save with the current version of "
                "titans to stop this warning. See MIGRATIONS.md."
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        return

    file_version = int(payload[_SCHEMA_VERSION_KEY])
    if file_version == TITANS_SCHEMA_VERSION:
        return
    if file_version > TITANS_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint schema {file_version} > code schema "
            f"{TITANS_SCHEMA_VERSION}; upgrade titans. See "
            "MIGRATIONS.md for the per-version change log."
        )
    # Older than current: walker raises when no migration is registered.
    # In v1 there are no migrations, so this unconditionally errors out
    # with a clear "no migration available" message. When a future
    # breaking change lands, register the step in ``_CHECKPOINT_MIGRATIONS``
    # and the same path will apply it instead of raising.
    _migrate_payload_to_current(
        payload,
        from_version=file_version,
        current_version=TITANS_SCHEMA_VERSION,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint with automatic format detection.

    Args:
        path: Path to the checkpoint file. If no extension is provided,
            tries ``.safetensors`` first, then ``.pt``.
        device: Device to map tensors to.
        weights_only: Passed to :func:`torch.load` for pt files.

    Returns:
        Dict with ``"model"`` key containing the state dict, plus any
        metadata keys (including ``titans_schema_version`` when present).

    Raises:
        FileNotFoundError: If no matching checkpoint file is found.
        RuntimeError: If the checkpoint carries a schema version that is
            newer than the current code, or older with no migration
            available. See :data:`titans.TITANS_SCHEMA_VERSION` and
            ``MIGRATIONS.md``.
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        result = _load_safetensors(path, device=device)
        _check_schema_version(result, path)
        return result
    elif path.suffix == ".pt":
        result = _load_pt(path, device=device, weights_only=weights_only)
        _check_schema_version(result, path)
        return result
    else:
        # Extensionless: try safetensors first, then pt
        sf_path = path.with_suffix(".safetensors")
        if sf_path.exists():
            result = _load_safetensors(sf_path, device=device)
            _check_schema_version(result, sf_path)
            return result
        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            result = _load_pt(pt_path, device=device, weights_only=weights_only)
            _check_schema_version(result, pt_path)
            return result
        raise FileNotFoundError(
            f"No checkpoint found at {path} (tried {sf_path} and {pt_path})"
        )


# ---------------------------------------------------------------------------
# Internal load helpers
# ---------------------------------------------------------------------------


def _load_pt(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a PyTorch native checkpoint.

    Args:
        path: Path to the ``.pt`` file.
        device: Device to map tensors to.
        weights_only: Passed to :func:`torch.load`.

    Returns:
        Normalized dict with ``"model"`` key.
    """
    data = torch.load(path, map_location=device, weights_only=weights_only)
    if "model" not in data and all(isinstance(v, torch.Tensor) for v in data.values()):
        # Bare state dict — wrap it
        data = {"model": data}
    logger.info("Loaded pt checkpoint: %s", path)
    return data


def _load_safetensors(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a safetensors checkpoint with optional metadata sidecar.

    Args:
        path: Path to the ``.safetensors`` file.
        device: Device to map tensors to.

    Returns:
        Dict with ``"model"`` key and any metadata from the sidecar.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for format='safetensors'. "
            "Install with: pip install safetensors"
        ) from exc

    tensors = load_file(path, device=str(device))
    result: dict[str, Any] = {"model": tensors}

    sidecar = path.with_suffix(".meta.pt")
    if sidecar.exists():
        meta = torch.load(sidecar, map_location=device, weights_only=False)
        result.update(meta)
        logger.info("Loaded safetensors checkpoint with sidecar: %s", path)
    else:
        logger.info("Loaded safetensors checkpoint (no sidecar): %s", path)

    return result
