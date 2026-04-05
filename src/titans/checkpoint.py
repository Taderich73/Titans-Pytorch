# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Centralized checkpoint save/load with safetensors support.

Provides format-agnostic save and load functions that handle both PyTorch
native (``.pt``) and ``safetensors`` formats, including automatic format
detection on load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

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
    payload: dict[str, Any] = {"model": state_dict}
    if metadata:
        payload.update(metadata)
    torch.save(payload, pt_path)
    logger.info("Saved pt checkpoint: %s", pt_path)
    return [pt_path]


def _save_safetensors(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, Any] | None,
) -> list[Path]:
    """Save a checkpoint in safetensors format with optional metadata sidecar.

    Args:
        state_dict: Model state dict to save.
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

    sf_path = path.with_suffix(".safetensors")
    # safetensors requires contiguous tensors; clone shared tensors to avoid
    # duplicate-memory errors (e.g. tied embed/head weights)
    seen: dict[int, str] = {}
    prepared: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        data_ptr = v.data_ptr()
        if data_ptr in seen:
            prepared[k] = v.clone().contiguous()
        else:
            seen[data_ptr] = k
            prepared[k] = v.contiguous()
    save_file(prepared, sf_path)
    written: list[Path] = [sf_path]
    logger.info("Saved safetensors checkpoint: %s", sf_path)

    if metadata:
        sidecar = path.with_suffix(".meta.pt")
        torch.save(metadata, sidecar)
        written.append(sidecar)
        logger.info("Saved metadata sidecar: %s", sidecar)

    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    path: str | Path,
    *,
    format: str = "pt",
    metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Save a model checkpoint in the specified format.

    Args:
        state_dict: Model state dict to save.
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
            f"Unsupported checkpoint format {format!r}. "
            "Must be 'pt' or 'safetensors'."
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
        metadata keys.

    Raises:
        FileNotFoundError: If no matching checkpoint file is found.
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        return _load_safetensors(path, device=device)
    elif path.suffix == ".pt":
        return _load_pt(path, device=device, weights_only=weights_only)
    else:
        # Extensionless: try safetensors first, then pt
        sf_path = path.with_suffix(".safetensors")
        if sf_path.exists():
            return _load_safetensors(sf_path, device=device)
        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            return _load_pt(pt_path, device=device, weights_only=weights_only)
        raise FileNotFoundError(
            f"No checkpoint found at {path} "
            f"(tried {sf_path} and {pt_path})"
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
