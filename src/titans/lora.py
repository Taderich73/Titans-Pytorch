# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
LoRA (Low-Rank Adaptation) for Titans PyTorch models.

Provides LoRALinear module, layer wrapping, adapter save/load, and weight merging.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Target name sets (matching actual Titans layer names)
_ATTN_NAMES = {"proj_q", "proj_k", "proj_v", "proj_out"}
_FFN_NAMES = {"gate_proj", "up_proj", "down_proj"}

# Memory module projection names (NeuralLongTermMemory, tnt_memory)
_MEMORY_NAMES = {"proj_q", "proj_k", "proj_v", "proj_out"}

# Layers that must never be wrapped (embedding / LM-head weight-tied pair)
_NEVER_WRAP = {"embed", "head", "lm_head", "embed_tokens", "token_embedding"}


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear.

    At initialization, lora_B is zero so the output equals the base layer.
    During training only lora_A and lora_B are updated.

    Args:
        base: The frozen base nn.Linear layer.
        rank: LoRA rank (r).
        alpha: LoRA scaling alpha.  Effective scale = alpha / rank.
        dropout: Dropout probability applied to the input before the LoRA path.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        # Freeze base parameters — only LoRA matrices are trainable
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        # Kaiming-style init for A, zero init for B (standard LoRA)
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank) * (1.0 / math.sqrt(rank))
        )
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank
        self.dropout: nn.Module = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen base output with LoRA delta.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scale
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.lora_A.shape[1]}, "
            f"scale={self.scale:.4f}, "
            f"enabled={self.enabled}"
        )


def _resolve_target_names(targets: str) -> set[str]:
    """Resolve a comma-separated target string into a set of layer name keys.

    Args:
        targets: Comma-separated target groups, e.g. "attn,ffn" or "all".
            Valid tokens: "attn", "ffn", "memory", "all".

    Returns:
        Set of attribute name strings that should be wrapped.

    Raises:
        ValueError: If an unknown target token is encountered.
    """
    resolved: set[str] = set()
    for token in targets.split(","):
        token = token.strip().lower()
        if token == "attn":
            resolved |= _ATTN_NAMES
        elif token == "ffn":
            resolved |= _FFN_NAMES
        elif token == "memory":
            resolved |= _MEMORY_NAMES
        elif token == "all":
            resolved |= _ATTN_NAMES | _FFN_NAMES | _MEMORY_NAMES
        else:
            raise ValueError(
                f"Unknown LoRA target '{token}'. "
                "Valid tokens: 'attn', 'ffn', 'memory', 'all'."
            )
    return resolved


def wrap_lora_layers(
    model: nn.Module,
    targets: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Walk the model tree and replace targeted nn.Linear layers with LoRALinear.

    Only layers whose attribute name (the last component of their dotted path)
    is in the resolved target set are wrapped.  Layers whose path contains any
    token from _NEVER_WRAP are skipped unconditionally.

    Args:
        model: The model to modify in-place.
        targets: Comma-separated target groups ("attn", "ffn", "memory", "all").
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: Dropout probability for the LoRA input path.

    Returns:
        Sorted list of dotted module paths that were wrapped.
    """
    target_names = _resolve_target_names(targets)
    wrapped_paths: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            full_path = f"{prefix}.{name}" if prefix else name

            # Skip embed / head layers regardless of depth
            if any(skip in full_path.split(".") for skip in _NEVER_WRAP):
                continue

            if isinstance(child, nn.Linear) and name in target_names:
                wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
                wrapped_paths.append(full_path)
                logger.debug(f"Wrapped '{full_path}' with LoRALinear(rank={rank})")
            else:
                # Recurse into non-Linear children (and skip LoRALinear already done)
                if not isinstance(child, LoRALinear):
                    _walk(child, full_path)

    _walk(model, "")
    wrapped_paths.sort()
    logger.info(f"LoRA: wrapped {len(wrapped_paths)} layers — targets={targets!r}")
    return wrapped_paths


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle all LoRALinear adapters in the model on or off.

    Useful for computing a reference model forward pass (e.g. DPO) without
    needing a separate copy of the model.

    Args:
        model: The model containing LoRALinear modules.
        enabled: If True, LoRA deltas are added; if False, only the base
            layer output is returned.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled
            count += 1
    logger.debug(f"set_lora_enabled({enabled}) applied to {count} LoRALinear layers")


def save_adapters(model: nn.Module, path: Path, meta: dict[str, Any]) -> None:
    """Save only LoRA A/B parameters to a safetensors file.

    Writes two files:
    - ``path`` — adapter weights in safetensors format.
    - ``path.with_suffix('.json')`` — JSON metadata sidecar.

    Args:
        model: The model containing LoRALinear modules.
        path: Destination file path (e.g. ``adapters/lora.safetensors``).
        meta: Arbitrary metadata dict to save as a JSON sidecar.

    Raises:
        ImportError: If ``safetensors`` is not installed.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for save_adapters. "
            "Install with: pip install safetensors"
        ) from exc

    tensors: dict[str, torch.Tensor] = {}
    for module_path, module in model.named_modules():
        if isinstance(module, LoRALinear):
            tensors[f"{module_path}.lora_A"] = module.lora_A.data.contiguous()
            tensors[f"{module_path}.lora_B"] = module.lora_B.data.contiguous()

    if not tensors:
        logger.warning("save_adapters: no LoRALinear layers found — nothing saved")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Saved {len(tensors) // 2} LoRA adapter pairs to {path} "
        f"(metadata → {meta_path})"
    )


def load_adapters(model: nn.Module, path: Path) -> dict[str, Any]:
    """Load adapter weights from a safetensors file into a wrapped model.

    The model must already have LoRALinear wrappers at the same paths that
    were present when the adapters were saved.

    Args:
        model: The model with LoRALinear wrappers to load weights into.
        path: Path to the ``.safetensors`` adapter file.

    Returns:
        Metadata dict loaded from the JSON sidecar, or an empty dict if
        no sidecar is found.

    Raises:
        ImportError: If ``safetensors`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for load_adapters. "
            "Install with: pip install safetensors"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found: {path}")

    tensors = load_file(str(path))

    # Build a lookup of existing LoRALinear modules
    lora_modules: dict[str, LoRALinear] = {
        module_path: module
        for module_path, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

    loaded_count = 0
    for key, tensor in tensors.items():
        # key format: "<module_path>.lora_A" or "<module_path>.lora_B"
        if key.endswith(".lora_A"):
            module_path = key[: -len(".lora_A")]
            param_name = "lora_A"
        elif key.endswith(".lora_B"):
            module_path = key[: -len(".lora_B")]
            param_name = "lora_B"
        else:
            logger.warning(f"load_adapters: unexpected key '{key}' — skipping")
            continue

        if module_path not in lora_modules:
            logger.warning(
                f"load_adapters: no LoRALinear at '{module_path}' — skipping"
            )
            continue

        param = getattr(lora_modules[module_path], param_name)
        if tensor.dtype != param.dtype:
            logger.warning(
                "load_adapters: dtype mismatch for %s (ckpt=%s, "
                "param=%s); casting to param dtype",
                key, tensor.dtype, param.dtype,
            )
        with torch.no_grad():
            param.data.copy_(tensor.to(param.device, param.dtype))
        loaded_count += 1

    logger.info(f"Loaded {loaded_count} LoRA tensors from {path}")

    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def merge_lora_weights(model: nn.Module) -> None:
    """Fold LoRA A @ B * scale into each base layer weight permanently.

    After merging, LoRALinear wrappers are replaced with plain nn.Linear
    modules whose weights incorporate the learned delta.  This is useful
    before exporting the model for inference.

    Args:
        model: The model containing LoRALinear modules, modified in-place.
    """
    # Collect replacements first to avoid mutating while iterating
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for module_path, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        base = module.base
        with torch.no_grad():
            # delta shape: (in_features, out_features) → transpose for weight
            delta = (module.lora_A @ module.lora_B) * module.scale
            # base.weight is (out_features, in_features)
            base.weight.add_(delta.t())

        # Re-enable requires_grad on base weight now that it owns the delta
        base.weight.requires_grad_(True)

        # Locate parent module to perform setattr replacement
        parts = module_path.split(".")
        attr_name = parts[-1]
        parent_path = ".".join(parts[:-1])

        if parent_path:
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        else:
            parent = model

        replacements.append((parent, attr_name, base))

    for parent, attr_name, merged_linear in replacements:
        setattr(parent, attr_name, merged_linear)

    logger.info(f"Merged {len(replacements)} LoRA adapter(s) into base weights")


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters in the model.

    Args:
        model: Any nn.Module (typically one with LoRALinear wrappers).

    Returns:
        A tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
