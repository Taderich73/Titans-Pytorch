# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
LoRA (Low-Rank Adaptation) for Titans MLX models.

Provides:
- LoRALinear: Low-rank adapter wrapping an nn.Linear
- wrap_lora_layers: Inject LoRA into targeted layers of a model
- save_adapters / load_adapters: Persist and restore adapter weights
- merge_lora_weights: Fold LoRA weights into base Linear layers
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Generator

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear.

    At initialization, lora_B is zero so the output equals the base layer.
    During training only lora_A and lora_B are updated.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.freeze()

        # base.weight is (out_dim, in_dim) in MLX
        out_dim, in_dim = base.weight.shape

        self.lora_A = mx.random.normal((in_dim, rank)) * (1.0 / math.sqrt(rank))
        self.lora_B = mx.zeros((rank, out_dim))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_input = self.dropout(x) if self.dropout else x
        lora_out = (lora_input @ self.lora_A) @ self.lora_B * self.scale
        return base_out + lora_out


# ---------------------------------------------------------------------------
# Recursive tree walk
# ---------------------------------------------------------------------------

def _recursive_find_linear(
    module: nn.Module,
    prefix: str = "",
) -> Generator[tuple[str, str, nn.Module, nn.Linear], None, None]:
    """Walk the module tree and yield every nn.Linear found.

    Yields:
        (full_dotted_path, attr_name, parent_module, linear_instance)
    """
    children = module.children()
    for attr_name, child in children.items():
        if isinstance(child, list):
            for i, item in enumerate(child):
                path = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                if isinstance(item, nn.Module):
                    if isinstance(item, nn.Linear):
                        yield (path, str(i), child, item)  # parent is the list
                    yield from _recursive_find_linear(item, path)
        elif isinstance(child, dict):
            for k, v in child.items():
                path = f"{prefix}.{attr_name}.{k}" if prefix else f"{attr_name}.{k}"
                if isinstance(v, nn.Module):
                    if isinstance(v, nn.Linear):
                        yield (path, k, child, v)
                    yield from _recursive_find_linear(v, path)
        elif isinstance(child, nn.Module):
            path = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(child, nn.Linear):
                yield (path, attr_name, module, child)
            yield from _recursive_find_linear(child, path)


# ---------------------------------------------------------------------------
# LoRA wrapping
# ---------------------------------------------------------------------------

# Target name sets
_ATTN_NAMES = {"proj_q", "proj_k", "proj_v", "proj_out"}
_FFN_NAMES = {"gate_proj", "up_proj", "down_proj"}


def wrap_lora_layers(
    model: nn.Module,
    targets: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Replace targeted nn.Linear layers with LoRALinear wrappers.

    Args:
        model: The model to modify in-place.
        targets: Comma-separated target groups: "attn", "ffn", "memory", "all".
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: Dropout applied to LoRA input.

    Returns:
        List of full dotted paths that were wrapped.
    """
    target_set = {t.strip() for t in targets.split(",")}
    want_all = "all" in target_set
    want_attn = "attn" in target_set
    want_ffn = "ffn" in target_set
    want_memory = "memory" in target_set

    wrapped: list[str] = []

    for full_path, attr_name, parent, linear in list(
        _recursive_find_linear(model)
    ):
        # Never wrap embed or head layers
        if "embed" in full_path or "head" in full_path:
            continue

        # Determine the leaf name (last component of the path)
        leaf = full_path.rsplit(".", 1)[-1]

        should_wrap = False
        if want_all:
            should_wrap = True
        else:
            if want_attn and leaf in _ATTN_NAMES:
                should_wrap = True
            if want_ffn and leaf in _FFN_NAMES:
                should_wrap = True
            if want_memory and "memory" in full_path:
                should_wrap = True

        if not should_wrap:
            continue

        lora = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)

        # Replace on parent
        if isinstance(parent, list):
            idx = int(attr_name)
            parent[idx] = lora
        elif isinstance(parent, dict):
            parent[attr_name] = lora
        else:
            setattr(parent, attr_name, lora)

        wrapped.append(full_path)

    return wrapped


# ---------------------------------------------------------------------------
# Find LoRA modules
# ---------------------------------------------------------------------------

def _find_lora_modules(
    model: nn.Module,
    prefix: str = "",
) -> list[tuple[str, LoRALinear]]:
    """Walk the tree and collect all LoRALinear instances.

    Returns:
        List of (full_dotted_path, LoRALinear_instance).
    """
    results: list[tuple[str, LoRALinear]] = []
    children = model.children()
    for attr_name, child in children.items():
        if isinstance(child, list):
            for i, item in enumerate(child):
                path = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                if isinstance(item, LoRALinear):
                    results.append((path, item))
                if isinstance(item, nn.Module):
                    results.extend(_find_lora_modules(item, path))
        elif isinstance(child, dict):
            for k, v in child.items():
                path = f"{prefix}.{attr_name}.{k}" if prefix else f"{attr_name}.{k}"
                if isinstance(v, LoRALinear):
                    results.append((path, v))
                if isinstance(v, nn.Module):
                    results.extend(_find_lora_modules(v, path))
        elif isinstance(child, nn.Module):
            path = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(child, LoRALinear):
                results.append((path, child))
            results.extend(_find_lora_modules(child, path))
    return results


# ---------------------------------------------------------------------------
# Save / Load / Merge
# ---------------------------------------------------------------------------

def save_adapters(model: nn.Module, path: Path, meta: dict) -> None:
    """Save LoRA adapter weights and metadata.

    Writes:
        - ``path.with_suffix('.safetensors')``: adapter weight arrays
        - ``path.with_suffix('.meta.json')``: metadata dict
    """
    path = Path(path)
    weights: dict[str, mx.array] = {}
    for full_path, lora_mod in _find_lora_modules(model):
        weights[f"{full_path}.lora_A"] = lora_mod.lora_A
        weights[f"{full_path}.lora_B"] = lora_mod.lora_B

    sf_path = path.with_suffix(".safetensors")
    mx.save_safetensors(str(sf_path), weights)

    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))


def load_adapters(model: nn.Module, path: Path) -> dict:
    """Load LoRA adapter weights from disk into a wrapped model.

    Args:
        model: Model already wrapped with LoRALinear via ``wrap_lora_layers``.
        path: Base path (without extension).

    Returns:
        Metadata dict read from the ``.meta.json`` sidecar.
    """
    path = Path(path)
    sf_path = path.with_suffix(".safetensors")
    weights = mx.load(str(sf_path))

    for full_path, lora_mod in _find_lora_modules(model):
        a_key = f"{full_path}.lora_A"
        b_key = f"{full_path}.lora_B"
        if a_key in weights:
            lora_mod.lora_A = weights[a_key]
        if b_key in weights:
            lora_mod.lora_B = weights[b_key]

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    return meta


def merge_lora_weights(model: nn.Module) -> None:
    """Fold LoRA weights into the base Linear layers in-place.

    After merging, each LoRALinear is replaced by a plain nn.Linear
    whose weight equals ``base.weight + (lora_A @ lora_B).T * scale``.
    """
    for full_path, attr_name, parent, child in list(
        _recursive_find_lora(model)
    ):
        lora_mod = child
        base = lora_mod.base
        merged_weight = base.weight + (lora_mod.lora_A @ lora_mod.lora_B).T * lora_mod.scale

        new_linear = nn.Linear(
            input_dims=base.weight.shape[1],
            output_dims=base.weight.shape[0],
            bias=False,
        )
        new_linear.weight = merged_weight

        if isinstance(parent, list):
            idx = int(attr_name)
            parent[idx] = new_linear
        elif isinstance(parent, dict):
            parent[attr_name] = new_linear
        else:
            setattr(parent, attr_name, new_linear)


def _recursive_find_lora(
    module: nn.Module,
    prefix: str = "",
) -> Generator[tuple[str, str, object, LoRALinear], None, None]:
    """Walk tree yielding (full_path, attr_name, parent, LoRALinear).

    Similar to _recursive_find_linear but finds LoRALinear instances,
    and also tracks parent+attr_name for replacement.
    """
    children = module.children()
    for attr_name, child in children.items():
        if isinstance(child, list):
            for i, item in enumerate(child):
                path = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                if isinstance(item, LoRALinear):
                    yield (path, str(i), child, item)
                if isinstance(item, nn.Module):
                    yield from _recursive_find_lora(item, path)
        elif isinstance(child, dict):
            for k, v in child.items():
                path = f"{prefix}.{attr_name}.{k}" if prefix else f"{attr_name}.{k}"
                if isinstance(v, LoRALinear):
                    yield (path, k, child, v)
                if isinstance(v, nn.Module):
                    yield from _recursive_find_lora(v, path)
        elif isinstance(child, nn.Module):
            path = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(child, LoRALinear):
                yield (path, attr_name, module, child)
            yield from _recursive_find_lora(child, path)
