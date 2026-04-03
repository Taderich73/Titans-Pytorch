#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans MLX models on Apple Silicon.

Optimized for M1/M2/M3/M4 GPUs with unified memory and lazy evaluation.

Usage:
    # Generate from trained model
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.safetensors --prompt "Hello"

    # Generate with HuggingFace tokenizer
    uv run python scripts/inference.py --checkpoint model.safetensors --tokenizer meta-llama/Llama-2-7b-hf

    # Interactive mode with streaming
    uv run python scripts/inference.py --checkpoint model.safetensors --interactive --stream

    # Quantized inference (4-bit or 8-bit)
    uv run python scripts/inference.py --checkpoint model.safetensors --quantize 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans_mlx.memory import MemoryState, load_memory_states, save_memory_states

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Optional HuggingFace transformers
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Tokenizers
# =============================================================================


class SimpleTokenizer:
    """Simple character-level tokenizer for demo purposes."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        """Encode text to token IDs."""
        ids = [self.char_to_id.get(c, 0) for c in text]
        if return_tensors == "mlx":
            return {"input_ids": mx.array([ids])}
        return ids

    def decode(
        self, ids: list[int] | mx.array, skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, mx.array):
            ids = ids.tolist()
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def __call__(self, text: str, return_tensors: str | None = None) -> Any:
        return self.encode(text, return_tensors=return_tensors)


def load_tokenizer(
    tokenizer_name: str | None,
    vocab_size: int,
) -> SimpleTokenizer | PreTrainedTokenizerBase:
    """Load tokenizer - HuggingFace or simple character-level."""
    if tokenizer_name and HAS_TRANSFORMERS:
        logger.info(f"Loading HuggingFace tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif tokenizer_name and not HAS_TRANSFORMERS:
        logger.warning("transformers not installed, using simple tokenizer")

    logger.info("Using simple character-level tokenizer")
    return SimpleTokenizer(vocab_size)


# =============================================================================
# Chat Template Support
# =============================================================================

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_prompt_for_chat(prompt: str) -> str:
    """Wrap a raw prompt as ChatML user message + assistant prefix."""
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"


def strip_chat_delimiters(text: str) -> str:
    """Remove ChatML tokens from generated text."""
    return text.replace(IM_START, "").replace(IM_END, "").strip()


def should_use_chat(chat_template: str | None, cli_override: bool | None) -> bool:
    """Auto-detect chat mode. cli_override=True/False overrides, None=auto."""
    if cli_override is not None:
        return cli_override
    if chat_template is None or chat_template == "none":
        return False
    return True


def ensure_chat_tokens(tokenizer: Any) -> None:
    """Add ChatML special tokens if missing."""
    if not HAS_TRANSFORMERS or isinstance(tokenizer, SimpleTokenizer):
        return
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = [t for t in [IM_START, IM_END] if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def get_im_end_token_id(tokenizer: Any) -> int | None:
    """Get <|im_end|> token ID."""
    try:
        ids = tokenizer.encode(IM_END, add_special_tokens=False)
        return ids[0] if ids else None
    except Exception:
        return None


# =============================================================================
# LoRA Support
# =============================================================================


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

    for full_path, attr_name, parent, linear in list(_recursive_find_linear(model)):
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


def save_adapters(model: nn.Module, path: Path, meta: dict) -> None:
    """Save LoRA adapter weights and metadata."""
    path = Path(path)
    weights: dict[str, mx.array] = {}
    for full_path, lora_mod in _find_lora_modules(model):
        weights[f"{full_path}.lora_A"] = lora_mod.lora_A
        weights[f"{full_path}.lora_B"] = lora_mod.lora_B

    sf_path = path.with_suffix(".safetensors")
    mx.save_safetensors(str(sf_path), weights)

    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))


# =============================================================================
# Model Loading
# =============================================================================


def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Create Titans model based on type."""
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](config)


def load_model(
    checkpoint_path: Path,
    quantize: int | None = None,
) -> tuple[nn.Module, TitansConfig, str, str | None, str]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        quantize: Quantization bits (None, 4, or 8)

    Returns:
        Tuple of (model, config, model_type, tokenizer_name, chat_template)
    """
    # Load metadata
    meta_path = checkpoint_path.with_suffix(".meta.npz")
    if meta_path.exists():
        meta = np.load(str(meta_path))
        model_type = str(meta["model_type"][0])
        dim = int(meta["dim"][0])
        num_heads = int(meta["num_heads"][0])
        num_layers = int(meta["num_layers"][0])
        vocab_size = int(meta["vocab_size"][0])
        chunk_size = int(meta.get("chunk_size", [512])[0])
        window_size = int(meta.get("window_size", [512])[0])
        num_persistent_tokens = int(meta.get("num_persistent_tokens", [16])[0])
        num_memory_layers = int(meta.get("num_memory_layers", [2])[0])
        tokenizer_name = str(meta.get("tokenizer_name", [None])[0])
        if tokenizer_name == "None":
            tokenizer_name = None

        chat_template = str(meta.get("chat_template", ["none"])[0])
        if chat_template == "None":
            chat_template = "none"

        # TNT flags
        use_tnt = str(meta.get("use_tnt", ["False"])[0]) == "True"
        global_chunk_size = int(meta.get("global_chunk_size", [2048])[0])
        local_chunk_sizes_raw = str(meta.get("local_chunk_sizes", ["8,16"])[0])
        local_chunk_sizes = [int(c) for c in local_chunk_sizes_raw.split(",")]
        local_shard_length = int(meta.get("local_shard_length", [2048])[0])
        use_qk_projection = str(meta.get("use_qk_projection", ["True"])[0]) == "True"
        tnt_stage = int(meta.get("tnt_stage", [1])[0])

        # AttnRes flags
        use_attn_res = str(meta.get("use_attn_res", ["False"])[0]) == "True"
        num_attnres_blocks = int(meta.get("num_attnres_blocks", [8])[0])
        attnres_warmup_steps = int(meta.get("attnres_warmup_steps", [0])[0])
        attnres_modulate_global = (
            str(meta.get("attnres_modulate_global_memory", ["True"])[0]) == "True"
        )
        attnres_modulate_local = (
            str(meta.get("attnres_modulate_local_memory", ["False"])[0]) == "True"
        )

        # Memory objective (Yaad)
        memory_objective = str(meta.get("memory_objective", ["l2"])[0])
        huber_delta_init = float(meta.get("huber_delta_init", [0.0])[0])

        # Adaptive window flags
        adaptive_window = str(meta.get("adaptive_window", ["False"])[0]) == "True"
        adaptive_window_min = int(meta.get("adaptive_window_min", [64])[0])
        adaptive_window_max_raw = str(meta.get("adaptive_window_max", ["None"])[0])
        adaptive_window_max = None if adaptive_window_max_raw == "None" else int(adaptive_window_max_raw)
        adaptive_window_temperature = float(meta.get("adaptive_window_temperature", [10.0])[0])
    else:
        # Try to infer from checkpoint name
        logger.warning("No metadata file found, using default configuration")
        model_type = "mac"
        dim = 512
        num_heads = 8
        num_layers = 12
        vocab_size = 32000
        chunk_size = 512
        window_size = 512
        num_persistent_tokens = 16
        num_memory_layers = 2
        tokenizer_name = None
        chat_template = "none"
        use_tnt = False
        global_chunk_size = 2048
        local_chunk_sizes = [8, 16]
        local_shard_length = 2048
        use_qk_projection = True
        tnt_stage = 1
        use_attn_res = False
        num_attnres_blocks = 8
        attnres_warmup_steps = 0
        attnres_modulate_global = True
        attnres_modulate_local = False
        memory_objective = "l2"
        huber_delta_init = 0.0
        adaptive_window = False
        adaptive_window_min = 64
        adaptive_window_max = None
        adaptive_window_temperature = 10.0

    config = TitansConfig(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        window_size=window_size,
        num_persistent_tokens=num_persistent_tokens,
        num_memory_layers=num_memory_layers,
        dropout=0.0,
        use_conv=False,  # Disable conv for compatibility
        use_tnt=use_tnt,
        global_chunk_size=global_chunk_size,
        local_chunk_sizes=local_chunk_sizes,
        local_shard_length=local_shard_length,
        use_qk_projection=use_qk_projection,
        tnt_stage=tnt_stage,
        use_attn_res=use_attn_res,
        num_attnres_blocks=num_attnres_blocks,
        attnres_warmup_steps=attnres_warmup_steps,
        attnres_modulate_global_memory=attnres_modulate_global,
        attnres_modulate_local_memory=attnres_modulate_local,
        memory_objective=memory_objective,
        huber_delta_init=huber_delta_init,
        adaptive_window=adaptive_window,
        adaptive_window_min=adaptive_window_min,
        adaptive_window_max=adaptive_window_max,
        adaptive_window_temperature=adaptive_window_temperature,
    )

    model = create_model(model_type, config)

    # Load weights
    weights_path = checkpoint_path.with_suffix(".safetensors")
    if weights_path.exists():
        model.load_weights(str(weights_path))
    elif checkpoint_path.suffix == ".safetensors" and checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))
    else:
        # Try npz format
        checkpoint = np.load(str(checkpoint_path), allow_pickle=True)
        weights = {}
        for k in checkpoint.files:
            if not k.startswith("_"):
                weights[k] = mx.array(checkpoint[k])
        model.update(weights)

    # NOTE: Do NOT re-tie head.weight = embed.weight here.
    # For checkpoints trained with properly tied weights (post-fix), head and
    # embed are identical in the safetensors file, so loading both is fine.
    # For older checkpoints where the tie was broken during training, re-tying
    # would discard the independently trained output projection.

    # Apply quantization if requested
    if quantize:
        model = quantize_model(model, quantize)
        logger.info(f"Applied {quantize}-bit quantization")

    mx.eval(model.parameters())

    flags = []
    if config.use_tnt:
        flags.append("TNT")
    if config.use_attn_res:
        flags.append("AttnRes")
    flag_str = f" [{'+'.join(flags)}]" if flags else ""
    logger.info(f"Loaded {model_type.upper()}{flag_str} model from {checkpoint_path}")

    return model, config, model_type, tokenizer_name, chat_template


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """Apply quantization to model.

    Args:
        model: Model to quantize
        bits: Quantization bits (4 or 8)

    Returns:
        Quantized model
    """
    if bits not in (4, 8):
        logger.warning(f"Unsupported quantization bits: {bits}, using 8")
        bits = 8

    # MLX supports quantization via mlx.nn.quantize
    try:
        nn.quantize(model, bits=bits)
        logger.info(f"Quantized model to {bits} bits")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    return model


def load_lora_model(
    adapters_path: Path,
    checkpoint_override: str | None = None,
    quantize: int | None = None,
) -> tuple[nn.Module, TitansConfig, str, str | None, str]:
    """Load a base model and apply LoRA adapter weights.

    Reads ``adapters.meta.json`` for model config and LoRA parameters,
    creates the model, loads base weights, wraps with LoRA, then loads
    adapter weights.

    Args:
        adapters_path: Path to adapter files (base name, without extension).
        checkpoint_override: Override base checkpoint path from metadata.
        quantize: Quantization bits (None, 4, or 8).

    Returns:
        Tuple of (model, config, model_type, tokenizer_name, chat_template)
    """
    adapters_path = Path(adapters_path)

    # Read adapter metadata
    meta_path = adapters_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Adapter metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    # Extract model config from metadata
    model_type = meta.get("model_type", "mac")
    tokenizer_name = meta.get("tokenizer_name")
    if tokenizer_name == "None":
        tokenizer_name = None
    chat_template = meta.get("chat_template", "none")
    if chat_template == "None":
        chat_template = "none"

    config = TitansConfig(
        dim=meta.get("dim", 512),
        num_heads=meta.get("num_heads", 8),
        num_layers=meta.get("num_layers", 12),
        vocab_size=meta.get("vocab_size", 32000),
        chunk_size=meta.get("chunk_size", 512),
        window_size=meta.get("window_size", 512),
        num_persistent_tokens=meta.get("num_persistent_tokens", 16),
        num_memory_layers=meta.get("num_memory_layers", 2),
        dropout=0.0,
        use_conv=False,
        use_tnt=meta.get("use_tnt", False),
        global_chunk_size=meta.get("global_chunk_size", 2048),
        local_chunk_sizes=meta.get("local_chunk_sizes", [8, 16]),
        local_shard_length=meta.get("local_shard_length", 2048),
        use_qk_projection=meta.get("use_qk_projection", True),
        tnt_stage=meta.get("tnt_stage", 1),
        use_attn_res=meta.get("use_attn_res", False),
        num_attnres_blocks=meta.get("num_attnres_blocks", 8),
        attnres_warmup_steps=meta.get("attnres_warmup_steps", 0),
        attnres_modulate_global_memory=meta.get("attnres_modulate_global_memory", True),
        attnres_modulate_local_memory=meta.get("attnres_modulate_local_memory", False),
        memory_objective=meta.get("memory_objective", "l2"),
        huber_delta_init=meta.get("huber_delta_init", 0.0),
        adaptive_window=meta.get("adaptive_window", False),
        adaptive_window_min=meta.get("adaptive_window_min", 64),
        adaptive_window_max=meta.get("adaptive_window_max", None),
        adaptive_window_temperature=meta.get("adaptive_window_temperature", 10.0),
    )

    model = create_model(model_type, config)

    # Load base weights
    base_checkpoint = checkpoint_override or meta.get("base_checkpoint")
    if base_checkpoint:
        base_path = Path(base_checkpoint)
        weights_path = base_path.with_suffix(".safetensors")
        if weights_path.exists():
            model.load_weights(str(weights_path))
        elif base_path.suffix == ".safetensors" and base_path.exists():
            model.load_weights(str(base_path))
        else:
            raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
        logger.info(f"Loaded base weights from {base_checkpoint}")

    # Wrap with LoRA
    lora_targets = meta.get("lora_targets", "attn")
    lora_rank = meta.get("lora_rank", 8)
    lora_alpha = meta.get("lora_alpha", 16.0)
    wrapped = wrap_lora_layers(model, lora_targets, lora_rank, lora_alpha, dropout=0.0)
    logger.info(f"Wrapped {len(wrapped)} layers with LoRA (rank={lora_rank})")

    # Load adapter weights
    load_adapters(model, adapters_path)
    logger.info(f"Loaded adapter weights from {adapters_path}")

    # Apply quantization if requested
    if quantize:
        model = quantize_model(model, quantize)
        logger.info(f"Applied {quantize}-bit quantization")

    mx.eval(model.parameters())

    flags = []
    if config.use_tnt:
        flags.append("TNT")
    if config.use_attn_res:
        flags.append("AttnRes")
    flags.append("LoRA")
    flag_str = f" [{'+'.join(flags)}]"
    logger.info(f"Loaded {model_type.upper()}{flag_str} model with adapters")

    return model, config, model_type, tokenizer_name, chat_template


# =============================================================================
# Text Generation
# =============================================================================


def sample_top_p(probs: mx.array, p: float) -> mx.array:
    """Sample from top-p (nucleus) distribution."""
    sorted_indices = mx.argsort(-probs)
    sorted_probs = mx.take(probs, sorted_indices)
    cumulative_probs = mx.cumsum(sorted_probs)

    # Find cutoff
    cutoff_mask = cumulative_probs <= p
    # Always keep at least one token
    cutoff_mask = mx.concatenate([mx.array([True]), cutoff_mask[:-1]])

    # Zero out low probability tokens
    filtered_probs = mx.where(
        mx.take(cutoff_mask, mx.argsort(sorted_indices)),
        probs,
        mx.array(0.0),
    )

    # Renormalize
    filtered_probs = filtered_probs / (mx.sum(filtered_probs) + 1e-10)

    # Sample
    return mx.random.categorical(mx.log(filtered_probs + 1e-10))


def generate(
    model: nn.Module,
    input_ids: mx.array,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    states: list[MemoryState] | None = None,
    stream: bool = False,
) -> (
    Iterator[tuple[mx.array, list[MemoryState] | None]]
    | tuple[mx.array, list[MemoryState] | None]
):
    """Generate tokens autoregressively.

    Args:
        model: Titans model
        input_ids: Input token IDs (batch, seq)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        eos_token_id: Stop generation at this token
        states: Initial memory states
        stream: If True, yield tokens one by one

    Returns:
        Generated token IDs and final memory states
    """
    generated = input_ids
    chunk_size = getattr(model, "config", None)
    chunk_size = chunk_size.chunk_size if chunk_size else 512

    def _generate_step():
        nonlocal generated, states

        for _ in range(max_new_tokens):
            # Re-run the full context each step with fresh states so that:
            # 1. Attention sees all prior tokens (not just 1)
            # 2. Memory processes tokens exactly once (no redundant updates)
            # For sequences > chunk_size, only re-run the last chunk_size tokens
            # with states carried from prior complete chunks.
            context_len = generated.shape[1]
            tail_len = context_len % chunk_size
            if context_len <= chunk_size:
                # Fits in one chunk — run from scratch with initial states
                logits, final_states = model(generated, states=states)
            elif tail_len == 0:
                # Exact multiple of chunk_size — process all chunks
                logits, final_states = model(generated, states=states)
            else:
                # Process complete chunks to build up memory state,
                # then run the tail chunk for logits
                tail_start = context_len - tail_len
                complete = generated[:, :tail_start]
                chunk_states = states
                logits, chunk_states = model(complete, states=chunk_states)
                # Process tail with those states
                tail = generated[:, tail_start:]
                logits, final_states = model(tail, states=chunk_states)

            mx.eval(logits)

            # Get logits for last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated.shape[0] == 1:
                gen_tokens = set(generated[0].tolist())
                if gen_tokens:
                    vocab_size = next_logits.shape[-1]
                    # Build penalty multiplier: divide positive logits, multiply negative
                    factors = [1.0] * vocab_size
                    for tid in gen_tokens:
                        val = next_logits[0, tid].item()
                        factors[tid] = (
                            1.0 / repetition_penalty if val > 0 else repetition_penalty
                        )
                    next_logits = next_logits * mx.array([factors])

            # Apply top-k filtering
            if top_k > 0 and top_k < next_logits.shape[-1]:
                topk_values = mx.sort(next_logits)[:, -top_k:][:, 0]
                mask = next_logits < topk_values[:, None]
                next_logits = mx.where(mask, mx.array(-float("inf")), next_logits)

            # Softmax to get probabilities
            probs = mx.softmax(next_logits, axis=-1)

            # Apply top-p (nucleus) filtering and sample
            if top_p < 1.0:
                next_token = sample_top_p(probs[0], top_p).reshape(1, 1)
            else:
                next_token = mx.random.categorical(mx.log(probs + 1e-10)).reshape(1, 1)

            mx.eval(next_token)

            # Append to generated
            generated = mx.concatenate([generated, next_token], axis=1)

            # Check for EOS
            if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
                break

            if stream:
                yield generated, final_states

        if not stream:
            yield generated, final_states

    if stream:
        return _generate_step()
    else:
        return next(_generate_step())


def generate_streaming(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    states: list[MemoryState] | None = None,
) -> Iterator[str]:
    """Generate tokens with streaming output.

    Yields decoded text incrementally.
    """
    # Encode prompt
    if hasattr(tokenizer, "__call__"):
        if HAS_TRANSFORMERS and not isinstance(tokenizer, SimpleTokenizer):
            encoded = tokenizer(prompt, return_tensors="np")
            input_ids = mx.array(encoded["input_ids"])
        else:
            encoded = tokenizer(prompt, return_tensors="mlx")
            input_ids = encoded["input_ids"]
    else:
        input_ids = mx.array([tokenizer.encode(prompt)])

    generator = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        states=states,
        stream=True,
    )

    for generated, _ in generator:
        # Decode only the new token
        new_token = generated[0, -1:].tolist()
        text = tokenizer.decode(new_token, skip_special_tokens=True)
        yield text


# =============================================================================
# Benchmark
# =============================================================================


def benchmark_generation(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = "Hello, world!",
    num_tokens: int = 100,
    warmup: int = 2,
    repeat: int = 5,
) -> dict[str, float]:
    """Benchmark generation speed."""
    # Encode prompt
    if HAS_TRANSFORMERS and not isinstance(tokenizer, SimpleTokenizer):
        encoded = tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(encoded["input_ids"])
    else:
        encoded = tokenizer(prompt, return_tensors="mlx")
        input_ids = encoded["input_ids"]

    # Warmup
    for _ in range(warmup):
        output, _ = generate(model, input_ids, max_new_tokens=10)
        mx.eval(output)

    # Benchmark
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        output, _ = generate(model, input_ids, max_new_tokens=num_tokens)
        mx.eval(output)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = num_tokens / avg_time

    return {
        "avg_time_s": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": avg_time / num_tokens * 1000,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Titans MLX models")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Path to LoRA adapters (loads adapter metadata, wraps model)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g., meta-llama/Llama-2-7b-hf)",
    )

    # Chat template arguments
    parser.add_argument(
        "--chat",
        action="store_true",
        default=None,
        help="Force chat template formatting (auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--no-chat",
        dest="chat",
        action="store_false",
        help="Disable chat template formatting",
    )

    # Generation arguments
    parser.add_argument("--prompt", type=str, default="", help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )

    # Mode arguments
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--stream", action="store_true", help="Stream output token by token"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run generation benchmark"
    )

    # Optimization arguments
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantization bits (4 or 8)",
    )
    parser.add_argument(
        "--quantize-memory-state",
        action="store_true",
        default=False,
        help="Enable memory state quantization for reduced memory usage",
    )
    parser.add_argument(
        "--memory-state-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Bit-width for memory state weight quantization (default: 4)",
    )
    parser.add_argument(
        "--adaptive-window", action="store_true",
        help="Enable adaptive window sizing (must match training config)"
    )
    parser.add_argument(
        "--adaptive-window-min", type=int, default=64, help="Min adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-max", type=int, default=None, help="Max adaptive window"
    )
    parser.add_argument(
        "--adaptive-window-temperature", type=float, default=10.0,
        help="Soft mask temperature"
    )

    # Memory persistence arguments
    parser.add_argument(
        "--load-memory",
        type=str,
        default=None,
        help="Load memory state from file at startup",
    )
    parser.add_argument(
        "--save-memory",
        type=str,
        default=None,
        help="Save memory state to file on exit",
    )

    args = parser.parse_args()

    # Load model
    if args.adapters:
        model, config, model_type, saved_tokenizer, chat_template = load_lora_model(
            Path(args.adapters),
            checkpoint_override=args.checkpoint,
            quantize=args.quantize,
        )
    elif args.checkpoint:
        model, config, model_type, saved_tokenizer, chat_template = load_model(
            Path(args.checkpoint), args.quantize
        )
    else:
        parser.error("Either --checkpoint or --adapters is required")

    if args.quantize_memory_state:
        config.quantize_memory_state = True
        config.memory_state_weight_bits = args.memory_state_bits
        config.memory_state_momentum_bits = min(args.memory_state_bits * 2, 8)

    if args.adaptive_window:
        config.adaptive_window = True
        config.adaptive_window_min = args.adaptive_window_min
        config.adaptive_window_max = args.adaptive_window_max
        config.adaptive_window_temperature = args.adaptive_window_temperature

    # Load tokenizer (prefer command line, fallback to saved, then simple)
    tokenizer_name = args.tokenizer or saved_tokenizer
    tokenizer = load_tokenizer(tokenizer_name, config.vocab_size)

    # Determine chat mode
    use_chat = should_use_chat(chat_template, args.chat)
    if use_chat:
        ensure_chat_tokens(tokenizer)
        im_end_id = get_im_end_token_id(tokenizer)
        logger.info(f"Chat mode enabled (template: {chat_template})")
    else:
        im_end_id = None

    if args.benchmark:
        # Run benchmark
        logger.info("Running generation benchmark...")
        results = benchmark_generation(
            model, tokenizer, args.prompt or "Hello, world!", args.max_tokens
        )
        print("\n" + "=" * 50)
        print("  Generation Benchmark Results")
        print("=" * 50)
        print(f"  Tokens generated: {args.max_tokens}")
        print(f"  Average time: {results['avg_time_s']:.3f}s")
        print(f"  Tokens/sec: {results['tokens_per_sec']:.1f}")
        print(f"  ms/token: {results['ms_per_token']:.2f}")
        print("=" * 50)
        return

    if args.interactive:
        # Interactive mode
        logger.info(
            "Interactive mode. Commands: 'quit', 'reset', 'save [path]', 'load [path]'."
        )
        states = None

        # Load memory state from file if requested
        if args.load_memory:
            try:
                states = load_memory_states(Path(args.load_memory))
                logger.info(f"Loaded memory state from {args.load_memory}")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load memory state: {e}")

        while True:
            try:
                prompt = input("\nYou: ")
                cmd = prompt.strip().lower()
                if cmd == "quit":
                    break
                if cmd == "reset":
                    states = None
                    logger.info("Memory cleared.")
                    continue
                if cmd.startswith("save"):
                    parts = prompt.strip().split(maxsplit=1)
                    save_path = parts[1] if len(parts) > 1 else "memory_state.npz"
                    if states is None:
                        logger.warning(
                            "No memory state to save (run some prompts first)."
                        )
                    else:
                        save_memory_states(states, Path(save_path))
                        logger.info(f"Memory state saved to {save_path}")
                    continue
                if cmd.startswith("load"):
                    parts = prompt.strip().split(maxsplit=1)
                    load_path = parts[1] if len(parts) > 1 else "memory_state.npz"
                    try:
                        states = load_memory_states(Path(load_path))
                        logger.info(f"Loaded memory state from {load_path}")
                    except (FileNotFoundError, ValueError) as e:
                        logger.error(f"Failed to load memory state: {e}")
                    continue

                gen_prompt = format_prompt_for_chat(prompt) if use_chat else prompt
                eos_id = im_end_id or getattr(tokenizer, "eos_token_id", None)

                if args.stream:
                    # Streaming output
                    print("Model: ", end="", flush=True)
                    start_time = time.time()
                    token_count = 0
                    collected = []

                    for text in generate_streaming(
                        model,
                        tokenizer,
                        gen_prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        states=states,
                    ):
                        collected.append(text)
                        print(text, end="", flush=True)
                        token_count += 1

                    elapsed = time.time() - start_time
                    if use_chat:
                        # Print cleaned version on new line
                        raw = "".join(collected)
                        cleaned = strip_chat_delimiters(raw)
                        if cleaned != raw:
                            print(f"\n[cleaned: {cleaned}]")
                    print(
                        f"\n[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
                    )
                else:
                    # Normal generation
                    if HAS_TRANSFORMERS and not isinstance(tokenizer, SimpleTokenizer):
                        encoded = tokenizer(gen_prompt, return_tensors="np")
                        input_ids = mx.array(encoded["input_ids"])
                    else:
                        encoded = tokenizer(gen_prompt, return_tensors="mlx")
                        input_ids = encoded["input_ids"]

                    start_time = time.time()
                    output_ids, states = generate(
                        model,
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        eos_token_id=eos_id,
                        states=states,
                    )
                    mx.eval(output_ids)
                    elapsed = time.time() - start_time

                    generated_text = tokenizer.decode(
                        output_ids[0].tolist(),
                        skip_special_tokens=True,
                    )
                    if use_chat:
                        generated_text = strip_chat_delimiters(generated_text)
                    new_tokens = output_ids.shape[1] - input_ids.shape[1]
                    print(f"Model: {generated_text}")
                    print(
                        f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
                    )

            except KeyboardInterrupt:
                break

        # Auto-save on exit if --save-memory is set
        if args.save_memory and states is not None:
            save_memory_states(states, Path(args.save_memory))
            logger.info(f"Memory state saved to {args.save_memory}")

        logger.info("Goodbye!")

    else:
        # Single generation
        if not args.prompt:
            logger.warning("No prompt provided. Using empty prompt.")

        gen_prompt = format_prompt_for_chat(args.prompt) if use_chat else args.prompt
        eos_id = im_end_id or getattr(tokenizer, "eos_token_id", None)

        if HAS_TRANSFORMERS and not isinstance(tokenizer, SimpleTokenizer):
            encoded = tokenizer(gen_prompt, return_tensors="np")
            input_ids = mx.array(encoded["input_ids"])
        else:
            encoded = tokenizer(gen_prompt, return_tensors="mlx")
            input_ids = encoded["input_ids"]

        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generating {args.max_tokens} tokens...")

        start_time = time.time()

        if args.stream:
            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(args.prompt, end="", flush=True)

            token_count = 0
            collected = []
            for text in generate_streaming(
                model,
                tokenizer,
                gen_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                collected.append(text)
                print(text, end="", flush=True)
                token_count += 1

            elapsed = time.time() - start_time
            if use_chat:
                raw = "".join(collected)
                cleaned = strip_chat_delimiters(raw)
                if cleaned != raw:
                    print(f"\n[cleaned: {cleaned}]")
            print(f"\n{'=' * 50}")
            print(
                f"[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
            )
        else:
            output_ids, _ = generate(
                model,
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=eos_id,
            )
            mx.eval(output_ids)
            elapsed = time.time() - start_time

            generated_text = tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True,
            )
            if use_chat:
                generated_text = strip_chat_delimiters(generated_text)

            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)

            new_tokens = output_ids.shape[1] - input_ids.shape[1]
            print(
                f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
            )


if __name__ == "__main__":
    main()
