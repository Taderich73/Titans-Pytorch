#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Convert Titans checkpoints between on-disk formats.

Single entry point that supersedes the legacy
``scripts/convert_checkpoint.py`` (pt <-> safetensors) and
``scripts/convert_to_hf.py`` (native -> HuggingFace directory) pair. The
``--to`` flag selects the target format.

Examples:
    # Default (safetensors) -- toggles between .pt and .safetensors based
    # on the input extension, preserving metadata.
    uv run python scripts/convert.py checkpoints/final.pt

    # Explicit target (identical behavior for pt/safetensors -- the script
    # toggles to the other format when ``--to`` matches the input).
    uv run python scripts/convert.py checkpoints/final.pt --to safetensors
    uv run python scripts/convert.py checkpoints/final.safetensors --to pt

    # HuggingFace directory (previously convert_to_hf.py).
    uv run python scripts/convert.py checkpoints/final.pt --to hf \\
        --output-dir ./hf_model --tokenizer gpt2

Migration note:
    ``scripts/convert_checkpoint.py`` and ``scripts/convert_to_hf.py`` are
    preserved as deprecated shims that forward to this script. They will
    be removed in 0.8.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow running from repo root without pre-install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from titans import TITANS_SCHEMA_VERSION
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.config import TitansConfig
from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Registry of HF model types -> (config_cls, model_cls)
# Extend this dict when adding MAG/MAL/LMM variants.
MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "mac": (TitansMACConfig, TitansMACForCausalLM),
}


CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


# ---------------------------------------------------------------------------
# pt <-> safetensors path
# ---------------------------------------------------------------------------


def convert_format(
    input_path: Path,
    target_format: str,
    output_stem: Path | None = None,
    weights_only: bool = False,
) -> list[Path]:
    """Convert a checkpoint between ``.pt`` and ``.safetensors`` formats.

    Args:
        input_path: Input checkpoint (``.pt`` or ``.safetensors``).
        target_format: Either ``"pt"`` or ``"safetensors"``. If it matches
            the input extension, the output is the opposite format (matches
            the legacy ``convert_checkpoint.py`` toggle behavior).
        output_stem: Optional output stem (without extension). Defaults to
            the input path with its extension stripped.
        weights_only: If True, drop optimizer/scheduler metadata and only
            persist model weights.

    Returns:
        List of :class:`~pathlib.Path` objects for every file written.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    # Toggle behavior: when target matches the input extension, emit the
    # opposite format. This preserves the legacy ``convert_checkpoint.py``
    # semantics where the default was "flip the format".
    input_ext = input_path.suffix
    if target_format == "pt" and input_ext == ".pt":
        effective_target = "safetensors"
    elif target_format == "safetensors" and input_ext == ".safetensors":
        effective_target = "pt"
    else:
        effective_target = target_format

    if effective_target not in ("pt", "safetensors"):
        raise ValueError(
            f"Unsupported target format {effective_target!r}. "
            "Must be 'pt' or 'safetensors'."
        )

    stem = Path(output_stem) if output_stem is not None else input_path.with_suffix("")

    logger.info("Loading %s ...", input_path)
    ckpt = load_checkpoint(input_path, weights_only=False)

    metadata: dict | None = None
    if not weights_only:
        metadata = {k: v for k, v in ckpt.items() if k != "model"}
        if metadata:
            logger.info("Metadata keys: %s", list(metadata.keys()))

    paths = save_checkpoint(
        ckpt["model"],
        stem,
        format=effective_target,
        metadata=metadata if metadata else None,
    )
    for p in paths:
        logger.info("Written: %s", p)
    return paths


# ---------------------------------------------------------------------------
# HF directory path (ported verbatim from convert_to_hf.py so the
# semantics -- key remapping, tied-weight cloning, generation config --
# stay identical across the transition).
# ---------------------------------------------------------------------------


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add ``model.`` prefix to native checkpoint keys.

    Native keys:  ``embed.weight``, ``blocks.0.memory...``
    HF keys:      ``model.embed.weight``, ``model.blocks.0.memory...``
    """
    return {f"model.{k}": v for k, v in state_dict.items()}


def convert_to_hf(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_type: str = "mac",
    tokenizer_name: str | None = None,
    torch_dtype: str = "float32",
    add_chat_template: bool = False,
    push_to_hub: str | None = None,
    upload_model_code: bool = False,
) -> None:
    """Convert a native Titans checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to native .pt or .safetensors checkpoint.
        output_dir: Directory to write the HF model files to.
        model_type: Model variant key (default: "mac").
        tokenizer_name: HF tokenizer to include (e.g. "gpt2"). None to skip.
        torch_dtype: Dtype string for config metadata.
        add_chat_template: If True, add ChatML special tokens and template.
        push_to_hub: Hub repo ID to push to. None to skip.
        upload_model_code: If True, upload Python source for trust_remote_code.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config_cls, model_cls = MODEL_REGISTRY[model_type]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path, weights_only=False)

    # 2. Extract config
    if "config" in ckpt:
        titans_config = TitansConfig.from_dict(ckpt["config"])
    elif "titans_config" in ckpt:
        titans_config = TitansConfig.from_dict(ckpt["titans_config"])
    else:
        raise ValueError(
            "Checkpoint does not contain config metadata. "
            "Pass config values via CLI args."
        )

    # 3. Create HF config
    hf_config = config_cls.from_titans_config(titans_config)
    hf_config.torch_dtype = torch_dtype
    hf_config.architectures = [model_cls.__name__]
    hf_config.auto_map = {
        "AutoConfig": f"titans.hf.configuration.{config_cls.__name__}",
        "AutoModelForCausalLM": f"titans.hf.modeling.{model_cls.__name__}",
    }
    # Stamp the schema version explicitly so the emitted config.json
    # carries it even when an older TitansConfig was read from disk (the
    # kwarg default does this too, but being explicit keeps the
    # convert-script contract obvious when the output is inspected).
    hf_config.titans_schema_version = TITANS_SCHEMA_VERSION

    # 4. Remap state dict keys
    remapped = remap_state_dict_keys(ckpt["model"])

    # 5. Save config
    hf_config.save_pretrained(str(output_path))
    logger.info("Saved config.json to %s", output_path)

    # 6. Save model weights as safetensors
    from safetensors.torch import save_file

    # Handle tied weights: clone shared tensors to avoid safetensors errors
    seen: dict[int, str] = {}
    prepared: dict[str, torch.Tensor] = {}
    for k, v in remapped.items():
        data_ptr = v.data_ptr()
        if data_ptr in seen:
            prepared[k] = v.clone().contiguous()
        else:
            seen[data_ptr] = k
            prepared[k] = v.contiguous()

    sf_path = output_path / "model.safetensors"
    save_file(prepared, sf_path)
    logger.info("Saved model.safetensors (%.1f MB)", sf_path.stat().st_size / 1e6)

    # 7. Save tokenizer (optional)
    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if add_chat_template:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
                }
            )
            tokenizer.chat_template = CHATML_TEMPLATE

        tokenizer.save_pretrained(str(output_path))
        logger.info("Saved tokenizer files to %s", output_path)

    # 8. Save generation config
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=1.0,
        do_sample=True,
        eos_token_id=titans_config.vocab_size - 1,
        pad_token_id=titans_config.vocab_size - 1,
    )
    gen_config.save_pretrained(str(output_path))
    logger.info("Saved generation_config.json to %s", output_path)

    # 9. Push to Hub
    if push_to_hub is not None:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(push_to_hub, exist_ok=True)

        if upload_model_code:
            src_dir = Path(__file__).resolve().parent.parent / "src" / "titans" / "hf"
            for py_file in src_dir.glob("*.py"):
                api.upload_file(
                    path_or_fileobj=str(py_file),
                    path_in_repo=py_file.name,
                    repo_id=push_to_hub,
                )

        api.upload_folder(
            folder_path=str(output_path),
            repo_id=push_to_hub,
        )
        logger.info("Pushed to Hub: %s", push_to_hub)

    logger.info("Conversion complete.")


# Back-compat alias -- ``tests/test_hf_convert.py`` imports
# ``convert_checkpoint`` from the old script by name. The shim re-exports
# this alias so the test keeps working during the deprecation window.
convert_checkpoint = convert_to_hf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the unified convert CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert Titans checkpoints between on-disk formats "
            "(pt <-> safetensors <-> HuggingFace directory)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Input checkpoint path (.pt or .safetensors).",
    )
    parser.add_argument(
        "--to",
        dest="target",
        choices=("pt", "safetensors", "hf"),
        default="safetensors",
        help=(
            "Output format. Default: safetensors. For pt/safetensors the "
            "script toggles to the opposite format when ``--to`` matches "
            "the input extension, matching legacy convert_checkpoint.py "
            "behavior."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path stem for --to pt/safetensors "
            "(default: same directory and stem as input)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (required for --to hf).",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="For --to pt/safetensors: skip optimizer/scheduler metadata.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mac",
        choices=list(MODEL_REGISTRY.keys()),
        help="HF model variant (default: mac). Only used with --to hf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer identifier/path (e.g. gpt2). Only used with --to hf.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float32",
        help="Dtype for HF config metadata. Only used with --to hf.",
    )
    parser.add_argument(
        "--add-chat-template",
        action="store_true",
        help="Add ChatML special tokens and template. Only used with --to hf.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Hub repo ID to push to. Only used with --to hf.",
    )
    parser.add_argument(
        "--upload-model-code",
        action="store_true",
        help=(
            "Upload Python source files for trust_remote_code. Only used with --to hf."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch on ``--to`` and run the corresponding converter."""
    args = build_parser().parse_args(argv)

    if args.target in ("pt", "safetensors"):
        convert_format(
            input_path=args.checkpoint,
            target_format=args.target,
            output_stem=args.output,
            weights_only=args.weights_only,
        )
    elif args.target == "hf":
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for --to hf")
        convert_to_hf(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            model_type=args.model_type,
            tokenizer_name=args.tokenizer,
            torch_dtype=args.torch_dtype,
            add_chat_template=args.add_chat_template,
            push_to_hub=args.push_to_hub,
            upload_model_code=args.upload_model_code,
        )
    else:  # pragma: no cover - argparse ``choices`` enforces this already
        raise SystemExit(f"unknown --to value {args.target!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
