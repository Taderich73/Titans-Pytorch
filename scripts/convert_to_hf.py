#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Deprecated -- use ``scripts/convert.py --to hf`` instead.

Thin shim that forwards the legacy ``convert_to_hf.py`` CLI to the
unified ``scripts/convert.py`` entry point. The shim preserves the
original flag set so existing workflows keep running while emitting a
``DeprecationWarning`` on every invocation. Will be removed in 0.8.

For back-compat with ``tests/test_hf_convert.py``, the module re-exports
``remap_state_dict_keys``, ``convert_checkpoint`` (aliased to
``convert_to_hf``), ``MODEL_REGISTRY``, and ``CHATML_TEMPLATE`` from the
new ``scripts/convert.py`` module.

Example migration:

    # Old
    uv run python scripts/convert_to_hf.py --checkpoint ckpt.pt \\
        --output-dir ./hf --tokenizer gpt2

    # New
    uv run python scripts/convert.py ckpt.pt --to hf \\
        --output-dir ./hf --tokenizer gpt2
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Ensure the sibling scripts/ directory is importable, then surface the
# shared helpers from the unified script so legacy ``from convert_to_hf
# import convert_checkpoint`` (used by tests/test_hf_convert.py) keeps
# working through the deprecation window.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import convert as _convert  # noqa: E402

CHATML_TEMPLATE = _convert.CHATML_TEMPLATE
MODEL_REGISTRY = _convert.MODEL_REGISTRY
remap_state_dict_keys = _convert.remap_state_dict_keys
convert_to_hf = _convert.convert_to_hf
# Legacy entry point name (kept for back-compat with the old tests).
convert_checkpoint = _convert.convert_to_hf


def _parse_legacy_args(argv: list[str]) -> argparse.Namespace:
    """Re-parse the legacy CLI so we can translate it to the new one."""
    parser = argparse.ArgumentParser(
        description="[deprecated] Convert Titans checkpoints to HuggingFace format.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        default="mac",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default="float32")
    parser.add_argument("--add-chat-template", action="store_true")
    parser.add_argument("--push-to-hub", type=str, default=None)
    parser.add_argument("--upload-model-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Forward to ``scripts/convert.py --to hf`` with a deprecation warning."""
    warnings.warn(
        "scripts/convert_to_hf.py is deprecated; use "
        "`scripts/convert.py <checkpoint> --to hf --output-dir <dir>` instead. "
        "This shim will be removed in 0.8.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "[deprecated] scripts/convert_to_hf.py is a shim; forwarding to "
        "scripts/convert.py --to hf. Will be removed in 0.8.",
        file=sys.stderr,
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = _parse_legacy_args(argv)

    new_argv: list[str] = [
        legacy.checkpoint,
        "--to", "hf",
        "--output-dir", legacy.output_dir,
        "--model-type", legacy.model_type,
        "--torch-dtype", legacy.torch_dtype,
    ]
    if legacy.tokenizer is not None:
        new_argv.extend(["--tokenizer", legacy.tokenizer])
    if legacy.add_chat_template:
        new_argv.append("--add-chat-template")
    if legacy.push_to_hub is not None:
        new_argv.extend(["--push-to-hub", legacy.push_to_hub])
    if legacy.upload_model_code:
        new_argv.append("--upload-model-code")

    return _convert.main(new_argv)


if __name__ == "__main__":
    raise SystemExit(main())
