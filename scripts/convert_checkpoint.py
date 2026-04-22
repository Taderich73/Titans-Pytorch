#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Deprecated -- use ``scripts/convert.py`` instead.

Thin shim that forwards the legacy ``convert_checkpoint.py`` CLI to the
unified ``scripts/convert.py --to safetensors`` entry point. The shim
preserves the original ``<input> [--output <stem>] [--weights-only]``
signature so existing workflows keep running while emitting a
``DeprecationWarning`` on every invocation. Will be removed in 0.8.

Example migration:

    # Old
    uv run python scripts/convert_checkpoint.py ckpt.pt

    # New
    uv run python scripts/convert.py ckpt.pt --to safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path


def _parse_legacy_args(argv: list[str]) -> argparse.Namespace:
    """Re-parse the legacy CLI so we can translate it to the new one."""
    parser = argparse.ArgumentParser(
        description="[deprecated] Convert checkpoints between .pt and .safetensors.",
    )
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--weights-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Forward to ``scripts/convert.py`` with a deprecation warning."""
    warnings.warn(
        "scripts/convert_checkpoint.py is deprecated; use "
        "`scripts/convert.py <input> --to safetensors` instead. "
        "This shim will be removed in 0.8.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Print to stderr too -- tests and users without warnings filters
    # still want to see the notice.
    print(
        "[deprecated] scripts/convert_checkpoint.py is a shim; forwarding "
        "to scripts/convert.py --to safetensors. Will be removed in 0.8.",
        file=sys.stderr,
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = _parse_legacy_args(argv)

    # Translate to the new CLI. Legacy script toggled between pt and
    # safetensors based on the input extension; ``scripts/convert.py``
    # preserves that toggle when --to matches the input extension, so we
    # can unconditionally pass --to safetensors.
    new_argv: list[str] = [legacy.input, "--to", "safetensors"]
    if legacy.output is not None:
        new_argv.extend(["--output", legacy.output])
    if legacy.weights_only:
        new_argv.append("--weights-only")

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import convert  # noqa: PLC0415 -- deferred so the warning fires first

    return convert.main(new_argv)


if __name__ == "__main__":
    raise SystemExit(main())
