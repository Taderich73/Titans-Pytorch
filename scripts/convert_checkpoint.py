#!/usr/bin/env python3
"""Convert checkpoints between .pt and .safetensors formats.

Usage:
    python scripts/convert_checkpoint.py checkpoints/final.pt
    python scripts/convert_checkpoint.py checkpoints/final.safetensors
    python scripts/convert_checkpoint.py checkpoints/final.pt --output other/model
    python scripts/convert_checkpoint.py checkpoints/final.pt --weights-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from titans.checkpoint import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and run conversion."""
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between .pt and .safetensors formats."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input checkpoint path (.pt or .safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path without extension (default: same directory and stem "
            "as input, with the target format extension)."
        ),
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only convert model weights — skip optimizer/scheduler metadata.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input checkpoint not found: {input_path}")
        sys.exit(1)

    # Determine target format (opposite of input)
    if input_path.suffix == ".pt":
        target_format = "safetensors"
    elif input_path.suffix == ".safetensors":
        target_format = "pt"
    else:
        logger.error(
            f"Cannot determine format from extension: {input_path.suffix!r}. "
            "Use .pt or .safetensors."
        )
        sys.exit(1)

    output_stem = Path(args.output) if args.output else input_path.with_suffix("")

    # Load
    logger.info(f"Loading {input_path} ...")
    ckpt = load_checkpoint(input_path, weights_only=False)

    # Prepare metadata. ``titans_schema_version`` is preserved when
    # present on the input and stamped to the current version by
    # ``save_checkpoint`` otherwise — see titans.checkpoint for details.
    metadata = None
    if not args.weights_only:
        metadata = {k: v for k, v in ckpt.items() if k != "model"}
        if metadata:
            logger.info(f"Metadata keys: {list(metadata.keys())}")

    # Save in target format. ``save_checkpoint`` always emits
    # ``titans_schema_version`` (top-level for .pt, via the .meta.pt
    # sidecar for safetensors) so the converted output is
    # self-describing regardless of what was in the input.
    paths = save_checkpoint(
        ckpt["model"],
        output_stem,
        format=target_format,
        metadata=metadata if metadata else None,
    )

    for p in paths:
        logger.info(f"Written: {p}")

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
