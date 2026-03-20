#!/usr/bin/env python3
"""Convert structured JSONL dataset to plain text for pretraining/fine-tuning.

Reads a JSONL file where each line is a JSON object with fields like
topic, question, ethical_considerations, balanced_perspective, and
key_principles. Outputs a plain text file suitable for next-token
prediction training via `pretrain_mlx.py --data <output.txt>`.

Usage:
    python scripts/prepare_jsonl.py input.jsonl -o output.txt
    python scripts/prepare_jsonl.py input.jsonl -o output.txt --format raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_structured(entry: dict) -> str:
    """Format an entry using structured field extraction."""
    parts: list[str] = []

    topic = entry.get("topic", "")
    if topic:
        parts.append(f"Topic: {topic}")

    question = entry.get("question", "")
    if question:
        parts.append(f"Question: {question}")

    considerations = entry.get("ethical_considerations", "")
    if considerations:
        parts.append(f"Ethical Considerations:\n{considerations}")

    perspective = entry.get("balanced_perspective", "")
    if perspective:
        parts.append(f"Balanced Perspective:\n{perspective}")

    principles = entry.get("key_principles", [])
    if principles:
        formatted = "\n".join(f"- {p}" for p in principles)
        parts.append(f"Key Principles:\n{formatted}")

    return "\n\n".join(parts)


def format_raw(entry: dict) -> str:
    """Use the metadata.raw_response field directly."""
    metadata = entry.get("metadata", {})
    raw = metadata.get("raw_response", "")
    if raw:
        return raw.strip()
    # Fall back to structured format if raw_response is missing
    return format_structured(entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert structured JSONL to plain text for training"
    )
    parser.add_argument("input", type=str, help="Path to input JSONL file")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output text file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["structured", "raw"],
        default="structured",
        help="Output format: 'structured' extracts fields with labels, "
        "'raw' uses metadata.raw_response (default: structured)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n---\n\n",
        help="Separator between entries (default: '\\n\\n---\\n\\n')",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    formatter = format_raw if args.format == "raw" else format_structured

    entries: list[str] = []
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                text = formatter(entry)
                if text:
                    entries.append(text)
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_num}: {e}", file=sys.stderr)
                skipped += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(args.separator.join(entries))

    total_chars = sum(len(e) for e in entries)
    approx_tokens = total_chars // 4  # rough estimate

    print(f"Processed {len(entries)} entries ({skipped} skipped)")
    print(f"Output: {output_path} ({total_chars:,} chars, ~{approx_tokens:,} tokens)")


if __name__ == "__main__":
    main()
