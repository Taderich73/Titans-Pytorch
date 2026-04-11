#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Pre-tokenize a HuggingFace dataset to disk for faster training.

Tokenizes all texts once and saves to .npy shards, eliminating the
tokenization bottleneck during training.

Tokens are packed into fixed-length (seq_len + 1) windows for
next-token-prediction. Incomplete trailing tokens are discarded.

Usage:
    # Pre-tokenize FineWeb-Edu (streaming, avoids full download)
    uv run python scripts/pretokenize.py \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --subset sample-10BT \\
        --tokenizer NousResearch/Llama-2-7b-hf \\
        --output data/fineweb-tokenized \\
        --seq-len 2048

    # Then train with:
    uv run python scripts/pretrain.py \\
        --local-dataset data/fineweb-tokenized \\
        --tokenizer NousResearch/Llama-2-7b-hf \\
        --seq-len 2048 \\
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _save_shard(sequences: list[list[int]], output_path: Path, shard_idx: int) -> None:
    """Write a list of token sequences to a .npy shard file.

    Args:
        sequences: List of token ID lists, each of length seq_len+1.
        output_path: Directory to write the shard into.
        shard_idx: Zero-based shard index (used in filename).
    """
    arr = np.array(sequences, dtype=np.int32)
    shard_path = output_path / f"shard_{shard_idx:05d}.npy"
    np.save(str(shard_path), arr)
    logger.info(f"  Saved shard {shard_idx:05d}: {arr.shape} -> {shard_path}")


def pretokenize(
    dataset_name: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    seq_len: int = 2048,
    subset: Optional[str] = None,
    split: str = "train",
    max_tokens: int = -1,
    shard_size: int = 100_000,
    text_field: str = "text",
) -> dict:
    """Pre-tokenize a HuggingFace dataset and write packed .npy shards.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb-edu').
        output_dir: Directory to write shards and metadata.
        tokenizer_name: HuggingFace tokenizer name or local path.
        seq_len: Sequence length for each packed window (input+label = seq_len+1).
        subset: Dataset config/subset name, or None for the default.
        split: Dataset split to tokenize (default: 'train').
        max_tokens: Maximum total tokens to process. -1 means no limit.
        shard_size: Maximum number of sequences per shard file.
        text_field: Column name to read text from (falls back to 'content').

    Returns:
        Metadata dict with sequence counts, token counts, and shard info.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer: {tokenizer_name} (vocab_size={tokenizer.vocab_size})")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {dataset_name} (subset={subset!r}, split={split!r}, streaming=True)")
    ds = load_dataset(dataset_name, subset, split=split, streaming=True)

    buffer: list[int] = []
    shard_idx = 0
    total_tokens = 0
    total_sequences = 0
    sequences: list[list[int]] = []
    raw_text_chars = 0

    for example in tqdm(ds, desc="Tokenizing"):
        text: str = example.get(text_field, example.get("content", ""))
        if not text:
            continue

        raw_text_chars += len(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)

        # Pack buffer into fixed-length windows (stride = seq_len, not seq_len+1)
        while len(buffer) >= seq_len + 1:
            sequences.append(buffer[: seq_len + 1])
            buffer = buffer[seq_len:]  # overlap by 1 for next-token targets
            total_tokens += seq_len + 1
            total_sequences += 1

            if len(sequences) >= shard_size:
                _save_shard(sequences, output_path, shard_idx)
                shard_idx += 1
                sequences = []

        if max_tokens > 0 and total_tokens >= max_tokens:
            logger.info(f"Reached max_tokens={max_tokens:,}. Stopping.")
            break

    # Flush remaining complete sequences
    if sequences:
        _save_shard(sequences, output_path, shard_idx)
        shard_idx += 1

    total_shards = shard_idx

    # Compression ratio: raw chars vs token storage
    token_bytes = total_tokens * 4  # int32
    compression_ratio = raw_text_chars / max(token_bytes, 1)

    metadata = {
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": seq_len,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "total_shards": total_shards,
        "shard_size": shard_size,
        "token_bytes": token_bytes,
        "raw_text_chars": raw_text_chars,
        "compression_ratio": round(compression_ratio, 3),
    }

    meta_path = output_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata written to {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a HuggingFace dataset to .npy shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., HuggingFaceFW/fineweb-edu).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset config/subset name (e.g., sample-10BT).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to tokenize.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for .npy shards and metadata.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for each packed window (seq_len+1 tokens stored).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="Maximum total tokens to process. -1 means no limit.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Maximum number of sequences per shard file.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Column name containing the text to tokenize (fallback: 'content').",
    )

    args = parser.parse_args()

    meta = pretokenize(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        subset=args.subset,
        split=args.split,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        text_field=args.text_field,
    )

    logger.info("=" * 55)
    logger.info("Pre-tokenization complete!")
    logger.info(f"  Sequences : {meta['total_sequences']:>15,}")
    logger.info(f"  Tokens    : {meta['total_tokens']:>15,}  ({meta['total_tokens'] / 1e9:.3f}B)")
    logger.info(f"  Shards    : {meta['total_shards']:>15,}")
    logger.info(f"  Token bytes: {meta['token_bytes']:>14,}  ({meta['token_bytes'] / 1024**3:.2f} GB)")
    logger.info(f"  Compression ratio (chars/bytes): {meta['compression_ratio']:.2f}x")
    logger.info(f"  Output    : {args.output_dir}")
    logger.info("=" * 55)
    logger.info("")
    logger.info("Load shards in training with:")
    logger.info(f"    np.load('{args.output_dir}/shard_00000.npy')  # shape (N, {args.seq_len + 1})")


if __name__ == "__main__":
    main()
