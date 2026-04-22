#!/bin/bash
# Example: generate text from a trained checkpoint.
# Adjust --checkpoint, --tokenizer, --device, and --prompt for your setup.

uv run python scripts/inference.py \
  --checkpoint checkpoints/final.pt \
  --tokenizer gpt2 --device mps \
  --prompt "The capital of France is" \
  --temperature 0.7 --top-k 50 --max-new-tokens 60
