#!/bin/bash
# Example: supervised fine-tuning from a pretrained AttnRes checkpoint.

uv run python scripts/sft.py --model mac \
  --init-weights checkpoints/small-attnres/best_model \
  --use-attn-res \
  --dataset allenai/Dolci-Instruct-SFT \
  --tokenizer openai-community/gpt2 \
  --lr 2e-5
