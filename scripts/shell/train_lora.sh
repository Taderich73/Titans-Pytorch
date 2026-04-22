#!/bin/bash
# Example: LoRA fine-tuning on an instruction-tuning dataset.
# Point --init-weights at a pretrained checkpoint, and swap --dataset /
# --tokenizer for your own data.

uv run python scripts/lora.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer openai-community/gpt2 \
    --lora-targets attn --lora-rank 8 --lora-alpha 16 \
    --lr 1e-4
