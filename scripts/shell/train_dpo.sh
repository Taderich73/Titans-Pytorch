#!/bin/bash
# DPO training with LoRA (recommended for memory efficiency)
uv run python scripts/dpo.py \
    --model mac \
    --dataset allenai/Dolci-Instruct-DPO \
    --tokenizer openai-community/gpt2 \
    --loss-type dpo \
    --beta 0.1 \
    --use-lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --dim 512 \
    --num-layers 12 \
    --num-heads 8 \
    --seq-len 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 5e-7 \
    --warmup-ratio 0.1 \
    --grad-clip 1.0 \
    --epochs 3 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/dpo-lora \
    --seed 42
