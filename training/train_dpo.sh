#!/bin/bash
# DPO training with LoRA (recommended for memory efficiency)
uv run python scripts/dpo.py \
    --model mac \
    --dataset allenai/Dolci-Instruct-DPO \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --method dpo \
    --beta 0.1 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --dim 512 \
    --num-layers 12 \
    --num-heads 8 \
    --max-len 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 5e-7 \
    --warmup-ratio 0.1 \
    --grad-clip 1.0 \
    --epochs 3 \
    --save-every 1000 \
    --eval-every 500 \
    --checkpoint-dir checkpoints/dpo-lora \
    --seed 42
