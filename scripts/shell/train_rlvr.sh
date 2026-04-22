#!/bin/bash
# RLVR training with GRPO (offline mode)
uv run python scripts/rlvr.py \
    --model mac \
    --dataset allenai/Dolci-Think-RL-7B \
    --tokenizer openai-community/gpt2 \
    --offline \
    --loss-type grpo \
    --group-size 8 \
    --clip-ratio 0.2 \
    --dim 512 \
    --num-layers 12 \
    --num-heads 8 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lr 1e-6 \
    --warmup-ratio 0.03 \
    --grad-clip 1.0 \
    --max-steps 5000 \
    --save-every 1000 \
    --checkpoint-dir checkpoints/rlvr-grpo \
    --seed 42
