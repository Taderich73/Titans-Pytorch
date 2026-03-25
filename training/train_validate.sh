#!/bin/bash
# Phase 1: Validate TNT architecture works
# 121M params, ~500 steps, should show clear loss decrease
# Expected runtime: ~1-2 hours on M1 Max
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-10BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 12 --num-heads 8 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --seq-len 1024 --chunk-size 512 \
  --max-steps 500 \
  --lr 6e-4 \
  --eval-every 100 --eval-buffer-size 50 \
  --save-every 250 \
  --log-every 5 \
  --dtype bfloat16 \
  --checkpoint-dir checkpoints/validate
  
  #--use-tnt \
  #--local-chunk-sizes 8 16 \
  #--local-shard-length 2048 \
  #--global-chunk-size 2048 \
  #--use-attn-res \
  #--num-attnres-blocks 4 \
  #--attnres-warmup-steps 100 \
