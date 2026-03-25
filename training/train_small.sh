#!/bin/bash
# Phase 2: Pretrain 536M param Titans MAC model
# dim=1024, 16 layers, 8 heads
# Effective batch: 32,768 tokens (batch=2 * accum=16 * seq=1024)
# Expected: ~6-8s per micro-step on M1 Max -> ~2min per optimizer step
# 10K steps ≈ ~14 days. Adjust --max-steps as needed.
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-10BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 12 --num-heads 8 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --seq-len 2048 --chunk-size 512 \
  --max-steps 10000 \
  --lr 4e-4 \
  --eval-every 200 --eval-buffer-size 100 \
  --save-every 200 \
  --log-every 10 \
  --dtype bfloat16 \
  --checkpoint-dir checkpoints/small
