#!/bin/bash
# Pretrain small Titans MAC model with TNT hierarchical memory + AttnRes
# dim=512, 12 layers, 8 heads
# TNT: global memory + 2 local memories (chunk sizes 8, 16)
# AttnRes: 4 blocks, 200-step warmup before memory gating activates
# Effective batch: 32,768 tokens (batch=2 * accum=16 * seq=1024)
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
  --use-tnt \
  --local-chunk-sizes 8 16 \
  --local-shard-length 2048 \
  --global-chunk-size 2048 \
  --checkpoint-dir checkpoints/small-tnt
