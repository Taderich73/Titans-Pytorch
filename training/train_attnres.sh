#!/bin/bash
# Pretrain Titans MAC model with AttnRes
# dim=768, 16 layers, 16 heads
# Effective batch: ~98K tokens (batch=2 * accum=24 * seq=2048)
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-10BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 16 --num-heads 16 \
  --batch-size 2 --gradient-accumulation-steps 24 \
  --seq-len 2048 --chunk-size 512 \
  --max-steps 10000 \
  --lr 4e-4 \
  --eval-every 200 --eval-buffer-size 100 \
  --save-every 200 \
  --log-every 10 \
  --dtype bfloat16 \
  --use-attn-res \
  --num-attnres-blocks 8 \
  --attnres-warmup-steps 200 \
  --checkpoint-dir checkpoints/mac-attnres
