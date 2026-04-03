#!/bin/bash
# Pretrain Titans MAC model with Yaad (Huber attentional bias)
# Same architecture as train_small_attnres.sh but with Huber loss
# Effective batch: 65,536 tokens (batch=2 * accum=16 * seq=2048)
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-100BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 16 --num-heads 16 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --seq-len 2048 --chunk-size 512 \
  --max-steps 1000 \
  --lr 3e-4 \
  --eval-every 100 --eval-buffer-size 100 \
  --save-every 200 \
  --log-every 10 \
  --dtype bfloat16 \
  --memory-objective huber \
  --use-attn-res \
  --num-attnres-blocks 4 \
  --attnres-warmup-steps 100 \
  --checkpoint-dir checkpoints/small-yaad
