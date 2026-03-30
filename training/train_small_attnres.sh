#!/bin/bash
# Pretrain Titans MAC model with AttnRes (no TNT)                                                                                                                                       
# dim=768, 16 layers, 16 heads
# AttnRes: 4 blocks, 150-step warmup                                                                                                                                                    
# Effective batch: 65,536 tokens (batch=2 * accum=16 * seq=2048)                                                                                                                        
uv run --extra train python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-100BT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 768 --num-layers 20 --num-heads 16 \
  --batch-size 2 --gradient-accumulation-steps 16 \
  --seq-len 2048 --chunk-size 512 \
  --max-steps 10000 \
  --lr 3e-4 \
  --eval-every 200 --eval-buffer-size 100 \
  --save-every 200 \
  --log-every 10 \
  --dtype bfloat16 \
  --use-attn-res \
  --num-attnres-blocks 4 \
  --attnres-warmup-steps 100 \
  --checkpoint-dir checkpoints/small-attnres
