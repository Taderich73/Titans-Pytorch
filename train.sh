uv run --extra train python scripts/pretrain.py --model mac \
      --dataset HuggingFaceFW/fineweb-edu \
      --dataset-subset sample-10BT \
      --tokenizer NousResearch/Llama-2-7b-hf \
      --dim 512 --num-layers 16 --num-heads 8 \
      --batch-size 2 --gradient-accumulation-steps 32 \
      --seq-len 4096 --max-steps 10000 \
      --eval-every 200 --eval-buffer-size 100 \
      --save-every 500 \
      --dtype bfloat16
      
#--resume checkpoints/best_model.meta.npz \
