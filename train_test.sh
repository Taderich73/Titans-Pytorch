
uv run --extra train python scripts/pretrain.py --model mac \
        --dataset HuggingFaceFW/fineweb-edu \
        --dataset-subset sample-10BT \
        --tokenizer NousResearch/Llama-2-7b-hf \
        --dim 512 --num-layers 16 --num-heads 8 \
        --batch-size 2 --gradient-accumulation-steps 32 \
        --seq-len 2048 --max-steps 10 \
        --eval-every 2 --eval-buffer-size 10 \
        --dtype bfloat16 2>&1
