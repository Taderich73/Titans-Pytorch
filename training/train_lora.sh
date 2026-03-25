uv run python scripts/lora.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --lora-targets attn --lora-rank 8 --lora-alpha 16 \
    --lr 1e-4
