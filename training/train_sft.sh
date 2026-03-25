
uv run python scripts/sft.py --model mac \
  --init-weights checkpoints/small-attnres/best_model \
  --use-attn-res \
  --dataset allenai/Dolci-Instruct-SFT \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --lr 2e-5
