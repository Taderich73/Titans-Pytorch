#!/bin/bash
##
# Train a 1.5B parameter Titans model with TNT, AttnRes, MCA,
# Yaad Huber attentional bias, and Adaptive Window Sizing on HuggingFace Jobs.
#
# Replace --hub-repo with your own org/repo before running.
# --titans-sha pins the code revision — update when rerunning against HEAD.
##

uv run python -u scripts/launch_pretraining_job.py \
      --dim 1024 \
      --num-heads 16 \
      --num-layers 20 \
      --tokenizer gpt2 \
      --vocab-size 50257 \
      --seq-len 2048 \
      --chunk-size 512 \
      --window-size 512 \
      --rope-proportion 0.25 \
      --use-tnt true \
      --use-attn-res true \
      --use-mca true \
      --adaptive-window true \
      --memory-objective huber \
      --dataset HuggingFaceFW/fineweb-edu \
      --dataset-subset sample-10BT \
      --max-steps 10000 \
      --save-every 2500 \
      --titans-sha de2ea62 \
      --reset-global-state false \
      --hub-repo FlatFootInternational/titans-mac-1.5B \
      --save-format safetensors \
      --flavor a100-large \
      --timeout 24h

      #--titans-sha 0377b60 \ # auto-checkpointing
      #--titans-sha 26c4b7e \
