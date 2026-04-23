#!/bin/bash
##
# Train a ~145M parameter Titans-MAC model (dim=512, 16 layers) with
# AttnRes, MCA, Yaad Huber attentional bias, and Adaptive Window Sizing
# on HuggingFace Jobs. TNT is disabled for this tiny variant.
#
# Replace --hub-repo with your own org/repo before running.
# --titans-sha pins the code revision — update when rerunning against HEAD.
##

uv run python -u scripts/launch_pretraining_job.py \
      --dim 512 \
      --num-heads 8 \
      --num-layers 16 \
      --tokenizer gpt2 \
      --vocab-size 50257 \
      --seq-len 2048 \
      --chunk-size 256 \
      --window-size 256 \
      --rope-proportion 0.25 \
      --use-tnt false \
      --use-attn-res true \
      --use-mca true \
      --adaptive-window false \
      --memory-objective huber \
      --dataset HuggingFaceFW/fineweb-edu \
      --dataset-subset sample-10BT \
      --max-steps 100000 \
      --save-every 2500 \
      --titans-sha f534903 \
      --reset-global-state false \
      --resume checkpoints/step_50000.pt \
      --hub-repo FlatFootInternational/opentitans-mac-145m \
      --flavor a10g-large \
      --timeout 24h

