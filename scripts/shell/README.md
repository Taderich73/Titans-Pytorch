# Shell examples

Reference commands for the Python scripts in `scripts/`. Copy, adapt, run.

Every script here expects to be launched from the repository root:

```bash
./scripts/shell/train_titans_tiny.sh
```

## What's here

| Script                    | Purpose                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| `inference.sh`            | Generate text from a trained checkpoint                                     |
| `train_sft.sh`            | Supervised fine-tuning from a pretrained checkpoint                         |
| `train_lora.sh`           | LoRA fine-tuning (parameter-efficient)                                      |
| `train_dpo.sh`            | DPO preference optimization (LoRA-as-reference)                             |
| `train_rlvr.sh`           | RLVR with GRPO on a reasoning dataset                                       |
| `train_titans_tiny.sh`    | ~145M MAC pretraining on HF Jobs — sensible "first run" starting point      |
| `train_titans_full.sh`    | 1.5B MAC pretraining on HF Jobs — full feature set (TNT + AttnRes + MCA + …) |

## Before running

- **`--hub-repo <org/name>`** in the pretraining scripts points at a
  HuggingFace Hub repo. Replace the `FlatFootInternational/…` defaults
  with your own org and model name.
- **`--titans-sha <git-sha>`** in the pretraining scripts pins the code
  revision the HF Jobs runner will clone. Update to a current SHA before
  launching, or remove if you want HEAD.
- **`--checkpoint` / `--init-weights`** paths assume checkpoints live
  under `./checkpoints/`. Adjust as needed.
- **`--device mps`** in `inference.sh` is Apple Silicon. Change to `cuda`
  or `cpu` depending on your hardware.

## Adapting vs. forking

These scripts are intentionally terse — they exist to show which flags
matter for each workflow, not to be a substitute for reading the actual
Python CLI. For anything beyond a quick run, call the underlying script
(`scripts/pretrain.py`, `scripts/sft.py`, etc.) directly and use `--help`.
