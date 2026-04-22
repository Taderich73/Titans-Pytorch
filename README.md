# OpenTitans

![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.2-red.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
[![CI](https://github.com/Taderich73/OpenTitans/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Taderich73/OpenTitans/actions/workflows/ci.yml)

A PyTorch implementation of the **Titans** architecture (Google Research,
2024) with composable extensions: **TNT** hierarchical memory, **Attention
Residuals (AttnRes)**, **Memory Cross-Attention (MCA)**, **Yaad** Huber
attentional bias, **Adaptive Window Sizing**, and **Proportional RoPE
(p-RoPE)**.

Titans introduce a Neural Long-term Memory module that learns to memorize
historical context at **test time** via gradient descent with momentum and
weight decay. All extensions above are independent flags that compose freely
on top of the MAC, MAG, MAL, and LMM block variants. See
[`docs/`](docs/README.md) for per-feature deep-dives.

## Project Status

Alpha / research. The core Titans mechanics (MAC/MAG/MAL/LMM, deep memory,
chunked updates) and the extensions below are implemented and tested; paper
alignment is tracked equation-by-equation in
[`docs/paper_alignment.md`](docs/paper_alignment.md). Expect configuration
churn, non-stable training defaults, and occasional breaking API changes
before v1.0.

## Installation

```bash
# From source (recommended)
git clone https://github.com/Taderich73/OpenTitans.git
cd OpenTitans
uv sync                      # base
uv sync --extra train        # + training deps
uv sync --extra hf           # + HuggingFace integration
uv sync --all-extras         # everything (dev)

# Or via pip, directly from GitHub
pip install "titans[hf] @ git+https://github.com/Taderich73/OpenTitans.git"
```

Requires Python 3.12+ and PyTorch ≥ 2.2. Runs on CPU or CUDA.

## Quick Start

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=6, vocab_size=32000,
    chunk_size=512,
)
model = TitansMAC(config)

input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)

# Memory threads across calls — keep `states` to continue the same session
input_ids_next = torch.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

Other block variants (`TitansMAG`, `TitansMAL`, `TitansLMM`) share the same
constructor and forward signature. For chat/inference-time memory dumps,
HuggingFace interop, and training scripts, see the documentation links below.

## Feature Matrix

All flags are independent and compose with any block type (unless noted).

| Feature | Flag | Status | Doc |
| --- | --- | --- | --- |
| MAC / MAG / MAL / LMM blocks | `TitansMAC(...)` etc. | Stable | [configuration_guide.md](docs/configuration_guide.md) |
| TNT hierarchical memory | `use_tnt=True` | Stable | [tnt_hierarchical_memory.md](docs/tnt_hierarchical_memory.md) |
| Attention Residuals | `use_attn_res=True` | Stable | [attention_residuals.md](docs/attention_residuals.md) |
| Memory Cross-Attention | `use_mca=True` | Stable | [memory_cross_attention.md](docs/memory_cross_attention.md) |
| Yaad / Huber memory objective | `memory_objective="huber"` | Stable | [yaad_huber_bias.md](docs/yaad_huber_bias.md) |
| Adaptive window sizing (MAG/MAL) | `adaptive_window=True` | Experimental | [adaptive_window_sizing.md](docs/adaptive_window_sizing.md) |
| Proportional RoPE | `rope_proportion=0.25` | Stable | [proportional_rope.md](docs/proportional_rope.md) |
| Memory state persistence | `save_memory_states` / `load_memory_states` | Stable | [memory_persistence.md](docs/memory_persistence.md) |
| Memory auto-checkpointing | `auto_checkpoint=True` | Experimental | [memory_auto_checkpointing.md](docs/memory_auto_checkpointing.md) |
| HuggingFace integration | `from titans.hf import ...` | Stable (MAC only) | [huggingface_integration.md](docs/huggingface_integration.md) |
| LoRA / SFT / DPO / RLVR scripts | `scripts/{lora,sft,dpo,rlvr}.py` | Stable | [configuration_guide.md](docs/configuration_guide.md) |

## Documentation

Full docs live in [`docs/`](docs/README.md).

### Architecture
- [TNT: Hierarchical Memory](docs/tnt_hierarchical_memory.md) — global + local memories at different resolutions.
- [Attention Residuals (AttnRes)](docs/attention_residuals.md) — depth-wise softmax residuals.
- [Memory Cross-Attention (MCA)](docs/memory_cross_attention.md) — second read interface into the memory MLP.
- [Yaad: Huber Attentional Bias](docs/yaad_huber_bias.md) — robust memory updates via Huber loss.
- [Adaptive Window Sizing](docs/adaptive_window_sizing.md) — per-layer learned sliding window.
- [Proportional RoPE](docs/proportional_rope.md) — partial rotary-position rotation.
- [Paper Alignment and Deviations](docs/paper_alignment.md) — equation-level faithfulness tracking.

### Training and Configuration
- [Configuration Guide](docs/configuration_guide.md) — every flag, with Paper Origin Tags.
- [Memory State Persistence](docs/memory_persistence.md) — save/load memory across sessions.
- [Memory Auto-Checkpointing](docs/memory_auto_checkpointing.md) — novelty-triggered state capture.

### Ecosystem
- [HuggingFace Integration](docs/huggingface_integration.md) — `from_pretrained`, `Trainer`, Hub upload.

## Training and Inference Scripts

```bash
# Pretraining with HF Accelerate (streaming FineWeb-Edu)
uv run python scripts/launch_pretraining_job.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu --tokenizer gpt2 \
    --dim 512 --num-layers 12

# SFT / LoRA / DPO
uv run python scripts/sft.py  --init-weights checkpoints/final.pt --dataset HuggingFaceH4/ultrachat_200k --tokenizer meta-llama/Llama-2-7b-hf
uv run python scripts/lora.py --init-weights checkpoints/final.pt --dataset allenai/Dolci-Instruct-SFT --tokenizer gpt2
uv run python scripts/dpo.py  --init-weights checkpoints/sft/final.pt --dataset Anthropic/hh-rlhf --tokenizer gpt2 --loss-type dpo

# Inference (memory persistence + novelty-triggered checkpointing)
uv run python scripts/inference.py --checkpoint checkpoints/final.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --memory-dump session.npz --auto-checkpoint

# Convert a native checkpoint to HuggingFace format
uv run python scripts/convert_to_hf.py --checkpoint checkpoints/final.pt \
    --tokenizer gpt2 --output-dir ./hf_model
```

See the HF integration doc for `Trainer` usage and the config guide for
flag-level help. Run the test suite with `uv run pytest tests/`.

## Citation

```bibtex
@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}

@article{li2025tnt,
  title={TNT: Improving Chunkwise Training for Test-Time Memorization},
  author={Li, Shuo and Bick, Ari and Lucchi, Aurelien and Behrouz, Ali},
  journal={arXiv preprint arXiv:2511.07343},
  year={2025}
}

@techreport{kimi2025attnres,
  title={Attention Residuals},
  author={Kimi Team},
  institution={Moonshot AI},
  year={2025},
  note={arXiv:2603.15031}
}

@article{behrouz2025miras,
  title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.13173},
  year={2025}
}
```

## License

Apache License 2.0

Copyright (c) 2026 Delanoe Pirard / Aedelon
