# Titans for MLX


[![MLX](https://img.shields.io/badge/mlx-apple%20silicon-black.svg)](https://ml-explore.github.io/mlx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-289%20passed-brightgreen.svg)](tests/)

A complete **MLX** (Apple Silicon) implementation of the **Titans** architecture from Google Research, with **TNT** hierarchical memory and **Attention Residuals (AttnRes)** as composable, independent features.

Titans introduce a **Neural Long-term Memory** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. **TNT** adds a hierarchical memory system — one global memory for long-range context and N local memories at different resolutions. **AttnRes** replaces fixed residual connections with learned depth-wise softmax attention, mitigating PreNorm dilution.

Both TNT and AttnRes are **independent flags** that work with any block type (MAC, MAG, MAL):

| Flags | Memory | Residuals |
|-------|--------|-----------|
| (default) | Single NeuralLongTermMemory | Standard |
| `--use-tnt` | Hierarchical (global + local) | Standard |
| `--use-attn-res` | Single NeuralLongTermMemory | AttnRes |
| `--use-tnt --use-attn-res` | Hierarchical (global + local) | AttnRes |

---

## Table of Contents

- [Paper References](#paper-references)
- [Architecture Overview](#architecture-overview)
- [TNT: Hierarchical Memory](#tnt-hierarchical-memory)
- [Attention Residuals (AttnRes)](#attention-residuals-attnres)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretraining](#pretraining)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [Inference](#inference)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Paper References

> **Titans**: Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663

> **Titans Revisited**: Di Nepi, G., Siciliano, F., & Silvestri, F. (2025). *Titans Revisited*. arXiv:2510.09551

> **TNT**: Li, S., Bick, A., Lucchi, A., & Behrouz, A. (2025). *TNT: Improving Chunkwise Training for Test-Time Memorization*. [arXiv:2511.07343](https://arxiv.org/pdf/2511.07343)

> **AttnRes**: Kimi Team (2025). *Attention Residuals*. [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)

---

## Architecture Overview

### Memory Perspective

Titans are designed around a memory perspective inspired by human cognition:

| Memory Type | Module | Behavior at Test Time | Characteristics |
|-------------|--------|----------------------|-----------------|
| **Short-term** | Attention (limited window) | In-context learning (fixed weights) | Precise, limited capacity |
| **Long-term** | Neural Memory (LMM) | **Still learning** (weight updates via gradient descent) | Fading, unlimited capacity |
| **Persistent** | Learnable tokens | Fixed (task knowledge) | Stable, task-specific |

### Variants

| Aspect | MAC | MAG | MAL | LMM |
|--------|-----|-----|-----|-----|
| **Architecture** | Memory -> Attention -> Memory | Attention x Memory | Memory -> Attention | Memory only |
| **Attention Type** | Segmented (full causal per chunk) | Sliding Window | Sliding Window | None |
| **Long-context** | Best | Good | Baseline | Good |
| **Training Speed** | Medium | Fast | Fastest | Fast |

All three variants (MAC, MAG, MAL) support both `--use-tnt` and `--use-attn-res` independently.

### Neural Long-term Memory

**Associative Memory Loss** (Eq. 12):
```
l(M; x_t) = ||M(k_t) - v_t||^2
```

**Memory Update with Forgetting** (Eq. 13-14):
```
M_t = (1 - alpha_t) * M_{t-1} + S_t
S_t = eta_t * S_{t-1} - theta_t * grad(l(M_{t-1}; x_t))
```

Where alpha_t (decay), eta_t (momentum), theta_t (learning rate) are all data-dependent gates.

### Sub-layer Architecture

Each block exposes two sub-layers for composability:

- **`core_forward(h, state, memory_gate)`** — attention + memory update + gating
- **`ffn_forward(h)`** — feed-forward network

The orchestrator decides how to connect them — standard residuals or AttnRes. Blocks are agnostic to which is used.

---

## TNT: Hierarchical Memory

TNT extends blocks with a hierarchical memory system that separates long-range context from fine-grained detail.

```
Input x
  |
  +---> Global Memory (V)     --- large chunks (C_G=2048) --- long-range context
  |        sequential across global chunks
  |
  +---> Local Memory 1 (W^1)  --- small chunks (C_L=8)   --- fine detail
  |        resets every S_L tokens to learnable W_init
  |
  +---> Local Memory 2 (W^2)  --- small chunks (C_L=16)  --- medium detail
  |        resets every S_L tokens to learnable W_init
  |
  +---> ... N local memories at different resolutions
```

**Retrieval** (TNT Eq. 15):
```
o_t = f(V, q_t) + sum_i f(W^(i), M_t^(i) * q_t)
```

Enable with `--use-tnt`. Works with any block type:

```python
from titans_mlx import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=6, vocab_size=32000,
    chunk_size=512,
    use_tnt=True,
    local_chunk_sizes=[8, 16],
    local_shard_length=2048,
)
model = TitansMAC(config)
```

### Two-Stage Training

| Stage | Purpose | Local Chunk Sizes | Compute |
|-------|---------|-------------------|---------|
| **Stage 1** (pre-training) | Throughput | Moderate (C_L = {8, 16}) | Baseline |
| **Stage 2** (fine-tuning) | Quality | Smaller (C_L' = {4, 8}) | +5% |

```python
stage1 = TitansConfig.tnt_stage1(dim=512, num_heads=8, num_layers=12, vocab_size=32000)
stage2 = TitansConfig.tnt_stage2(stage1)  # halves local chunk sizes
```

---

## Attention Residuals (AttnRes)

AttnRes replaces fixed residual connections with learned depth-wise softmax attention (from the Kimi team's Attention Residuals paper). Instead of uniformly accumulating all prior layer outputs, each sub-layer selectively aggregates earlier representations using a learned pseudo-query vector.

```
Standard:  h_l = sum(v_i)                    (fixed unit weights)
AttnRes:   h_l = sum(alpha_{i->l} * v_i)     (learned softmax weights)
```

Key properties:
- **Two AttnRes calls per block** (before core, before FFN), each with its own pseudo-query
- **Embedding as standalone source** (b_0) — always available for attention
- **Block boundaries** at sub-layer granularity for coarse-grained depth grouping
- **Memory gating** — AttnRes attention weights modulate the memory learning rate

Enable with `--use-attn-res`. Works with any block type, independently of `--use-tnt`:

```python
from titans_mlx import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    use_attn_res=True,
    num_attnres_blocks=4,
    attnres_warmup_steps=200,
)
model = TitansMAC(config)
```

### Training with AttnRes

```bash
uv run python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 12 --num-heads 8 \
  --use-attn-res \
  --num-attnres-blocks 4 \
  --attnres-warmup-steps 200
```

### Combined TNT + AttnRes

```bash
uv run python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 512 --num-layers 12 --num-heads 8 \
  --use-tnt \
  --local-chunk-sizes 8 16 \
  --use-attn-res \
  --num-attnres-blocks 4 \
  --attnres-warmup-steps 200
```

---

## Installation

```bash
git clone https://github.com/dlattka/titans-tnt-mlx.git
cd titans-tnt-mlx
uv sync

# With training dependencies
uv sync --extra train

# With all extras (development)
uv sync --all-extras
```

Requires macOS 13.5+ and Apple Silicon (M1/M2/M3/M4).

---

## Quick Start

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=6, vocab_size=32000,
    chunk_size=512,
)

model = TitansMAC(config)
mx.eval(model.parameters())

input_ids = mx.random.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
mx.eval(logits)

# Continue with states (memory threads across calls)
input_ids_next = mx.random.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

---

## Pretraining

```bash
# Demo with synthetic data
uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

# Train with FineWeb-Edu
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 512 --num-layers 12

# Resume from checkpoint
uv run python scripts/pretrain.py --model mac \
    --resume checkpoints/latest.safetensors
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| **Model** |
| `--model` | `mac` | Variant: mac, mag, mal, lmm |
| `--dim` | `512` | Model dimension |
| `--num-heads` | `8` | Attention heads |
| `--num-layers` | `12` | Number of layers |
| `--chunk-size` | `512` | Chunk size (MAC) |
| `--window-size` | `512` | Window size (MAG/MAL) |
| **TNT** |
| `--use-tnt` | `False` | Enable hierarchical memory |
| `--local-chunk-sizes` | `8 16` | Chunk sizes per local memory |
| `--local-shard-length` | `2048` | Local memory reset period |
| `--global-chunk-size` | `2048` | Global memory chunk size |
| **AttnRes** |
| `--use-attn-res` | `False` | Enable Attention Residuals |
| `--num-attnres-blocks` | `8` | AttnRes block count (N) |
| `--attnres-warmup-steps` | `0` | Steps before memory gating activates |
| `--attnres-modulate-global` | `True` | Gate global memory LR |
| `--attnres-modulate-local` | `False` | Gate local memory LR |
| **Data** |
| `--dataset` | - | HuggingFace dataset name (streaming) |
| `--dataset-subset` | - | Dataset subset |
| `--data` | - | Local text file path |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--seq-len` | `4096` | Sequence length |
| **Training** |
| `--batch-size` | `4` | Batch size |
| `--gradient-accumulation-steps` | `32` | Gradient accumulation |
| `--lr` | `4e-4` | Learning rate |
| `--dtype` | `float16` | float32, float16, bfloat16 |
| `--max-steps` | `-1` | Max steps (-1 = use epochs) |
| **Checkpointing** |
| `--checkpoint-dir` | `checkpoints/` | Checkpoint directory |
| `--save-every` | `1000` | Save every N steps |
| `--eval-every` | `500` | Eval every N steps |
| `--resume` | - | Resume from checkpoint |
| **Logging** |
| `--log-every` | `10` | Log every N steps |
| `--wandb` | `False` | Enable W&B logging |

---

## Supervised Fine-Tuning (SFT)

Fine-tune a pretrained model on instruction-following data using streamed HuggingFace datasets with chat template formatting.

```bash
# SFT with Dolci-Instruct
uv run python scripts/sft.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --lr 2e-5 --seq-len 2048

# With TNT stage 2 fine-tuning
uv run python scripts/sft.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --use-tnt --tnt-stage 2
```

**Checkpoint compatibility**: The model architecture flags (`--use-tnt`, `--use-attn-res`, etc.) must match the checkpoint being loaded via `--init-weights`. For example, a checkpoint trained with `--use-attn-res` requires `--use-attn-res` during SFT, otherwise loading will fail with extra parameter errors.

**Chat template**: Uses the tokenizer's built-in `chat_template` if available, falls back to ChatML.

**Loss masking**: By default, only assistant response tokens contribute to the loss. Use `--train-on-all` to train on the full conversation.

### SFT Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | *(required)* | HuggingFace dataset (messages format) |
| `--init-weights` | - | Pretrained checkpoint to fine-tune |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--lr` | `2e-5` | Learning rate |
| `--seq-len` | `2048` | Max sequence length |
| `--train-on-all` | `False` | Train on all tokens (not just assistant) |
| `--messages-field` | `messages` | Field name for messages in dataset |
| `--tnt-stage` | `1` | TNT stage (2 = halved local chunks) |
| `--gradient-accumulation-steps` | `8` | Gradient accumulation |

All model architecture flags from pretraining are supported (`--dim`, `--num-heads`, `--use-tnt`, `--use-attn-res`, etc.).

---

## LoRA Fine-Tuning

Low-rank adapter fine-tuning — trains small adapter matrices while keeping the base model frozen. Produces lightweight adapter files that can be saved independently or merged into the base model.

```bash
# LoRA with attention projections (default)
uv run python scripts/lora.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --lora-targets attn --lora-rank 8 --lr 1e-4

# LoRA with attention + FFN layers
uv run python scripts/lora.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --lora-targets attn,ffn

# Train and merge adapters into base model
uv run python scripts/lora.py --model mac \
    --init-weights checkpoints/best_model \
    --dataset allenai/Dolci-Instruct-SFT \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --merge-and-save checkpoints/merged_model
```

**Adapters** are saved as `adapters.safetensors` + `adapters.meta.json` in the checkpoint directory. They can be loaded onto any compatible base model or merged permanently.

### LoRA Target Layers

| Preset | Layers |
|--------|--------|
| `attn` (default) | `proj_q`, `proj_k`, `proj_v`, `proj_out` |
| `ffn` | `gate_proj`, `up_proj`, `down_proj` |
| `memory` | Memory MLP layers |
| `all` | All linear layers (except embed/head) |

Combine presets with commas: `--lora-targets attn,ffn`

### LoRA Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-rank` | `8` | LoRA rank |
| `--lora-alpha` | `16` | LoRA scaling factor |
| `--lora-dropout` | `0.05` | Dropout on LoRA path |
| `--lora-targets` | `attn` | Target layer presets |
| `--merge-and-save` | - | Merge adapters and save full model |
| `--lr` | `1e-4` | Learning rate (higher than SFT) |
| `--weight-decay` | `0.01` | Weight decay (lower than SFT) |

All SFT and model architecture flags are also supported.

---

## Inference

```bash
# Generate from pretrained model
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --prompt "Once upon a time" \
    --max-tokens 100

# Interactive mode
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --interactive

# Chat mode with SFT model (auto-detected from checkpoint)
uv run python scripts/inference.py \
    --checkpoint checkpoints/sft_model.safetensors \
    --interactive --stream

# Force chat mode on/off
uv run python scripts/inference.py \
    --checkpoint checkpoints/model.safetensors \
    --interactive --chat

# Inference with LoRA adapters
uv run python scripts/inference.py \
    --adapters checkpoints/final_adapters \
    --interactive --stream

# LoRA with different base checkpoint
uv run python scripts/inference.py \
    --adapters checkpoints/final_adapters \
    --checkpoint checkpoints/different_base.safetensors \
    --interactive
```

**Chat mode** is auto-detected from checkpoint metadata. SFT and LoRA checkpoints save `chat_template: "chatml"`, which enables ChatML formatting automatically. Override with `--chat` or `--no-chat`.

**LoRA adapters** are loaded via `--adapters`, which reads the adapter metadata to reconstruct the model. Use `--checkpoint` alongside `--adapters` to override the base model path.

---

## Configuration Reference

### TitansConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Architecture** |
| `dim` | 512 | Model dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 12 | Number of blocks |
| `vocab_size` | 32000 | Vocabulary size |
| **Memory** |
| `num_memory_layers` | 2 | Memory MLP depth |
| `memory_lr` | 0.1 | Memory learning rate (theta_t) |
| `memory_momentum` | 0.9 | Memory momentum (eta_t) |
| `memory_decay` | 0.01 | Memory decay (alpha_t) |
| **Attention** |
| `num_persistent_tokens` | 16 | Persistent memory tokens |
| `chunk_size` | 512 | Segment size for MAC |
| `window_size` | 512 | Sliding window for MAG/MAL |
| **TNT Hierarchical Memory** |
| `use_tnt` | False | Enable hierarchical memory |
| `global_chunk_size` | 2048 | Global memory chunk size |
| `local_chunk_sizes` | [8, 16] | Chunk sizes per local memory |
| `local_shard_length` | 2048 | Local memory reset period |
| `use_qk_projection` | True | Q-K projection for local retrieval |
| **Attention Residuals** |
| `use_attn_res` | False | Enable AttnRes |
| `num_attnres_blocks` | 8 | Number of AttnRes blocks (N) |
| `attnres_warmup_steps` | 0 | Steps before memory gating activates |
| `attnres_modulate_global_memory` | True | Gate global memory LR |
| `attnres_modulate_local_memory` | False | Gate local memory LR |

---

## API Reference

```python
from titans_mlx import (
    # Configuration
    TitansConfig,

    # Models — use_tnt and use_attn_res flags in config
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansLMM,

    # Shared orchestrator
    process_chunk,

    # Memory Components
    NeuralLongTermMemory,
    MemoryState,
    HierarchicalMemory,
    GlobalMemory,
    LocalMemory,
    TNTMemoryState,
    QKProjection,

    # AttnRes
    BlockAttnRes,
    AttnResMemoryGate,

    # Attention
    SlidingWindowAttention,
    SegmentedAttention,
    PersistentMemory,

    # State Persistence
    save_memory_states,
    load_memory_states,
    save_tnt_memory_states,
    load_tnt_memory_states,
)
```

---

## Development

### Project Structure

```
titans-tnt-mlx/
+-- src/titans_mlx/
|   +-- config.py           # TitansConfig
|   +-- memory.py           # NeuralLongTermMemory, MemoryState, TNTMemoryState
|   +-- tnt_memory.py       # GlobalMemory, LocalMemory, HierarchicalMemory
|   +-- attn_res.py         # BlockAttnRes, AttnResMemoryGate
|   +-- models.py           # MAC/MAG/MAL/LMM blocks and models, process_chunk
|   +-- attention.py        # SegmentedAttention, SlidingWindowAttention
|   +-- persistent.py       # PersistentMemory
|   +-- qk_projection.py    # QKProjection
|   +-- optimizations.py
|   +-- metal_kernels.py
|
+-- scripts/
|   +-- pretrain.py         # Pretraining on raw text
|   +-- sft.py              # Supervised fine-tuning (chat data)
|   +-- lora.py             # LoRA fine-tuning (adapters)
|   +-- inference.py        # Text generation
|
+-- tests/                  # 289 tests
```

### Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=titans_mlx --cov-report=term-missing
```

---

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
```

---

## License

Apache License 2.0

Copyright (c) 2026 Delanoe Pirard / Aedelon
