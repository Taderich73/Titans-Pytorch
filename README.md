# Titans for MLX


[![MLX](https://img.shields.io/badge/mlx-apple%20silicon-black.svg)](https://ml-explore.github.io/mlx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-403%20passed-brightgreen.svg)](tests/)

A complete **MLX** (Apple Silicon) implementation of the **Titans** architecture from Google Research, with **TNT** hierarchical memory, **Attention Residuals (AttnRes)**, **Memory Cross-Attention (MCA)**, and **Memory State Persistence** as composable, independent features.

Titans introduce a **Neural Long-term Memory** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. **TNT** adds a hierarchical memory system — one global memory for long-range context and N local memories at different resolutions. **AttnRes** replaces fixed residual connections with learned depth-wise softmax attention, mitigating PreNorm dilution. **MCA** adds cross-attention to the memory's weight rows, giving the model a second read interface into learned associations. **Memory State Persistence** enables dumping, loading, forking, and merging memory states across sessions for inference-time continual learning.

TNT, AttnRes, and MCA are **independent flags** that work with any block type (MAC, MAG, MAL) and compose freely:

| Flags | Memory | Residuals | Memory Read |
|-------|--------|-----------|-------------|
| (default) | Single NeuralLongTermMemory | Standard | MLP only |
| `--use-tnt` | Hierarchical (global + local) | Standard | MLP only |
| `--use-attn-res` | Single NeuralLongTermMemory | AttnRes | MLP only |
| `--use-mca` | Single NeuralLongTermMemory | Standard | MLP + Cross-Attention |
| `--use-tnt --use-attn-res --use-mca` | Hierarchical (global + local) | AttnRes | MLP + Cross-Attention |

---

## Table of Contents

- [Paper References](#paper-references)
- [Architecture Overview](#architecture-overview)
- [TNT: Hierarchical Memory](#tnt-hierarchical-memory)
- [Attention Residuals (AttnRes)](#attention-residuals-attnres)
- [Memory Cross-Attention (MCA)](#memory-cross-attention-mca)
- [Memory State Persistence](#memory-state-persistence)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretraining](#pretraining)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
- [RLVR (Reinforcement Learning with Verifiable Rewards)](#rlvr-reinforcement-learning-with-verifiable-rewards)
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

All three variants (MAC, MAG, MAL) support `--use-tnt`, `--use-attn-res`, and `--use-mca` independently. LMM supports `--use-tnt` but not `--use-attn-res` or `--use-mca` (it has no attention mechanism).

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

Each block exposes sub-layers for composability:

- **`core_forward(h, state, memory_gate)`** — attention + memory update + gating
- **`mca_forward(h, mem_state)`** — cross-attention to memory weight rows (at MCA insertion layers)
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

## Memory Cross-Attention (MCA)

MCA adds cross-attention to NeuralLongTermMemory's weight matrix rows at configurable insertion layers. This gives the model a second read interface into the same memory that's already being written to by the surprise-driven update mechanism.

| | MLP Retrieval (existing) | Cross-Attention (MCA) |
|---|---|---|
| Operation | `output = MLP(query)` | `softmax(Q @ K^T) @ V` |
| Nature | Nonlinear function of query | Linear blend of memory directions |
| What it captures | Precise key-value lookup | Soft discovery of relevant associations |

Key properties:
- **Reads from existing memory** — no new state, no separate memory bank
- **Gated** — sigmoid gate initialized near-zero (-3.0 bias), so MCA has no effect until the gate learns to open
- **Configurable insertion** — defaults to midpoint layer, supports explicit multi-layer placement
- **AttnRes integration** — MCA becomes a third sub-layer in the AttnRes framework when both are enabled

Enable with `--use-mca`. Works with MAC, MAG, MAL (not LMM — stays pure memory):

```python
from titans_mlx import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    use_mca=True,
    mca_num_heads=8,
    # mca_insertion_layers=[6],  # auto = [num_layers // 2]
)
model = TitansMAC(config)
```

---

## Memory State Persistence

The `MemoryDumpManager` provides serialization, inspection, and management of NeuralLTM memory states. This enables inference-time continual learning across sessions without modifying model weights.

```python
from titans_mlx import TitansConfig, TitansMAC, MemoryDumpManager

config = TitansConfig(dim=512, num_heads=8, num_layers=6, vocab_size=32000)
model = TitansMAC(config)
mgr = MemoryDumpManager(config)

# Process some data — memory learns
logits, states = model(input_ids)

# Dump memory state to disk
dump_path = mgr.dump(states, step_count=100, description="after legal corpus")

# Later: load into a fresh session
restored_states = mgr.load(dump_path)
logits, states = model(new_input_ids, states=restored_states)
```

### Operations

| Operation | Description |
|-----------|-------------|
| `dump(states)` | Serialize memory states to timestamped directory (.npz + metadata) |
| `load(path)` | Restore memory states (strict=True validates dimensions) |
| `inspect(path)` | Per-layer weight norms, momentum norms, metadata |
| `diff(path_a, path_b)` | Per-layer Frobenius distance between two dumps |
| `merge(paths, strategy)` | Combine multiple dumps (weighted_mean, max_norm, recency) |
| `reset(states, layers)` | Zero out weights/momentum (all or specific layers) |
| `fork(states)` | Snapshot current state without altering live state |

### Use Cases

- **Session continuity** — dump at end of session, load at start of next
- **Domain adaptation** — process a specialized corpus, dump the resulting memory state, load it for domain-specific inference
- **Memory forking** — fork before processing uncertain data, revert if the domain shift was too aggressive
- **Memory merging** — combine memory states from multiple processing runs

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
| **MCA** |
| `--use-mca` | `False` | Enable Memory Cross-Attention |
| `--mca-num-heads` | `8` | MCA attention heads |
| `--mca-insertion-layers` | auto | Insertion layers (default: midpoint) |
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

## DPO (Direct Preference Optimization)

Align a model using preference pairs — chosen vs rejected responses. Supports standard DPO (with reference model) and SimPO (reference-free).

```bash
# DPO with LoRA (recommended — base model serves as reference, single model in memory)
uv run python scripts/dpo.py --model mac \
    --resume checkpoints/sft_model \
    --dataset allenai/Dolci-Instruct-DPO \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --method dpo --beta 0.1 \
    --lora --lora-targets attn,ffn

# SimPO (no reference model needed)
uv run python scripts/dpo.py --model mac \
    --resume checkpoints/sft_model \
    --dataset allenai/Dolci-Instruct-DPO \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --method simpo --beta 0.1 --gamma 1.0
```

**Methods:**
- **Standard DPO** — Rafailov et al. Uses a reference model to compute KL-constrained preference loss. With `--lora`, the frozen base model automatically serves as the reference (no extra memory).
- **SimPO** — Reference-free. Uses length-normalized average log-probabilities with a reward margin. Half the memory of standard DPO in full-parameter mode.

**LoRA-as-reference trick**: When using `--lora` with standard DPO, the base model (without LoRA adapters) acts as the reference model. A `set_lora_enabled()` toggle computes reference log-probs without loading a second model copy — making DPO practical on memory-constrained Apple Silicon.

### DPO Options

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `dpo` | `dpo` or `simpo` |
| `--beta` | `0.1` | KL penalty / scaling strength |
| `--gamma` | `1.0` | SimPO reward margin |
| `--lora` | `False` | Enable LoRA (base model = reference) |
| `--lora-rank` | `8` | LoRA rank |
| `--lora-alpha` | `16` | LoRA scaling |
| `--lora-targets` | `attn,ffn` | Target layers |
| `--max-len` | `2048` | Max sequence length |
| `--lr` | `5e-7` | Learning rate (lower than SFT) |
| `--chosen-field` | `chosen` | Dataset field for preferred responses |
| `--rejected-field` | `rejected` | Dataset field for rejected responses |

All model architecture flags are supported (`--use-tnt`, `--use-attn-res`, etc.).

---

## RLVR (Reinforcement Learning with Verifiable Rewards)

Train with reward signals from deterministic verifiers — no learned reward model needed. Supports GRPO (group relative policy optimization) and REINFORCE with EMA baseline.

```bash
# GRPO with offline rollouts (pre-computed)
uv run python scripts/rlvr.py --model mac \
    --resume checkpoints/sft_model \
    --dataset allenai/Dolci-Think-RL-7B \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --mode offline --method grpo

# REINFORCE with live generation + verification
uv run python scripts/rlvr.py --model mac \
    --resume checkpoints/sft_model \
    --dataset my/prompts-dataset \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --mode live --method reinforce \
    --verifier exact_match --num-rollouts 4

# Custom verifier function
uv run python scripts/rlvr.py --model mac \
    --resume checkpoints/sft_model \
    --dataset my/prompts-dataset \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --mode live --verifier path/to/verifier.py:my_function
```

**Methods:**
- **GRPO** — Group Relative Policy Optimization (per DeepSeekMath). Uses clipped importance ratios and group-relative baselines. No value model needed — baselines are estimated from the rollout group. Requires >= 2 rollouts per prompt.
- **REINFORCE** — Single-sample policy gradient with exponential moving average baseline. Lower compute cost, works with `--num-rollouts 1`.

**Operating Modes:**
- **Offline** — Uses pre-computed rollouts from the dataset (e.g., `allenai/Dolci-Think-RL-7B` with `prompt`, `ground_truth`, and `outputs` fields). Faster iteration, no generation overhead.
- **Live** — Generates fresh rollouts at training time using temperature sampling, then scores them with a verifier function against `ground_truth`. More flexible but slower.

**Built-in Verifiers:**
- `exact_match` — Case-insensitive, whitespace-stripped string comparison
- `numeric_match` — Extracts the last number from the response, compares within tolerance

**Custom verifiers** via `--verifier path/to/module.py:function_name`. The function must have signature `(response: str, ground_truth: list[str]) -> float` returning a reward in [0, 1].

### RLVR Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `offline` | `offline` or `live` |
| `--method` | `grpo` | `grpo` or `reinforce` |
| `--num-rollouts` | `8` | Rollouts per prompt |
| `--epsilon` | `0.2` | GRPO clipping range |
| `--kl-beta` | `0.0` | KL penalty (0 = disabled) |
| `--ema-decay` | `0.99` | REINFORCE baseline decay |
| `--verifier` | `exact_match` | Verifier name or path:function |
| `--temperature` | `0.7` | Generation temperature (live mode) |
| `--max-new-tokens` | `2048` | Generation cap (live mode) |
| `--lr` | `1e-6` | Learning rate |
| `--max-steps` | `5000` | Training steps |

All model architecture and LoRA flags are supported.

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

**Memory state quantization** reduces the memory footprint of the persistent state (weights + momentum) carried between chunks. Uses 4-bit weights and mixed-precision momentum (float16 for linear memory, 8-bit for deep memory). Saves 60-75% of state memory with minimal retrieval distortion.

```bash
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --quantize-memory-state \
    --memory-state-bits 4 \
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
| **Memory State Quantization** |
| `quantize_memory_state` | False | Enable state quantization (inference) |
| `memory_state_weight_bits` | 4 | Bit-width for weights/Q-K projections |
| `memory_state_momentum_bits` | 8 | Bit-width for momentum (deep memory) |
| **Memory Cross-Attention** |
| `use_mca` | False | Enable MCA at insertion layers |
| `mca_insertion_layers` | None | Insertion layers (None = auto midpoint) |
| `mca_num_heads` | 8 | Cross-attention heads |
| `mca_gate_type` | "scalar" | Gate type: "scalar" or "vector" |
| `mca_gate_bias_init` | -3.0 | Gate bias init (sigmoid(-3) ~ 0.05) |
| **Memory Dump** |
| `mca_auto_dump` | False | Enable automatic memory dumps |
| `mca_dump_trigger` | "session_end" | Trigger: session_end, every_n, surprise_threshold |
| `mca_dump_path` | "./memory_dumps/" | Dump output directory |
| `mca_dump_keep_last_n` | 10 | Number of dumps to retain |

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

    # Memory Cross-Attention
    MemoryCrossAttention,

    # State Persistence
    save_memory_states,
    load_memory_states,
    save_tnt_memory_states,
    load_tnt_memory_states,
    MemoryDumpManager,

    # Memory State Quantization
    QuantizedTensor,
    QuantizedMemoryState,
    quantize_tensor,
    quantize_memory_state,
    get_weights,
    get_momentum,
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
|   +-- mca.py              # MemoryCrossAttention
|   +-- memory_dump.py      # MemoryDumpManager (dump/load/inspect/diff/merge/fork)
|   +-- quantize_state.py   # QuantizedTensor, QuantizedMemoryState
|   +-- optimizations.py
|   +-- metal_kernels.py
|
+-- scripts/
|   +-- pretrain.py         # Pretraining on raw text
|   +-- sft.py              # Supervised fine-tuning (chat data)
|   +-- lora.py             # LoRA fine-tuning (adapters)
|   +-- dpo.py              # DPO / SimPO preference optimization
|   +-- rlvr.py             # GRPO / REINFORCE with verifiable rewards
|   +-- inference.py        # Text generation
|
+-- tests/                  # 403 tests
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
