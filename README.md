# TitansTNT for MLX


[![MLX](https://img.shields.io/badge/mlx-apple%20silicon-black.svg)](https://ml-explore.github.io/mlx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-228%20passed-brightgreen.svg)](tests/)

A complete **MLX** (Apple Silicon) implementation of the **Titans** architecture from Google Research, extended with **TNT** hierarchical memory and **Attention Residuals (AttnRes)**.

Titans introduce a **Neural Long-term Memory (LMM)** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. **TNT** builds on this with a hierarchical memory system — one global memory for long-range context and N local memories at different resolutions for fine-grained detail — with periodic resets and Q-K projection for domain-aligned retrieval. **AttnRes** replaces fixed residual connections with learned depth-wise softmax attention, mitigating PreNorm dilution and enabling architecturally-aware memory gating.

---

## Table of Contents

- [Paper References](#paper-references)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
  - [Memory Perspective](#memory-perspective)
  - [Architecture Variants](#architecture-variants)
  - [Neural Long-term Memory](#neural-long-term-memory)
- [TNT: Hierarchical Memory](#tnt-hierarchical-memory)
  - [Architecture](#tnt-architecture)
  - [Two-Stage Training](#two-stage-training)
  - [TNT Quick Start](#tnt-quick-start)
- [Attention Residuals (AttnRes)](#attention-residuals-attnres)
  - [How It Works](#how-it-works)
  - [Memory Gating](#memory-gating)
  - [AttnRes Quick Start](#attnres-quick-start)
  - [Training with AttnRes](#training-with-attnres)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretraining](#pretraining)
- [Inference](#inference)
- [Benchmarks](#benchmarks)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [MLX Optimizations](#mlx-optimizations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Paper References

> **Original Paper**: Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv preprint arXiv:2501.00663

> **Analysis Paper**: Di Nepi, G., Siciliano, F., & Silvestri, F. (2025). *Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model*. arXiv preprint arXiv:2510.09551

> **TNT Paper**: Li, S., Bick, A., Lucchi, A., & Behrouz, A. (2025). *TNT: Improving Chunkwise Training for Test-Time Memorization*. [arXiv:2511.07343](https://arxiv.org/pdf/2511.07343)

> **AttnRes Paper**: Kimi Team (2025). *Attention Residuals*. [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)

---

## Features

### Core Features

| Feature | MLX |
|---------|-----|
| MAC (Memory as Context) | ✅ |
| MAG (Memory as Gate) | ✅ |
| MAL (Memory as Layer) | ✅ |
| LMM (Memory Only) | ✅ |
| **TNT Hierarchical Memory** | ✅ |
| **TNT MAC / MAG / MAL variants** | ✅ |
| **Q-K Projection (local retrieval)** | ✅ |
| **Two-Stage Training (pre-train + fine-tune)** | ✅ |
| **Attention Residuals (AttnRes)** | ✅ MAC |
| **AttnRes Memory Gating** | ✅ MAC |
| Deep Memory (L_M >= 1) | ✅ |
| Data-dependent Gating | ✅ |
| RoPE (Rotary Embeddings) | ✅ |
| 1D Depthwise Convolution | ✅ |
| Mixed Precision Training | ✅ fp16/bf16 |
| Gradient Accumulation | ✅ |
| Streaming Datasets | ✅ |
| W&B Logging | ✅ |

### MLX-Specific Features

| Feature | MLX |
|---------|-----|
| Metal Kernels | ✅ |
| Unified Memory | ✅ |

### Test Coverage

- **228 unit tests** covering all modules
- **Integration tests** for all model variants (including TNT and AttnRes)

---

## Architecture Overview

<p align="left">
<img src="assets/figures/fig1_memory_training.png" alt="Neural Memory Training" width="600"/>
</p>
<p align="left"><em>Figure 1: Neural memory training with efficient parallelization via matmul operations (from paper)</em></p>

### Memory Perspective

Titans are designed around a **memory perspective** inspired by human cognition (Section 1 of paper):

| Memory Type | Module | Behavior at Test Time | Characteristics |
|-------------|--------|----------------------|-----------------|
| **Short-term** | Attention (limited window) | In-context learning (fixed weights) | Precise, limited capacity |
| **Long-term** | Neural Memory (LMM) | **Still learning** (weight updates via gradient descent) | Fading, unlimited capacity |
| **Persistent** | Learnable tokens | Fixed (task knowledge) | Stable, task-specific |

### Architecture Variants

#### Quick Comparison

| Aspect | MAC | MAG | MAL | LMM |
|--------|-----|-----|-----|-----|
| **Architecture** | Memory → Attention → Memory | Attention ⊗ Memory | Memory → Attention | Memory only |
| **Attention Type** | Segmented (full causal per chunk) | Sliding Window | Sliding Window | None |
| **Memory-Attention** | Bidirectional | Parallel (gating) | Sequential | N/A |
| **Chunking Required** | Yes | No | No | No |
| **Long-context** | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐ Baseline | ⭐⭐ Good |
| **Training Speed** | Medium | Fast | Fastest | Fast |

#### When to Use Each Variant

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Needle-in-haystack retrieval | **MAC** | Attention decides when to query long-term memory |
| Long document QA (>100K tokens) | **MAC** | Best BABILong benchmark results (97.95%) |
| Language modeling (perplexity) | **MAG** | Slightly better perplexity than MAC |
| Real-time / streaming inference | **MAG** | No chunking, constant memory footprint |
| Maximum training throughput | **MAL** | Simplest architecture |
| Existing hybrid model replacement | **MAL** | Same architecture as Griffin/Samba |
| Pure sequence modeling | **LMM** | Tests memory capability alone |

#### MAC: Memory as Context (Section 4.1)

<p align="left">
<img src="assets/figures/fig2_mac.png" alt="MAC Architecture" width="700"/>
</p>
<p align="left"><em>Figure 2: MAC (Memory as Context) - Bidirectional interaction between memory and attention</em></p>

```
h_t = M*_{t-1}(q_t)                              # Eq. 21: Retrieve from memory
S̃^(t) = [persistent] || h_t || x                # Eq. 22: Concatenate
y_t = Attn(S̃^(t))                               # Eq. 23: Segmented attention
M_t = M_{t-1}(y_t)                               # Eq. 24: Update memory
o_t = y_t ⊗ M*_t(y_t)                            # Eq. 25: Output gating
```

**Advantages**: Best long-context performance, bidirectional memory-attention interaction
**Disadvantages**: Requires chunking, slightly slower training

#### MAG: Memory as Gate (Section 4.2)

<p align="left">
<img src="assets/figures/fig4_mag_mal.png" alt="MAG and MAL Architecture" width="700"/>
</p>
<p align="left"><em>Figure 4-5: MAG (Memory as Gate) and MAL (Memory as Layer) architectures</em></p>

```
x̃ = [persistent] || x                           # Eq. 26: Add persistent tokens
y = SW-Attn*(x̃)                                  # Eq. 27: Sliding window attention
o = y ⊗ M(x̃)                                     # Eq. 28: Element-wise gating
```

**Advantages**: No chunking, best perplexity, good balance
**Disadvantages**: Memory and attention don't directly communicate

#### MAL: Memory as Layer (Section 4.3)

```
x̃ = [persistent] || x                           # Eq. 29: Add persistent tokens
y = M(x̃)                                         # Eq. 30: Memory layer
o = SW-Attn(y)                                   # Eq. 31: Attention on memory output
```

**Advantages**: Fastest training, simplest architecture
**Disadvantages**: Weaker long-context performance

### Neural Long-term Memory

<p align="left">
<img src="assets/figures/fig3a_lstm_forget.png" alt="LSTM-inspired Gating" width="400"/>
<img src="assets/figures/fig3b_lstm_update.png" alt="Memory Update" width="400"/>
</p>
<p align="left"><em>Figure 3: LSTM-inspired gating mechanism for memory forgetting (left) and update (right)</em></p>

#### Core Equations (Section 3.1)

**Associative Memory Loss** (Eq. 12):
```
ℓ(M; x_t) = ||M(k_t) - v_t||²
```

**Memory Update with Forgetting** (Eq. 13):
```
M_t = (1 - α_t) · M_{t-1} + S_t
```

**Surprise with Momentum** (Eq. 14):
```
S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)
      \_________/   \____________________/
      Past Surprise   Momentary Surprise
```

Where:
- `α_t` ∈ [0,1]: Forgetting/decay factor (data-dependent)
- `η_t` ∈ [0,1): Surprise decay / momentum coefficient (data-dependent)
- `θ_t` > 0: Learning rate for momentary surprise (data-dependent)

#### Key Innovations

1. **Momentum-based surprise**: Unlike DeltaNet/TTT which use momentary surprise only
2. **Forgetting mechanism**: Weight decay for memory management on long sequences
3. **Deep memory**: MLP with L_M >= 2 layers for more expressive power
4. **Data-dependent gates**: α, η, θ are functions of input, not fixed hyperparameters

---

## TNT: Hierarchical Memory

TNT extends Titans with a **hierarchical memory system** that separates long-range context from fine-grained detail, enabling both better memorization and massive parallelism through periodic local memory resets.

### TNT Architecture

```
Input x
  │
  ├──► Global Memory (V)     ─── large chunks (C_G=2048) ─── long-range context
  │        sequential across global chunks
  │
  ├──► Local Memory 1 (W¹)   ─── small chunks (C_L=8)   ─── fine detail
  │        resets every S_L tokens to learnable W_init
  │
  ├──► Local Memory 2 (W²)   ─── small chunks (C_L=16)  ─── medium detail
  │        resets every S_L tokens to learnable W_init
  │
  └──► ... N local memories at different resolutions
```

**Retrieval** (Eq. 15 from TNT paper):
```
o_t = f(V, q_t) + Σ_{i=1}^{N} f(W^(i), M_t^(i) · q_t)
      └─ global ─┘   └──── local with Q-K projection ────┘
```

Key innovations:
- **Learnable W_init**: Local memories reset to trained initial weights (not zeros), preserving task knowledge
- **Q-K Projection**: Projects queries onto the key subspace (`M_t = Σ k_τ k_τᵀ`), resolving the domain mismatch between memory compression and retrieval
- **Periodic resets**: Breaking sequential dependencies across shards enables context parallelism

### Two-Stage Training

| Stage | Purpose | Local Chunk Sizes | Compute |
|-------|---------|-------------------|---------|
| **Stage 1** (pre-training) | Throughput | Moderate (C_L = {8, 16}) | Baseline |
| **Stage 2** (fine-tuning) | Quality | Smaller (C_L' = {4, 8}) | +5% |

```python
from titans_mlx import TitansConfig

# Stage 1: pre-training config
stage1 = TitansConfig.tnt_stage1(dim=512, num_heads=8, num_layers=12, vocab_size=32000)

# Stage 2: derive fine-tuning config (halves local chunk sizes)
stage2 = TitansConfig.tnt_stage2(stage1)
# stage2.active_local_chunk_sizes → [4, 8]
```

### TNT Quick Start

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansTNT

# Configure TNT with hierarchical memory
config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,
    window_size=512,
    use_tnt=True,
    local_chunk_sizes=[8, 16],       # N=2 local memories
    local_shard_length=2048,          # Reset period
    global_chunk_size=2048,           # Global memory chunk size
)

# Create model — supports "mac", "mag", "mal" variants
model = TitansTNT(config, variant="mac")
mx.eval(model.parameters())

# Forward pass
input_ids = mx.random.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
mx.eval(logits)

# Continue with state (memory threads across calls)
input_ids_next = mx.random.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

See [`examples/tnt_usage.py`](examples/tnt_usage.py) for more detailed examples including multi-resolution memories and state persistence.

---

## Attention Residuals (AttnRes)

AttnRes replaces the fixed residual connections between TNT blocks with **learned softmax attention over depth**. Instead of uniformly accumulating all prior layer outputs (`h_l = h_{l-1} + f(h_{l-1})`), each layer selectively aggregates earlier representations using a learned pseudo-query vector.

This addresses **PreNorm dilution** — the progressive weakening of early-layer contributions as depth grows — and provides an architecturally-aware signal for memory gating.

### How It Works

**Block AttnRes** groups layers into N blocks (default 8). Within each block, standard residuals apply. Across blocks, learned softmax attention selects which prior block representations matter most:

```
Standard:  h_l = Σ v_i                    (fixed unit weights)
AttnRes:   h_l = Σ α_{i→l} · v_i          (learned softmax weights)
```

Each layer has a single pseudo-query vector `w_l ∈ R^d` (768 parameters). Total overhead for 16 layers: ~24K parameters.

### Memory Gating

The AttnRes attention weights also modulate the memory update learning rate. When a block's AttnRes weight is high (the model values current processing), memory writes are stronger. When low, memory writes are suppressed — keeping the Neural Memory cleaner:

```
effective_lr = theta × attnres_importance_weight
```

This is configurable independently for global and local memories.

### AttnRes Quick Start

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansTNT

config = TitansConfig(
    dim=768,
    num_heads=16,
    num_layers=16,
    vocab_size=32000,
    chunk_size=512,
    use_tnt=True,
    local_chunk_sizes=[8, 16],
    # Enable AttnRes
    use_attn_res=True,
    num_attnres_blocks=8,          # N=8 blocks, S=2 layers per block
    attnres_warmup_steps=500,      # Fixed LR for first 500 steps
    attnres_modulate_global_memory=True,
    attnres_modulate_local_memory=False,
)

model = TitansTNT(config, variant="mac")
mx.eval(model.parameters())

input_ids = mx.random.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
```

### Training with AttnRes

```bash
uv run python scripts/pretrain.py --model mac \
  --dataset HuggingFaceFW/fineweb-edu \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --dim 768 --num-layers 16 --num-heads 16 \
  --use-attn-res \
  --num-attnres-blocks 8 \
  --attnres-warmup-steps 500
```

---

## Installation

### Basic Installation

```bash
git clone https://github.com/yourusername/Google-Titans-replication.git
cd Google-Titans-replication
uv sync
```

### With Training Dependencies

```bash
uv sync --extra train
```

### With All Extras (Development)

```bash
uv sync --all-extras
```

### Requirements

MLX requires macOS 13.5+ and Apple Silicon (M1/M2/M3/M4):

```bash
# MLX is included in default dependencies
uv sync
```

---

## Quick Start

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansMAC, TitansMAG, TitansMAL

# Configuration
config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,
    window_size=512,
    num_persistent_tokens=16,
    num_memory_layers=2,
)

# Create model
model = TitansMAC(config)
mx.eval(model.parameters())  # Evaluate parameters

# Forward pass
input_ids = mx.random.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
mx.eval(logits)  # Force evaluation

# Continue with states
input_ids_next = mx.random.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

### Standalone Neural Memory

```python
from titans_mlx import TitansConfig, NeuralLongTermMemory
import mlx.core as mx

config = TitansConfig(dim=512, num_memory_layers=2)
memory = NeuralLongTermMemory(config)
mx.eval(memory.parameters())

x = mx.random.normal((2, 100, 512))
output, state = memory(x)
mx.eval(output)
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

# Full training
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 1024 --num-layers 24 --num-heads 16 \
    --batch-size 4 --gradient-accumulation-steps 32 \
    --dtype float16 --wandb

# Train with local text
uv run python scripts/pretrain.py --model mag \
    --data path/to/corpus.txt

# Resume from checkpoint
uv run python scripts/pretrain.py --model mac \
    --resume checkpoints/latest.safetensors
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| **Model Architecture** |
| `--model` | `mac` | Model variant: mac, mag, mal, lmm |
| `--dim` | `512` | Model dimension |
| `--num-heads` | `8` | Attention heads |
| `--num-layers` | `12` | Number of layers |
| `--vocab-size` | `32000` | Vocabulary size |
| `--chunk-size` | `512` | Chunk size for MAC |
| `--window-size` | `512` | Window size for MAG/MAL |
| `--use-attn-res` | `False` | Enable Attention Residuals |
| `--num-attnres-blocks` | `8` | AttnRes block count (N) |
| `--attnres-warmup-steps` | `0` | Steps before AttnRes memory gating activates |
| `--attnres-modulate-global` | `True` | Gate global memory LR with AttnRes |
| `--attnres-modulate-local` | `False` | Gate local memory LR with AttnRes |
| **Data** |
| `--dataset` | - | HuggingFace dataset name (streaming) |
| `--dataset-subset` | - | Dataset subset (e.g., sample-10BT) |
| `--data` | - | Local text file path |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--seq-len` | `4096` | Sequence length |
| **Training** |
| `--epochs` | `1` | Number of epochs |
| `--max-steps` | `-1` | Max steps (-1 = use epochs) |
| `--batch-size` | `4` | Batch size |
| `--gradient-accumulation-steps` | `32` | Gradient accumulation steps |
| `--lr` | `4e-4` | Learning rate |
| `--weight-decay` | `0.1` | Weight decay |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--warmup-ratio` | `0.03` | Warmup ratio |
| `--dtype` | `float16` | float32, float16, bfloat16 |
| **Checkpointing** |
| `--checkpoint-dir` | `checkpoints/` | Checkpoint directory |
| `--save-every` | `1000` | Save every N steps |
| `--eval-every` | `500` | Eval every N steps |
| `--resume` | - | Resume from checkpoint (.safetensors) |
| **Logging** |
| `--log-every` | `10` | Log every N steps |
| `--wandb` | `False` | Enable W&B logging |
| `--wandb-project` | `titans-mlx` | W&B project name |
| `--wandb-run-name` | - | W&B run name |
| `--seed` | `42` | Random seed |

---

## Inference

```bash
# Generate text
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --prompt "Once upon a time" \
    --max-tokens 100

# Interactive mode
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --interactive

# With quantization and benchmark
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.safetensors \
    --prompt "Hello" \
    --quantize 8 \
    --benchmark
```

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | **required** | Path to model checkpoint (.safetensors) |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--prompt` | - | Input prompt |
| `--max-tokens` | `100` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `0.9` | Top-p (nucleus) sampling |
| `--repetition-penalty` | `1.0` | Repetition penalty |
| `--interactive` | `False` | Interactive mode |
| `--stream` | `False` | Stream output token by token |
| `--quantize` | - | Quantization bits: 4 or 8 |
| `--benchmark` | `False` | Run generation benchmark |

---

## Benchmarks

### Model Quality (from Paper Table 1 & 5)

**Language Modeling (340M params, 15B tokens)**:

| Model | Wiki ppl ↓ | Avg Accuracy ↑ |
|-------|------------|----------------|
| MAC | 25.43 | 47.36 |
| MAG | **25.07** | **47.54** |
| MAL | 24.69 | 46.55 |
| LMM | 26.18 | 46.17 |

**Long Context (BABILong benchmark)**:

| Model | Accuracy ↑ |
|-------|------------|
| MAC | **97.95** |
| MAG | 96.70 |
| MAL | 96.91 |
| LMM | 92.68 |

### Inference Speed (Apple M4 Pro)

Configuration: batch=4, seq_len=256, dim=256, 4 layers

| Model | MLX (ms) |
|-------|----------|
| MAC | **19.89** |
| MAG | **9.72** |
| MAL | **9.75** |
| LMM | **7.11** |

---

## Configuration Reference

### TitansConfig Parameters

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| **Model Architecture** |
| `dim` | 512 | Model dimension (d_in) | - |
| `num_heads` | 8 | Number of attention heads | - |
| `num_layers` | 12 | Number of Titans blocks | Stackable |
| `vocab_size` | 32000 | Vocabulary size | - |
| `max_seq_len` | 8192 | Maximum sequence length | - |
| **Memory** |
| `num_memory_layers` | 2 | Memory MLP depth (L_M >= 1) | Section 3.1 |
| `memory_hidden_mult` | 4.0 | Memory hidden dim multiplier | - |
| `memory_lr` | 0.1 | Learning rate θ_t (scaled by gate) | Eq. 14 |
| `memory_momentum` | 0.9 | Momentum η_t (scaled by gate) | Eq. 14 |
| `memory_decay` | 0.01 | Forgetting α_t (scaled by gate) | Eq. 13 |
| **Attention** |
| `num_persistent_tokens` | 16 | Persistent memory tokens (N_p) | Eq. 19 |
| `chunk_size` | 512 | Segment size for MAC | Section 4.1 |
| `window_size` | 512 | Sliding window for MAG/MAL | Section 4.2-4.3 |
| **Architecture Options** |
| `use_conv` | True | 1D depthwise convolution | Section 4.4 |
| `conv_kernel_size` | 4 | Convolution kernel size | Section 4.4 |
| `use_rope` | True | Rotary Position Embeddings | - |
| `activation` | "silu" | Activation function | Section 4.4 |
| `dropout` | 0.0 | Dropout rate | - |
| **TNT Hierarchical Memory** |
| `use_tnt` | False | Enable TNT hierarchical memory | TNT paper |
| `global_chunk_size` | 2048 | Global memory chunk size (C_G) | TNT Eq. 5 |
| `local_chunk_sizes` | [8, 16] | Chunk sizes per local memory (C_L) | TNT Eq. 6 |
| `local_shard_length` | 2048 | Local memory reset period (S_L) | TNT Eq. 6 |
| `use_qk_projection` | True | Q-K projection for local retrieval | TNT Eq. 7 |
| `tnt_stage` | 1 | Training stage (1=pre-train, 2=fine-tune) | TNT Sec. 4.2 |
| `finetune_local_chunk_sizes` | None | Smaller C_L' for stage 2 | TNT Sec. 4.2 |
| **Attention Residuals (AttnRes)** |
| `use_attn_res` | False | Enable Block Attention Residuals | AttnRes paper |
| `num_attnres_blocks` | 8 | Number of AttnRes blocks (N) | AttnRes Sec. 3.2 |
| `attnres_warmup_steps` | 0 | Steps before AttnRes memory gating activates | - |
| `attnres_modulate_global_memory` | True | Gate global memory LR with AttnRes weights | - |
| `attnres_modulate_local_memory` | False | Gate local memory LR with AttnRes weights | - |
| **FFN** |
| `ffn_mult` | 4.0 | FFN hidden dim multiplier | - |
| `init_std` | 0.02 | Weight initialization std | - |

---

## API Reference

```python
from titans_mlx import (
    # Configuration
    TitansConfig,

    # Models (original Titans)
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansLMM,

    # TNT Models (hierarchical memory)
    TitansTNT,          # Full model — pass variant="mac"/"mag"/"mal"
    TNTMACBlock,
    TNTMAGBlock,
    TNTMALBlock,

    # TNT Memory Components
    HierarchicalMemory,
    GlobalMemory,
    LocalMemory,
    TNTMemoryState,
    QKProjection,

    # TNT State Persistence
    save_tnt_memory_states,
    load_tnt_memory_states,

    # AttnRes (Attention Residuals)
    BlockAttnRes,
    AttnResMemoryGate,

    # Core Components
    NeuralLongTermMemory,
    MemoryState,
    SlidingWindowAttention,
    SegmentedAttention,
    PersistentMemory,

    # State Persistence
    save_memory_states,
    load_memory_states,

    # Optimizations
    compile_model,      # Note: Limited support
    compile_function,
    get_device_info,

    # Metal Kernels (benchmarking only)
    metal_silu_gate,
    metal_memory_update,
    metal_rope,
)
```

---

## MLX Optimizations

### Gradient Computation

The MLX implementation uses **analytical gradients** instead of `mx.grad` for the memory update:

```python
# Efficient gradient via matmul (avoids huge intermediate tensors)
# Instead of: expand_dims + outer product + sum
# We use: reshape + matmul

delta_flat = delta.reshape(batch_seq, -1)  # (B*S, D_out)
act_flat = act.reshape(batch_seq, -1)      # (B*S, D_in)
grad_w = delta_flat.T @ act_flat           # (D_out, D_in)
```

This optimization provides **5x speedup** for MAC.

### Why Not mx.compile?

`mx.compile` cannot compile full Titans models because:

1. **MemoryState**: Dataclasses are not supported
2. **Dynamic loops**: Python for-loops for chunk processing
3. **Mutable state**: Memory state updates

Individual components (FFN, attention) can be compiled for marginal gains.

### Metal Kernels

Custom Metal kernels are available but **not faster** than native MLX for typical tensor sizes:

| Operation | Metal Kernel | Native MLX | Verdict |
|-----------|--------------|------------|---------|
| SiLU Gate | 0.44ms | 0.26ms | Native faster |
| Memory Update | 0.20ms | 0.23ms | ~Equal |
| RoPE | 0.23ms | 0.23ms | ~Equal |

MLX already optimizes well for Apple Silicon. Use native operations.

### Recommended Practices

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansMAC

# 1. Use float16 for training (default)
# Saves memory, marginal speed difference on Apple Silicon

# 2. Evaluate parameters after creation
model = TitansMAC(config)
mx.eval(model.parameters())

# 3. Evaluate outputs when needed
logits, states = model(input_ids)
mx.eval(logits)  # Force computation

# 4. Use larger batches to amortize overhead
# batch_size=4 or higher recommended

# 5. Disable convolution if dimensions mismatch
config = TitansConfig(..., use_conv=False)
```

---

## Troubleshooting

### Common Issues

**Issue**: `ValueError: conv1d groups` error
```python
# Disable convolution
config = TitansConfig(..., use_conv=False)
```

**Issue**: Slow first iteration
```python
# Normal - MLX compiles on first call
# Subsequent iterations will be faster
```

**Issue**: Memory not releasing
```python
# Force garbage collection
import gc
gc.collect()
mx.metal.clear_cache()  # If available
```

---

## Development

### Project Structure

```
titans-tnt-mlx/
├── src/
│   └── titans_mlx/             # MLX implementation
│       ├── __init__.py
│       ├── config.py
│       ├── memory.py           # MemoryState, TNTMemoryState, serialization
│       ├── attention.py
│       ├── persistent.py
│       ├── models.py           # TitansMAC, TitansMAG, TitansMAL, TitansLMM
│       ├── qk_projection.py    # Q-K Projection for TNT local retrieval
│       ├── tnt_memory.py       # GlobalMemory, LocalMemory, HierarchicalMemory
│       ├── tnt_models.py       # TNTMACBlock, TNTMAGBlock, TNTMALBlock, TitansTNT
│       ├── attn_res.py         # BlockAttnRes, AttnResMemoryGate
│       ├── optimizations.py    # MLX optimizations
│       └── metal_kernels.py    # Metal kernels
│
├── scripts/
│   ├── pretrain.py         # MLX training
│   └── inference.py        # MLX inference
│
├── tests/
│   ├── test_memory.py
│   ├── test_attention.py
│   ├── test_models.py
│   ├── test_persistent.py
│   ├── test_config.py
│   └── test_tnt.py             # TNT: config, Q-K proj, hierarchical memory, models
│
├── examples/
│   ├── basic_usage.py
│   ├── long_sequence.py
│   └── tnt_usage.py            # TNT hierarchical memory examples
│
└── pyproject.toml
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_memory.py -v

# With coverage
uv run pytest tests/ --cov=titans_mlx --cov-report=term-missing
```

### Linting

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
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

@article{dinepi2025titans,
  title={Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model},
  author={Di Nepi, Gavriel and Siciliano, Federico and Silvestri, Fabrizio},
  journal={arXiv preprint arXiv:2510.09551},
  year={2025}
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
