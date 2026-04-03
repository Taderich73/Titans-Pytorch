# Titans for PyTorch


[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.2-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen.svg)](tests/)

A complete **PyTorch** implementation of the **Titans** architecture from Google Research, with **TNT** hierarchical memory, **Attention Residuals (AttnRes)**, **Memory Cross-Attention (MCA)**, **Yaad Huber attentional bias**, and **Adaptive Window Sizing** as composable, independent features.

Titans introduce a **Neural Long-term Memory** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. **TNT** adds a hierarchical memory system — one global memory for long-range context and N local memories at different resolutions. **AttnRes** replaces fixed residual connections with learned depth-wise softmax attention, mitigating PreNorm dilution. **MCA** adds cross-attention to the memory's weight rows, giving the model a second read interface into learned associations. **Yaad** (from the Miras framework) replaces the standard L2 attentional bias with a Huber loss that is robust to outlier tokens. **Adaptive Window Sizing** lets each layer learn its own effective sliding window size via soft masking, balancing local context richness against compute cost.

TNT, AttnRes, MCA, Yaad, and Adaptive Window are **independent flags** that work with any block type (MAC, MAG, MAL) and compose freely:

| Flags | Memory | Residuals | Memory Read | Attentional Bias | Window |
|-------|--------|-----------|-------------|------------------|--------|
| (default) | Single NeuralLongTermMemory | Standard | MLP only | L2 | Fixed |
| `use_tnt=True` | Hierarchical (global + local) | Standard | MLP only | L2 | Fixed |
| `use_attn_res=True` | Single NeuralLongTermMemory | AttnRes | MLP only | L2 | Fixed |
| `use_mca=True` | Single NeuralLongTermMemory | Standard | MLP + Cross-Attention | L2 | Fixed |
| `memory_objective="huber"` | Single NeuralLongTermMemory | Standard | MLP only | Huber (Yaad) | Fixed |
| `adaptive_window=True` | Single NeuralLongTermMemory | Standard | MLP only | L2 | Learned |
| `use_tnt=True, use_attn_res=True` | Hierarchical (global + local) | AttnRes | MLP only | L2 | Fixed |

---

## Table of Contents

- [Paper References](#paper-references)
- [Architecture Overview](#architecture-overview)
- [TNT: Hierarchical Memory](#tnt-hierarchical-memory)
- [Attention Residuals (AttnRes)](#attention-residuals-attnres)
- [Memory Cross-Attention (MCA)](#memory-cross-attention-mca)
- [Yaad: Huber Attentional Bias](#yaad-huber-attentional-bias)
- [Adaptive Window Sizing](#adaptive-window-sizing)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretraining](#pretraining)
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

> **Miras (Yaad)**: Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)

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

All three variants (MAC, MAG, MAL) support `use_tnt`, `use_attn_res`, and `use_mca` independently. `adaptive_window` applies to MAG and MAL (sliding window variants). LMM is a standalone memory-only model.

### Neural Long-term Memory

**Associative Memory Loss** — configurable attentional bias (Eq. 12):
```
L2 (default):  l(M; x_t) = ||M(k_t) - v_t||^2
Huber (Yaad):  l(M; x_t) = { L2  if |error| <= delta_t
                            { L1  if |error| > delta_t
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

The orchestrator (`process_chunk`) decides how to connect them — standard residuals or AttnRes. Blocks are agnostic to which is used.

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

Enable with `use_tnt=True`. Works with any block type:

```python
from titans import TitansConfig, TitansMAC

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

Enable with `use_attn_res=True`. Works with any block type, independently of `use_tnt`:

```python
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    use_attn_res=True,
    num_attnres_blocks=4,
    attnres_warmup_steps=200,
)
model = TitansMAC(config)
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

Enable with `use_mca=True`. Works with MAC, MAG, MAL:

```python
from titans import TitansConfig, TitansMAC

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

## Yaad: Huber Attentional Bias

Yaad is a variant from the **Miras** framework (Behrouz et al., 2025) that replaces the standard L2 attentional bias with a **Huber loss**. This makes memory updates robust to outlier tokens — small errors use L2 gradients (precise), while large errors switch to L1 gradients (bounded magnitude).

```
Standard (L2):  loss = ||M(k_t) - v_t||^2         — all errors treated equally
Yaad (Huber):   loss = { L2  if |error| <= delta   — precise for normal tokens
                       { L1  if |error| > delta     — bounded for outliers
```

The threshold `delta` is **data-dependent** — a learned gate that adapts per chunk based on the input.

Enable with `memory_objective="huber"`. Composes with all other flags:

```python
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    chunk_size=512,
    memory_objective="huber",
)
model = TitansMAC(config)
```

---

## Adaptive Window Sizing

Standard sliding window attention uses a fixed window for all layers and all content. Adaptive window sizing lets each layer **learn its own effective window size** from the input, using differentiable soft masking.

Each layer gets a lightweight predictor (single linear projection) that outputs a "falloff center" per query position. A temperature-controlled sigmoid converts query-key distances into soft mask weights — positions within the falloff center attend normally, positions beyond decay smoothly. An efficiency regularization term penalizes large windows during training, encouraging the model to use smaller windows where local context suffices.

| Component | Role |
|-----------|------|
| `AdaptiveWindowPredictor` | Per-layer linear -> sigmoid -> soft mask |
| Soft masking | Differentiable alternative to hard window cutoff |
| Efficiency regularization | `lambda * mean(falloff_centers / max_window)` added to loss |

Supported for **MAG** and **MAL** blocks (both use sliding window attention).

---

## Installation

```bash
git clone https://github.com/dlattka/titans-pytorch.git
cd titans-pytorch
uv sync

# With training dependencies
uv sync --extra train

# With all extras (development)
uv sync --all-extras
```

Requires Python 3.12+ and PyTorch >= 2.2.0. Supports CPU and CUDA GPUs.

---

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

# Continue with states (memory threads across calls)
input_ids_next = torch.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

### Other Variants

```python
from titans import TitansMAG, TitansMAL, TitansLMM

# Memory as Gate — sliding window + memory in parallel
model_mag = TitansMAG(config)

# Memory as Layer — memory preprocesses before attention
model_mal = TitansMAL(config)

# Long-term Memory Module — memory only, no attention
model_lmm = TitansLMM(config)
```

---

## Pretraining

```bash
# Train with HuggingFace Accelerate
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer gpt2 \
    --dim 512 --num-layers 12

# Demo with synthetic data
uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

# Resume from checkpoint
uv run python scripts/pretrain.py --model mac \
    --resume checkpoints/latest.pt
```

The pretraining script supports HuggingFace Accelerate for multi-GPU training, mixed precision, gradient accumulation, cosine annealing with warmup, and optional WandB logging.

---

## Inference

```bash
# Generate from pretrained model
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 100

# With memory state persistence across sessions
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --memory-dump memory_states.npz \
    --prompt "Continue the story"
```

Memory states can be saved and loaded via `save_memory_states()` / `load_memory_states()` for inference-time continual learning across sessions.

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
| **Memory Objective (Attentional Bias)** |
| `memory_objective` | "l2" | `"l2"` (Titans default) or `"huber"` (Yaad) |
| `huber_delta_init` | 0.0 | Bias init for Huber delta gate |
| **Attention Residuals** |
| `use_attn_res` | False | Enable AttnRes |
| `num_attnres_blocks` | 8 | Number of AttnRes blocks (N) |
| `attnres_warmup_steps` | 0 | Steps before memory gating activates |
| `attnres_modulate_global_memory` | True | Gate global memory LR |
| `attnres_modulate_local_memory` | False | Gate local memory LR |
| **Memory Cross-Attention** |
| `use_mca` | False | Enable MCA at insertion layers |
| `mca_insertion_layers` | None | Insertion layers (None = auto midpoint) |
| `mca_num_heads` | 8 | Cross-attention heads |
| `mca_gate_type` | "scalar" | Gate type: "scalar" or "vector" |
| `mca_gate_bias_init` | -3.0 | Gate bias init (sigmoid(-3) ~ 0.05) |
| **Adaptive Window Sizing** |
| `adaptive_window` | False | Enable per-layer learned window sizing |
| `adaptive_window_min` | 64 | Minimum window size floor |
| `adaptive_window_max` | None | Maximum window size (None = `window_size`) |
| `adaptive_window_temperature` | 10.0 | Sigmoid sharpness at boundary |
| `adaptive_window_lambda` | 0.01 | Efficiency regularization weight |

---

## API Reference

```python
from titans import (
    # Configuration
    TitansConfig,

    # Models
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
    RotaryPositionEmbedding,
    PersistentMemory,

    # Memory Cross-Attention
    MemoryCrossAttention,

    # Adaptive Window Sizing
    AdaptiveWindowPredictor,
    compute_window_regularization,

    # State Persistence
    save_memory_states,
    load_memory_states,
)
```

---

## Development

### Project Structure

```
titans-pytorch/
+-- src/titans/
|   +-- config.py           # TitansConfig
|   +-- memory.py           # NeuralLongTermMemory, MemoryState, TNTMemoryState
|   +-- tnt_memory.py       # GlobalMemory, LocalMemory, HierarchicalMemory
|   +-- attn_res.py         # BlockAttnRes, AttnResMemoryGate
|   +-- models.py           # MAC/MAG/MAL/LMM blocks and models, process_chunk
|   +-- attention.py        # SegmentedAttention, SlidingWindowAttention
|   +-- persistent.py       # PersistentMemory
|   +-- qk_projection.py    # QKProjection
|   +-- mca.py              # MemoryCrossAttention
|   +-- adaptive_window.py  # AdaptiveWindowPredictor, compute_window_regularization
|   +-- memory_dump.py      # save/load memory states (.npz)
|
+-- scripts/
|   +-- pretrain.py         # Pretraining with HuggingFace Accelerate
|   +-- inference.py        # Text generation with memory persistence
|   +-- hf_pretrain.py      # HuggingFace Jobs training script
|   +-- launch_hf_job.py    # HF Jobs launcher
|
+-- tests/                  # 101 tests
```

### Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=titans --cov-report=term-missing
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

@article{behrouz2025miras,
  title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.13173},
  year={2025}
}
```

---

## License

Apache License 2.0

Copyright (c) 2026 Delanoe Pirard / Aedelon
