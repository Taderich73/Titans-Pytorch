# TNT: Hierarchical Memory

> **Paper**: Li, S., Bick, A., Lucchi, A., & Behrouz, A. (2025). *TNT: Improving Chunkwise Training for Test-Time Memorization*. [arXiv:2511.07343](https://arxiv.org/pdf/2511.07343)

> **Paper alignment:** Li et al., 2025 (TNT)
>
> **Implementation status:** Faithful — three deviations were fixed in Plan 6.
>
> **Details:** Three gaps were closed to match the paper exactly:
> 1. **Learnable `W_init`** (§4.1.1) — the initial local-memory state is now an `nn.Parameter` that receives gradients. Previously it was assigned via `.data`, which silently froze it.
> 2. **Per-position causal Q-K projection** (App. C) — the projection is now a per-position prefix-sum scan (linear-attention style). Previously a chunk-mean was used, which leaks future information within a chunk and violates causality.
> 3. **Reset cadence** (Eq. 6) — local memory now resets at every token index `t ≡ 0 (mod S_L)`. Previously it only reset at chunk boundaries, so resets near the middle of a chunk were missed.
>
> **Interaction with AttnRes:** When `use_tnt=True` and `use_attn_res=True`, the `AttnResMemoryGate` described in `docs/attention_residuals.md` modulates the TNT local memory's learning rate. That gate is a project-specific addition — see the AttnRes doc.

## Overview

TNT extends Titans blocks with a hierarchical memory system that separates long-range context from fine-grained detail. Instead of a single `NeuralLongTermMemory`, each layer maintains:

- **One global memory (V)** -- processes large chunks (default 2048 tokens) and persists across the entire sequence, capturing long-range patterns.
- **N local memories (W^1, W^2, ...)** -- process small chunks (e.g., 8, 16 tokens) at different resolutions and periodically reset to a learned initial state, capturing fine-grained detail within local windows.

```
Input x
  |
  +---> Global Memory (V)     --- C_G=2048 --- long-range context
  |        sequential across global chunks
  |
  +---> Local Memory 1 (W^1)  --- C_L=8    --- fine detail
  |        resets every S_L tokens to learnable W_init
  |
  +---> Local Memory 2 (W^2)  --- C_L=16   --- medium detail
  |        resets every S_L tokens to learnable W_init
```

## Retrieval

Retrieval combines the global and local outputs (TNT Eq. 15):

```
o_t = f(V, q_t) + sum_i f(W^(i), M_t^(i) * q_t)
```

Each local memory applies an optional learned Q-K projection (`M_t^(i)`) to the query before retrieval, allowing each resolution to attend to different aspects of the input.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_tnt` | `False` | Enable hierarchical memory |
| `global_chunk_size` | 2048 | Chunk size for global memory |
| `local_chunk_sizes` | `[8, 16]` | Chunk sizes per local memory (one memory per entry) |
| `local_shard_length` | 2048 | Tokens before local memory resets to `W_init` |
| `use_qk_projection` | `True` | Q-K projection for local retrieval |

## Usage

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512, num_heads=8, num_layers=6, vocab_size=32000,
    chunk_size=512,
    use_tnt=True,
    local_chunk_sizes=[8, 16],
    local_shard_length=2048,
)
model = TitansMAC(config)

input_ids = torch.randint(0, 32000, (2, 2048))
logits, states = model(input_ids)
```

Works with any block type (MAC, MAG, MAL).

## Two-Stage Training

TNT recommends a two-stage training schedule:

| Stage | Purpose | Local Chunk Sizes | Compute |
|-------|---------|-------------------|---------|
| **Stage 1** (pre-training) | Throughput | Moderate (e.g., [8, 16]) | Baseline |
| **Stage 2** (fine-tuning) | Quality | Halved (e.g., [4, 8]) | +5% |

```python
stage1 = TitansConfig.tnt_stage1(dim=512, num_heads=8, num_layers=12, vocab_size=32000)
stage2 = TitansConfig.tnt_stage2(stage1)  # halves local chunk sizes
```

## Key Classes

- `HierarchicalMemory` -- top-level module combining global and local memories
- `GlobalMemory` -- wrapper around `NeuralLongTermMemory` for global chunks
- `LocalMemory` -- memory with periodic reset and optional Q-K projection
- `TNTMemoryState` -- state dataclass holding global state, local states, step counters, and Q-K projections

## Design Notes

- Local memories reset to a **learned** `W_init` (not zeros) at shard boundaries, so the model can encode prior knowledge into the reset state.
- The global memory never resets -- it accumulates information across the entire sequence.
- Q-K projections are per-local-memory and per-layer, stored in `TNTMemoryState.qk_projections`.
- When saving/loading memory states, `TNTMemoryState` is serialized transparently by `save_memory_states` / `load_memory_states`. The `reset_for_inference` flag (default `True`) zeros local step counters on load to prevent silent resets.

---

[Back to docs index](README.md) · [Back to project README](../README.md)
