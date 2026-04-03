# MLX to PyTorch Migration — Design Specification

**Date:** 2026-04-03
**Status:** Approved
**Approach:** Vertical Slice (MAC first)

## Motivation

The current MLX implementation is limited to Apple Silicon with unified memory.
Models beyond dim:512 / 16 layers are impractical due to memory constraints.
Migrating to PyTorch enables:

1. **Hardware portability** — Training on CUDA GPUs (A100, H100) via HuggingFace Jobs, AWS, etc.
2. **Ecosystem access** — HuggingFace Accelerate, FSDP, DeepSpeed, FlashAttention, `torch.compile()`.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| MLX code fate | Archive to `archive/titans_mlx/` | Valuable reference during migration; no dual-backend maintenance |
| Package name | `titans` at `src/titans/` | Clean imports; repo name already signals PyTorch |
| Migration strategy | Vertical slice — MAC end-to-end first | Validates memory update mechanism early; composable features slot in later |
| Target scale | Single GPU first, structured for distributed | Even single A100 unblocks much larger models; `nn.Module` conventions support FSDP later |
| Training scripts | pretrain + inference first; sft/dpo/lora/rlvr deferred | Core model is the hard part; fine-tuning scripts are variations on the loop |
| Metal kernels | Drop entirely; native PyTorch ops | `F.scaled_dot_product_attention` + `torch.compile()` cover the critical paths |
| Optimizations | Deferred; correctness first | `torch.compile()`, mixed precision, mask caching added after core works |
| Test approach | Port coverage, write PyTorch-idiomatically; device-parametrized | Not mechanical find-and-replace; proper `torch.testing`, CPU + CUDA paths |

## Design Principle: Composable Feature Isolation

The MLX codebase achieves clean isolation of composable features (TNT, AttnRes, MCA,
Yaad, Adaptive Windows). Each feature is:

- A self-contained module with its own file
- Activated via config flags
- Integrated at well-defined insertion points in the model

**This architecture must be preserved in PyTorch.** Deferred features are defined as
config fields (accepted by `TitansConfig`) but raise `NotImplementedError` at model
construction time when enabled. This keeps the config interface stable and makes each
feature independently portable.

## Project Structure

```
titans-pytorch/
├── archive/                     # MLX code preserved for reference
│   └── titans_mlx/             # Full copy of current src/titans_mlx/
├── src/titans/                  # New PyTorch package
│   ├── __init__.py             # Public API exports
│   ├── config.py               # TitansConfig (dataclass, framework-agnostic)
│   ├── memory.py               # MemoryState, MemoryMLP, NeuralLongTermMemory
│   ├── attention.py            # RoPE, SegmentedAttention, SlidingWindowAttention
│   ├── persistent.py           # PersistentMemory
│   ├── models.py               # FeedForward, RMSNorm, MACBlock, TitansMAC
│   │                           #   + MAG/MAL/LMM stubs (NotImplementedError)
│   └── memory_dump.py          # State serialization (save/load .npz)
├── scripts/
│   ├── pretrain.py             # PyTorch + HuggingFace Accelerate
│   └── inference.py            # Inference with memory persistence
├── tests/
│   ├── conftest.py             # Fixtures: config, device, batch_size, seq_len
│   ├── test_config.py          # Config validation, defaults, computed properties
│   ├── test_memory.py          # MemoryState, MemoryMLP, NeuralLongTermMemory
│   ├── test_attention.py       # RoPE, SlidingWindowAttention, SegmentedAttention
│   ├── test_models.py          # TitansMAC forward, chunking, weight tying, stubs
│   └── test_memory_dump.py     # Round-trip serialization
├── pyproject.toml              # Updated dependencies
└── README.md
```

## Dependencies

```toml
dependencies = [
    "torch>=2.2.0",
    "numpy>=2.0.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
    "tqdm>=4.67.1",
]

[project.optional-dependencies]
train = [
    "accelerate>=0.27.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "wandb>=0.16.0",
]
```

## Framework Translation Map

### Core Type Mappings

| MLX | PyTorch |
|---|---|
| `mx.array` | `torch.Tensor` |
| `mx.nn.Module` (`__call__`) | `torch.nn.Module` (`forward`) |
| `mx.nn.Linear` | `torch.nn.Linear` |
| `mx.nn.Embedding` | `torch.nn.Embedding` |
| `mx.nn.Conv1d` (channels-last) | `torch.nn.Conv1d` (channels-first, needs transpose) |
| `mx.stop_gradient(x)` | `x.detach()` |
| `mx.expand_dims(x, axis)` | `x.unsqueeze(axis)` |
| `mx.broadcast_to(x, shape)` | `x.expand(shape)` |
| `mx.concatenate(tensors, axis)` | `torch.cat(tensors, dim)` |
| `mx.clip(x, lo, hi)` | `torch.clamp(x, lo, hi)` |
| `x.astype(mx.float32)` | `x.float()` or `x.to(torch.float32)` |

### Attention

| MLX | PyTorch |
|---|---|
| `mx.fast.scaled_dot_product_attention(q, k, v, mask="causal")` | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| Boolean mask (True=attend) | Additive float mask (0.0=attend, -inf=block) |

### Parameter Registration

| MLX Pattern | PyTorch Pattern |
|---|---|
| `self.attr = mx.array(...)` (auto-discovered) | `self.attr = nn.Parameter(...)` (trainable) or `self.register_buffer(...)` (non-trainable) |
| `self.blocks = [Block(...)]` | `self.blocks = nn.ModuleList([Block(...)])` |
| Inline `nn.Dropout(p)(x)` | Store as `self.dropout = nn.Dropout(p)`, call `self.dropout(x)` |

### Initialization

| MLX | PyTorch |
|---|---|
| `layer.weight = mx.random.normal(shape) * std` | `nn.init.normal_(layer.weight, std=std)` |

## Module Designs

### config.py — TitansConfig

Framework-agnostic dataclass. Preserved from MLX version with no structural changes.
All fields retained (including TNT, AttnRes, MCA, adaptive window) so the interface
is stable. Computed properties (`head_dim`, `ffn_dim`, `memory_hidden_dim`, etc.)
unchanged. `to_dict()` / `from_dict()` / factory methods preserved.

### memory.py — Neural Long-term Memory

**MemoryState:** Dataclass holding `list[torch.Tensor]` for weights and momentum.
`detach()` calls `w.detach()` on each tensor. `clone()` calls `w.detach().clone()`.
State tensors are plain `torch.Tensor` (not `nn.Parameter`) — updated by the
analytical gradient mechanism, not by the optimizer.

**MemoryMLP:** Standard `nn.Module` with `nn.ModuleList` of `nn.Linear` layers.
`forward()` replaces `__call__()`. `forward_with_weights(x, weights)` uses explicit
weight matrices via `F.linear(x, w)` — same pattern as MLX but with PyTorch ops.

**NeuralLongTermMemory:** The core memory module.

Gradient flow (two distinct paths):

1. **Main training graph** (optimizer backprop):
   `input -> proj_k/v/q -> retrieve(q, state.weights) -> proj_out -> output`.
   Projections are trainable `nn.Parameter`. Retrieval treats `state.weights`
   as detached tensors. Gate projections train via the main loss.

2. **Memory self-update** (analytical, within forward):
   `k, v, state.weights -> compute_gradients() -> new_weights/momentum -> detach()`.
   Closed computation producing next state. Does not connect to training graph.

Key methods preserved with identical math:
- `_compute_gradients_linear()` — closed-form gradient for 1-layer memory
- `_compute_gradients_deep()` — analytical backprop for multi-layer memory
- `_parallel_memory_update_linear()` — tensorized closed-form update (Eq 16-18)
- `_activation_derivative()` — SiLU/GELU/ReLU derivatives
- `init_state()` — creates state on same device as model parameters

Conv1d translation: transpose `(B, L, C)` -> `(B, C, L)` before conv, transpose back after.

Deferred: `QuantizedMemoryState` handling raises `NotImplementedError`.

### attention.py

**RotaryPositionEmbedding:** `inv_freq`, `cos_cached`, `sin_cached` stored via
`register_buffer` (moves with `.to(device)`, not trainable). Cache rebuilt when
sequence exceeds current max. Rotation math identical.

**SlidingWindowAttention:** Uses `F.scaled_dot_product_attention` with additive
float mask. Boolean mask converted: `torch.where(bool_mask, 0.0, float("-inf"))`.
Prefix tokens get full attention. RoPE applied to Q and K with offset.

**SegmentedAttention:** Full causal on `[persistent || memory || input]`.
Uses `is_causal=True`. Returns only input positions (slice off prefix).

Helper functions `_rearrange_to_heads` / `_rearrange_from_heads` use
`reshape` + `permute` (PyTorch's `transpose` equivalent).

### persistent.py — PersistentMemory

`self.tokens` becomes `nn.Parameter(torch.randn(num_tokens, dim) * init_std)`.
`forward(batch_size)` uses `unsqueeze(0).expand(batch_size, -1, -1)`.

### models.py

**RMSNorm:** `self.weight = nn.Parameter(torch.ones(dim))`. Computation in float32,
cast back to input dtype.

**FeedForward:** Three `nn.Linear` layers. `self.dropout = nn.Dropout(p)` stored
as module attribute.

**MACBlock:** Same sub-layer structure: `core_forward`, `ffn_forward`, `mca_forward`.
`self.memory_query = nn.Parameter(torch.randn(1, 1, dim) * init_std)`.
All sub-modules stored as attributes for proper parameter discovery.
Deferred features (`use_tnt`, `use_attn_res`, `use_mca`) raise `NotImplementedError`
at construction time.

**TitansMAC:** `nn.ModuleList` for blocks. Weight tying: `self.head.weight = self.embed.weight`.
Forward pass: embed -> chunk processing -> norm -> head -> logits.
`process_chunk()` preserved as standalone function with standard residual path.

**MAG/MAL/LMM stubs:** Classes defined, raise `NotImplementedError` in `__init__`
with pointer to `archive/titans_mlx/models.py` for reference.

### memory_dump.py — State Serialization

`save_memory_states()`: Converts `torch.Tensor` to `np.ndarray` via `.cpu().numpy()`.
Same `.npz` key naming scheme. Forward-compatible with MLX-generated dumps.

`load_memory_states()`: Loads numpy arrays, converts via
`torch.from_numpy(arr).to(device)`. Device parameter added to load function signature.

`save_tnt_memory_states()` / `load_tnt_memory_states()`: Preserved but deferred
(raise `NotImplementedError` until TNT is ported).

### scripts/pretrain.py

HuggingFace Accelerate-based training loop:

```
Accelerator(gradient_accumulation_steps=..., mixed_precision=...)
    -> prepare(model, optimizer, scheduler, dataloader)
    -> training loop with accelerator.accumulate(model)
    -> accelerator.backward(loss)
    -> gradient clipping
    -> memory state detachment at step boundaries
```

Datasets: Standard PyTorch `Dataset` / `DataLoader`. Synthetic, text file, and
HuggingFace streaming datasets. Accelerate handles distributed data sharding.

Optimizer: `torch.optim.AdamW` with cosine annealing + warmup via
`torch.optim.lr_scheduler`.

Checkpointing: `torch.save(model.state_dict())` for model weights. Memory states
saved separately as `.npz`.

WandB: Optional, same integration pattern. `accelerator.log()` for distributed safety.

### scripts/inference.py

Load model checkpoint + optional memory state dump. Run generation with memory
persistence across chunks. Save updated memory state after inference.

## Test Suite

### Fixtures (conftest.py)

```python
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)

@pytest.fixture
def default_config():
    return TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=256)
```

### Test Coverage

| File | Validates |
|---|---|
| `test_config.py` | Creation, validation, defaults, `to_dict`/`from_dict`, computed properties, deferred flags accepted |
| `test_memory.py` | MemoryState detach/clone. MemoryMLP forward shapes + forward_with_weights. NeuralLongTermMemory: init_state, forward shapes, state updates produce different weights, linear parallel update, deep gradient computation, Huber path, conv branch, gate values bounded |
| `test_attention.py` | RoPE cache + rotation shapes + position-dependence. SlidingWindowAttention: output shapes, window masking. SegmentedAttention: output shapes, prefix handling, causal property |
| `test_models.py` | TitansMAC: forward shapes, single-chunk + multi-chunk, memory state carryover, weight tying. Stubs raise NotImplementedError |
| `test_memory_dump.py` | Round-trip save/load. Values match after deserialization. Cross-device load |

All tests parametrized for device (CPU + CUDA when available).
Use `torch.testing.assert_close` for numerical comparisons.

## pyproject.toml Changes

- Package name: `titans` (was `titans`)
- Package path: `src/titans` (was `src/titans_mlx`)
- Core dep: `torch>=2.2.0` replaces `mlx>=0.18.0`
- New train dep: `accelerate>=0.27.0`
- Keywords: replace `mlx`, `apple-silicon` with `pytorch`, `cuda`, `gpu`
- mypy overrides: replace `mlx.*` with `torch.*`
- isort known-first-party: `titans` (was `titans`, `titans_mlx`)
- Ruff config: unchanged
- Tox config: unchanged structure

## Deferred Work

### Phase 2: Remaining Model Variants

| Item | Complexity | Reference |
|---|---|---|
| TitansMAG | Medium | `archive/titans_mlx/models.py` lines 461-694. Memory receives normed input (pre-attention). Gating combines attention and memory outputs. |
| TitansMAL | Medium | `archive/titans_mlx/models.py`. Memory as layer before attention. Sequential rather than parallel. |
| TitansLMM | Low | Memory-only model, no attention. Simplest variant. |

### Phase 3: Composable Features

| Item | Complexity | Reference |
|---|---|---|
| TNT Hierarchical Memory | High | `archive/titans_mlx/tnt_memory.py` (403 lines). Global + N local memories, periodic resets, Q-K projection. Also requires `qk_projection.py`. |
| AttnRes (Attention Residuals) | Medium | `archive/titans_mlx/attn_res.py` (131 lines). Learned softmax over prior block outputs. Modifies `process_chunk` flow. |
| MCA (Memory Cross-Attention) | Medium | `archive/titans_mlx/mca.py` (104 lines). Cross-attend to memory weight rows. Gated, inserted at configurable layers. |
| Yaad / Huber Loss | Low | Config fields and gates already ported. Verify Huber error capping path works correctly. |
| Adaptive Window Sizing | Low | `archive/titans_mlx/adaptive_window.py` (109 lines). Learned soft masking, integrates with SlidingWindowAttention. |

### Phase 4: Training Scripts

| Item | Complexity | Reference |
|---|---|---|
| SFT | Medium | `archive/scripts/sft.py`. Standard supervised fine-tuning with chat templates. |
| DPO | Medium | `archive/scripts/dpo.py`. Reference + policy model forward passes. |
| LoRA | Medium | `archive/scripts/lora.py`. Consider using HuggingFace `peft` library. |
| RLVR | High | `archive/scripts/rlvr.py`. Reinforcement learning with verifiable rewards. |

### Phase 5: Optimization

| Item | Complexity | Description |
|---|---|---|
| `torch.compile()` | Low | Decorate model/modules. May need to avoid dynamic control flow in hot paths. Profile first. |
| Mixed precision tuning | Low | Accelerate handles it; memory update numerics (L2 norm, geometric series) may need float32 enforcement. |
| FlashAttention verification | Free | Verify `F.scaled_dot_product_attention` activates FlashAttention on CUDA. |
| Custom Triton kernels | High | Only if profiling shows bottlenecks. Candidate: parallel memory update batched matmul. |
| Mask caching | Low | LRU cache for sliding window masks. |
| Memory state quantization | Medium | `archive/titans_mlx/quantize_state.py` (215 lines). 4-bit weights / 8-bit momentum. |
| FSDP / DeepSpeed | Medium | Accelerate config-driven. Model structured correctly (standard nn.Module, no global state). |

### Phase 6: Infrastructure

| Item | Complexity | Description |
|---|---|---|
| Gradient diagnostics | Low | Port `scripts/diagnose_gradients.py`. |
| Memory dump manager | Low | Port `MemoryDumpManager` auto-dump triggers and retention policy. |
| Pretokenization | Low | Port `scripts/pretokenize.py` — mostly framework-agnostic numpy. |
