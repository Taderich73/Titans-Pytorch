# Public API (stable surface for 0.7.x)

This page is the contract for what `from titans import X` is allowed to
be. Everything listed under [Stable top-level exports](#stable-top-level-exports)
is part of the supported surface for the 0.7.x line. Anything else —
attention primitives, LoRA helpers, auto-checkpointing, quantization —
lives behind a sub-module and should be imported from there directly
(e.g. `from titans.lora import LoRALinear`).

OpenTitans is still Alpha (`Development Status :: 3 - Alpha`). The
stable surface is deliberately small so we can iterate on the long tail
without breaking users on every point release.

## Stable top-level exports

`len(titans.__all__)` is frozen at **12** for 0.7.x. A test in
`tests/test_public_api.py` enforces the exact set.

### Config

| Symbol | Purpose |
| --- | --- |
| [`TitansConfig`](configuration_guide.md) | Single source of truth for all model / memory / attention hyper-parameters. Pydantic-validated. |

### Models

All four model variants share the same `TitansConfig` schema and return
`(logits, states)` from `forward`.

| Symbol | Purpose |
| --- | --- |
| `TitansMAC` | Memory-as-Context — memory reads prepended to the attention window. |
| `TitansMAG` | Memory-as-Gate — sigmoid-gated fusion of attention and memory outputs. |
| `TitansMAL` | Memory-as-Layer — memory acts as a standalone layer in the stack. |
| `TitansLMM` | LMM variant with deeper long-term memory plumbing. |

See [`docs/tnt_hierarchical_memory.md`](tnt_hierarchical_memory.md) and
[`docs/paper_alignment.md`](paper_alignment.md) for the architectural
differences.

### Memory

| Symbol | Purpose |
| --- | --- |
| `NeuralLongTermMemory` | The memory MLP that learns online at test time. |
| `MemoryState` | Per-layer memory state carried across chunks (parameters, momentum, optimizer scalars). |
| `TNTMemoryState` | Composite state for TNT hierarchical memory — pairs local + global states. |

### Persistence

| Symbol | Purpose |
| --- | --- |
| `save_memory_states` | Serialize a list of `MemoryState` / `TNTMemoryState` to `.npz`. See [Memory State Persistence](memory_persistence.md). |
| `load_memory_states` | Inverse of `save_memory_states`. |
| `save_checkpoint` | Save model + memory + optimizer to a single checkpoint file. |
| `load_checkpoint` | Inverse of `save_checkpoint`. |

## Deprecated top-level imports (removed in 0.8)

Every name below used to live in `titans.__all__`. In 0.7.x each one is
still importable via `from titans import X`, but access emits a
`DeprecationWarning` pointing at the canonical sub-module. The shims
will be deleted in 0.8 — migrate now.

| Old import | New import |
| --- | --- |
| `from titans import AdaptiveWindowPredictor` | `from titans.adaptive_window import AdaptiveWindowPredictor` |
| `from titans import compute_window_regularization` | `from titans.adaptive_window import compute_window_regularization` |
| `from titans import RotaryPositionEmbedding` | `from titans.attention import RotaryPositionEmbedding` |
| `from titans import SegmentedAttention` | `from titans.attention import SegmentedAttention` |
| `from titans import SlidingWindowAttention` | `from titans.attention import SlidingWindowAttention` |
| `from titans import log_sdpa_backend` | `from titans.attention import log_sdpa_backend` |
| `from titans import BlockAttnRes` | `from titans.attn_res import BlockAttnRes` |
| `from titans import MemoryCrossAttention` | `from titans.mca import MemoryCrossAttention` |
| `from titans import QKProjection` | `from titans.qk_projection import QKProjection` |
| `from titans import GlobalMemory` | `from titans.tnt_memory import GlobalMemory` |
| `from titans import HierarchicalMemory` | `from titans.tnt_memory import HierarchicalMemory` |
| `from titans import LocalMemory` | `from titans.tnt_memory import LocalMemory` |
| `from titans import PersistentMemory` | `from titans.persistent import PersistentMemory` |
| `from titans import QuantizedMemoryState` | `from titans.quantize_state import QuantizedMemoryState` |
| `from titans import QuantizedTensor` | `from titans.quantize_state import QuantizedTensor` |
| `from titans import quantize_memory_state` | `from titans.quantize_state import quantize_memory_state` |
| `from titans import quantize_tensor` | `from titans.quantize_state import quantize_tensor` |
| `from titans import LoRALinear` | `from titans.lora import LoRALinear` |
| `from titans import wrap_lora_layers` | `from titans.lora import wrap_lora_layers` |
| `from titans import set_lora_enabled` | `from titans.lora import set_lora_enabled` |
| `from titans import save_adapters` | `from titans.lora import save_adapters` |
| `from titans import load_adapters` | `from titans.lora import load_adapters` |
| `from titans import merge_lora_weights` | `from titans.lora import merge_lora_weights` |
| `from titans import count_lora_parameters` | `from titans.lora import count_lora_parameters` |
| `from titans import CheckpointEntry` | `from titans.checkpoint_types import CheckpointEntry` |
| `from titans import GateSnapshot` | `from titans.checkpoint_types import GateSnapshot` |
| `from titans import MemoryCheckpointConfig` | `from titans.checkpoint_types import MemoryCheckpointConfig` |
| `from titans import SignalFrame` | `from titans.checkpoint_types import SignalFrame` |
| `from titans import TransitionRecord` | `from titans.checkpoint_types import TransitionRecord` |
| `from titans import build_signal_frame` | `from titans.checkpoint_signals import build_signal_frame` |
| `from titans import compute_momentum_norms` | `from titans.checkpoint_signals import compute_momentum_norms` |
| `from titans import compute_momentum_shift` | `from titans.checkpoint_signals import compute_momentum_shift` |
| `from titans import compute_weight_delta` | `from titans.checkpoint_signals import compute_weight_delta` |
| `from titans import compute_weight_norms` | `from titans.checkpoint_signals import compute_weight_norms` |
| `from titans import StatisticalNoveltyDetector` | `from titans.novelty_detector import StatisticalNoveltyDetector` |
| `from titans import TriggerDecision` | `from titans.novelty_detector import TriggerDecision` |
| `from titans import MemoryCheckpointer` | `from titans.memory_checkpointer import MemoryCheckpointer` |

> A follow-up refactor (P9) will consolidate `checkpoint_types`,
> `checkpoint_signals`, `novelty_detector`, and `memory_checkpointer`
> under a single `titans.checkpointing` package. When that lands the
> "New import" column for those rows will point at
> `titans.checkpointing` and the current paths will themselves be
> deprecated for one more release.

## Stability policy

- **0.7.x (current).** Names in `titans.__all__` are frozen. Signatures
  may gain new optional keyword arguments but existing positional /
  required arguments won't change.
- **0.8.** Deprecation shims in this document are removed. Anything
  still imported via `from titans import X` where `X` is not in
  `__all__` becomes `ImportError`.
- **1.0 (future).** Current stable surface becomes a semver contract.
  Until then, assume anything outside `__all__` can move.

---

[Back to docs index](README.md)
