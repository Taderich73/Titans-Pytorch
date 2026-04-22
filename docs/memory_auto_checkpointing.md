# Memory Auto-Checkpointing

> **Paper alignment:** N/A — novel engineering layer.
>
> **Implementation status:** Novel extension. No reference paper specifies any part of this subsystem.
>
> **Details:** `MemoryCheckpointer`, `StatisticalNoveltyDetector`, `SignalFrame`, `TransitionRecord`, and the four-state ring-buffer machine (MONITORING → TRIGGERED → CAPTURING_AFTER → COOLDOWN) are all project-specific. Their purpose is training observability (capturing before/during/after snapshots of grokking-like transitions) and crash resilience during long inference runs. The Titans paper offers no mechanism for either; this is plumbing the project chose to build on top. Disabled by default (`auto_checkpoint=False`); zero overhead when off.

Automatic, novelty-triggered checkpointing of Titans neural memory state during inference. Designed for two use cases:

1. **Crash resilience** — preserve accumulated memory state during long inference runs so it survives power loss, OOM, or process kill without re-processing the full input stream
2. **Phase transition research** — capture before/during/after snapshots when the memory undergoes grokking (sudden generalization), with full gate decay trajectories for studying the dynamics

Disabled by default (`auto_checkpoint=False`). Zero overhead when off.

## Quick Start

```bash
# Enable auto-checkpointing during inference
uv run python scripts/inference.py \
    --checkpoint checkpoints/final.pt \
    --prompt "The theory of everything" \
    --max-new-tokens 2000 \
    --auto-checkpoint

# Also enable the signal log for post-hoc analysis
uv run python scripts/inference.py \
    --checkpoint checkpoints/final.pt \
    --prompt "The theory of everything" \
    --max-new-tokens 2000 \
    --auto-checkpoint --signal-log
```

## How It Works

### Novelty Detection

The system monitors three signals from the memory update at each chunk:

| Signal | What It Measures | Trigger Meaning |
|--------|-----------------|-----------------|
| **Prediction error** (primary) | `\|\|M(k) - v\|\|` — how wrong the memory's predictions are | Spike: novel input. Drop: memory just learned (grokking) |
| **Weight delta** (fallback) | `\|\|W_new - W_old\|\|` — how much memory weights changed | Large delta: memory reorganizing |
| **Momentum shift** (fallback) | `\|\|S_new - S_old\|\|` — change in update direction | Direction change: phase transition |

The detection uses a **cascade**: prediction error is checked first. If unavailable (e.g., zero throughout), the system falls back to weight delta OR momentum shift, whichever has data.

**Bidirectional detection:**
- **Spikes** (value > mean + kσ) detect novel input the memory can't handle yet
- **Drops** use rate-of-change (first derivative) to detect sudden cliffs — a sharp error reduction indicates grokking, distinct from gradual learning

**Per-layer independence:** Each memory-carrying block is monitored independently. A single layer grokking triggers a capture even if other layers are stable.

### State Machine

```
MONITORING → TRIGGERED → CAPTURING_AFTER → COOLDOWN → MONITORING
```

- **MONITORING**: Each chunk pushes state to a ring buffer and feeds signals to the detector
- **TRIGGERED**: Detector fires. The calmest (lowest signal magnitude) entry in the ring buffer becomes the "before" snapshot
- **CAPTURING_AFTER**: Saves the next M states as "after" entries to capture settling
- **COOLDOWN**: Suppresses triggers for N chunks to avoid cascading on the same event

### Ring Buffer

A fixed-size in-memory FIFO of recent committed states. When a trigger fires, the "before" snapshot is selected as the entry with the lowest total signal magnitude — the most "quiescent" state before the transition. This gives a cleaner baseline for diffing against the "during" state.

## Gate Snapshots

When `auto_checkpoint=True`, the `NeuralLongTermMemory.forward()` method packages the data-dependent gate values into a `GateSnapshot` alongside the memory state:

| Gate | Symbol | What It Controls |
|------|--------|-----------------|
| alpha | Decay | How aggressively memory forgets old state |
| theta | Learning rate | How fast memory learns from errors |
| eta | Momentum | How much prior update direction carries forward |
| delta | Huber knee | Where L2 switches to L1 (Huber objective only) |

These gates are computed every forward pass as part of the memory update math — packaging them adds zero computational overhead. Each saved transition snapshot includes both the memory state (weights + momentum) and the gate profile that produced it.

For researchers studying grokking, the gate trajectories reveal *why* a transition happened: did the decay gate suddenly drop (memory retaining more)? Did the learning rate spike? The signal log captures the full per-chunk gate evolution.

## Disk Layout

```
memory_checkpoints/
├── transitions/
│   ├── tr_20260414_153022_prediction_error/
│   │   ├── before.npz              # MemoryState + GateSnapshot
│   │   ├── during.npz              # State at trigger point
│   │   ├── after_001.npz           # Settling snapshots
│   │   ├── after_002.npz
│   │   ├── after_003.npz
│   │   ├── signal_window.jsonl.gz  # SignalFrames for this transition
│   │   └── metadata.json           # Trigger details, magnitude, timing
│   └── tr_20260414_160847_weight_delta/
│       └── ...
├── ring_buffer_final.npz           # Last ring buffer on clean exit
├── signal_log/                     # Only when --signal-log active
│   ├── signals_000001.jsonl.gz
│   └── signals_000002.jsonl.gz     # Rotated at max_frames
└── session.json                    # Session metadata
```

### Transition metadata.json

```json
{
    "transition_id": "tr_20260414_153022_prediction_error",
    "trigger": {
        "signal_source": "prediction_error",
        "confidence": 0.92,
        "reason": "prediction_error 3.1σ above running mean (layer 4)"
    },
    "transition_magnitude": 0.847,
    "duration_chunks": 31,
    "before_chunk_index": 1042,
    "during_chunk_index": 1067,
    "after_chunk_range": [1068, 1072],
    "config_hash": "a3f8c2..."
}
```

## Configuration

### CLI Flags

| Flag | Effect |
|------|--------|
| `--auto-checkpoint` | Enable novelty-triggered checkpointing |
| `--signal-log` | Enable WAL signal log (requires `--auto-checkpoint`) |

If `--signal-log` is used without `--auto-checkpoint`, a warning is logged and the flag is ignored.

### MemoryCheckpointConfig

For programmatic use, pass a `MemoryCheckpointConfig` to fine-tune behavior:

```python
from titans import TitansConfig
from titans.checkpoint_types import MemoryCheckpointConfig

config = TitansConfig(
    dim=512, num_heads=8, num_layers=12, vocab_size=32000,
    auto_checkpoint=True,
    checkpoint_config=MemoryCheckpointConfig(
        checkpoint_dir="my_checkpoints",
        ring_size=25,              # Ring buffer size
        sigma_threshold=2.0,       # Z-score trigger threshold
        window_size=50,            # Sliding window for statistics
        min_observations=10,       # Warmup before arming
        cooldown_chunks=20,        # Suppress triggers after capture
        after_capture_count=5,     # "After" snapshots per transition
        keep_last_n_transitions=10,# Retention policy
        signal_log_enabled=True,   # Enable WAL
        signal_log_format="jsonl", # "jsonl" or "parquet"
        signal_log_max_frames=100_000,  # Rotation threshold
    ),
)
```

All config fields are JSON-serializable primitives that round-trip through `TitansConfig.to_dict()` / `from_dict()` and work with HuggingFace `push_to_hub` / `from_pretrained`.

## Programmatic Usage

```python
import torch
from titans import TitansConfig, TitansMAC
from titans.checkpoint_types import MemoryCheckpointConfig
from titans.memory_checkpointer import MemoryCheckpointer

config = TitansConfig(
    dim=512, num_heads=8, num_layers=6, vocab_size=32000,
    chunk_size=512, auto_checkpoint=True,
)
model = TitansMAC(config)
model.eval()

ckpt_config = MemoryCheckpointConfig(checkpoint_dir="./checkpoints")
checkpointer = MemoryCheckpointer(ckpt_config, config_hash="abc123")

states = None
with torch.no_grad():
    for i, chunk in enumerate(chunks):
        logits, states, gate_snapshots = model(chunk, states=states)
        states = [s.detach() for s in states]
        checkpointer.on_chunk_commit(states, gate_snapshots, chunk_index=i)

checkpointer.flush()
```

## TNT Awareness

For TNT hierarchical memory models, the detector maintains separate signal windows per memory tier:

- **Global memory window**: Continuous, never reset. Monitors long-range pattern learning.
- **Local memory windows**: Reset when `local_step_counters` hits a shard boundary (matching the memory's own reset cycle). Prevents false triggers from shard resets.

Global and local signals are evaluated independently — a grokking event in global memory (long-range pattern suddenly clicks) is a different phenomenon from one in local memory and both are captured.

## Signal Log

When `--signal-log` is enabled, a compressed JSONL file streams per-chunk signal frames to disk. Each frame contains:

- Per-layer prediction error, weight delta, momentum shift, gradient norms
- Per-layer absolute weight and momentum norms
- Per-layer gate means (alpha, theta, eta)
- Batch variance of error signal
- TNT-specific: global vs. local tier breakdowns, local reset flags

The log flushes every 50 frames for crash resilience and rotates at `signal_log_max_frames`. This provides a complete timeline for post-hoc analysis — plot the signal trajectory, identify transitions the detector missed, or use it as training data for a future learned novelty detector.

## Crash Behavior

| Exit Type | What Survives |
|-----------|---------------|
| **Clean** (`flush()`) | Ring buffer final, session.json, all completed transitions, signal log |
| **Unclean** (crash/OOM/kill) | Completed transitions already written, signal log (up to last 50-frame flush), incremental session.json |

The ring buffer and any in-progress transition capture are lost on unclean exit. The signal log serves as the recovery trail — the trajectory leading up to the crash is preserved.

## NoveltyDetector Protocol

The detection system is designed for extensibility. `StatisticalNoveltyDetector` is the v1 implementation. A future learned detector (RNN/LSTM trained on signal log data) can plug in behind the same protocol:

```python
from titans.novelty_detector import NoveltyDetector  # Protocol

class LearnedDetector:
    """Future: trained RNN that recognizes pre-grokking trajectories."""

    def observe(self, frame: SignalFrame) -> TriggerDecision:
        ...

    def reset(self) -> None:
        ...
```

## API Reference

```python
from titans import (
    # Data structures
    GateSnapshot,
    SignalFrame,
    CheckpointEntry,
    TransitionRecord,
    MemoryCheckpointConfig,

    # Core components
    MemoryCheckpointer,
    StatisticalNoveltyDetector,
    TriggerDecision,

    # Signal extraction
    build_signal_frame,
    compute_weight_delta,
    compute_momentum_shift,
    compute_weight_norms,
    compute_momentum_norms,
)
```

---

[Back to docs index](README.md) · [Back to project README](../README.md)
