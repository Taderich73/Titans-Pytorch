# Memory State Persistence

> **Paper alignment:** Titans (Behrouz et al., 2024)
>
> **Implementation status:** Deviation (deliberate) — the paper is silent on initialization and on between-session persistence; both are local engineering choices.
>
> **Details:** Persistent memory weights are Gaussian-initialized with `std * init_std` at model construction. The papers do not specify a distribution or scale for this initialization, so the choice is project-local and deliberately conservative. Session-to-session persistence via `save_memory_states` / `load_memory_states` is also project-local — Titans the paper evaluates in-session only.

The Titans architecture's memory module updates its weights during inference (test-time learning). By default these learned states exist only in RAM. Memory persistence lets you save and reload them across sessions.

## API

### `save_memory_states(states, path)`
Serialize all layer memory states to a single `.npz` file. Handles both `MemoryState` and `TNTMemoryState` transparently.

```python
from titans.memory_dump import save_memory_states

# After running inference
save_memory_states(states, "my_memory.npz")
```

### `load_memory_states(path, device=None, *, reset_for_inference=False)`
Deserialize memory states from a `.npz` file. Validates structure and raises `ValueError` on shape/key mismatches. Logs a warning for any loaded tensor whose Frobenius norm is near-zero (indicating decay collapse).

```python
from titans.memory_dump import load_memory_states

states = load_memory_states("my_memory.npz")
logits, states = model(input_ids, states=states)
```

When `reset_for_inference=True`, TNT local step counters and Q-K projections are zeroed to prevent local memories from being silently wiped on the first forward pass. The default is `False` so training-resume paths preserve live TNT state; inference callers should pass `reset_for_inference=True` explicitly.

## CLI Usage

```bash
# Save memory on exit
uv run python scripts/inference.py --checkpoint model.pt --memory-dump session.npz --prompt "Hello"

# Resume from saved memory
uv run python scripts/inference.py --checkpoint model.pt --memory-dump session.npz --prompt "Continue"
```

## MemoryDumpManager

For more advanced workflows, `MemoryDumpManager` wraps the low-level save/load with timestamped dump directories, per-layer inspection, state diffing, merging, and forking.

```python
from titans.memory_dump import MemoryDumpManager

manager = MemoryDumpManager("./memory_dumps", keep_last_n=5)

# Save with a tag
path = manager.save(states, tag="step_1000")

# Load the most recent dump
loaded = manager.load_latest()

# Inspect per-layer norms
info = manager.inspect(states)

# Diff two state sets
delta = manager.diff(states_a, states_b)

# Merge multiple state sets
merged = manager.merge([states_a, states_b], strategy="weighted_mean")

# Deep-copy without disk I/O
forked = manager.fork(states)
```

### Merge Strategies

| Strategy | Description |
|----------|-------------|
| `weighted_mean` | Weighted average of tensors (uniform weights if none given) |
| `max_norm` | Per-sublayer, keep the tensor with the highest Frobenius norm |
| `recency` | Linearly increasing weights so more recent states dominate |

## File Format

The `.npz` file contains:
- `num_layers` -- number of model layers
- `layer_{i}_type` -- 0 = plain `MemoryState`, 1 = `TNTMemoryState`

For plain `MemoryState`:
- `layer_{i}_num_memory_layers` -- MLP depth
- `layer_{i}_weight_{j}` -- weight matrix for layer i, memory sub-layer j
- `layer_{i}_momentum_{j}` -- momentum matrix for layer i, memory sub-layer j

For `TNTMemoryState` (type=1), additional keys:
- `layer_{i}_global_*` -- global memory weights/momentum
- `layer_{i}_num_locals` -- number of local memories
- `layer_{i}_local_{k}_*` -- per-local-memory weights/momentum
- `layer_{i}_local_init_{k}_*` -- local init tensors
- `layer_{i}_qk_{k}` -- Q-K projection tensors
- `layer_{i}_step_counters` -- local step counters
