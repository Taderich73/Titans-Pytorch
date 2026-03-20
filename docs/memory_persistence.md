# Memory State Persistence

The Titans architecture's memory module updates its weights during inference (test-time learning). By default these learned states exist only in RAM. Memory persistence lets you save and reload them across sessions.

## API

### `save_memory_states(states, path)`
Serialize all layer memory states to a single `.npz` file.

```python
from titans_mlx.memory import save_memory_states

# After running inference
save_memory_states(states, "my_memory.npz")
```

### `load_memory_states(path)`
Deserialize memory states from a `.npz` file. Validates structure and raises `ValueError` on shape/key mismatches.

```python
from titans_mlx.memory import load_memory_states

states = load_memory_states("my_memory.npz")
logits, states = model(input_ids, states=states)
```

## CLI Usage

```bash
# Save memory on exit
uv run python scripts/inference.py --checkpoint model.safetensors --interactive --save-memory session.npz

# Resume from saved memory
uv run python scripts/inference.py --checkpoint model.safetensors --interactive --load-memory session.npz

# Both (load at start, auto-save on quit)
uv run python scripts/inference.py --checkpoint model.safetensors --interactive \
    --load-memory session.npz --save-memory session.npz
```

## Interactive Commands

During interactive mode:

| Command | Description |
|---|---|
| `save` | Save memory state to `memory_state.npz` |
| `save path/to/file.npz` | Save to a specific path |
| `load` | Load memory state from `memory_state.npz` |
| `load path/to/file.npz` | Load from a specific path |
| `reset` | Clear memory to fresh init (not loaded state) |
| `quit` | Exit (auto-saves if `--save-memory` is set) |

## File Format

The `.npz` file contains:
- `num_layers` — number of model layers (for validation)
- `num_memory_layers_{i}` — MLP depth per layer (for validation)
- `layer_{i}_weight_{j}` — weight matrix for layer i, memory sub-layer j
- `layer_{i}_momentum_{j}` — momentum matrix for layer i, memory sub-layer j
