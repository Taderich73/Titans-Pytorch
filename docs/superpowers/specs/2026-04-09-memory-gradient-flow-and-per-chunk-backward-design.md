# Memory Gradient Flow Fix + Per-Chunk Backward

**Date:** 2026-04-09
**Status:** Draft
**Scope:** Correctness, observability, effectiveness, performance

## Problem Statement

The Titans training pipeline has several interacting issues that prevent the data-dependent
memory gates (alpha/decay, theta/learning-rate, eta/momentum, delta/Huber-knee) from
functioning as designed:

1. **NLTM retrieval order violates the paper.** `NeuralLongTermMemory.forward`
   (`memory.py:374`) retrieves from the input state (`state.weights`) instead of the
   updated state (`new_state.weights`). The Titans paper Eq. 3-4 specifies
   o_t = f(W_t, q_t) — retrieval from the updated state. Because `output` never depends
   on `new_state`, the gate projections (gate_decay_proj, gate_lr_proj,
   gate_momentum_proj, gate_delta_proj) receive no gradient signal through the output
   path. In TNT mode this is accidentally masked by `HierarchicalMemory.forward`
   (`tnt_memory.py:216`) which discards NLTM's output and does its own retrieval from
   `new_state`, creating a gradient path. But the standalone NLTM path is broken, and the
   TNT workaround wastes a full retrieval pass.

2. **g_norm logging reads post-reset state.** `hf_pretrain.py:614-616` reads
   `memory_states[0].global_state.weights[0]` after the global state has been reset to
   fresh init (line 594). The logged norm is always the norm of freshly initialized
   weights — a constant. It has no relationship to training dynamics.

3. **Alpha logging lacks precision.** `f"{alpha0:.4f}"` at `hf_pretrain.py:628` rounds
   sigmoid(-6.0) and sigmoid(-5.96) both to `0.0025`. Any movement from the optimizer is
   invisible at 4 decimal places in the deep sigmoid saturation regime.

4. **Gate decay init is too saturated.** `gate_decay_proj.bias` initializes to -6, giving
   sigmoid(-6) ~ 0.0025 with a sigmoid derivative of ~0.0025 — a 400x gradient scaling
   penalty. The gate can barely move even when gradients do reach it.

5. **Per-batch global state reset limits memory utility.** `hf_pretrain.py:583-598`
   unconditionally resets global memory state to fresh init every batch. Global memory
   sees only 4 chunks (seq_len=2048 / chunk_size=512) before being wiped. The TNT
   paper's Eq. 5 describes global state evolving sequentially with no per-batch reset.

6. **Multi-chunk forward forces concatenate-then-loss.** The internal chunk loop in
   `TitansMAC/MAG/MAL.forward()` concatenates all chunk outputs before returning,
   preventing per-chunk backward. This forces peak activation memory proportional to
   num_chunks and required activation checkpointing (added in `e9b9885`) to avoid OOM,
   costing ~33% compute overhead per step.

## Paper References

- **Titans** (Behrouz, Zhong, Mirrokni 2025): Eq. 3-4 (chunkwise compression/retrieval),
  Eq. 13-14 (decay/momentum update), Eq. 15 (retrieval from updated state),
  Eq. 21/24/25 (MAC variant: MCA retrieves from M_{t-1}, output retrieves from M_t)
- **TNT** (Li, Behrouz et al. 2025): Eq. 5 (global memory update), Eq. 7 (hierarchical
  retrieval rule)

## Design

### Change 1: Reorder NLTM Retrieval (memory.py)

Reorder `NeuralLongTermMemory.forward` so the state update happens before retrieval.
The output is then computed from the updated state, aligning with Titans Eq. 3-4.

**Current flow (memory.py:374-408):**
```
retrieve(q, state.weights)       # OLD state
compute gates (alpha, theta, eta)
update state -> new_state
output = proj_out(retrieved)     # from OLD retrieval
```

**New flow:**
```
compute gates (alpha, theta, eta)
update state -> new_state
retrieve(q, new_state.weights)   # UPDATED state
output = proj_out(retrieved)
```

The standalone `retrieve` method (`memory.py:522-536`) is NOT changed — it takes an
explicit `state` parameter and retrieves from whatever state the caller passes. This is
correct: MCA context retrieval in `MACBlock.core_forward` (`models.py:447`) calls
`self.memory.retrieve(query, state)` with the OLD state, which aligns with Titans
Eq. 21 (h_t = M*_{t-1}(q_t) — MCA retrieves from the previous state). Only the
internal retrieval inside `forward()` is reordered.

**Gradient consequence:** `output` now depends on `new_state.weights` which depends on
alpha, theta, eta. Combined with the `338f990` fix (conditional detach of returned
state), gate projections receive gradients via:
loss -> output -> proj_out -> retrieved -> forward_with_weights -> new_state.weights
-> parallel_memory_update(alpha, theta, eta) -> gate_decay_proj.bias

**Backward compatibility:** The `detach_memory_state_in_forward` flag from `338f990`
remains. After the reorder, this flag only affects whether the RETURNED state has its
autograd graph intact — it no longer controls whether gates receive gradients within the
current chunk. With the reorder, output depends on new_state.weights regardless of the
flag (the retrieval happens before the detach decision). The flag's effect is now limited
to preventing cross-chunk gradient flow through state, which is also handled by
chunk-boundary detach. It is kept for checkpoint-loading compatibility (the flag is
stored in TitansConfig which is serialized alongside weights).

### Change 2: Clean Up HierarchicalMemory (tnt_memory.py)

After Change 1, NLTM's output IS the correct retrieval from the updated state.
`HierarchicalMemory.forward` no longer needs its own separate retrieval.

**Current flow (tnt_memory.py:145-217):**
```
_, new_global = global_memory(x, state.global_state)   # output discarded
for local_mem: local_out, new_local = local_mem(x, ...)
output = self.retrieve(x, new_state)                    # redundant second retrieval
```

**New flow:**
```
global_out, new_global = global_memory(x, state.global_state)  # output used
for local_mem: local_out, new_local = local_mem(x, ...)
output = self.proj_out(global_out + sum(local_outs))            # combine directly
```

`HierarchicalMemory.retrieve()` stays as a public method for external callers (eval,
inference, MCA context retrieval) but is no longer called during the internal forward
pass.

The local memories follow the same pattern — after Change 1, each `local_mem(x, ...)`
returns output already retrieved from the updated local state.

### Change 3: Deprecate Multi-Chunk Forward (models.py)

`TitansMAC/MAG/MAL.forward()` becomes a single-chunk operation. All chunking, state
threading, detaching, and loss computation move to the caller.

**Removed from forward():**
- `chunks = x.split(chunk_size)` loop
- Chunk-boundary `new_states = [s.detach() for s in new_states]`
- `_run_process_chunk_checkpointed` wrapper calls
- Output concatenation `torch.cat(outputs, dim=1)`

**Retained:** The single-chunk path — `process_chunk(self.blocks, x, states, ...)` with
the embedding + LM head wrapping. This is what forward already does when
`seq_len <= chunk_size`.

**Forward signature** (unchanged shape, but input is now always one chunk):
```python
def forward(
    self,
    input_ids: torch.Tensor,              # (B, chunk_size)
    states: list[MemoryState] | None = None,
) -> tuple[torch.Tensor, list[MemoryState]]:
    """Process a single chunk. Returns (logits, new_states)."""
```

**Removed helpers:**
- `_run_process_chunk_checkpointed`
- `_flatten_states_to_tuples`
- `_unflatten_tuples_to_states`
- `use_chunk_checkpointing` config flag

### Change 4: Per-Chunk Backward Training Loop (hf_pretrain.py)

The training loop chunks input_ids and labels, calls `model()` per chunk, computes
per-chunk CE loss, runs backward immediately, then detaches state before the next chunk.

```python
with accelerator.accumulate(model):
    chunks = batch["input_ids"].split(CHUNK_SIZE, dim=1)
    label_chunks = batch["labels"].split(CHUNK_SIZE, dim=1)
    num_chunks = len(chunks)
    batch_loss = 0.0

    for chunk_ids, chunk_labels in zip(chunks, label_chunks):
        logits, memory_states = model(chunk_ids, states=memory_states)
        chunk_loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), chunk_labels.view(-1)
        )
        accelerator.backward(chunk_loss / num_chunks)
        batch_loss += chunk_loss.item() / num_chunks

        # Truncated BPTT: detach state at chunk boundary
        if memory_states is not None:
            memory_states = [
                s.detach() if s is not None else None for s in memory_states
            ]

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**Key properties:**
- `chunk_loss / num_chunks` distributes loss evenly so total gradient magnitude is
  comparable to the old single-backward path.
- `accelerator.backward()` accumulates gradients across chunks (PyTorch default).
- `optimizer.step()` fires once per micro-batch (or per accumulation window).
- `accelerator.accumulate(model)` handles cross-micro-batch gradient accumulation as
  before.
- Peak activation memory is bounded to one chunk (no recompute needed), eliminating the
  need for activation checkpointing.

**Eval loop (no backward):**
```python
chunks = input_ids.split(CHUNK_SIZE, dim=1)
all_logits = []
with torch.no_grad():
    for chunk_ids in chunks:
        logits, memory_states = model(chunk_ids, states=memory_states)
        all_logits.append(logits)
full_logits = torch.cat(all_logits, dim=1)
```

**bf16 scaler interaction:** `accelerator.backward()` handles mixed-precision scaling
internally. Multiple calls per step accumulate scaled gradients correctly — Accelerate's
GradScaler is designed for this pattern (gradient accumulation across micro-batches is
the same mechanism).

### Change 5: Fix g_norm Logging (hf_pretrain.py)

Capture global memory state norm *before* the detach/reset block:

```python
# Before detach/reset (insert above line 574)
_pre_reset_g_norm = None
if memory_states is not None:
    g_state = getattr(memory_states[0], "global_state", None)
    if g_state is not None and hasattr(g_state, "weights") and len(g_state.weights) > 0:
        _pre_reset_g_norm = g_state.weights[0].detach().float().norm().item()

# ... existing detach + reset block ...

# In the logging block (line 604+):
if _pre_reset_g_norm is not None:
    postfix["g_norm"] = f"{_pre_reset_g_norm:.2e}"
```

### Change 6: Improve Alpha and Gate Gradient Logging (hf_pretrain.py)

Log the raw bias value, increase alpha precision, and add gate gradient norm:

```python
if gate_proj is not None:
    raw_bias = gate_proj.bias.item()
    alpha0 = torch.sigmoid(gate_proj.bias).item()
    postfix["alpha"] = f"{alpha0:.6f}"
    postfix["decay_bias"] = f"{raw_bias:.4f}"
    if gate_proj.bias.grad is not None:
        postfix["gate_grad"] = f"{gate_proj.bias.grad.item():.2e}"
```

### Change 7: Configurable Gate Decay Init (config.py, memory.py)

Add to `TitansConfig`:
```python
gate_decay_init: float = -2.0
```

Apply in `NeuralLongTermMemory.__init__`:
```python
nn.init.constant_(self.gate_decay_proj.bias, config.gate_decay_init)
```

Default -2.0 gives sigmoid(-2) ~ 0.12, gradient scaling ~0.10 (40x improvement over
-6). Memory retains ~88% of state per chunk step. Follows the existing `huber_delta_init`
pattern.

Old checkpoints trained with bias=-6 remain loadable — the init only affects newly
constructed models. The loaded bias value from the checkpoint overrides the init.

### Change 8: Configurable Per-Batch Global State Reset (hf_pretrain.py)

Add to the training constants block:
```python
RESET_GLOBAL_STATE_PER_BATCH = True  # default preserves existing behavior
```

Expose as CLI arg in launch_hf_job.py. Gate the reset block:
```python
if RESET_GLOBAL_STATE_PER_BATCH and memory_states is not None:
    # existing reset logic (lines 583-598)
    ...
```

This is a training policy flag, not a model architecture field — it lives in the
training script constants, not in `TitansConfig`.

### Change 9: Remove Chunk-Checkpointing Helpers (models.py, config.py)

With per-chunk backward, peak memory is bounded to one chunk's activations. Activation
checkpointing inside the model is no longer needed.

**Remove from config.py:**
- `use_chunk_checkpointing: bool = False`

**Remove from models.py:**
- `_run_process_chunk_checkpointed()` function
- `_flatten_states_to_tuples()` function
- `_unflatten_tuples_to_states()` function
- All references to `use_chunk_checkpointing` in TitansMAC/MAG/MAL

**Remove from hf_pretrain.py:**
- `PYTORCH_CUDA_ALLOC_CONF` env var set (added in `e9b9885` specifically for the
  multi-chunk activation graph — no longer needed with per-chunk backward)

### Change 10: Update Tests

**New tests:**
- `test_nltm_retrieval_uses_updated_state`: Verify that NLTM.forward output changes when
  state is updated (not just a function of input and old state).
- `test_single_chunk_forward_api`: Verify forward() with seq_len == chunk_size works as
  before.
- `test_training_loop_per_chunk_backward`: Integration test with 2+ chunks, verify gate
  gradients are nonzero, verify loss decreases.

**Updated tests:**
- `test_tnt_gate_projections_receive_gradients`: Update to use single-chunk forward API
  if it currently relies on multi-chunk forward.
- `test_tnt_gate_gradients_with_multi_chunk_seq`: Rewrite to call model per-chunk in a
  loop (simulating training loop pattern) instead of relying on internal chunk loop.
- `test_checkpointed_chunk_propagates_gradients`: Remove (checkpointing helpers removed).
- `test_titans_mac_multi_chunk_checkpointed_gate_gradients`: Remove or rewrite as
  per-chunk-backward test.
- `test_state_detached`: Update to verify chunk-boundary detach happens in the caller,
  not inside forward.

**Kept tests:**
- All tests that use seq_len <= chunk_size (single-chunk) should pass with minimal
  changes.
- The unit-level NLTM tests in `test_memory.py` should be updated to verify the
  reordered retrieval.

## Files Changed

| File | Changes |
|------|---------|
| `src/titans/config.py` | Add `gate_decay_init`, remove `use_chunk_checkpointing` |
| `src/titans/memory.py` | Reorder NLTM.forward (retrieval after update) |
| `src/titans/tnt_memory.py` | HierarchicalMemory uses NLTM output directly |
| `src/titans/models.py` | Deprecate multi-chunk forward, remove checkpoint helpers |
| `scripts/hf_pretrain.py` | Per-chunk backward loop, logging fixes, reset flag, remove CUDA_ALLOC_CONF |
| `scripts/launch_hf_job.py` | Add `--reset-global-state` CLI flag |
| `tests/test_memory.py` | Update for retrieval reorder |
| `tests/test_models.py` | Rewrite multi-chunk tests for new API |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Existing checkpoints incompatible | `gate_decay_init` only affects new models; loaded bias overrides init. `detach_memory_state_in_forward=True` preserves legacy behavior. |
| Per-chunk backward produces different gradients than full-sequence backward | This is expected (truncated BPTT). The chunk-boundary detach was already present; per-chunk backward just makes the truncation explicit. |
| Multiple `accelerator.backward()` calls per step interact with grad scaler | This is the standard gradient accumulation pattern. Accelerate handles it. |
| Removing CUDA_ALLOC_CONF causes fragmentation | Per-chunk backward reduces peak memory below what the old multi-chunk path needed, so fragmentation headroom increases. Monitor on first HF Jobs run. |
| Global state carry (reset=False) causes numerical drift | Alpha (decay gate) now actually learns, providing a forgetting mechanism. Monitor g_norm across batches. |
| NLTM retrieval reorder changes model output for all configs | Yes, this is intentional — the old output was wrong per the paper. New runs will produce different (better) results. |
