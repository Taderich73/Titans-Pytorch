# Memory Gradient Flow Fix + Per-Chunk Backward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix data-dependent gate gradient flow (paper Eq. 3-4 alignment), simplify model to single-chunk forward, restructure training for per-chunk backward, and fix logging/init issues.

**Architecture:** Reorder NLTM retrieval to use updated state, remove multi-chunk forward from model classes (TitansMAC/MAG/MAL), move chunking + TBPTT to the training loop, add configurable gate init and state reset policy.

**Tech Stack:** PyTorch, HuggingFace Accelerate, pytest

---

### Task 1: Add `gate_decay_init` Config Field

**Files:**
- Modify: `src/titans/config.py:88-89`
- Modify: `src/titans/config.py:238`
- Test: `tests/test_config.py` (if exists, otherwise `tests/test_memory.py`)

- [ ] **Step 1: Add the config field**

In `src/titans/config.py`, replace:

```python
    # Gate initialization
    gate_decay_bias_init: float = -6.0
```

with:

```python
    # Gate initialization
    gate_decay_bias_init: float = -2.0
```

This changes the default from -6.0 (sigmoid ≈ 0.0025, gradient scaling ≈ 0.0025) to -2.0 (sigmoid ≈ 0.12, gradient scaling ≈ 0.10). The field name stays `gate_decay_bias_init` since it already exists — we're just changing the default. The `to_dict` and `from_dict` methods already serialize it.

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/ -x -q 2>&1 | tail -20`

Expected: All tests pass (the init value only affects newly constructed models; tests that check specific alpha values at init will need updating if any exist).

- [ ] **Step 3: Commit**

```bash
git add src/titans/config.py
git commit -m "feat(config): change gate_decay_bias_init default from -6 to -2

sigmoid(-2) ≈ 0.12 with gradient scaling ~0.10 — a 40x improvement over
the previous sigmoid(-6) ≈ 0.0025. The old init put the decay gate deep
in sigmoid saturation where the optimizer could barely move it.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Reorder NLTM Retrieval to Use Updated State

**Files:**
- Modify: `src/titans/memory.py:374-408`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write failing test — NLTM output depends on gate parameters**

Add to `tests/test_memory.py`:

```python
class TestNLTMRetrievalOrder:
    """Verify NeuralLongTermMemory retrieves from the UPDATED state (Eq. 3-4)."""

    def test_output_depends_on_gate_decay(self):
        """NLTM output must change when gate_decay_proj bias changes.

        If retrieval uses the old (input) state, the output is independent of
        alpha and this test fails. If retrieval uses new_state (as Eq. 3-4
        requires), the output depends on alpha via new_state.weights.
        """
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
        )
        mem = NeuralLongTermMemory(config)
        mem.eval()

        x = torch.randn(1, 16, 32)
        state = mem.init_state(1)

        # Forward with original bias
        out1, _ = mem(x, state=state, return_state=True)

        # Change gate_decay_proj bias and forward again with same input + state
        with torch.no_grad():
            mem.gate_decay_proj.bias.fill_(5.0)  # very different from init
        out2, _ = mem(x, state=state, return_state=True)

        # If retrieval uses updated state, outputs must differ
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "NLTM output is independent of gate_decay_proj — retrieval likely "
            "uses old state instead of updated state (violates Eq. 3-4)"
        )

    def test_gate_projections_receive_gradients_through_output(self):
        """Gate projections must have nonzero gradients from loss through output.

        This tests the direct gradient path: loss -> output -> proj_out ->
        retrieved -> forward_with_weights(new_state.weights) -> alpha ->
        gate_decay_proj.bias.
        """
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
            memory_objective="huber", huber_delta_init=-10.0,
        )
        mem = NeuralLongTermMemory(config)
        mem.train()

        x = torch.randn(2, 16, 32)
        state = mem.init_state(2)

        output, new_state = mem(x, state=state, return_state=True)
        loss = output.sum()
        loss.backward()

        gate_names = ["gate_decay_proj", "gate_lr_proj", "gate_momentum_proj",
                      "gate_delta_proj"]
        for name in gate_names:
            proj = getattr(mem, name, None)
            if proj is None:
                continue
            assert proj.bias.grad is not None, f"{name}.bias.grad is None"
            assert proj.bias.grad.abs().max() > 0, f"{name}.bias.grad is zero"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/test_memory.py::TestNLTMRetrievalOrder -x -v 2>&1 | tail -20`

Expected: FAIL — `test_output_depends_on_gate_decay` fails because the current code retrieves from old state, making output independent of gate_decay_proj.

- [ ] **Step 3: Reorder retrieval in NLTM.forward**

In `src/titans/memory.py`, replace lines 374-408 (the block from `retrieved = ...` through `output = self.proj_out(retrieved)`):

```python
        retrieved = self.memory.forward_with_weights(q, state.weights)

        # Data-dependent gates
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.sigmoid(self.gate_decay_proj(x_mean))
        theta = torch.sigmoid(self.gate_lr_proj(x_mean)) * self.config.memory_lr
        eta = torch.sigmoid(self.gate_momentum_proj(x_mean)) * self.config.memory_momentum
        alpha = torch.mean(alpha)
        theta = torch.mean(theta)
        eta = torch.mean(eta)

        if self.memory_objective == "huber":
            delta_val = torch.sigmoid(self.gate_delta_proj(x_mean))
            delta_val = torch.mean(delta_val) * self.config.memory_error_clip
            self._current_delta = delta_val

        if memory_gate is not None:
            lr_scale = memory_gate

        theta = theta * lr_scale

        if len(state.weights) == 1:
            new_state = self._parallel_memory_update_linear(
                k, v, state, alpha, theta, eta
            )
        else:
            delta_val = getattr(self, "_current_delta", None)
            grads = self._compute_gradients(k, v, state.weights, delta=delta_val)
            new_momentum = [eta * m - theta * g for m, g in zip(state.momentum, grads)]
            new_weights = [
                (1 - alpha) * w + s for w, s in zip(state.weights, new_momentum)
            ]
            new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        output = self.proj_out(retrieved)
```

with:

```python
        # Data-dependent gates
        x_mean = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.sigmoid(self.gate_decay_proj(x_mean))
        theta = torch.sigmoid(self.gate_lr_proj(x_mean)) * self.config.memory_lr
        eta = torch.sigmoid(self.gate_momentum_proj(x_mean)) * self.config.memory_momentum
        alpha = torch.mean(alpha)
        theta = torch.mean(theta)
        eta = torch.mean(eta)

        if self.memory_objective == "huber":
            delta_val = torch.sigmoid(self.gate_delta_proj(x_mean))
            delta_val = torch.mean(delta_val) * self.config.memory_error_clip
            self._current_delta = delta_val

        if memory_gate is not None:
            lr_scale = memory_gate

        theta = theta * lr_scale

        if len(state.weights) == 1:
            new_state = self._parallel_memory_update_linear(
                k, v, state, alpha, theta, eta
            )
        else:
            delta_val = getattr(self, "_current_delta", None)
            grads = self._compute_gradients(k, v, state.weights, delta=delta_val)
            new_momentum = [eta * m - theta * g for m, g in zip(state.momentum, grads)]
            new_weights = [
                (1 - alpha) * w + s for w, s in zip(state.weights, new_momentum)
            ]
            new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        # Retrieve from the UPDATED state (Titans Eq. 3-4: o_t = f(W_t, q_t)).
        # This puts alpha, theta, eta in the output's computation graph so gate
        # projections receive gradients from the LM loss.
        retrieved = self.memory.forward_with_weights(q, new_state.weights)
        output = self.proj_out(retrieved)
```

Also update the comment block at line 411-418 to reflect the new semantics:

```python
        if return_state:
            # Detach controls whether the RETURNED state has its autograd graph.
            # After the retrieval reorder, output already depends on new_state
            # (via forward_with_weights), so gates receive gradients within the
            # current chunk regardless of this flag. The flag now only controls
            # whether cross-chunk gradient flow is possible through the returned
            # state. Kept for config serialization compatibility.
            must_detach = (
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/test_memory.py -x -v 2>&1 | tail -30`

Expected: All tests pass, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add src/titans/memory.py tests/test_memory.py
git commit -m "fix(memory): reorder NLTM retrieval to use updated state (Eq. 3-4)

NeuralLongTermMemory.forward now computes the state update BEFORE
retrieval, so output = proj_out(forward_with_weights(q, new_state.weights))
instead of forward_with_weights(q, state.weights). This aligns with
Titans paper Eq. 3-4 (o_t = f(W_t, q_t)) and puts alpha, theta, eta in
the output computation graph, enabling gate gradient flow through loss.

The standalone retrieve() method is NOT changed — it takes an explicit
state parameter for MCA context retrieval (Eq. 21: h_t = M*_{t-1}(q_t)).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Clean Up HierarchicalMemory

**Files:**
- Modify: `src/titans/tnt_memory.py:145-234`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing test — HierarchicalMemory uses NLTM output directly**

Add to `tests/test_models.py`:

```python
class TestHierarchicalMemoryCleanup:
    """Verify HierarchicalMemory uses NLTM outputs directly (no redundant retrieve)."""

    def test_tnt_forward_output_matches_combined_nltm_outputs(self):
        """HierarchicalMemory.forward output should be proj_out of combined
        global + local NLTM outputs, not a separate retrieval pass."""
        config = TitansConfig(
            dim=32, num_heads=4, num_layers=1, vocab_size=64,
            chunk_size=16, num_memory_layers=1, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
        )
        from titans.tnt_memory import HierarchicalMemory
        hm = HierarchicalMemory(config)
        hm.train()

        x = torch.randn(2, 16, 32)
        output, new_state = hm(x, state=None)

        # Output must have gradient path to gate projections
        loss = output.sum()
        loss.backward()

        global_nltm = hm.global_memory.memory
        assert global_nltm.gate_decay_proj.bias.grad is not None
        assert global_nltm.gate_decay_proj.bias.grad.abs().max() > 0
```

- [ ] **Step 2: Run test to verify it passes (it should already pass with Task 2's NLTM fix)**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/test_models.py::TestHierarchicalMemoryCleanup -x -v 2>&1 | tail -20`

Expected: PASS (the NLTM fix from Task 2 already gives correct outputs; this test locks in the behavior before refactoring).

- [ ] **Step 3: Refactor HierarchicalMemory.forward to use NLTM outputs directly**

In `src/titans/tnt_memory.py`, replace the `forward` method (lines 145-217):

```python
    def forward(
        self, x: torch.Tensor, state: TNTMemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TNTMemoryState]:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if state is None:
            state = self.init_state(batch_size)

        global_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_global_memory
            else 1.0
        )
        local_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_local_memory
            else 1.0
        )

        # 1. Update global memory
        _, new_global_state = self.global_memory(
            x, state=state.global_state, lr_scale=global_lr_scale
        )

        # 2. Update each local memory
        new_local_states = []
        new_qk_projections = []
        new_step_counters = []

        for i, local_mem in enumerate(self.local_memories):
            local_state, counter = local_mem.maybe_reset(
                state.local_states[i],
                state.local_step_counters[i],
                batch_size=batch_size,
            )

            if counter == 0 and state.local_step_counters[i] > 0:
                qk_carry = torch.zeros(self.config.dim, self.config.dim, device=x.device)
            else:
                qk_carry = state.qk_projections[i]

            needs_keys = local_mem.qk_proj is not None
            if needs_keys:
                local_out, new_local_state, normed_keys = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale, return_keys=True,
                )
            else:
                local_out, new_local_state = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale,
                )
            new_local_states.append(new_local_state)

            if needs_keys:
                new_carry = local_mem.qk_proj.update_carry(normed_keys, qk_carry)
                new_qk_projections.append(new_carry)
            else:
                new_qk_projections.append(qk_carry)

            new_step_counters.append(counter + seq_len)

        new_state = TNTMemoryState(
            global_state=new_global_state,
            local_states=new_local_states,
            local_inits=state.local_inits,
            qk_projections=new_qk_projections,
            local_step_counters=new_step_counters,
        )

        # 3. Retrieve
        output = self.retrieve(x, new_state)
        return output, new_state
```

with:

```python
    def forward(
        self, x: torch.Tensor, state: TNTMemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TNTMemoryState]:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if state is None:
            state = self.init_state(batch_size)

        global_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_global_memory
            else 1.0
        )
        local_lr_scale = (
            memory_gate
            if memory_gate is not None and self.config.attnres_modulate_local_memory
            else 1.0
        )

        # 1. Global memory: NLTM.forward now retrieves from updated state
        #    (Eq. 3-4), so global_out is the correct retrieval output.
        global_out, new_global_state = self.global_memory(
            x, state=state.global_state, lr_scale=global_lr_scale
        )

        # 2. Update each local memory — local_out is likewise already the
        #    correct retrieval from the updated local state.
        local_outs = []
        new_local_states = []
        new_qk_projections = []
        new_step_counters = []

        for i, local_mem in enumerate(self.local_memories):
            local_state, counter = local_mem.maybe_reset(
                state.local_states[i],
                state.local_step_counters[i],
                batch_size=batch_size,
            )

            if counter == 0 and state.local_step_counters[i] > 0:
                qk_carry = torch.zeros(self.config.dim, self.config.dim, device=x.device)
            else:
                qk_carry = state.qk_projections[i]

            needs_keys = local_mem.qk_proj is not None
            if needs_keys:
                local_out, new_local_state, normed_keys = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale, return_keys=True,
                )
            else:
                local_out, new_local_state = local_mem(
                    x, state=local_state, lr_scale=local_lr_scale,
                )
            local_outs.append(local_out)
            new_local_states.append(new_local_state)

            if needs_keys:
                new_carry = local_mem.qk_proj.update_carry(normed_keys, qk_carry)
                new_qk_projections.append(new_carry)
            else:
                new_qk_projections.append(qk_carry)

            new_step_counters.append(counter + seq_len)

        new_state = TNTMemoryState(
            global_state=new_global_state,
            local_states=new_local_states,
            local_inits=state.local_inits,
            qk_projections=new_qk_projections,
            local_step_counters=new_step_counters,
        )

        # 3. Combine NLTM outputs directly instead of a separate retrieval pass.
        combined = global_out
        for local_out in local_outs:
            combined = combined + local_out
        output = self.proj_out(combined)
        return output, new_state
```

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/ -x -v 2>&1 | tail -30`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/titans/tnt_memory.py tests/test_models.py
git commit -m "refactor(tnt_memory): use NLTM outputs directly in HierarchicalMemory

After the NLTM retrieval reorder, each memory module's forward() already
returns output retrieved from the updated state. HierarchicalMemory no
longer needs its own separate retrieve() call — it combines the NLTM
outputs directly via proj_out(global_out + sum(local_outs)). Eliminates
one redundant retrieval pass per forward call.

The standalone retrieve() method is kept for external callers (MCA
context retrieval, eval).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Simplify Model Forward to Single-Chunk

**Files:**
- Modify: `src/titans/models.py:1-214` (remove checkpoint helpers)
- Modify: `src/titans/models.py:507-564` (TitansMAC.forward)
- Modify: `src/titans/models.py:706-755` (TitansMAG.forward)
- Modify: `src/titans/models.py:894-943` (TitansMAL.forward)
- Modify: `src/titans/config.py:104-114` (remove use_chunk_checkpointing)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write test — single-chunk forward produces correct output**

Add to `tests/test_models.py`:

```python
class TestSingleChunkForward:
    """Verify the simplified single-chunk forward API."""

    def test_forward_single_chunk(self):
        """forward() with seq_len == chunk_size works and returns logits + states."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
        )
        model = TitansMAC(config)
        model.train()

        ids = torch.randint(0, 256, (2, 32))
        logits, states = model(ids, states=None)

        assert logits.shape == (2, 32, 256)
        assert len(states) == 2
        assert states[0] is not None

    def test_forward_rejects_multi_chunk_input(self):
        """forward() should raise when seq_len > chunk_size."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
        )
        model = TitansMAC(config)

        ids = torch.randint(0, 256, (2, 64))  # 2 chunks
        with pytest.raises(ValueError, match="chunk_size"):
            model(ids, states=None)

    def test_manual_chunk_loop_matches_gate_gradients(self):
        """Manual chunk loop (simulating training) must give gate gradients."""
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, num_memory_layers=2, num_persistent_tokens=4,
            use_tnt=True, local_chunk_sizes=[8], local_shard_length=128,
            memory_objective="huber", huber_delta_init=-10.0,
        )
        model = TitansMAC(config)
        model.train()

        # Simulate 2-chunk training step
        ids = torch.randint(0, 256, (2, 64))
        chunks = ids.split(32, dim=1)
        states = None
        total_loss = 0.0

        for chunk in chunks:
            logits, states = model(chunk, states=states)
            loss = F.cross_entropy(logits.view(-1, 256), chunk.view(-1))
            (loss / len(chunks)).backward()
            total_loss += loss.item()
            states = [s.detach() if s is not None else None for s in states]

        # Gate gradients must be nonzero
        block0 = model.blocks[0]
        global_nltm = block0.memory.global_memory.memory
        assert global_nltm.gate_decay_proj.bias.grad is not None
        assert global_nltm.gate_decay_proj.bias.grad.abs().max() > 0
```

- [ ] **Step 2: Run test to see current state**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/test_models.py::TestSingleChunkForward::test_forward_single_chunk -x -v 2>&1 | tail -20`

Expected: `test_forward_single_chunk` passes (single-chunk already works). `test_forward_rejects_multi_chunk_input` fails (model currently accepts multi-chunk). `test_manual_chunk_loop_matches_gate_gradients` may pass or fail depending on import.

- [ ] **Step 3: Remove checkpoint helpers and multi-chunk forward**

In `src/titans/config.py`, remove lines 104-114 (the `use_chunk_checkpointing` field and its docstring). Also remove `"use_chunk_checkpointing": self.use_chunk_checkpointing,` from `to_dict()` at line 240.

In `src/titans/models.py`, remove:
- Lines 17-213: `_flatten_states_to_tuples`, `_unflatten_tuples_to_states`, `_run_process_chunk_checkpointed` and all their docstrings/comments
- The `from titans.memory import ... TNTMemoryState` can stay if other code uses it

Replace TitansMAC.forward (lines 507-564) with:

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState | TNTMemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState | TNTMemoryState]]:
        """Process a single chunk. Returns (logits, new_states).

        Args:
            input_ids: Token IDs, shape (B, seq_len) where seq_len <= chunk_size.
            states: Per-block memory states from a previous chunk, or None.

        Raises:
            ValueError: If seq_len > chunk_size. Callers must chunk externally.
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if seq_len > chunk_size:
            raise ValueError(
                f"seq_len ({seq_len}) > chunk_size ({chunk_size}). "
                f"Multi-chunk input is no longer supported in forward(). "
                f"Split input_ids into chunks and call forward() per chunk."
            )

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)
        x, new_states = process_chunk(
            self.blocks, x, states, self.config, self._step_count
        )

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states
```

Apply the identical pattern to TitansMAG.forward (lines 706-755) and TitansMAL.forward (lines 894-943). Same structure — add the ValueError for seq_len > chunk_size, remove the multi-chunk branch.

- [ ] **Step 4: Remove old multi-chunk tests, run suite**

Remove or update these tests in `tests/test_models.py`:
- `test_checkpointed_chunk_propagates_gradients` — remove (checkpointing helpers removed)
- `test_titans_mac_multi_chunk_checkpointed_gate_gradients` — remove
- `test_tnt_gate_gradients_with_multi_chunk_seq` — rewrite to use manual chunk loop (like `test_manual_chunk_loop_matches_gate_gradients` above)

Update `test_tnt_gate_projections_receive_gradients` if it uses `seq_len > chunk_size` — change to `seq_len == chunk_size`.

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/ -x -v 2>&1 | tail -40`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/titans/config.py src/titans/models.py tests/test_models.py
git commit -m "refactor(models): simplify forward to single-chunk, remove checkpointing

TitansMAC/MAG/MAL.forward() now only processes single chunks (seq_len <=
chunk_size) and raises ValueError otherwise. All chunking, state
threading, and TBPTT detach logic moves to the training loop.

Removes _flatten_states_to_tuples, _unflatten_tuples_to_states,
_run_process_chunk_checkpointed, and use_chunk_checkpointing config flag.
Per-chunk backward in the training loop bounds peak memory to one chunk
without activation checkpointing's ~33% compute overhead.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Per-Chunk Backward Training Loop

**Files:**
- Modify: `scripts/hf_pretrain.py:27-34` (remove CUDA_ALLOC_CONF)
- Modify: `scripts/hf_pretrain.py:233` (remove USE_CHUNK_CHECKPOINTING)
- Modify: `scripts/hf_pretrain.py:542-577` (training step)
- Modify: `scripts/hf_pretrain.py:265-291` (add RESET_GLOBAL_STATE_PER_BATCH)

- [ ] **Step 1: Remove CUDA_ALLOC_CONF early-set**

In `scripts/hf_pretrain.py`, remove lines 25-34 (the `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)` block and its comment). Move `import os` back to its alphabetical position with other imports.

- [ ] **Step 2: Remove USE_CHUNK_CHECKPOINTING constant**

Remove line 233: `USE_CHUNK_CHECKPOINTING = True`

Remove the corresponding `use_chunk_checkpointing=USE_CHUNK_CHECKPOINTING` kwarg from the TitansConfig construction (search for where config is built and remove this kwarg).

- [ ] **Step 3: Add RESET_GLOBAL_STATE_PER_BATCH constant**

Add near the other training constants (around line 268):

```python
RESET_GLOBAL_STATE_PER_BATCH = True  # set False to let global state carry across batches
```

- [ ] **Step 4: Rewrite the training step for per-chunk backward**

Replace the training step block (lines 542-577, from `try:` through the `except torch.cuda.OutOfMemoryError` handler, plus the memory_states detach/reset block through line 598):

```python
        try:
            with accelerator.accumulate(model):
                chunks = batch["input_ids"].split(CHUNK_SIZE, dim=1)
                label_chunks = batch["labels"].split(CHUNK_SIZE, dim=1)
                num_chunks = len(chunks)
                batch_loss = 0.0

                for chunk_ids, chunk_labels in zip(chunks, label_chunks):
                    logits, memory_states = model(chunk_ids, states=memory_states)
                    if _profile_this_step:
                        _log_mem(f"step {global_step:03d}: after forward")
                    chunk_loss = F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE), chunk_labels.view(-1)
                    )
                    accelerator.backward(chunk_loss / num_chunks)
                    if _profile_this_step:
                        _log_mem(f"step {global_step:03d}: after backward")
                    batch_loss += chunk_loss.item() / num_chunks

                    # Truncated BPTT: detach state at chunk boundary
                    if memory_states is not None:
                        memory_states = [
                            s.detach() if s is not None else None
                            for s in memory_states
                        ]

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if _profile_this_step:
                    _log_mem(f"step {global_step:03d}: after optimizer step")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"\n[mem] CUDA OOM at step {global_step}", flush=True)
            print(f"[mem] {oom}", flush=True)
            if torch.cuda.is_available():
                summary = torch.cuda.memory_summary(abbreviated=True)
                print(f"[mem] memory_summary:\n{summary}", flush=True)
            raise

        # Capture g_norm BEFORE optional global state reset
        _pre_reset_g_norm = None
        if memory_states is not None:
            try:
                g_state = getattr(memory_states[0], "global_state", None)
                if g_state is not None and hasattr(g_state, "weights") and len(g_state.weights) > 0:
                    _pre_reset_g_norm = g_state.weights[0].detach().float().norm().item()
            except Exception:
                pass

        # Optional per-batch global memory state reset
        if RESET_GLOBAL_STATE_PER_BATCH and memory_states is not None:
            unwrapped_for_reset = accelerator.unwrap_model(model)
            reset_batch_size = batch["input_ids"].shape[0]
            new_memory_states = []
            for block, state in zip(unwrapped_for_reset.blocks, memory_states):
                if state is None:
                    new_memory_states.append(None)
                    continue
                global_mem = getattr(
                    getattr(block, "memory", None), "global_memory", None
                )
                if global_mem is not None and hasattr(state, "global_state"):
                    fresh_global = global_mem.init_state(reset_batch_size)
                    new_memory_states.append(replace(state, global_state=fresh_global))
                else:
                    new_memory_states.append(state)
            memory_states = new_memory_states
```

- [ ] **Step 5: Update loss tracking**

Replace `running_loss += loss.item()` (line 600) with `running_loss += batch_loss` (since `batch_loss` is already the averaged loss from the chunk loop).

- [ ] **Step 6: Commit**

```bash
git add scripts/hf_pretrain.py
git commit -m "feat(training): per-chunk backward with configurable state reset

Restructure training step for per-chunk backward (truncated BPTT):
each chunk computes CE loss and runs backward immediately, then
detaches state before the next chunk. Peak activation memory is
bounded to one chunk without activation checkpointing overhead.

Add RESET_GLOBAL_STATE_PER_BATCH flag (default True) to control
whether global memory resets to fresh init each batch.

Remove CUDA_ALLOC_CONF early-set and USE_CHUNK_CHECKPOINTING
(no longer needed with per-chunk backward).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Fix Logging — g_norm, Alpha Precision, Gate Gradient

**Files:**
- Modify: `scripts/hf_pretrain.py:604-632` (logging block)

- [ ] **Step 1: Rewrite the logging block**

Replace the logging block (from `if global_step % LOG_EVERY == 0:` through `running_loss = 0.0`):

```python
        running_loss += batch_loss
        global_step += 1
        pbar.update(1)

        if global_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            lr = optimizer.param_groups[0]["lr"]
            postfix = {"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"}

            # Global memory state norm (captured BEFORE reset in the step above)
            if _pre_reset_g_norm is not None:
                postfix["g_norm"] = f"{_pre_reset_g_norm:.2e}"

            # Gate decay instrumentation: raw bias, sigmoid(bias), gradient
            try:
                unwrapped_for_log = accelerator.unwrap_model(model)
                block0 = unwrapped_for_log.blocks[0]
                gate_proj = getattr(
                    getattr(getattr(block0, "memory", None), "global_memory", None),
                    "memory",
                    None,
                )
                gate_proj = getattr(gate_proj, "gate_decay_proj", None)
                if gate_proj is not None:
                    raw_bias = gate_proj.bias.item()
                    alpha0 = torch.sigmoid(gate_proj.bias).item()
                    postfix["alpha"] = f"{alpha0:.6f}"
                    postfix["decay_bias"] = f"{raw_bias:.4f}"
                    if gate_proj.bias.grad is not None:
                        postfix["gate_grad"] = f"{gate_proj.bias.grad.item():.2e}"
            except Exception:
                pass

            pbar.set_postfix(**postfix)
            running_loss = 0.0
```

- [ ] **Step 2: Commit**

```bash
git add scripts/hf_pretrain.py
git commit -m "fix(training): fix g_norm logging, improve alpha and gate grad visibility

g_norm now reads global state norm BEFORE the per-batch reset (was
reading post-reset, showing a constant). Alpha logged at 6 decimal
places (was 4, hiding movement in sigmoid saturation). Raw decay
bias value and gate gradient norm added to postfix for direct
observability of optimizer dynamics.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update launch_hf_job.py CLI

**Files:**
- Modify: `scripts/launch_hf_job.py`

- [ ] **Step 1: Add --reset-global-state flag**

Find the training argument group in `scripts/launch_hf_job.py` and add:

```python
    train.add_argument(
        "--reset-global-state", type=str, default=None,
        choices=["true", "false"],
        help="Override RESET_GLOBAL_STATE_PER_BATCH (default: true)",
    )
```

Find where the launcher injects overrides into the training script and add:

```python
    if args.reset_global_state is not None:
        # Inject as a constant override at top of hf_pretrain.py
        # (follow the pattern used for other overrides)
```

Check the existing override injection pattern in the launcher and follow it exactly.

- [ ] **Step 2: Remove --chunk-checkpointing flag if present**

Search for `chunk-checkpointing` or `chunk_checkpointing` in `launch_hf_job.py` and remove the argument and any injection logic.

- [ ] **Step 3: Bump the titans SHA default**

After all changes are committed and pushed, update `scripts/launch_hf_job.py:135` default SHA to point to the new HEAD. **Do this as the final step after pushing the branch.**

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_hf_job.py
git commit -m "feat(scripts): add --reset-global-state flag, remove --chunk-checkpointing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Final Test Suite Pass + Cleanup

**Files:**
- Modify: `tests/test_models.py` (any remaining failures)
- Modify: `tests/test_memory.py` (any remaining failures)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/ -v 2>&1 | tail -60`

Fix any remaining failures. Common issues:
- Tests that construct TitansConfig with `use_chunk_checkpointing=True` — remove the kwarg
- Tests that pass `seq_len > chunk_size` to model.forward() — split into chunk loop
- Tests checking alpha ≈ 0.0025 (sigmoid(-6)) — update to ≈ 0.12 (sigmoid(-2))

- [ ] **Step 2: Run full suite again to confirm clean**

Run: `cd /Users/dlattka/Projects/titans-pytorch/.claude/worktrees/gallant-curran && python -m pytest tests/ -v 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 3: Commit any test fixes**

```bash
git add tests/
git commit -m "test: update test suite for new single-chunk API and gate init

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
