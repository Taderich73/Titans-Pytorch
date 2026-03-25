# SFT & LoRA Training Scripts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two self-contained training scripts (`scripts/sft.py`, `scripts/lora.py`) for supervised fine-tuning of Titans MLX models using streamed HuggingFace datasets with chat template formatting and loss masking.

**Architecture:** Both scripts are self-contained — no shared utils module. Each copies ~100 lines of training boilerplate from `scripts/pretrain.py` and adds SFT-specific data pipeline (streaming chat datasets, template formatting, loss masking). The LoRA script adds a `LoRALinear` wrapper and adapter save/load/merge utilities. Tests cover the new components (chat formatting, loss masking, LoRA module, adapter I/O) without requiring network access or GPU training.

**Tech Stack:** MLX, HuggingFace `datasets` (streaming), HuggingFace `transformers` (tokenizers), safetensors

**Spec:** `docs/superpowers/specs/2026-03-25-sft-lora-training-scripts-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `scripts/sft.py` | Full-parameter SFT: chat data pipeline, masked loss, training loop |
| Create | `scripts/lora.py` | LoRA fine-tuning: LoRALinear, adapter save/load/merge, training loop |
| Create | `tests/test_sft.py` | Tests for chat formatting, loss masking, streaming dataset |
| Create | `tests/test_lora.py` | Tests for LoRALinear, layer wrapping, adapter save/load/merge |

No existing files are modified.

---

### Task 1: Chat Template Formatting & Loss Mask Generation

**Files:**
- Create: `tests/test_sft.py`

This task builds and tests the two core SFT functions in isolation before they go into the full script. We write the tests first, then write the functions directly in `scripts/sft.py` in Task 2.

- [ ] **Step 1: Write tests for ChatML fallback formatting**

Create `tests/test_sft.py`:

```python
"""Tests for SFT chat formatting and loss masking."""

import pytest


class TestFormatChatML:
    """Tests for ChatML fallback formatter."""

    def test_single_turn(self):
        """Single user/assistant turn produces correct ChatML."""
        # Import will fail until sft.py exists — that's expected
        from scripts.sft import format_chatml

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_chatml(messages)
        expected = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>\n"
        )
        assert result == expected

    def test_multi_turn(self):
        """Multi-turn conversation."""
        from scripts.sft import format_chatml

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        result = format_chatml(messages)
        assert result.count("<|im_start|>user") == 2
        assert result.count("<|im_start|>assistant") == 2
        assert result.count("<|im_end|>") == 4

    def test_system_message(self):
        """System messages are formatted correctly."""
        from scripts.sft import format_chatml

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = format_chatml(messages)
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result


class TestBuildLossMask:
    """Tests for loss mask generation."""

    def test_assistant_only_mask(self):
        """Mask is 1 only for assistant content tokens."""
        from scripts.sft import build_loss_mask

        # Simulate token_ids and role boundaries
        # Format: [user_start, user_content..., user_end, asst_start, asst_content..., asst_end]
        # Role spans: list of (role, start_idx, end_idx) — end_idx is exclusive
        role_spans = [
            ("user", 0, 5),       # tokens 0-4: user turn (including delimiters)
            ("assistant", 5, 10), # tokens 5-9: assistant turn (including delimiters)
        ]
        # assistant delimiters: token 5 is <|im_start|>assistant\n, token 9 is <|im_end|>
        assistant_content_spans = [(6, 9)]  # content only, excluding role prefix
        seq_len = 10

        mask = build_loss_mask(seq_len, assistant_content_spans, include_eos=True, eos_positions=[9])
        assert len(mask) == seq_len
        # User tokens: all 0
        assert mask[0:5] == [0, 0, 0, 0, 0]
        # Assistant role prefix (token 5): 0
        assert mask[5] == 0
        # Assistant content (tokens 6-8): 1
        assert mask[6:9] == [1, 1, 1]
        # EOS token (token 9): 1 (model should learn to stop)
        assert mask[9] == 1

    def test_train_on_all(self):
        """When train_on_all=True, entire mask is 1."""
        from scripts.sft import build_loss_mask

        mask = build_loss_mask(10, [], include_eos=False, eos_positions=[], train_on_all=True)
        assert mask == [1] * 10

    def test_multi_turn_mask(self):
        """Multiple assistant turns each get masked correctly."""
        from scripts.sft import build_loss_mask

        # Two assistant turns
        assistant_content_spans = [(3, 5), (8, 10)]
        eos_positions = [5, 10]
        mask = build_loss_mask(11, assistant_content_spans, include_eos=True, eos_positions=eos_positions)
        # Tokens 3,4 = content, 5 = eos -> 1
        assert mask[3:6] == [1, 1, 1]
        # Tokens 8,9 = content, 10 = eos -> 1
        assert mask[8:11] == [1, 1, 1]
        # Everything else: 0
        assert sum(mask) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_sft.py -v 2>&1 | head -20`
Expected: FAIL (ImportError — `scripts.sft` doesn't exist yet)

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_sft.py
git commit -m "test: add SFT chat formatting and loss mask tests"
```

---

### Task 2: SFT Script — Core Data Pipeline

**Files:**
- Create: `scripts/sft.py`

Build the SFT script with chat formatting, loss masking, and the streaming dataset class. This task focuses on the data pipeline — the training loop comes in Task 3.

- [ ] **Step 1: Create `scripts/sft.py` with imports, ChatML formatting, and loss mask functions**

Write the file with:
- All imports (mlx, numpy, transformers, datasets, tqdm, etc.) with optional import guards (same pattern as `scripts/pretrain.py`)
- `format_chatml(messages) -> str` — formats messages list into ChatML string
- `build_loss_mask(seq_len, assistant_content_spans, include_eos, eos_positions, train_on_all=False) -> list[int]` — returns per-token mask
- `tokenize_chat(messages, tokenizer, max_len, train_on_all=False) -> dict` — tokenizes a messages list, applies chat template (tokenizer's if available, ChatML fallback), returns `{"input_ids": [...], "labels": [...], "loss_mask": [...]}` with proper masking

Key implementation details for `tokenize_chat`:
- Check `getattr(tokenizer, 'chat_template', None) is not None` before using `apply_chat_template`
- For ChatML fallback: add `<|im_start|>`, `<|im_end|>` as special tokens if not present
- Build the mask by tokenizing each role turn individually to track token boundaries
- Truncate to `max_len`, pad if needed
- Return `input_ids[:-1]`, `labels[1:]`, `loss_mask[1:]` (shifted for next-token prediction)

- [ ] **Step 2: Run tests to verify chat formatting and loss mask pass**

Run: `uv run python -m pytest tests/test_sft.py -v`
Expected: All tests in `TestFormatChatML` and `TestBuildLossMask` PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/sft.py
git commit -m "feat(sft): add chat template formatting and loss mask generation"
```

---

### Task 3: SFT Script — Streaming Dataset & Training Loop

**Files:**
- Modify: `scripts/sft.py`

Add the streaming dataset class, masked loss function, training loop, checkpointing, and CLI. This copies the training boilerplate from `scripts/pretrain.py` and adapts it.

- [ ] **Step 1: Add `SFTStreamingDataset` class**

Append to `scripts/sft.py`:
- `SFTStreamingDataset` class that:
  - Takes `dataset_name, tokenizer, max_len, subset=None, split="train", seed=42, messages_field="messages", train_on_all=False, buffer_size=1000`
  - Streams from HuggingFace with shuffle buffer
  - Yields tokenized chat examples via `tokenize_chat()`
  - `get_batch(batch_size)` method returns `{"input_ids": mx.array, "labels": mx.array, "loss_mask": mx.array}` or `None` if exhausted
  - Pads batch sequences to the longest in the batch (not max_len) for efficiency

- [ ] **Step 2: Add masked loss function**

```python
def masked_loss_fn(model, input_ids, labels, loss_mask):
    """Cross-entropy loss masked to assistant tokens only."""
    logits, _ = model(input_ids)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    per_token = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="none")
    loss = (per_token * mask_flat).sum() / mask_flat.sum().clip(min=1)
    return loss, logits
```

- [ ] **Step 3: Copy training utilities from pretrain.py**

Copy these functions verbatim from `scripts/pretrain.py` into `scripts/sft.py`:
- `create_model` (lines 391-403)
- `count_parameters` (lines 406-421)
- `get_lr_schedule` (lines 429-441)
- `_tree_add`, `_tree_scale`, `_eval_grads` (lines 469-504)
- `sanitize_and_clip_grads` (lines 530-601)
- `apply_gradients` (lines 604-625) — includes weight tying re-application
- `save_checkpoint` (lines 666-728) — add `chat_template` to metadata dict
- `prune_checkpoints` (lines 731-742)
- `_remap_tnt_keys` (lines 745-754)
- `load_checkpoint` (lines 757-805)
- `evaluate` (lines 628-658) — **adapt for streaming**: the pretrain.py version uses `len(dataset)` and `get_batch(indices)` with index-based access. Rewrite to consume N batches from a streaming dataset's `get_batch(batch_size)` method instead. Accept a `num_batches` parameter, loop `num_batches` times calling `dataset.get_batch(batch_size)`, and average the loss. The loss_mask from each batch should be used (call `masked_loss_fn` not `loss_fn`).

Modify `compute_grads` to use `masked_loss_fn` instead of `loss_fn`:
```python
def compute_grads(model, input_ids, labels, loss_mask):
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: masked_loss_fn(m, input_ids, labels, loss_mask)[0]
    )
    loss, grads = loss_and_grad_fn(model)
    return loss, grads
```

- [ ] **Step 4: Add `SFTConfig` dataclass**

Similar to pretrain.py's `TrainingConfig` but with SFT-specific defaults:
- `lr: float = 2e-5`
- `gradient_accumulation_steps: int = 8`
- `seq_len: int = 2048`
- `train_on_all: bool = False`
- `messages_field: str = "messages"`
- `chat_template: str = "auto"` (auto = use tokenizer's if available, else chatml)
- Remove `data_path`, `synthetic_samples` (not needed for SFT)
- Keep all other fields from pretrain.py's `TrainingConfig`

- [ ] **Step 5: Add training loop**

Copy the streaming dataset branch of pretrain.py's `train()` function (lines 1068-1229) and adapt:
- Pass `loss_mask` from batch to `compute_grads`
- Add `chat_template` field to checkpoint metadata
- Remove fixed-size dataset branch (SFT is streaming only)
- Log `masked_tokens_ratio` (fraction of tokens contributing to loss) periodically

- [ ] **Step 6: Add `main()` with argparse**

Argparse following pretrain.py's pattern. Key differences:
- `--dataset` is required (no synthetic/local-file fallback)
- `--lr` default is `2e-5`
- `--gradient-accumulation-steps` default is `8`
- `--seq-len` default is `2048`
- Add `--train-on-all` flag
- Add `--messages-field` flag (default: `"messages"`)
- Add `--tnt-stage` flag (default: 1)
- No `--data` or `--synthetic-samples` flags

- [ ] **Step 7: Run full test suite**

Run: `uv run python -m pytest tests/test_sft.py -v`
Expected: All PASS

- [ ] **Step 8: Smoke test the script's argparse**

Run: `uv run python scripts/sft.py --help`
Expected: Shows help text with all flags, no import errors

- [ ] **Step 9: Commit**

```bash
git add scripts/sft.py
git commit -m "feat(sft): add streaming dataset, masked loss, and training loop"
```

---

### Task 4: LoRA Module & Layer Wrapping

**Files:**
- Create: `tests/test_lora.py`
- Create: `scripts/lora.py` (partial — LoRA module + wrapping only)

- [ ] **Step 1: Write LoRA tests**

Create `tests/test_lora.py`:

```python
"""Tests for LoRA module, layer wrapping, and adapter I/O."""

import json
import math
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestLoRALinear:
    """Tests for the LoRALinear wrapper module."""

    def test_identity_at_init(self):
        """LoRA output equals base output at init (B is zeros)."""
        from scripts.lora import LoRALinear

        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        mx.eval(lora.parameters())

        x = mx.random.normal((2, 10, 32))
        base_out = base(x)
        lora_out = lora(x)
        mx.eval(base_out, lora_out)

        np.testing.assert_allclose(
            np.array(lora_out), np.array(base_out), atol=1e-6
        )

    def test_output_shape(self):
        """LoRA produces same shape as base linear."""
        from scripts.lora import LoRALinear

        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        mx.eval(lora.parameters())

        x = mx.random.normal((2, 10, 32))
        out = lora(x)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_trainable_params(self):
        """Only lora_A and lora_B should be trainable."""
        from scripts.lora import LoRALinear

        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        base.freeze()
        mx.eval(lora.parameters())

        trainable = lora.trainable_parameters()
        # Should have lora_A and lora_B
        flat = dict(tree_flatten(trainable))
        assert any("lora_A" in k for k in flat), f"No lora_A found in {list(flat.keys())}"
        assert any("lora_B" in k for k in flat), f"No lora_B found in {list(flat.keys())}"
        # Base weight should NOT be trainable
        assert not any("base" in k for k in flat), f"Base params found in trainable: {list(flat.keys())}"

    def test_nonzero_after_training_step(self):
        """After a gradient step, LoRA output should differ from base."""
        from scripts.lora import LoRALinear

        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        base.freeze()
        mx.eval(lora.parameters())

        x = mx.random.normal((2, 10, 32))
        target = mx.random.normal((2, 10, 64))

        def loss_fn(model):
            return mx.mean((model(x) - target) ** 2)

        loss_grad = nn.value_and_grad(lora, loss_fn)
        loss, grads = loss_grad(lora)
        optimizer = mx.optimizers.Adam(learning_rate=0.01)
        optimizer.update(lora, grads)
        mx.eval(lora.parameters(), optimizer.state)

        # Now lora_B should no longer be all zeros
        assert not mx.all(lora.lora_B == 0).item()


class TestWrapModel:
    """Tests for wrapping model layers with LoRA."""

    def test_wrap_attn_layers(self):
        """Wrapping with 'attn' preset targets proj_q/k/v/out."""
        from scripts.lora import wrap_lora_layers

        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model = TitansMAC(config)
        mx.eval(model.parameters())

        wrapped_names = wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0, dropout=0.0)

        assert len(wrapped_names) > 0
        # Should include attention projections
        assert any("proj_q" in n for n in wrapped_names)
        assert any("proj_k" in n for n in wrapped_names)
        assert any("proj_v" in n for n in wrapped_names)
        assert any("proj_out" in n for n in wrapped_names)
        # Should NOT include FFN
        assert not any("gate_proj" in n for n in wrapped_names)

    def test_wrap_ffn_layers(self):
        """Wrapping with 'ffn' preset targets gate/up/down projections."""
        from scripts.lora import wrap_lora_layers

        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model = TitansMAC(config)
        mx.eval(model.parameters())

        wrapped_names = wrap_lora_layers(model, targets="ffn", rank=4, alpha=8.0, dropout=0.0)

        assert any("gate_proj" in n for n in wrapped_names)
        assert any("up_proj" in n for n in wrapped_names)
        assert any("down_proj" in n for n in wrapped_names)
        assert not any("proj_q" in n for n in wrapped_names)

    def test_embed_head_never_wrapped(self):
        """embed and head layers must never be wrapped (weight tying)."""
        from scripts.lora import wrap_lora_layers

        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model = TitansMAC(config)
        mx.eval(model.parameters())

        wrapped_names = wrap_lora_layers(model, targets="all", rank=4, alpha=8.0, dropout=0.0)

        assert not any("embed" in n for n in wrapped_names)
        assert not any("head" in n for n in wrapped_names)

    def test_forward_after_wrap(self):
        """Model forward pass still works after LoRA wrapping."""
        from scripts.lora import wrap_lora_layers

        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model = TitansMAC(config)
        mx.eval(model.parameters())

        wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0, dropout=0.0)

        x = mx.random.randint(0, 50, (1, 16))
        logits, states = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 16, 50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_lora.py -v 2>&1 | head -20`
Expected: FAIL (ImportError — `scripts.lora` doesn't exist yet)

- [ ] **Step 3: Create `scripts/lora.py` with LoRALinear and wrap_lora_layers**

Write the initial `scripts/lora.py` with:

`LoRALinear(nn.Module)`:
- `__init__(self, base, rank, alpha, dropout)`: stores frozen base, creates `lora_A` (Kaiming init: `normal * 1/sqrt(rank)`), `lora_B` (zeros), scale = alpha/rank, optional `nn.Dropout`
- `__call__(self, x)`: `base(x) + dropout(x) @ A @ B * scale`

`wrap_lora_layers(model, targets, rank, alpha, dropout) -> list[str]`:
- Define target patterns:
  - `attn`: `{"proj_q", "proj_k", "proj_v", "proj_out"}`
  - `ffn`: `{"gate_proj", "up_proj", "down_proj"}`
  - `memory`: match `memory` in path AND layer is `nn.Linear`
  - `all`: any `nn.Linear`
- Skip any path containing `embed` or `head`
- Replace matching `nn.Linear` with `LoRALinear` wrapper
- Freeze base weights via `base.freeze()` inside `LoRALinear.__init__`
- Return list of wrapped layer full dotted paths

**Implementation note — MLX module tree traversal:** Do NOT use `model.named_modules()` — it returns short relative names, not full dotted paths. Instead, write a recursive walk that builds full paths:

```python
def _recursive_find_linear(module, prefix=""):
    """Walk module tree, yield (full_dotted_path, attr_name, parent, nn.Linear)."""
    for attr_name in dir(module):
        child = getattr(module, attr_name, None)
        if isinstance(child, nn.Linear):
            full_path = f"{prefix}.{attr_name}" if prefix else attr_name
            yield full_path, attr_name, module, child
        elif isinstance(child, nn.Module):
            full_path = f"{prefix}.{attr_name}" if prefix else attr_name
            yield from _recursive_find_linear(child, full_path)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    full_path = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                    if isinstance(item, nn.Linear):
                        yield full_path, i, child, item
                    else:
                        yield from _recursive_find_linear(item, full_path)
```

Then replace via `setattr(parent, attr_name, LoRALinear(...))` or `parent[i] = LoRALinear(...)` for list items.

- [ ] **Step 4: Run LoRA tests**

Run: `uv run python -m pytest tests/test_lora.py::TestLoRALinear tests/test_lora.py::TestWrapModel -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_lora.py scripts/lora.py
git commit -m "feat(lora): add LoRALinear module and model wrapping"
```

---

### Task 5: LoRA Adapter Save/Load/Merge

**Files:**
- Modify: `tests/test_lora.py` (add adapter I/O tests)
- Modify: `scripts/lora.py` (add save/load/merge functions)

- [ ] **Step 1: Add adapter I/O tests to `tests/test_lora.py`**

Append to `tests/test_lora.py`:

```python
class TestAdapterIO:
    """Tests for adapter saving, loading, and merging."""

    def _make_wrapped_model(self):
        """Helper: create a small wrapped MAC model."""
        from scripts.lora import wrap_lora_layers

        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model = TitansMAC(config)
        mx.eval(model.parameters())
        wrapped = wrap_lora_layers(model, targets="attn", rank=4, alpha=8.0, dropout=0.0)
        return model, config, wrapped

    def test_save_and_load_adapters(self):
        """Save adapters, load into fresh model, outputs match."""
        from scripts.lora import save_adapters, load_adapters, wrap_lora_layers, _find_lora_modules

        # Create base model and save its weights before wrapping
        config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, num_persistent_tokens=2, num_memory_layers=1,
            use_conv=False, use_rope=False, dropout=0.0,
        )
        model1 = TitansMAC(config)
        mx.eval(model1.parameters())

        # Save base weights before LoRA wrapping
        base_weights = list(tree_flatten(model1.parameters()))

        wrap_lora_layers(model1, targets="attn", rank=4, alpha=8.0, dropout=0.0)

        # Perturb lora_B so output differs from base
        for _, module in _find_lora_modules(model1):
            module.lora_B = mx.ones_like(module.lora_B) * 0.1
        mx.eval(model1.parameters())

        x = mx.random.randint(0, 50, (1, 16))
        out1, _ = model1(x)
        mx.eval(out1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_adapters(
                model1, Path(tmpdir) / "adapters",
                meta={
                    "rank": 4, "alpha": 8.0, "dropout": 0.0,
                    "lora_targets": "attn",
                    "model_type": "mac",
                    "base_checkpoint": "test",
                },
            )

            # Verify files exist
            assert (Path(tmpdir) / "adapters.safetensors").exists()
            assert (Path(tmpdir) / "adapters.meta.json").exists()

            # Load into fresh model using saved base weights
            model2 = TitansMAC(config)
            model2.load_weights(base_weights)
            mx.eval(model2.parameters())
            wrap_lora_layers(model2, targets="attn", rank=4, alpha=8.0, dropout=0.0)
            load_adapters(model2, Path(tmpdir) / "adapters")
            mx.eval(model2.parameters())

            out2, _ = model2(x)
            mx.eval(out2)

            np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-5)

    def test_merge_lora_weights(self):
        """After merging, output matches LoRA output but model has no LoRA modules."""
        from scripts.lora import merge_lora_weights, wrap_lora_layers, _find_lora_modules

        model, config, wrapped = self._make_wrapped_model()

        # Perturb lora_B
        for _, module in _find_lora_modules(model):
            module.lora_B = mx.ones_like(module.lora_B) * 0.1
        mx.eval(model.parameters())

        x = mx.random.randint(0, 50, (1, 16))
        out_before, _ = model(x)
        mx.eval(out_before)

        merge_lora_weights(model)

        out_after, _ = model(x)
        mx.eval(out_after)

        np.testing.assert_allclose(np.array(out_before), np.array(out_after), atol=1e-5)

    def test_adapter_meta_json(self):
        """Metadata JSON contains expected fields."""
        from scripts.lora import save_adapters

        model, config, wrapped = self._make_wrapped_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_adapters(
                model, Path(tmpdir) / "adapters",
                meta={
                    "rank": 4, "alpha": 8.0, "dropout": 0.0,
                    "lora_targets": "attn",
                    "model_type": "mac",
                    "base_checkpoint": "test",
                    "chat_template": "chatml",
                    "tokenizer": "gpt2",
                },
            )

            with open(Path(tmpdir) / "adapters.meta.json") as f:
                meta = json.load(f)

            assert meta["rank"] == 4
            assert meta["alpha"] == 8.0
            assert meta["lora_targets"] == "attn"
            assert meta["chat_template"] == "chatml"
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run python -m pytest tests/test_lora.py::TestAdapterIO -v 2>&1 | head -20`
Expected: FAIL (ImportError for `save_adapters`, `load_adapters`, `merge_lora_weights`)

- [ ] **Step 3: Implement adapter save/load/merge in `scripts/lora.py`**

Add to `scripts/lora.py`:

`_find_lora_modules(model) -> list[tuple[str, LoRALinear]]`:
- Reuse the same recursive walk from `_recursive_find_linear`, but yield `(full_path, module)` where module is `LoRALinear`
- This is the shared traversal used by save, load, merge, and tests

`save_adapters(model, path, meta)`:
- Use `_find_lora_modules` to collect all `lora_A` and `lora_B` arrays with their full dotted paths (e.g., `blocks.0.attention.proj_q.lora_A`)
- Save to `{path}.safetensors` using `mx.save_safetensors(str(path.with_suffix(".safetensors")), weights_dict)`
- Write `meta` dict to `{path}.with_suffix(".meta.json")`

`load_adapters(model, path)`:
- Load `{path}.safetensors` via `mx.load()`
- Use `_find_lora_modules` to walk model tree, set `lora_A` and `lora_B` from loaded weights by matching full dotted paths
- Read and return metadata from `{path}.meta.json`

`merge_lora_weights(model)`:
- Use the recursive walk to find `LoRALinear` instances and their parent + attr_name
- Compute merged weight: `base.weight + (lora_A @ lora_B).T * scale`
- Create a new `nn.Linear` with the merged weight
- Replace the `LoRALinear` on the parent via `setattr(parent, attr_name, new_linear)` (or `parent[i]` for list items)

- [ ] **Step 4: Run all LoRA tests**

Run: `uv run python -m pytest tests/test_lora.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_lora.py scripts/lora.py
git commit -m "feat(lora): add adapter save, load, and merge utilities"
```

---

### Task 6: LoRA Script — Training Loop & CLI

**Files:**
- Modify: `scripts/lora.py`

Add the full training loop, checkpoint handling, and CLI to make `scripts/lora.py` a complete runnable script.

- [ ] **Step 1: Copy training boilerplate from pretrain.py**

Same set of functions as Task 3 Step 3 — copy into `scripts/lora.py`:
- `create_model`, `count_parameters`, `get_lr_schedule`
- `_tree_add`, `_tree_scale`, `_eval_grads`
- `sanitize_and_clip_grads`, `apply_gradients`
- `save_checkpoint`, `prune_checkpoints`, `_remap_tnt_keys`, `load_checkpoint`
- `evaluate`

Note: `apply_gradients` MUST still re-tie `head.weight = embed.weight` unconditionally (same as pretrain.py). Even though LoRA doesn't wrap embed/head, the optimizer still creates new arrays on each update and breaks the reference. The weight tie is always needed.

- [ ] **Step 2: Copy chat data pipeline from sft.py**

Copy from `scripts/sft.py`:
- `format_chatml`
- `build_loss_mask`
- `tokenize_chat`
- `SFTStreamingDataset`
- `masked_loss_fn`
- `compute_grads` (masked version)

- [ ] **Step 3: Add `LoRAConfig` dataclass**

Extends SFT config with:
- `lora_rank: int = 8`
- `lora_alpha: float = 16.0`
- `lora_dropout: float = 0.05`
- `lora_targets: str = "attn"`
- `merge_and_save: str | None = None`
- `lr: float = 1e-4` (higher than SFT default)
- `weight_decay: float = 0.01` (lower than SFT default)

- [ ] **Step 4: Add training loop**

Similar to SFT's training loop but:
- After model creation and weight loading, call `wrap_lora_layers()`
- Freeze all base parameters, only LoRA params get gradients
- Log trainable param count vs total
- At end of training: save adapters (not full model) via `save_adapters()`
- If `--merge-and-save` is set: also call `merge_lora_weights()` and `save_checkpoint()` to that path

- [ ] **Step 5: Add `main()` with argparse**

Same as SFT's argparse plus:
- `--lora-rank` (default: 8)
- `--lora-alpha` (default: 16.0)
- `--lora-dropout` (default: 0.05)
- `--lora-targets` (default: `"attn"`, choices described in help)
- `--merge-and-save` (default: None)
- `--dtype` (default: `"float16"`, choices: `float32`, `float16`, `bfloat16`)
- `--lr` default: `1e-4`
- `--weight-decay` default: `0.01`

- [ ] **Step 6: Smoke test**

Run: `uv run python scripts/lora.py --help`
Expected: Shows help text with LoRA-specific flags, no import errors

- [ ] **Step 7: Run all tests**

Run: `uv run python -m pytest tests/test_sft.py tests/test_lora.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/lora.py
git commit -m "feat(lora): add complete LoRA training script with CLI"
```

---

### Task 7: Final Verification & Cleanup

**Files:**
- All created files

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All existing and new tests PASS. No regressions.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check scripts/sft.py scripts/lora.py tests/test_sft.py tests/test_lora.py`
Expected: No errors (or fix any that appear)

- [ ] **Step 3: Run type checker**

Run: `uv run mypy scripts/sft.py scripts/lora.py --ignore-missing-imports`
Expected: No errors (or fix any that appear — strict mypy may need type annotations on some copied functions)

- [ ] **Step 4: Verify both scripts show help**

Run: `uv run python scripts/sft.py --help && uv run python scripts/lora.py --help`
Expected: Both print usage without errors

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: lint and type fixes for SFT and LoRA scripts"
```
