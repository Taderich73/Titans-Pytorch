# DPO & RLVR Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/dpo.py` and `scripts/rlvr.py` for preference optimization and reinforcement learning with verifiable rewards.

**Architecture:** Two self-contained scripts following the existing one-script-per-mode convention. Both reuse the ChatML data pipeline pattern from `scripts/sft.py`, the LoRA integration from `scripts/lora.py`, and the training loop structure (gradient accumulation, cosine LR, checkpointing) from both. A small modification to `LoRALinear` in `scripts/lora.py` adds an `enabled` flag for the reference model trick.

**Tech Stack:** MLX, HuggingFace datasets (streaming), HuggingFace tokenizers, numpy, tqdm, wandb (optional)

**Spec:** `docs/superpowers/specs/2026-03-25-dpo-rlvr-training-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `scripts/lora.py` | Add `enabled` flag to `LoRALinear`, add `set_lora_enabled()` utility |
| Create | `scripts/dpo.py` | DPO/SimPO training script with preference data pipeline |
| Create | `scripts/rlvr.py` | GRPO/REINFORCE training with offline and live modes, verifiers |
| Modify | `tests/test_lora.py` | Add tests for LoRA enabled/disabled toggle |
| Create | `tests/test_dpo.py` | Tests for DPO data pipeline, loss functions, log-prob computation |
| Create | `tests/test_rlvr.py` | Tests for RLVR verifiers, loss functions, offline data pipeline |

---

### Task 1: Add `enabled` flag to LoRALinear

**Files:**
- Modify: `scripts/lora.py:84-114` (LoRALinear class)
- Test: `tests/test_lora.py`

- [ ] **Step 1: Write the failing test for LoRA enabled toggle**

Add to `tests/test_lora.py`:

```python
class TestLoRAEnabled:
    """Tests for LoRA enabled/disabled toggle."""

    def test_disabled_equals_base(self) -> None:
        """When enabled=False, output equals base layer only."""
        mx.random.seed(42)
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        # Make lora_B nonzero so LoRA would change the output
        lora.lora_B = mx.ones_like(lora.lora_B) * 0.1

        x = mx.random.normal((2, 10, 32))

        # Enabled: output differs from base
        enabled_out = lora(x)
        base_out = base(x)
        mx.eval(enabled_out, base_out)
        assert not np.allclose(np.array(enabled_out), np.array(base_out), atol=1e-6)

        # Disabled: output equals base
        lora.enabled = False
        disabled_out = lora(x)
        mx.eval(disabled_out)
        np.testing.assert_allclose(
            np.array(disabled_out), np.array(base_out), atol=1e-6
        )

    def test_enabled_default_true(self) -> None:
        """LoRALinear.enabled defaults to True."""
        base = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        assert lora.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lora.py::TestLoRAEnabled -v`
Expected: FAIL — `LoRALinear` has no `enabled` attribute

- [ ] **Step 3: Add `enabled` flag to LoRALinear**

In `scripts/lora.py`, modify `LoRALinear.__init__` to add `self.enabled = True` and modify `__call__`:

```python
def __init__(
    self,
    base: nn.Linear,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    super().__init__()
    self.base = base
    self.base.freeze()

    out_dim, in_dim = base.weight.shape

    self.lora_A = mx.random.normal((in_dim, rank)) * (1.0 / math.sqrt(rank))
    self.lora_B = mx.zeros((rank, out_dim))
    self.scale = alpha / rank
    self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    self.enabled = True

def __call__(self, x: mx.array) -> mx.array:
    base_out = self.base(x)
    if not self.enabled:
        return base_out
    lora_input = self.dropout(x) if self.dropout else x
    lora_out = (lora_input @ self.lora_A) @ self.lora_B * self.scale
    return base_out + lora_out
```

- [ ] **Step 4: Add `set_lora_enabled` utility**

Add after the `_find_lora_modules` function in `scripts/lora.py`:

```python
def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle all LoRALinear adapters on/off.

    When disabled, LoRALinear.__call__ returns only the base layer output.
    Useful for computing reference model log-probs in DPO without loading
    a second model.
    """
    for _path, lora_mod in _find_lora_modules(model):
        lora_mod.enabled = enabled
```

- [ ] **Step 5: Write test for `set_lora_enabled` on full model**

Add to `tests/test_lora.py`:

```python
from scripts.lora import set_lora_enabled

class TestSetLoRAEnabled:
    """Tests for the set_lora_enabled utility."""

    def test_toggle_on_model(self) -> None:
        """set_lora_enabled toggles all LoRA layers in a model."""
        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)
        wrap_lora_layers(model, "attn", rank=4, alpha=8.0)

        # All should be enabled
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is True

        set_lora_enabled(model, False)
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is False

        set_lora_enabled(model, True)
        for _path, lora_mod in _find_lora_modules(model):
            assert lora_mod.enabled is True
```

- [ ] **Step 6: Run all LoRA tests**

Run: `uv run pytest tests/test_lora.py -v`
Expected: All PASS (existing + new tests)

- [ ] **Step 7: Commit**

```bash
git add scripts/lora.py tests/test_lora.py
git commit -m "feat(lora): add enabled flag and set_lora_enabled() for reference model support"
```

---

### Task 2: DPO data pipeline and log-prob utilities

**Files:**
- Create: `scripts/dpo.py` (partial — data pipeline and utilities only)
- Create: `tests/test_dpo.py`

- [ ] **Step 1: Write failing tests for DPO data extraction and log-probs**

Create `tests/test_dpo.py`:

```python
"""Tests for DPO data pipeline and log-probability computation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestExtractMessages:
    """Tests for extracting role/content from Dolci-style message dicts."""

    def test_strips_metadata_fields(self) -> None:
        """Only role and content are kept from message dicts."""
        from scripts.dpo import extract_messages

        raw_messages = [
            {
                "role": "user",
                "content": "Hello",
                "country": "US",
                "hashed_ip": "abc123",
                "toxic": False,
                "redacted": False,
                "turn_identifier": 1,
                "header": {"accept-language": "en"},
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "country": None,
                "hashed_ip": None,
                "toxic": False,
                "redacted": False,
                "turn_identifier": 2,
                "header": {},
            },
        ]

        cleaned = extract_messages(raw_messages)
        assert len(cleaned) == 2
        assert cleaned[0] == {"role": "user", "content": "Hello"}
        assert cleaned[1] == {"role": "assistant", "content": "Hi there!"}

    def test_handles_minimal_messages(self) -> None:
        """Messages with only role/content pass through unchanged."""
        from scripts.dpo import extract_messages

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        cleaned = extract_messages(messages)
        assert cleaned == messages


class TestComputeLogprobs:
    """Tests for per-token log-probability computation."""

    def test_output_shape(self) -> None:
        """Log-probs shape is (batch, seq_len - 1)."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20], [2, 6, 11, 21]])
        log_probs = compute_logprobs(model, input_ids)
        mx.eval(log_probs)

        assert log_probs.shape == (2, 3)  # (batch=2, seq_len-1=3)

    def test_values_are_negative(self) -> None:
        """Log-probs should all be <= 0."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20]])
        log_probs = compute_logprobs(model, input_ids)
        mx.eval(log_probs)

        assert np.all(np.array(log_probs) <= 0.0 + 1e-6)

    def test_mask_zeroes_padding(self) -> None:
        """Masked positions produce zero log-probs."""
        from scripts.dpo import compute_logprobs

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        input_ids = mx.array([[1, 5, 10, 20]])
        # Mask: first token real, rest padded
        mask = mx.array([[1, 1, 0, 0]])
        log_probs = compute_logprobs(model, input_ids, mask=mask)
        mx.eval(log_probs)

        lp = np.array(log_probs)
        # mask[:, 1:] = [1, 0, 0] — positions 1,2 should be zero
        assert lp[0, 1] == 0.0
        assert lp[0, 2] == 0.0
        assert lp[0, 0] != 0.0  # first position is unmasked
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dpo.py -v`
Expected: FAIL — `scripts.dpo` does not exist

- [ ] **Step 3: Implement data pipeline and log-prob utilities**

Create `scripts/dpo.py` with the initial boilerplate, data utilities, and log-prob function. Follow the exact import/structure pattern from `scripts/sft.py`:

```python
#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Direct Preference Optimization (DPO) training for Titans MLX models.

Supports:
- Standard DPO (Rafailov et al.) with reference model
- SimPO (reference-free, length-normalized)
- LoRA mode with base-model-as-reference trick
- Streaming HuggingFace preference datasets
- Gradient accumulation, cosine LR, checkpointing

Usage:
    # DPO with LoRA (recommended — base model serves as reference)
    uv run python scripts/dpo.py --model mac --dataset allenai/Dolci-Instruct-DPO \\
        --tokenizer gpt2 --dim 256 --num-layers 4 --lora

    # SimPO (no reference model needed)
    uv run python scripts/dpo.py --model mac --dataset allenai/Dolci-Instruct-DPO \\
        --method simpo --tokenizer gpt2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tqdm import tqdm

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Optional imports
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# =============================================================================
# Data Utilities
# =============================================================================


def extract_messages(raw_messages: list[dict]) -> list[dict]:
    """Extract only role and content from message dicts.

    Dolci-Instruct-DPO messages contain metadata fields (country, hashed_ip,
    toxic, header, etc.) that we discard.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in raw_messages
    ]


def format_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML string."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def tokenize_sequence(
    messages: list[dict],
    tokenizer: Any,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a message list into input_ids and an attention mask.

    Returns:
        (token_ids, attention_mask) where mask is 1 for real, 0 for padding.
        Both are lists of length max_len.
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        input_ids: list[int] = tokenizer.apply_chat_template(
            messages, tokenize=True
        )
    else:
        special_tokens = []
        existing = set(tokenizer.additional_special_tokens or [])
        if IM_START not in existing:
            special_tokens.append(IM_START)
        if IM_END not in existing:
            special_tokens.append(IM_END)
        if special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
        formatted = format_chatml(messages)
        input_ids = tokenizer.encode(formatted)

    # Truncate
    input_ids = input_ids[:max_len]
    real_len = len(input_ids)

    # Pad to max_len
    pad_len = max_len - real_len
    attention_mask = [1] * real_len + [0] * pad_len
    input_ids = input_ids + [0] * pad_len

    return input_ids, attention_mask


# =============================================================================
# Log-Probability Computation
# =============================================================================


def compute_logprobs(
    model: nn.Module,
    input_ids: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute per-token log-probabilities of the actual next tokens.

    Args:
        model: Titans model returning (logits, states).
        input_ids: (batch, seq_len) token IDs.
        mask: (batch, seq_len) attention mask. If provided, positions where
            mask[:, 1:] == 0 will have their log-probs zeroed out.

    Returns:
        (batch, seq_len - 1) per-token log-probabilities.
    """
    logits, _ = model(input_ids)
    # Log-softmax via MLX primitives
    log_probs = logits[:, :-1] - mx.logsumexp(
        logits[:, :-1], axis=-1, keepdims=True
    )
    # Gather log-probs for actual next tokens
    token_log_probs = mx.take_along_axis(
        log_probs, input_ids[:, 1:, None], axis=-1
    ).squeeze(-1)
    if mask is not None:
        token_log_probs = token_log_probs * mask[:, 1:]
    return token_log_probs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dpo.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dpo.py tests/test_dpo.py
git commit -m "feat(dpo): add data pipeline, log-prob utilities, and tests"
```

---

### Task 3: DPO loss functions

**Files:**
- Modify: `scripts/dpo.py`
- Modify: `tests/test_dpo.py`

- [ ] **Step 1: Write failing tests for DPO and SimPO loss**

Add to `tests/test_dpo.py`:

```python
class TestDPOLoss:
    """Tests for DPO loss computation."""

    def test_prefers_chosen(self) -> None:
        """Loss is lower when policy assigns higher log-prob to chosen."""
        from scripts.dpo import dpo_loss

        # Policy strongly prefers chosen
        chosen_logps = mx.array([-1.0])   # high log-prob
        rejected_logps = mx.array([-5.0]) # low log-prob
        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-2.0])

        loss_good = dpo_loss(chosen_logps, rejected_logps,
                             ref_chosen_logps, ref_rejected_logps, beta=0.1)

        # Policy prefers rejected (bad)
        loss_bad = dpo_loss(rejected_logps, chosen_logps,
                            ref_chosen_logps, ref_rejected_logps, beta=0.1)
        mx.eval(loss_good, loss_bad)

        assert float(loss_good) < float(loss_bad)

    def test_beta_scaling(self) -> None:
        """Higher beta amplifies the loss signal."""
        from scripts.dpo import dpo_loss

        chosen_logps = mx.array([-1.0])
        rejected_logps = mx.array([-3.0])
        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-2.0])

        loss_low_beta = dpo_loss(chosen_logps, rejected_logps,
                                  ref_chosen_logps, ref_rejected_logps, beta=0.01)
        loss_high_beta = dpo_loss(chosen_logps, rejected_logps,
                                   ref_chosen_logps, ref_rejected_logps, beta=1.0)
        mx.eval(loss_low_beta, loss_high_beta)

        # With higher beta, the loss magnitude should differ
        assert float(loss_low_beta) != float(loss_high_beta)


class TestSimPOLoss:
    """Tests for SimPO loss computation."""

    def test_prefers_chosen(self) -> None:
        """Loss is lower when avg log-prob of chosen exceeds rejected."""
        from scripts.dpo import simpo_loss

        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-3.0])

        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps,
                          beta=0.1, gamma=1.0)
        mx.eval(loss)

        # Should be a valid finite number
        assert np.isfinite(float(loss))
        assert float(loss) > 0  # cross-entropy style loss is positive

    def test_no_reference_model(self) -> None:
        """SimPO loss takes only policy log-probs, no reference."""
        from scripts.dpo import simpo_loss

        # Just verifying the signature works without ref
        chosen_avg_logps = mx.array([-1.0])
        rejected_avg_logps = mx.array([-2.0])
        loss = simpo_loss(chosen_avg_logps, rejected_avg_logps,
                          beta=0.1, gamma=0.5)
        mx.eval(loss)
        assert np.isfinite(float(loss))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dpo.py::TestDPOLoss tests/test_dpo.py::TestSimPOLoss -v`
Expected: FAIL — `dpo_loss` and `simpo_loss` not defined

- [ ] **Step 3: Implement loss functions**

Add to `scripts/dpo.py` after the `compute_logprobs` function:

```python
# =============================================================================
# Loss Functions
# =============================================================================


def dpo_loss(
    chosen_logps: mx.array,
    rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """Standard DPO loss (Rafailov et al.).

    Args:
        chosen_logps: Sum of per-token log-probs for chosen, (batch,).
        rejected_logps: Sum of per-token log-probs for rejected, (batch,).
        ref_chosen_logps: Reference model log-probs for chosen, (batch,).
        ref_rejected_logps: Reference model log-probs for rejected, (batch,).
        beta: KL penalty strength.

    Returns:
        Scalar loss.
    """
    log_ratio_chosen = chosen_logps - ref_chosen_logps
    log_ratio_rejected = rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    # Numerically stable log-sigmoid: log(sigmoid(x)) = x - softplus(x)
    return -mx.mean(logits - mx.logaddexp(mx.zeros_like(logits), logits))


def simpo_loss(
    chosen_avg_logps: mx.array,
    rejected_avg_logps: mx.array,
    beta: float = 0.1,
    gamma: float = 1.0,
) -> mx.array:
    """SimPO loss (reference-free, length-normalized).

    Args:
        chosen_avg_logps: Mean per-token log-prob for chosen, (batch,).
        rejected_avg_logps: Mean per-token log-prob for rejected, (batch,).
        beta: Scaling factor.
        gamma: Reward margin.

    Returns:
        Scalar loss.
    """
    logits = beta * (chosen_avg_logps - rejected_avg_logps - gamma)
    return -mx.mean(logits - mx.logaddexp(mx.zeros_like(logits), logits))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dpo.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dpo.py tests/test_dpo.py
git commit -m "feat(dpo): add DPO and SimPO loss functions"
```

---

### Task 4: DPO streaming dataset

**Files:**
- Modify: `scripts/dpo.py`
- Modify: `tests/test_dpo.py`

- [ ] **Step 1: Write failing test for DPOStreamingDataset**

Add to `tests/test_dpo.py`:

```python
class TestDPOStreamingDataset:
    """Tests for DPO data loading (using mock data)."""

    def test_batch_shape(self) -> None:
        """Batch has correct keys and shapes."""
        from scripts.dpo import DPOStreamingDataset
        from unittest.mock import MagicMock

        # Create a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.additional_special_tokens = []
        tokenizer.add_special_tokens = MagicMock()
        # Simple encode: each char becomes an int
        tokenizer.encode = lambda text: list(range(len(text)))

        dataset = DPOStreamingDataset.__new__(DPOStreamingDataset)
        dataset.tokenizer = tokenizer
        dataset.max_len = 32

        # Test tokenize_pair directly
        from scripts.dpo import extract_messages, tokenize_sequence

        chosen_msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        rejected_msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Bye"},
        ]

        c_ids, c_mask = tokenize_sequence(chosen_msgs, tokenizer, 32)
        r_ids, r_mask = tokenize_sequence(rejected_msgs, tokenizer, 32)

        assert len(c_ids) == 32
        assert len(c_mask) == 32
        assert sum(c_mask) > 0  # some real tokens
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_dpo.py::TestDPOStreamingDataset -v`
Expected: FAIL — `DPOStreamingDataset` not defined

- [ ] **Step 3: Implement DPOStreamingDataset**

Add to `scripts/dpo.py`:

```python
# =============================================================================
# DPO Streaming Dataset
# =============================================================================


class DPOStreamingDataset:
    """Streaming dataset for DPO preference pairs.

    Expects HuggingFace datasets with 'chosen' and 'rejected' fields,
    each containing a list of message dicts with 'role' and 'content'.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        buffer_size: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.buffer_size = buffer_size
        self._iterator: Any = None

    def _create_iterator(self):
        """Create a fresh streaming iterator of preference pairs."""
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for example in ds:
            chosen_raw = example.get(self.chosen_field)
            rejected_raw = example.get(self.rejected_field)
            if chosen_raw is None or rejected_raw is None:
                continue

            try:
                chosen_msgs = extract_messages(chosen_raw)
                rejected_msgs = extract_messages(rejected_raw)

                c_ids, c_mask = tokenize_sequence(
                    chosen_msgs, self.tokenizer, self.max_len
                )
                r_ids, r_mask = tokenize_sequence(
                    rejected_msgs, self.tokenizer, self.max_len
                )
            except Exception:
                continue

            yield {
                "chosen_ids": c_ids,
                "chosen_mask": c_mask,
                "rejected_ids": r_ids,
                "rejected_mask": r_mask,
            }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Return a batch of preference pairs as mx.arrays.

        Returns:
            Dict with "chosen_ids", "chosen_mask", "rejected_ids",
            "rejected_mask" as mx.arrays of shape (batch, max_len),
            or None if exhausted.
        """
        if self._iterator is None:
            self._iterator = self._create_iterator()

        batch_items: list[dict] = []
        for _ in range(batch_size):
            try:
                item = next(self._iterator)
                batch_items.append(item)
            except StopIteration:
                self._iterator = self._create_iterator()
                if batch_items:
                    break
                return None

        if not batch_items:
            return None

        return {
            "chosen_ids": mx.array(
                np.array([item["chosen_ids"] for item in batch_items])
            ),
            "chosen_mask": mx.array(
                np.array(
                    [item["chosen_mask"] for item in batch_items],
                    dtype=np.float32,
                )
            ),
            "rejected_ids": mx.array(
                np.array([item["rejected_ids"] for item in batch_items])
            ),
            "rejected_mask": mx.array(
                np.array(
                    [item["rejected_mask"] for item in batch_items],
                    dtype=np.float32,
                )
            ),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dpo.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dpo.py tests/test_dpo.py
git commit -m "feat(dpo): add DPOStreamingDataset for preference pair loading"
```

---

### Task 5: DPO config, training loop, CLI, and main()

**Files:**
- Modify: `scripts/dpo.py`

This task adds the DPOConfig dataclass, the training loop, checkpoint functions, argparse CLI, and main() — mirroring the structure from `scripts/sft.py`. The training loop computes log-probs for chosen/rejected (and reference if DPO), computes loss, accumulates gradients, and applies them.

- [ ] **Step 1: Add DPOConfig dataclass**

Add to `scripts/dpo.py` after the loss functions. Follow the exact pattern from `SFTConfig` in `scripts/sft.py:272-336`, with these DPO-specific additions:

```python
@dataclass
class DPOConfig:
    """DPO training hyperparameters."""

    # Model (same fields as SFTConfig)
    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    use_tnt: bool = False
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    global_chunk_size: int = 2048
    tnt_stage: int = 1
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global: bool = True
    attnres_modulate_local: bool = False

    # DPO-specific
    method: str = "dpo"  # "dpo" or "simpo"
    beta: float = 0.1
    gamma: float = 1.0  # SimPO margin

    # LoRA
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: str = "attn,ffn"
    lora_dropout: float = 0.0

    # Data
    dataset: str | None = None
    dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    max_len: int = 2048
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    chat_template: str = "auto"

    # Training
    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lr: float = 5e-7
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    eval_dataset: str | None = None
    eval_split: str = "train"
    resume: str | None = None
    init_weights: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-mlx-dpo"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    dtype: str = "float16"
```

- [ ] **Step 2: Add model creation, LR schedule, gradient utilities**

Copy these exact functions from `scripts/sft.py` (line ranges for reference):
- `create_model` (sft.py:463-475) — model factory, unchanged
- `count_parameters` (sft.py:478-493) — param counting, unchanged
- `get_lr_schedule` (sft.py:501-513) — cosine + warmup, unchanged
- `_tree_add` (sft.py:543-554) — gradient tree addition, unchanged
- `_tree_scale` (sft.py:557-568) — gradient tree scaling, unchanged
- `_eval_grads` (sft.py:571-578) — force evaluation to prevent graph growth, unchanged
- `sanitize_and_clip_grads` (sft.py:600-669) — NaN replacement + norm clipping, unchanged
- `apply_gradients` (sft.py:672-686) — apply + re-tie head/embed weights, unchanged

All functions are copied verbatim. No modifications needed.

- [ ] **Step 3: Add checkpoint functions**

Copy these exact functions from `scripts/sft.py`:
- `save_checkpoint` (sft.py:727-777) — **one change**: add `"training_stage": "dpo"` to the metadata dict
- `prune_checkpoints` (sft.py:780-791) — unchanged
- `_remap_tnt_keys` (sft.py:794-800) — unchanged
- `load_checkpoint` (sft.py:803-847) — unchanged

- [ ] **Step 4: Add the DPO training loop**

```python
def compute_dpo_grads(
    model: nn.Module,
    ref_model: nn.Module | None,
    chosen_ids: mx.array,
    chosen_mask: mx.array,
    rejected_ids: mx.array,
    rejected_mask: mx.array,
    config: DPOConfig,
    use_lora_ref: bool = False,
) -> tuple[mx.array, dict]:
    """Compute DPO/SimPO loss and gradients.

    For DPO with LoRA: disables LoRA to get reference log-probs,
    re-enables for policy log-probs.
    For DPO without LoRA: uses separate ref_model.
    For SimPO: no reference model needed.
    """
    if config.method == "simpo":
        # SimPO: no reference model needed
        def simpo_loss_fn(m):
            chosen_lps = compute_logprobs(m, chosen_ids, mask=chosen_mask)
            rejected_lps = compute_logprobs(m, rejected_ids, mask=rejected_mask)
            chosen_lengths = mx.clip(chosen_mask[:, 1:].sum(axis=1), a_min=1, a_max=None)
            rejected_lengths = mx.clip(rejected_mask[:, 1:].sum(axis=1), a_min=1, a_max=None)
            chosen_avg = chosen_lps.sum(axis=1) / chosen_lengths
            rejected_avg = rejected_lps.sum(axis=1) / rejected_lengths
            return simpo_loss(chosen_avg, rejected_avg,
                              beta=config.beta, gamma=config.gamma)

        loss_and_grad_fn = nn.value_and_grad(model, simpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads
    else:
        # Standard DPO: compute reference log-probs first (no gradient)
        from scripts.lora import set_lora_enabled

        if use_lora_ref:
            set_lora_enabled(model, False)
            ref_chosen_lps = compute_logprobs(model, chosen_ids, mask=chosen_mask)
            ref_rejected_lps = compute_logprobs(model, rejected_ids, mask=rejected_mask)
            mx.eval(ref_chosen_lps, ref_rejected_lps)
            ref_chosen_sum = mx.stop_gradient(ref_chosen_lps.sum(axis=1))
            ref_rejected_sum = mx.stop_gradient(ref_rejected_lps.sum(axis=1))
            set_lora_enabled(model, True)
        elif ref_model is not None:
            ref_chosen_lps = compute_logprobs(ref_model, chosen_ids, mask=chosen_mask)
            ref_rejected_lps = compute_logprobs(ref_model, rejected_ids, mask=rejected_mask)
            mx.eval(ref_chosen_lps, ref_rejected_lps)
            ref_chosen_sum = mx.stop_gradient(ref_chosen_lps.sum(axis=1))
            ref_rejected_sum = mx.stop_gradient(ref_rejected_lps.sum(axis=1))
        else:
            raise ValueError("Standard DPO requires --lora or a reference model")

        # Now compute policy log-probs with gradient tracking
        def dpo_loss_fn(m):
            c_lps = compute_logprobs(m, chosen_ids, mask=chosen_mask)
            r_lps = compute_logprobs(m, rejected_ids, mask=rejected_mask)
            return dpo_loss(
                c_lps.sum(axis=1), r_lps.sum(axis=1),
                ref_chosen_sum, ref_rejected_sum,
                beta=config.beta,
            )

        loss_and_grad_fn = nn.value_and_grad(model, dpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads
```

Then add the `train()` function following the exact loop structure from `scripts/sft.py:855-1114` but calling `compute_dpo_grads` instead of `compute_grads`, and using `DPOStreamingDataset` instead of `SFTStreamingDataset`.

- [ ] **Step 5: Add argparse CLI and main()**

Follow the pattern from `scripts/sft.py:1126-1287`. Add DPO-specific args (`--method`, `--beta`, `--gamma`, `--lora`, `--lora-rank`, `--lora-alpha`, `--lora-targets`, `--lora-dropout`, `--chosen-field`, `--rejected-field`). In main():
- Build DPOConfig from args
- Create model, optionally wrap with LoRA
- Optionally load a frozen reference model (full-parameter DPO only)
- Create DPOStreamingDataset
- Call train()

Include the standard `if __name__ == "__main__": main()` guard.

- [ ] **Step 6: Smoke test the script help**

Run: `uv run python scripts/dpo.py --help`
Expected: Prints usage with all DPO-specific flags

- [ ] **Step 7: Run all DPO tests**

Run: `uv run pytest tests/test_dpo.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/dpo.py
git commit -m "feat(dpo): add complete DPO/SimPO training script with config, loop, and CLI"
```

---

### Task 6: RLVR verifiers

**Files:**
- Create: `scripts/rlvr.py` (partial — verifier framework only)
- Create: `tests/test_rlvr.py`

- [ ] **Step 1: Write failing tests for verifiers**

Create `tests/test_rlvr.py`:

```python
"""Tests for RLVR verifiers, loss functions, and data pipeline."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class TestExactMatchVerifier:
    """Tests for exact_match verifier."""

    def test_exact_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("42", ["42"]) == 1.0

    def test_case_insensitive(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("Hello World", ["hello world"]) == 1.0

    def test_strips_whitespace(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("  answer  ", ["answer"]) == 1.0

    def test_no_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("wrong", ["right"]) == 0.0

    def test_multiple_ground_truths(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("b", ["a", "b", "c"]) == 1.0


class TestNumericMatchVerifier:
    """Tests for numeric_match verifier."""

    def test_exact_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 42.", ["42"]) == 1.0

    def test_approximate(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("3.14159", ["3.14"], tolerance=0.01) == 1.0

    def test_no_number_found(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("no numbers here", ["42"]) == 0.0

    def test_wrong_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 7", ["42"]) == 0.0

    def test_extracts_last_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("Step 1: 10, Step 2: 20, Final: 42", ["42"]) == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_rlvr.py -v`
Expected: FAIL — `scripts.rlvr` does not exist

- [ ] **Step 3: Implement verifiers**

Create `scripts/rlvr.py` with initial boilerplate (same import pattern as `scripts/dpo.py`) and verifier functions:

```python
#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Reinforcement Learning with Verifiable Rewards (RLVR) for Titans MLX models.

Supports:
- GRPO (Group Relative Policy Optimization) with clipped importance ratios
- REINFORCE with EMA baseline
- Offline mode (pre-computed rollouts) and live mode (generate + verify)
- Pluggable verifier framework (exact_match, numeric_match, custom)
- LoRA and full-parameter training

Usage:
    # GRPO with offline rollouts
    uv run python scripts/rlvr.py --model mac --dataset allenai/Dolci-Think-RL-7B \\
        --mode offline --method grpo --tokenizer gpt2

    # REINFORCE with live generation
    uv run python scripts/rlvr.py --model mac --dataset my/prompts \\
        --mode live --method reinforce --verifier exact_match
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tqdm import tqdm

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Optional imports
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# =============================================================================
# Verifiers
# =============================================================================


def exact_match(response: str, ground_truth: list[str]) -> float:
    """Exact match verifier (case-insensitive, whitespace-stripped).

    Returns 1.0 if response matches any ground truth, else 0.0.
    """
    normalized = response.strip().lower()
    for gt in ground_truth:
        if normalized == gt.strip().lower():
            return 1.0
    return 0.0


def numeric_match(
    response: str,
    ground_truth: list[str],
    tolerance: float = 0.01,
) -> float:
    """Extract the last number from response and compare to ground truth.

    Returns 1.0 if within tolerance of any ground truth number, else 0.0.
    """
    # Find all numbers in response (int or float)
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if not numbers:
        return 0.0

    extracted = float(numbers[-1])

    for gt in ground_truth:
        try:
            gt_num = float(gt.strip())
            if abs(extracted - gt_num) <= tolerance:
                return 1.0
        except ValueError:
            continue

    return 0.0


def load_custom_verifier(spec: str) -> Callable:
    """Load a custom verifier from 'path/to/module.py:function_name'.

    The function must have signature:
        (response: str, ground_truth: list[str]) -> float
    """
    module_path, func_name = spec.rsplit(":", 1)
    spec_obj = importlib.util.spec_from_file_location("custom_verifier", module_path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return getattr(module, func_name)


BUILTIN_VERIFIERS = {
    "exact_match": exact_match,
    "numeric_match": numeric_match,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rlvr.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/rlvr.py tests/test_rlvr.py
git commit -m "feat(rlvr): add verifier framework with exact_match and numeric_match"
```

---

### Task 7: RLVR loss functions

**Files:**
- Modify: `scripts/rlvr.py`
- Modify: `tests/test_rlvr.py`

- [ ] **Step 1: Write failing tests for GRPO and REINFORCE losses**

Add to `tests/test_rlvr.py`:

```python
class TestGRPOLoss:
    """Tests for GRPO loss with clipped importance ratios."""

    def test_zero_advantage_zero_loss(self) -> None:
        """When all rewards are identical, advantages are zero → loss is zero."""
        from scripts.rlvr import grpo_loss

        log_probs = mx.array([[[-0.5, -0.3, -0.4]]])  # (1, 1, 3)
        log_probs_old = mx.array([[[-0.5, -0.3, -0.4]]])
        rewards = mx.array([[1.0]])  # (1, 1) — single rollout, no variance
        masks = mx.array([[[1.0, 1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        # Single rollout → std=0 → advantages=0 → loss=0
        assert abs(float(loss)) < 1e-6

    def test_positive_advantage_negative_loss(self) -> None:
        """Rollout with above-average reward should decrease loss."""
        from scripts.rlvr import grpo_loss

        # 2 rollouts: first has high reward, second has low
        log_probs = mx.array([[[-0.5, -0.3], [-0.5, -0.3]]])  # (1, 2, 2)
        log_probs_old = mx.array([[[-0.5, -0.3], [-0.5, -0.3]]])
        rewards = mx.array([[1.0, 0.0]])  # (1, 2)
        masks = mx.array([[[1.0, 1.0], [1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_clipping_bounds_ratio(self) -> None:
        """Large log-prob differences should be clipped."""
        from scripts.rlvr import grpo_loss

        # Very different log_probs vs old → large ratio
        log_probs = mx.array([[[0.0, 0.0], [-5.0, -5.0]]])
        log_probs_old = mx.array([[[-5.0, -5.0], [0.0, 0.0]]])
        rewards = mx.array([[1.0, 0.0]])
        masks = mx.array([[[1.0, 1.0], [1.0, 1.0]]])

        loss = grpo_loss(log_probs, log_probs_old, rewards, masks, epsilon=0.2)
        mx.eval(loss)
        assert np.isfinite(float(loss))


class TestREINFORCELoss:
    """Tests for REINFORCE with baseline loss."""

    def test_positive_advantage(self) -> None:
        """Positive advantage produces negative loss (encourages action)."""
        from scripts.rlvr import reinforce_loss

        log_probs = mx.array([[-0.5, -0.3, -0.4]])  # (1, 3)
        rewards = mx.array([1.0])  # (1,)
        baseline = 0.5  # advantage = 1.0 - 0.5 = 0.5
        masks = mx.array([[1.0, 1.0, 1.0]])

        loss = reinforce_loss(log_probs, rewards, baseline, masks)
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_zero_advantage_zero_loss(self) -> None:
        """When reward equals baseline, loss is zero."""
        from scripts.rlvr import reinforce_loss

        log_probs = mx.array([[-0.5, -0.3]])
        rewards = mx.array([0.5])
        baseline = 0.5
        masks = mx.array([[1.0, 1.0]])

        loss = reinforce_loss(log_probs, rewards, baseline, masks)
        mx.eval(loss)
        assert abs(float(loss)) < 1e-6
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_rlvr.py::TestGRPOLoss tests/test_rlvr.py::TestREINFORCELoss -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement GRPO and REINFORCE losses**

Add to `scripts/rlvr.py`:

```python
# =============================================================================
# Log-Probability Computation
# =============================================================================


def compute_logprobs(
    model: nn.Module,
    input_ids: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute per-token log-probabilities of actual next tokens.

    Args:
        model: Titans model returning (logits, states).
        input_ids: (batch, seq_len) token IDs.
        mask: (batch, seq_len) attention mask.

    Returns:
        (batch, seq_len - 1) per-token log-probabilities.
    """
    logits, _ = model(input_ids)
    log_probs = logits[:, :-1] - mx.logsumexp(
        logits[:, :-1], axis=-1, keepdims=True
    )
    token_log_probs = mx.take_along_axis(
        log_probs, input_ids[:, 1:, None], axis=-1
    ).squeeze(-1)
    if mask is not None:
        token_log_probs = token_log_probs * mask[:, 1:]
    return token_log_probs


# =============================================================================
# Loss Functions
# =============================================================================


def grpo_loss(
    log_probs: mx.array,
    log_probs_old: mx.array,
    rewards: mx.array,
    masks: mx.array,
    epsilon: float = 0.2,
    kl_beta: float = 0.0,
    ref_log_probs: mx.array | None = None,
) -> mx.array:
    """GRPO loss with clipped importance ratios (per DeepSeekMath).

    Args:
        log_probs: (batch, num_rollouts, seq_len-1) current policy log-probs
            (from compute_logprobs which strips one token).
        log_probs_old: (batch, num_rollouts, seq_len-1) old policy log-probs (no grad).
        rewards: (batch, num_rollouts) per-rollout rewards.
        masks: (batch, num_rollouts, seq_len-1) token masks (aligned with log-probs).
        epsilon: Clipping range for importance ratios.
        kl_beta: KL penalty coefficient (0 = disabled).
        ref_log_probs: (batch, num_rollouts, seq_len-1) reference model log-probs for KL.

    Returns:
        Scalar loss.
    """
    # Group-relative advantages
    mean_reward = mx.mean(rewards, axis=1, keepdims=True)
    std_reward = mx.sqrt(mx.var(rewards, axis=1, keepdims=True) + 1e-8)
    advantages = (rewards - mean_reward) / std_reward  # (batch, num_rollouts)

    # Sum log-probs over sequence per rollout
    # masks shape: (batch, num_rollouts, seq_len-1) — pre-sliced to match log-probs
    seq_log_probs = (log_probs * masks).sum(axis=-1)          # (batch, num_rollouts)
    seq_log_probs_old = (log_probs_old * masks).sum(axis=-1)  # (batch, num_rollouts)

    # Importance ratios
    ratio = mx.exp(seq_log_probs - mx.stop_gradient(seq_log_probs_old))
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -mx.mean(mx.minimum(surr1, surr2))

    # Optional KL penalty
    kl_loss = mx.array(0.0)
    if kl_beta > 0 and ref_log_probs is not None:
        kl = (log_probs - ref_log_probs) * masks
        kl_loss = kl_beta * mx.mean(kl.sum(axis=-1))

    return policy_loss + kl_loss


def reinforce_loss(
    log_probs: mx.array,
    rewards: mx.array,
    baseline: float,
    masks: mx.array,
) -> mx.array:
    """REINFORCE loss with EMA baseline.

    Args:
        log_probs: (batch, seq_len-1) per-token log-probs (from compute_logprobs).
        rewards: (batch,) per-example rewards.
        baseline: EMA baseline value.
        masks: (batch, seq_len-1) token masks (pre-sliced to match log-probs).

    Returns:
        Scalar loss.
    """
    advantages = rewards - baseline  # (batch,)
    seq_log_probs = (log_probs * masks).sum(axis=-1)  # (batch,)
    return -mx.mean(advantages * seq_log_probs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rlvr.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/rlvr.py tests/test_rlvr.py
git commit -m "feat(rlvr): add GRPO and REINFORCE loss functions with tests"
```

---

### Task 8: RLVR offline data pipeline

**Files:**
- Modify: `scripts/rlvr.py`
- Modify: `tests/test_rlvr.py`

- [ ] **Step 1: Write failing test for OfflineRLDataset**

Add to `tests/test_rlvr.py`:

```python
class TestOfflineRLDataset:
    """Tests for offline rollout data loading."""

    def test_reward_from_ground_truth(self) -> None:
        """Rewards are computed by verifier against ground_truth."""
        from scripts.rlvr import compute_rollout_rewards, exact_match

        outputs = ["42", "wrong", "42"]
        ground_truth = ["42"]

        rewards = compute_rollout_rewards(outputs, ground_truth, exact_match)
        assert rewards == [1.0, 0.0, 1.0]

    def test_empty_outputs(self) -> None:
        """No outputs produces empty rewards."""
        from scripts.rlvr import compute_rollout_rewards, exact_match

        rewards = compute_rollout_rewards([], ["42"], exact_match)
        assert rewards == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_rlvr.py::TestOfflineRLDataset -v`
Expected: FAIL

- [ ] **Step 3: Implement OfflineRLDataset and helpers**

Add to `scripts/rlvr.py`:

```python
# =============================================================================
# Data Utilities
# =============================================================================


def format_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML string."""
    parts: list[str] = []
    for message in messages:
        parts.append(f"{IM_START}{message['role']}\n{message['content']}{IM_END}\n")
    return "".join(parts)


def compute_rollout_rewards(
    outputs: list[str],
    ground_truth: list[str],
    verifier: Callable,
) -> list[float]:
    """Score each rollout output against ground truth using the verifier."""
    return [verifier(output, ground_truth) for output in outputs]


def tokenize_and_pad(
    text: str,
    tokenizer: Any,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize text, truncate/pad to max_len.

    Returns (token_ids, attention_mask).
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        # Wrap as assistant response for consistency
        ids = tokenizer.encode(text)
    else:
        special_tokens = []
        existing = set(tokenizer.additional_special_tokens or [])
        if IM_START not in existing:
            special_tokens.append(IM_START)
        if IM_END not in existing:
            special_tokens.append(IM_END)
        if special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
        ids = tokenizer.encode(text)

    ids = ids[:max_len]
    real_len = len(ids)
    pad_len = max_len - real_len
    mask = [1] * real_len + [0] * pad_len
    ids = ids + [0] * pad_len
    return ids, mask


# =============================================================================
# Offline RL Dataset
# =============================================================================


class OfflineRLDataset:
    """Streaming dataset for offline RL with pre-computed rollouts.

    Expects HuggingFace datasets with 'prompt', 'ground_truth', and
    'outputs' fields (e.g., allenai/Dolci-Think-RL-7B).
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_len: int,
        num_rollouts: int = 8,
        verifier: Callable = exact_match,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        prompt_field: str = "prompt",
        ground_truth_field: str = "ground_truth",
        outputs_field: str = "outputs",
        buffer_size: int = 1000,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_rollouts = num_rollouts
        self.verifier = verifier
        self.subset = subset
        self.split = split
        self.seed = seed
        self.prompt_field = prompt_field
        self.ground_truth_field = ground_truth_field
        self.outputs_field = outputs_field
        self.buffer_size = buffer_size
        self._iterator: Any = None

    def _create_iterator(self):
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        for example in ds:
            prompt = example.get(self.prompt_field, "")
            ground_truth = example.get(self.ground_truth_field, [])
            outputs = example.get(self.outputs_field, [])

            if not outputs or not ground_truth:
                continue

            # Take up to num_rollouts
            rollouts = outputs[:self.num_rollouts]
            if len(rollouts) < 2:
                continue  # GRPO needs >= 2 rollouts

            # Compute rewards
            rewards = compute_rollout_rewards(rollouts, ground_truth, self.verifier)

            # Tokenize prompt
            prompt_ids, _ = tokenize_and_pad(prompt, self.tokenizer, self.max_len)

            # Tokenize each rollout (prompt + response)
            rollout_ids_list = []
            rollout_masks_list = []
            for rollout_text in rollouts:
                full_text = prompt + rollout_text
                r_ids, r_mask = tokenize_and_pad(
                    full_text, self.tokenizer, self.max_len
                )
                rollout_ids_list.append(r_ids)
                rollout_masks_list.append(r_mask)

            yield {
                "prompt_ids": prompt_ids,
                "rollout_ids": rollout_ids_list,
                "rollout_masks": rollout_masks_list,
                "rewards": rewards,
            }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Return a batch of rollout groups.

        Returns dict with:
            prompt_ids: (batch, max_len)
            rollout_ids: (batch, num_rollouts, max_len)
            rollout_masks: (batch, num_rollouts, max_len)
            rewards: (batch, num_rollouts)
        """
        if self._iterator is None:
            self._iterator = self._create_iterator()

        batch_items: list[dict] = []
        for _ in range(batch_size):
            try:
                item = next(self._iterator)
                batch_items.append(item)
            except StopIteration:
                self._iterator = self._create_iterator()
                if batch_items:
                    break
                return None

        if not batch_items:
            return None

        # Pad rollout count to max in batch
        max_rollouts = max(len(item["rewards"]) for item in batch_items)

        padded_rollout_ids = []
        padded_rollout_masks = []
        padded_rewards = []

        for item in batch_items:
            n = len(item["rewards"])
            pad_n = max_rollouts - n

            r_ids = item["rollout_ids"] + [[0] * self.max_len] * pad_n
            r_masks = item["rollout_masks"] + [[0] * self.max_len] * pad_n
            rews = item["rewards"] + [0.0] * pad_n

            padded_rollout_ids.append(r_ids)
            padded_rollout_masks.append(r_masks)
            padded_rewards.append(rews)

        return {
            "prompt_ids": mx.array(
                np.array([item["prompt_ids"] for item in batch_items])
            ),
            "rollout_ids": mx.array(np.array(padded_rollout_ids)),
            "rollout_masks": mx.array(
                np.array(padded_rollout_masks, dtype=np.float32)
            ),
            "rewards": mx.array(
                np.array(padded_rewards, dtype=np.float32)
            ),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rlvr.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/rlvr.py tests/test_rlvr.py
git commit -m "feat(rlvr): add offline RL dataset and rollout reward computation"
```

---

### Task 9: RLVR live generation and config/training loop/CLI

**Files:**
- Modify: `scripts/rlvr.py`

This is the largest task — adds live rollout generation, the RLVRConfig dataclass, the training loop, and the full CLI. Follow the same patterns established in Tasks 5 and 8.

- [ ] **Step 1: Add temperature sampling for live generation**

Add to `scripts/rlvr.py`:

```python
def generate_rollout(
    model: nn.Module,
    prompt_ids: mx.array,
    max_new_tokens: int,
    temperature: float = 0.7,
    eos_token_id: int | None = None,
) -> mx.array:
    """Generate a single rollout via temperature sampling.

    Args:
        model: Titans model.
        prompt_ids: (1, prompt_len) token IDs.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        eos_token_id: Stop token (e.g., <|im_end|>).

    Returns:
        (1, prompt_len + generated_len) full sequence.
    """
    tokens = prompt_ids
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        # Temperature-scaled logits → categorical sampling
        next_logits = logits[:, -1, :] / temperature
        next_token = mx.random.categorical(next_logits)
        next_token = next_token.reshape(1, 1)
        tokens = mx.concatenate([tokens, next_token], axis=1)
        mx.eval(tokens)

        if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
            break

    return tokens
```

- [ ] **Step 1b: Write failing test for generate_rollout**

Add to `tests/test_rlvr.py`:

```python
class TestGenerateRollout:
    """Tests for live rollout generation."""

    def test_output_longer_than_prompt(self) -> None:
        """Generated sequence should be longer than prompt."""
        from scripts.rlvr import generate_rollout

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        prompt = mx.array([[1, 5, 10]])
        result = generate_rollout(model, prompt, max_new_tokens=5, temperature=0.7)
        mx.eval(result)

        assert result.shape[1] > prompt.shape[1]
        assert result.shape[1] <= prompt.shape[1] + 5

    def test_stops_at_eos(self) -> None:
        """Generation stops when EOS token is produced."""
        from scripts.rlvr import generate_rollout

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        prompt = mx.array([[1, 5, 10]])
        # Use a common token as EOS — may or may not stop early, but should not crash
        result = generate_rollout(model, prompt, max_new_tokens=20, temperature=0.7, eos_token_id=0)
        mx.eval(result)
        assert result.shape[1] >= prompt.shape[1]
```

- [ ] **Step 2: Add RLVRConfig dataclass**

```python
@dataclass
class RLVRConfig:
    """RLVR training hyperparameters."""

    # Model (same as DPOConfig/SFTConfig)
    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    use_tnt: bool = False
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    global_chunk_size: int = 2048
    tnt_stage: int = 1
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global: bool = True
    attnres_modulate_local: bool = False

    # RLVR-specific
    method: str = "grpo"  # "grpo" or "reinforce"
    mode: str = "offline"  # "offline" or "live"
    num_rollouts: int = 8
    epsilon: float = 0.2  # GRPO clipping range
    kl_beta: float = 0.0  # KL penalty (0 = disabled)
    ema_decay: float = 0.99  # REINFORCE baseline decay
    temperature: float = 0.7  # Live mode generation temperature
    max_new_tokens: int = 2048  # Live mode generation cap
    verifier: str = "exact_match"  # Verifier name or path:function

    # LoRA
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: str = "attn,ffn"
    lora_dropout: float = 0.0

    # Data
    dataset: str | None = None
    dataset_subset: str | None = None
    tokenizer: str = "gpt2"
    max_len: int = 2048
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"
    outputs_field: str = "outputs"
    chat_template: str = "auto"

    # Training
    max_steps: int = 5000
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lr: float = 1e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    eval_dataset: str | None = None
    eval_split: str = "train"
    resume: str | None = None
    init_weights: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-mlx-rlvr"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    dtype: str = "float16"
```

- [ ] **Step 3: Add model creation, LR schedule, gradient utilities, checkpoint functions**

Same as Task 5 step 2-3 — copy from `scripts/sft.py` with `"training_stage": "rlvr"` in checkpoint metadata.

- [ ] **Step 4: Add the RLVR training loop**

Add `compute_rlvr_grads` and `train` functions. The core gradient function:

```python
def compute_rlvr_grads(
    model: nn.Module,
    rollout_ids: mx.array,
    rollout_masks: mx.array,
    rewards: mx.array,
    config: RLVRConfig,
    ema_baseline: float = 0.0,
) -> tuple[mx.array, dict, float]:
    """Compute GRPO or REINFORCE gradients.

    Args:
        model: Titans model.
        rollout_ids: (batch, num_rollouts, seq_len) tokenized rollouts.
        rollout_masks: (batch, num_rollouts, seq_len-1) token masks.
        rewards: (batch, num_rollouts) per-rollout rewards.
        config: Training config.
        ema_baseline: Current EMA baseline (REINFORCE only).

    Returns:
        (loss, grads, updated_ema_baseline)
    """
    batch_size, num_rollouts, seq_len = rollout_ids.shape

    # Slice masks to match compute_logprobs output shape (seq_len-1)
    # Data pipeline produces (batch, num_rollouts, seq_len), log-probs are (batch, seq_len-1)
    shifted_masks = rollout_masks[:, :, 1:]

    if config.method == "grpo":
        # Skip if all rewards identical (zero variance → zero advantage)
        reward_std = mx.sqrt(mx.var(rewards, axis=1) + 1e-8)
        if float(mx.max(reward_std)) < 1e-6:
            logger.warning("All rollout rewards identical — skipping batch")
            return mx.array(0.0), {}, ema_baseline

        # Compute old log-probs (no gradient) for importance ratios
        log_probs_old_list = []
        for i in range(num_rollouts):
            lp = compute_logprobs(model, rollout_ids[:, i, :])
            log_probs_old_list.append(lp)
        log_probs_old = mx.stop_gradient(mx.stack(log_probs_old_list, axis=1))
        mx.eval(log_probs_old)

        def grpo_loss_fn(m):
            lp_list = []
            for i in range(num_rollouts):
                lp = compute_logprobs(m, rollout_ids[:, i, :])
                lp_list.append(lp)
            log_probs = mx.stack(lp_list, axis=1)
            return grpo_loss(
                log_probs, log_probs_old, rewards, shifted_masks,
                epsilon=config.epsilon, kl_beta=config.kl_beta,
            )

        loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
        loss, grads = loss_and_grad_fn(model)
        return loss, grads, ema_baseline

    else:  # reinforce
        # Process one rollout at a time, accumulate
        total_loss = mx.array(0.0)
        all_grads = None

        for i in range(num_rollouts):
            r_ids = rollout_ids[:, i, :]
            r_mask = shifted_masks[:, i, :]
            r_rewards = rewards[:, i]

            advantage = r_rewards - ema_baseline

            def reinforce_loss_fn(m):
                lp = compute_logprobs(m, r_ids, mask=r_mask)
                seq_lp = lp.sum(axis=-1)
                return -mx.mean(advantage * seq_lp)

            loss_and_grad_fn = nn.value_and_grad(model, reinforce_loss_fn)
            loss, grads = loss_and_grad_fn(model)
            total_loss = total_loss + loss

            if all_grads is None:
                all_grads = grads
            else:
                all_grads = _tree_add(all_grads, grads)

        # Average over rollouts
        scale = mx.array(1.0 / num_rollouts)
        avg_grads = _tree_scale(all_grads, scale)
        avg_loss = total_loss / num_rollouts

        # Update EMA baseline
        batch_mean_reward = float(mx.mean(rewards))
        new_baseline = config.ema_decay * ema_baseline + (1 - config.ema_decay) * batch_mean_reward

        return avg_loss, avg_grads, new_baseline
```

The `train()` function follows `scripts/sft.py:855-1114` exactly:
- Same outer while loop, gradient accumulation, optimizer step pattern
- For **offline mode**: call `dataset.get_batch()` each step
- For **live mode**: call `dataset.get_batch()` for prompts only, then call `generate_rollout` for each prompt × `num_rollouts`, tokenize results, compute rewards via verifier, assemble into the same batch shape
- Log `train/loss`, `train/reward_mean`, `train/reward_std`, `train/lr`
- Checkpoint with `"training_stage": "rlvr"` in metadata

- [ ] **Step 5: Add argparse CLI and main()**

Include all RLVR-specific flags: `--mode`, `--method`, `--num-rollouts`, `--epsilon`, `--kl-beta`, `--ema-decay`, `--temperature`, `--max-new-tokens`, `--verifier`, plus all standard model/training/LoRA/checkpoint flags.

In main():
- Build RLVRConfig
- Resolve verifier (builtin name or custom path:function)
- Create model, optionally wrap with LoRA
- Create OfflineRLDataset or LiveRLDataset based on mode
- Call train()

- [ ] **Step 6: Smoke test the script help**

Run: `uv run python scripts/rlvr.py --help`
Expected: Prints usage with all RLVR-specific flags

- [ ] **Step 7: Run all RLVR tests**

Run: `uv run pytest tests/test_rlvr.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/rlvr.py
git commit -m "feat(rlvr): add complete RLVR training script with GRPO, REINFORCE, live/offline modes"
```

---

### Task 10: Integration tests and final verification

**Files:**
- Modify: `tests/test_dpo.py`
- Modify: `tests/test_rlvr.py`

- [ ] **Step 1: Add DPO integration test with small model**

Add to `tests/test_dpo.py`:

```python
class TestDPOIntegration:
    """End-to-end DPO loss computation with a real model."""

    def test_dpo_loss_with_lora_reference(self) -> None:
        """Full DPO forward pass: policy + reference via LoRA toggle."""
        from scripts.dpo import compute_logprobs, dpo_loss
        from scripts.lora import wrap_lora_layers, set_lora_enabled

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)
        wrap_lora_layers(model, "attn", rank=4, alpha=8.0)

        chosen_ids = mx.array([[1, 5, 10, 20, 30]])
        rejected_ids = mx.array([[1, 5, 11, 21, 31]])

        # Policy log-probs (LoRA enabled)
        pi_chosen = compute_logprobs(model, chosen_ids)
        pi_rejected = compute_logprobs(model, rejected_ids)

        # Reference log-probs (LoRA disabled)
        set_lora_enabled(model, False)
        ref_chosen = compute_logprobs(model, chosen_ids)
        ref_rejected = compute_logprobs(model, rejected_ids)
        set_lora_enabled(model, True)

        mx.eval(pi_chosen, pi_rejected, ref_chosen, ref_rejected)

        loss = dpo_loss(
            pi_chosen.sum(axis=1), pi_rejected.sum(axis=1),
            ref_chosen.sum(axis=1), ref_rejected.sum(axis=1),
            beta=0.1,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_simpo_loss_no_reference(self) -> None:
        """SimPO forward pass without any reference model."""
        from scripts.dpo import compute_logprobs, simpo_loss

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        chosen_ids = mx.array([[1, 5, 10, 20, 30]])
        rejected_ids = mx.array([[1, 5, 11, 21, 31]])
        mask = mx.array([[1, 1, 1, 1, 1]])

        chosen_lps = compute_logprobs(model, chosen_ids, mask=mask)
        rejected_lps = compute_logprobs(model, rejected_ids, mask=mask)
        mx.eval(chosen_lps, rejected_lps)

        lengths = mx.array([4.0])  # seq_len - 1
        loss = simpo_loss(
            chosen_lps.sum(axis=1) / lengths,
            rejected_lps.sum(axis=1) / lengths,
            beta=0.1, gamma=1.0,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))
```

- [ ] **Step 2: Add RLVR integration test**

Add to `tests/test_rlvr.py`:

```python
from titans_mlx.config import TitansConfig
from titans_mlx.models import TitansMAC


class TestRLVRIntegration:
    """End-to-end RLVR loss with a real model."""

    def test_grpo_with_model(self) -> None:
        """GRPO loss computation with actual model log-probs."""
        from scripts.rlvr import compute_logprobs, grpo_loss

        mx.random.seed(42)
        config = TitansConfig(dim=64, num_heads=2, num_layers=2, vocab_size=128)
        model = TitansMAC(config)

        # 1 prompt, 2 rollouts, 5 tokens each
        rollout_ids = mx.array([
            [[1, 5, 10, 20, 30], [1, 5, 11, 21, 31]]
        ])
        masks = mx.ones((1, 2, 4))  # seq_len - 1 = 4

        # Compute log-probs for each rollout
        lps = []
        for i in range(2):
            lp = compute_logprobs(model, rollout_ids[:, i, :])
            lps.append(lp)
        log_probs = mx.stack(lps, axis=1)
        mx.eval(log_probs)

        rewards = mx.array([[1.0, 0.0]])

        loss = grpo_loss(
            log_probs, mx.stop_gradient(log_probs),
            rewards, masks, epsilon=0.2,
        )
        mx.eval(loss)
        assert np.isfinite(float(loss))
```

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/test_lora.py tests/test_dpo.py tests/test_rlvr.py -v`
Expected: All PASS

- [ ] **Step 4: Run existing test suite to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_dpo.py tests/test_rlvr.py
git commit -m "test: add integration tests for DPO and RLVR training"
```

---

### Task 11: Training shell scripts

**Files:**
- Create: `training/train_dpo.sh`
- Create: `training/train_rlvr.sh`

- [ ] **Step 1: Create DPO training shell script**

```bash
#!/bin/bash
# DPO training with LoRA (recommended for memory efficiency)
uv run python scripts/dpo.py \
    --model mac \
    --dataset allenai/Dolci-Instruct-DPO \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --method dpo \
    --beta 0.1 \
    --lora \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets attn,ffn \
    --dim 512 \
    --num-layers 12 \
    --num-heads 8 \
    --max-len 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --lr 5e-7 \
    --warmup-ratio 0.1 \
    --grad-clip 1.0 \
    --epochs 3 \
    --save-every 1000 \
    --eval-every 500 \
    --checkpoint-dir checkpoints/dpo-lora \
    --seed 42
```

- [ ] **Step 2: Create RLVR training shell script**

```bash
#!/bin/bash
# RLVR training with GRPO (offline mode)
uv run python scripts/rlvr.py \
    --model mac \
    --dataset allenai/Dolci-Think-RL-7B \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --mode offline \
    --method grpo \
    --num-rollouts 8 \
    --epsilon 0.2 \
    --dim 512 \
    --num-layers 12 \
    --num-heads 8 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lr 1e-6 \
    --warmup-ratio 0.03 \
    --grad-clip 1.0 \
    --max-steps 5000 \
    --save-every 1000 \
    --eval-every 500 \
    --checkpoint-dir checkpoints/rlvr-grpo \
    --seed 42
```

- [ ] **Step 3: Make scripts executable**

Run: `chmod +x training/train_dpo.sh training/train_rlvr.sh`

- [ ] **Step 4: Commit**

```bash
git add training/train_dpo.sh training/train_rlvr.sh
git commit -m "feat: add example training shell scripts for DPO and RLVR"
```
