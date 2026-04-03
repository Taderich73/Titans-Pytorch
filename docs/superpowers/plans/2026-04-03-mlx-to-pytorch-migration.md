# MLX to PyTorch Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Titans from MLX to PyTorch with a working MAC end-to-end pipeline (config → memory → attention → model → pretrain → inference).

**Architecture:** Vertical slice — port TitansMAC first with all supporting modules. Archive MLX code for reference. Composable features (TNT, AttnRes, MCA, adaptive windows) deferred with NotImplementedError stubs. Training uses HuggingFace Accelerate for single/multi-GPU.

**Tech Stack:** PyTorch >= 2.2, HuggingFace Accelerate, NumPy, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-04-03-mlx-to-pytorch-migration-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `archive/titans_mlx/` | Archived MLX code (copy of current `src/titans_mlx/`) |
| Create | `src/titans/__init__.py` | Public API exports |
| Create | `src/titans/config.py` | TitansConfig dataclass (framework-agnostic) |
| Create | `src/titans/memory.py` | MemoryState, MemoryMLP, NeuralLongTermMemory |
| Create | `src/titans/attention.py` | RoPE, SegmentedAttention, SlidingWindowAttention |
| Create | `src/titans/persistent.py` | PersistentMemory learnable tokens |
| Create | `src/titans/models.py` | RMSNorm, FeedForward, MACBlock, TitansMAC, stubs |
| Create | `src/titans/memory_dump.py` | State serialization (save/load .npz) |
| Create | `scripts/pretrain.py` | PyTorch + Accelerate training (replaces MLX version) |
| Create | `scripts/inference.py` | Inference with memory persistence (replaces MLX version) |
| Create | `tests/conftest.py` | Shared fixtures |
| Create | `tests/test_config.py` | Config tests |
| Create | `tests/test_memory.py` | Memory module tests |
| Create | `tests/test_attention.py` | Attention module tests |
| Create | `tests/test_models.py` | Model tests |
| Create | `tests/test_memory_dump.py` | Serialization round-trip tests |
| Modify | `pyproject.toml` | Dependencies, package path, keywords |

---

### Task 1: Archive MLX Code and Update Project Config

**Files:**
- Create: `archive/titans_mlx/` (copy of `src/titans_mlx/`)
- Create: `archive/scripts/` (copy of `scripts/`)
- Create: `archive/tests/` (copy of `tests/`)
- Modify: `pyproject.toml`

- [ ] **Step 1: Copy MLX source to archive**

```bash
mkdir -p archive
cp -r src/titans_mlx archive/titans_mlx
cp -r scripts archive/scripts
cp -r tests archive/tests
```

- [ ] **Step 2: Remove old source directories**

```bash
rm -rf src/titans_mlx
rm -rf tests/
rm -rf scripts/
mkdir -p src/titans
mkdir -p tests
mkdir -p scripts
```

- [ ] **Step 3: Create empty `src/titans/__init__.py`**

Create `src/titans/__init__.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time — PyTorch Implementation.

Usage:
    import torch
    from titans import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    x = torch.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

__version__ = "0.2.0"
```

- [ ] **Step 4: Update pyproject.toml**

Replace the full content of `pyproject.toml` with:

```toml
[project]
name = "titans"
version = "0.2.0"
description = "Implementation of Titans: Learning to Memorize at Test Time"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.12"
authors = [
    { name = "Delanoe Pirard", email = "contact@aedelon.com" }
]
keywords = ["deep-learning", "memory", "transformers", "pytorch", "cuda", "gpu"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "torch>=2.2.0",
    "numpy>=2.0.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
    "tqdm>=4.67.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "mypy>=1.8",
    "ruff>=0.2",
]
train = [
    "accelerate>=0.27.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "wandb>=0.16.0",
]
hub = [
    "huggingface_hub>=0.20.0",
]
all = [
    "titans[dev,train,hub]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/titans"]

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B905",   # zip() without strict=
    "C408",   # dict() call
    "SIM102", # nested if
    "SIM108", # ternary
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = [
    "ARG002",  # unused method arguments
    "E402",    # module-level imports
]

[tool.ruff.lint.isort]
known-first-party = ["titans"]

[tool.mypy]
python_version = "3.12"
warn_return_any = false
warn_unused_ignores = true
disallow_untyped_defs = false
check_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = ["torch.*", "transformers.*", "datasets.*", "wandb.*", "huggingface_hub.*", "accelerate.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[dependency-groups]
dev = [
    "mypy>=1.19.1",
    "numpy>=2.4.1",
    "pytest>=9.0.2",
    "pytest-cov>=7.0.0",
    "ruff>=0.14.9",
    "tox>=4.0",
    "tox-uv>=1.0",
]

[tool.tox]
env_list = ["lint", "typecheck", "test"]

[tool.tox.env_run_base]
runner = "uv-venv-lock-runner"
extras = ["dev"]

[tool.tox.env.lint]
description = "Run ruff linting and formatting checks"
commands = [
    ["ruff", "check", "src/", "tests/"],
    ["ruff", "format", "--check", "src/", "tests/"],
]

[tool.tox.env.typecheck]
description = "Run mypy type checking"
commands = [["mypy", "src/titans"]]

[tool.tox.env.test]
description = "Run pytest suite"
commands = [["pytest", "tests/", "-v", "--tb=short"]]

[tool.tox.env.test-cov]
description = "Run tests with coverage"
commands = [["pytest", "tests/", "--cov=titans", "--cov-report=term-missing"]]
```

- [ ] **Step 5: Commit**

```bash
git add archive/ src/titans/__init__.py pyproject.toml
git add -u  # stages deletions of src/titans_mlx/, old tests/, old scripts/
git commit -m "refactor: archive MLX code, scaffold PyTorch package structure"
```

---

### Task 2: TitansConfig (Framework-Agnostic)

**Files:**
- Create: `src/titans/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config test file**

Create `tests/test_config.py`:

```python
"""Tests for TitansConfig."""

import pytest

from titans.config import TitansConfig


class TestTitansConfig:
    def test_defaults(self):
        config = TitansConfig()
        assert config.dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 12
        assert config.vocab_size == 32000

    def test_computed_properties(self):
        config = TitansConfig(dim=256, num_heads=4, ffn_mult=4.0)
        assert config.head_dim == 64
        assert config.ffn_dim == 1024
        assert config.memory_hidden_dim == int(256 * 4.0)

    def test_to_dict_from_dict_roundtrip(self):
        config = TitansConfig(dim=128, num_heads=2, num_layers=4)
        d = config.to_dict()
        restored = TitansConfig.from_dict(d)
        assert restored.dim == 128
        assert restored.num_heads == 2
        assert restored.num_layers == 4

    def test_invalid_memory_objective(self):
        with pytest.raises(ValueError, match="memory_objective"):
            TitansConfig(memory_objective="invalid")

    def test_mca_insertion_layer_validation(self):
        with pytest.raises(ValueError, match="MCA insertion layer"):
            TitansConfig(use_mca=True, mca_insertion_layers=[99], num_layers=6)

    def test_tnt_factory_stage1(self):
        config = TitansConfig.tnt_stage1(dim=256)
        assert config.use_tnt is True
        assert config.tnt_stage == 1
        assert config.dim == 256

    def test_tnt_factory_stage2(self):
        s1 = TitansConfig.tnt_stage1()
        s2 = TitansConfig.tnt_stage2(s1)
        assert s2.tnt_stage == 2
        assert s2.finetune_local_chunk_sizes == [4, 8]

    def test_attnres_sub_layer_block_size(self):
        config = TitansConfig(num_layers=6, num_attnres_blocks=4)
        assert config.attnres_sub_layer_block_size >= 1

    def test_deferred_flags_accepted(self):
        """Config accepts all flags — errors happen at model construction."""
        config = TitansConfig(
            use_tnt=True,
            use_attn_res=True,
            use_mca=True,
            mca_insertion_layers=[3],
            adaptive_window=True,
        )
        assert config.use_tnt is True
        assert config.use_attn_res is True
```

- [ ] **Step 2: Write conftest.py**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for Titans PyTorch tests."""

import pytest
import torch

from titans.config import TitansConfig


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Test on CPU always, CUDA when available."""
    return torch.device(request.param)


@pytest.fixture
def default_config():
    """Small config for fast tests."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        max_seq_len=256,
        num_memory_layers=2,
        num_persistent_tokens=4,
    )


@pytest.fixture
def linear_memory_config():
    """Config with single-layer (linear) memory."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        max_seq_len=256,
        num_memory_layers=1,
        num_persistent_tokens=4,
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 16
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.config'`

- [ ] **Step 4: Write config.py**

Create `src/titans/config.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Configuration for Titans PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TitansConfig:
    """Configuration for Titans models.

    All fields are preserved from the MLX implementation. Features not yet
    ported (TNT, AttnRes, MCA, adaptive window) are accepted here but raise
    NotImplementedError at model construction time.
    """

    # Core dimensions
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    ffn_mult: float = 4.0

    # Memory configuration
    num_memory_layers: int = 2
    memory_hidden_mult: float = 4.0
    num_persistent_tokens: int = 16

    # Sequence configuration
    chunk_size: int = 512
    window_size: int = 512
    max_seq_len: int = 8192

    # Memory learning parameters
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_error_clip: float = 10.0
    memory_grad_clip: float = 1.0

    # Memory objective (attentional bias)
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0

    # Architecture options
    use_conv: bool = True
    conv_kernel_size: int = 4
    use_rope: bool = True

    # TNT Hierarchical Memory (deferred)
    use_tnt: bool = False
    global_chunk_size: int = 2048
    local_chunk_sizes: list[int] = field(default_factory=lambda: [8, 16])
    local_shard_length: int = 2048
    use_qk_projection: bool = True
    tnt_stage: int = 1
    finetune_local_chunk_sizes: list[int] | None = None

    # AttnRes (deferred)
    use_attn_res: bool = False
    num_attnres_blocks: int = 8
    attnres_warmup_steps: int = 0
    attnres_modulate_global_memory: bool = True
    attnres_modulate_local_memory: bool = False

    # Memory state quantization (deferred)
    quantize_memory_state: bool = False
    memory_state_weight_bits: int = 4
    memory_state_momentum_bits: int = 8

    # Adaptive window sizing (deferred)
    adaptive_window: bool = False
    adaptive_window_min: int = 64
    adaptive_window_max: int | None = None
    adaptive_window_temperature: float = 10.0
    adaptive_window_lambda: float = 0.01

    # Memory Cross-Attention (deferred)
    use_mca: bool = False
    mca_insertion_layers: list[int] | None = None
    mca_num_heads: int = 8
    mca_gate_type: str = "scalar"
    mca_gate_bias_init: float = -3.0

    # Gate initialization
    gate_decay_bias_init: float = -6.0

    # AttnRes numerical stability
    attnres_logit_clip: float = 30.0

    # Memory dump
    mca_auto_dump: bool = False
    mca_dump_trigger: str = "session_end"
    mca_dump_path: str = "./memory_dumps/"
    mca_dump_keep_last_n: int = 10

    # Training
    dropout: float = 0.0
    activation: str = "silu"
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return int(self.dim * self.ffn_mult)

    @property
    def memory_hidden_dim(self) -> int:
        return int(self.dim * self.memory_hidden_mult)

    @property
    def effective_adaptive_window_max(self) -> int:
        if self.adaptive_window_max is not None:
            return self.adaptive_window_max
        return self.window_size

    @property
    def num_local_memories(self) -> int:
        return len(self.local_chunk_sizes)

    @property
    def active_local_chunk_sizes(self) -> list[int]:
        if self.tnt_stage == 2 and self.finetune_local_chunk_sizes is not None:
            return self.finetune_local_chunk_sizes
        return self.local_chunk_sizes

    def __post_init__(self) -> None:
        valid_objectives = ("l2", "huber")
        if self.memory_objective not in valid_objectives:
            raise ValueError(
                f"memory_objective must be one of {valid_objectives}, "
                f"got '{self.memory_objective}'"
            )
        if self.use_mca:
            for idx in self.mca_active_insertion_layers:
                if idx >= self.num_layers:
                    raise ValueError(
                        f"MCA insertion layer {idx} >= num_layers {self.num_layers}"
                    )

    @property
    def mca_active_insertion_layers(self) -> list[int]:
        if not self.use_mca:
            return []
        if self.mca_insertion_layers is not None:
            return self.mca_insertion_layers
        return [self.num_layers // 2]

    @property
    def attnres_sub_layer_block_size(self) -> int:
        num_mca_layers = len(self.mca_active_insertion_layers) if self.use_mca else 0
        total_sub_layers = (self.num_layers * 2) + num_mca_layers
        return max(1, total_sub_layers // self.num_attnres_blocks)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "ffn_mult": self.ffn_mult,
            "num_memory_layers": self.num_memory_layers,
            "memory_hidden_mult": self.memory_hidden_mult,
            "num_persistent_tokens": self.num_persistent_tokens,
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "max_seq_len": self.max_seq_len,
            "memory_lr": self.memory_lr,
            "memory_momentum": self.memory_momentum,
            "memory_error_clip": self.memory_error_clip,
            "memory_grad_clip": self.memory_grad_clip,
            "memory_objective": self.memory_objective,
            "huber_delta_init": self.huber_delta_init,
            "use_conv": self.use_conv,
            "conv_kernel_size": self.conv_kernel_size,
            "use_rope": self.use_rope,
            "use_tnt": self.use_tnt,
            "global_chunk_size": self.global_chunk_size,
            "local_chunk_sizes": self.local_chunk_sizes,
            "local_shard_length": self.local_shard_length,
            "use_qk_projection": self.use_qk_projection,
            "tnt_stage": self.tnt_stage,
            "finetune_local_chunk_sizes": self.finetune_local_chunk_sizes,
            "use_attn_res": self.use_attn_res,
            "num_attnres_blocks": self.num_attnres_blocks,
            "attnres_warmup_steps": self.attnres_warmup_steps,
            "attnres_modulate_global_memory": self.attnres_modulate_global_memory,
            "attnres_modulate_local_memory": self.attnres_modulate_local_memory,
            "quantize_memory_state": self.quantize_memory_state,
            "memory_state_weight_bits": self.memory_state_weight_bits,
            "memory_state_momentum_bits": self.memory_state_momentum_bits,
            "adaptive_window": self.adaptive_window,
            "adaptive_window_min": self.adaptive_window_min,
            "adaptive_window_max": self.adaptive_window_max,
            "adaptive_window_temperature": self.adaptive_window_temperature,
            "adaptive_window_lambda": self.adaptive_window_lambda,
            "use_mca": self.use_mca,
            "mca_insertion_layers": self.mca_insertion_layers,
            "mca_num_heads": self.mca_num_heads,
            "mca_gate_type": self.mca_gate_type,
            "mca_gate_bias_init": self.mca_gate_bias_init,
            "gate_decay_bias_init": self.gate_decay_bias_init,
            "attnres_logit_clip": self.attnres_logit_clip,
            "mca_auto_dump": self.mca_auto_dump,
            "mca_dump_trigger": self.mca_dump_trigger,
            "mca_dump_path": self.mca_dump_path,
            "mca_dump_keep_last_n": self.mca_dump_keep_last_n,
            "dropout": self.dropout,
            "activation": self.activation,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TitansConfig:
        return cls(**d)

    @classmethod
    def tnt_stage1(cls, **kwargs) -> TitansConfig:
        defaults = dict(
            use_tnt=True,
            global_chunk_size=2048,
            local_chunk_sizes=[8, 16],
            local_shard_length=2048,
            tnt_stage=1,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def tnt_stage2(cls, stage1_config: TitansConfig) -> TitansConfig:
        d = stage1_config.to_dict()
        d["finetune_local_chunk_sizes"] = [
            max(1, cs // 2) for cs in d["local_chunk_sizes"]
        ]
        d["tnt_stage"] = 2
        return cls.from_dict(d)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/titans/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: add TitansConfig and test suite"
```

---

### Task 3: PersistentMemory Module

**Files:**
- Create: `src/titans/persistent.py`
- Create: `tests/test_persistent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persistent.py`:

```python
"""Tests for PersistentMemory module."""

import torch

from titans.config import TitansConfig
from titans.persistent import PersistentMemory


class TestPersistentMemory:
    def test_forward_shape(self, default_config, device):
        mem = PersistentMemory(default_config).to(device)
        result = mem(batch_size=2)
        assert result.shape == (2, default_config.num_persistent_tokens, default_config.dim)
        assert result.device.type == device.type

    def test_returns_none_when_zero_tokens(self, device):
        config = TitansConfig(dim=64, num_persistent_tokens=0)
        mem = PersistentMemory(config).to(device)
        assert mem(batch_size=2) is None

    def test_tokens_are_parameters(self, default_config):
        mem = PersistentMemory(default_config)
        assert isinstance(mem.tokens, torch.nn.Parameter)

    def test_batch_expansion_shares_data(self, default_config, device):
        mem = PersistentMemory(default_config).to(device)
        result = mem(batch_size=3)
        # expand shares memory — all batch items point to same data
        assert result.shape[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_persistent.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.persistent'`

- [ ] **Step 3: Write persistent.py**

Create `src/titans/persistent.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Persistent Memory Module for Titans (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from titans.config import TitansConfig


class PersistentMemory(nn.Module):
    """Persistent Memory tokens — learnable, data-independent.

    These tokens are prepended to the input sequence and encode
    task-specific knowledge. They remain fixed during inference.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.num_tokens = config.num_persistent_tokens
        self.dim = config.dim

        if self.num_tokens > 0:
            self.tokens = nn.Parameter(
                torch.randn(self.num_tokens, self.dim) * config.init_std
            )
        else:
            self.tokens = None

    def forward(self, batch_size: int) -> torch.Tensor | None:
        """Get persistent memory tokens expanded for batch.

        Args:
            batch_size: Batch size

        Returns:
            Persistent tokens (batch, num_tokens, dim) or None if num_tokens=0
        """
        if self.tokens is None:
            return None
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_persistent.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/titans/persistent.py tests/test_persistent.py
git commit -m "feat: add PersistentMemory module"
```

---

### Task 4: Attention Modules (RoPE, SWA, Segmented)

**Files:**
- Create: `src/titans/attention.py`
- Create: `tests/test_attention.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_attention.py`:

```python
"""Tests for attention modules."""

import torch

from titans.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans.config import TitansConfig


class TestRotaryPositionEmbedding:
    def test_output_shape(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q, q_rot)

    def test_cache_rebuild(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=8).to(device)
        q = torch.randn(1, 1, 16, 16, device=device)
        k = torch.randn(1, 1, 16, 16, device=device)
        # seq_len=16 exceeds max_seq_len=8 — should rebuild cache
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape

    def test_seq_offset(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(1, 2, 4, 16, device=device)
        k = torch.randn(1, 2, 4, 16, device=device)
        q_rot1, _ = rope(q, k, seq_offset=0)
        q_rot2, _ = rope(q, k, seq_offset=10)
        assert not torch.allclose(q_rot1, q_rot2)


class TestSlidingWindowAttention:
    def test_output_shape(self, default_config, device):
        attn = SlidingWindowAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_prefix(self, default_config, device):
        attn = SlidingWindowAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        prefix = torch.randn(2, 4, default_config.dim, device=device)
        out = attn(x, prefix=prefix)
        # Output shape matches input (not prefix+input)
        assert out.shape == x.shape

    def test_no_rope(self, device):
        config = TitansConfig(dim=64, num_heads=4, use_rope=False, window_size=32)
        attn = SlidingWindowAttention(config).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = attn(x)
        assert out.shape == x.shape


class TestSegmentedAttention:
    def test_output_shape(self, default_config, device):
        attn = SegmentedAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_persistent_and_memory(self, default_config, device):
        attn = SegmentedAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        persistent = torch.randn(2, 4, default_config.dim, device=device)
        memory = torch.randn(2, 1, default_config.dim, device=device)
        out = attn(x, persistent=persistent, memory=memory)
        # Output is for input positions only
        assert out.shape == x.shape

    def test_causal_property(self, default_config, device):
        """Early positions should not be affected by later positions."""
        attn = SegmentedAttention(default_config).to(device)
        attn.eval()
        x = torch.randn(1, 8, default_config.dim, device=device)
        out1 = attn(x)
        # Modify last position
        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(1, default_config.dim, device=device)
        out2 = attn(x_mod)
        # First position output should be unchanged
        torch.testing.assert_close(out1[:, 0, :], out2[:, 0, :])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_attention.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.attention'`

- [ ] **Step 3: Write attention.py**

Create `src/titans/attention.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Attention modules for Titans architecture (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self, dim: int, max_seq_len: int = 8192, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))
        self._max_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: (batch, heads, seq, head_dim)
            k: (batch, heads, seq, head_dim)
            seq_offset: Offset for position indices
        """
        seq_len = q.shape[2]
        if seq_offset + seq_len > self._max_seq_len:
            self._build_cache(seq_offset + seq_len)

        cos = self.cos_cached[seq_offset : seq_offset + seq_len]
        sin = self.sin_cached[seq_offset : seq_offset + seq_len]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        return q_rotated, k_rotated

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_even = x1 * cos - x2 * sin
        rotated_odd = x1 * sin + x2 * cos

        batch, heads, seq, half_dim = rotated_even.shape
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.reshape(batch, heads, seq, half_dim * 2)


def _rearrange_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, seq, dim) -> (batch, heads, seq, head_dim)"""
    batch, seq, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)


def _rearrange_from_heads(x: torch.Tensor) -> torch.Tensor:
    """(batch, heads, seq, head_dim) -> (batch, seq, dim)"""
    batch, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention for MAG/MAL variants."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.scale = self.head_dim**-0.5

        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim, max_seq_len=config.max_seq_len
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=std)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        row_idx = positions.unsqueeze(1)
        col_idx = positions.unsqueeze(0)
        causal_mask = col_idx <= row_idx
        window_mask = (row_idx - col_idx) < self.window_size
        bool_mask = causal_mask & window_mask
        return torch.where(bool_mask, 0.0, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        prefix: torch.Tensor | None = None,
        seq_offset: int = 0,
        adaptive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with sliding window attention.

        Args:
            x: Input (batch, seq, dim)
            prefix: Optional prefix tokens (batch, prefix_len, dim)
            seq_offset: Offset for rotary embeddings
            adaptive_mask: Optional soft mask (deferred — raises if provided)
        """
        if adaptive_mask is not None:
            raise NotImplementedError(
                "Adaptive window masking not yet ported. "
                "See archive/titans_mlx/adaptive_window.py"
            )

        batch_size, seq_len, _ = x.shape

        if prefix is not None:
            full_x = torch.cat([prefix, x], dim=1)
            prefix_len = prefix.shape[1]
        else:
            full_x = x
            prefix_len = 0

        q = self.proj_q(x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        if self.rope is not None:
            q, _ = self.rope(q, q, seq_offset=prefix_len + seq_offset)
            k, _ = self.rope(k, k, seq_offset=seq_offset)

        mask = self._create_extended_mask(seq_len, full_x.shape[1], prefix_len, x.device)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        output = _rearrange_from_heads(output)
        output = self.proj_out(output)
        return output

    def _create_extended_mask(
        self, query_len: int, key_len: int, prefix_len: int, device: torch.device
    ) -> torch.Tensor:
        # Queries can always attend to all prefix tokens
        prefix_mask = torch.zeros((query_len, prefix_len), device=device)

        if key_len > prefix_len:
            main_mask = self._create_sliding_window_mask(query_len, device)
        else:
            main_mask = torch.zeros((query_len, 0), device=device)

        mask = torch.cat([prefix_mask, main_mask], dim=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, query_len, key_len)


class SegmentedAttention(nn.Module):
    """Segmented/Chunked Attention for MAC variant."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim, max_seq_len=config.max_seq_len
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        x: torch.Tensor,
        persistent: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with segmented attention.

        Full sequence is: [persistent] || [memory] || [input]

        Returns output for input positions only.
        """
        batch_size, seq_len, _ = x.shape

        components = []
        if persistent is not None:
            components.append(persistent)
        if memory is not None:
            components.append(memory)
        components.append(x)

        full_x = torch.cat(components, dim=1)
        prefix_len = full_x.shape[1] - seq_len

        q = self.proj_q(full_x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        if self.rope is not None:
            q, k = self.rope(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )

        output = _rearrange_from_heads(output)
        output = self.proj_out(output)

        # Return only input positions
        return output[:, prefix_len:]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_attention.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/titans/attention.py tests/test_attention.py
git commit -m "feat: add attention modules (RoPE, SlidingWindow, Segmented)"
```

---

### Task 5: Memory Module (MemoryState, MemoryMLP, NeuralLongTermMemory)

This is the largest and most critical module. The analytical gradient computation and parallel linear memory update must be exact.

**Files:**
- Create: `src/titans/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory.py`:

```python
"""Tests for Neural Long-term Memory module."""

import torch

from titans.config import TitansConfig
from titans.memory import MemoryMLP, MemoryState, NeuralLongTermMemory


class TestMemoryState:
    def test_detach(self, device):
        w = torch.randn(4, 4, device=device, requires_grad=True)
        m = torch.randn(4, 4, device=device, requires_grad=True)
        state = MemoryState(weights=[w], momentum=[m])
        detached = state.detach()
        assert not detached.weights[0].requires_grad
        assert not detached.momentum[0].requires_grad

    def test_clone(self, device):
        w = torch.randn(4, 4, device=device)
        state = MemoryState(weights=[w], momentum=[torch.zeros_like(w)])
        cloned = state.clone()
        # Modify original — clone should be unaffected
        w.fill_(0)
        assert cloned.weights[0].abs().sum() > 0


class TestMemoryMLP:
    def test_forward_shape_linear(self, device):
        config = TitansConfig(dim=64, num_memory_layers=1)
        mlp = MemoryMLP(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = mlp(x)
        assert out.shape == (2, 8, 64)

    def test_forward_shape_deep(self, default_config, device):
        mlp = MemoryMLP(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        out = mlp(x)
        assert out.shape == (2, 8, default_config.dim)

    def test_forward_with_weights(self, device):
        config = TitansConfig(dim=32, num_memory_layers=1)
        mlp = MemoryMLP(config).to(device)
        x = torch.randn(2, 4, 32, device=device)
        weights = mlp.get_weights()
        out1 = mlp(x)
        out2 = mlp.forward_with_weights(x, weights)
        torch.testing.assert_close(out1, out2)


class TestNeuralLongTermMemory:
    def test_init_state(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        state = mem.init_state(batch_size=2)
        assert len(state.weights) == default_config.num_memory_layers
        assert state.weights[0].device.type == device.type
        # Momentum should be zeros
        assert state.momentum[0].abs().sum() == 0

    def test_forward_shape(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        output, new_state = mem(x)
        assert output.shape == x.shape
        assert new_state is not None

    def test_state_changes_after_forward(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        state = mem.init_state(batch_size=2)
        _, new_state = mem(x, state=state)
        # Weights should differ after update
        assert not torch.allclose(state.weights[0], new_state.weights[0])

    def test_linear_memory_forward(self, linear_memory_config, device):
        """Single-layer memory uses parallel update path."""
        mem = NeuralLongTermMemory(linear_memory_config).to(device)
        x = torch.randn(2, 8, linear_memory_config.dim, device=device)
        output, new_state = mem(x)
        assert output.shape == x.shape
        assert len(new_state.weights) == 1

    def test_forward_no_state(self, default_config, device):
        """Forward without explicit state should auto-initialize."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        output, state = mem(x, state=None)
        assert output.shape == x.shape
        assert state is not None

    def test_state_detached(self, default_config, device):
        """Returned state should be detached from computation graph."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        _, state = mem(x)
        for w in state.weights:
            assert not w.requires_grad
        for m in state.momentum:
            assert not m.requires_grad

    def test_with_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=True)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_without_conv(self, device):
        config = TitansConfig(dim=64, num_heads=4, num_memory_layers=1, use_conv=False)
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_huber_objective(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_memory_layers=1, memory_objective="huber"
        )
        mem = NeuralLongTermMemory(config).to(device)
        x = torch.randn(2, 8, 64, device=device)
        output, state = mem(x)
        assert output.shape == x.shape

    def test_retrieve(self, default_config, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        state = mem.init_state(batch_size=2)
        queries = torch.randn(2, 4, default_config.dim, device=device)
        retrieved = mem.retrieve(queries, state)
        assert retrieved.shape == queries.shape

    def test_gradients_flow_through_projections(self, default_config, device):
        """Main training graph: gradients should flow through proj_k/v/q/out."""
        mem = NeuralLongTermMemory(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device, requires_grad=True)
        output, _ = mem(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert mem.proj_k.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.memory'`

- [ ] **Step 3: Write memory.py**

Create `src/titans/memory.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans (PyTorch Implementation).

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.config import TitansConfig

_L2_NORM_EPS: float = 1e-8
_DEGENERATE_THRESHOLD: float = 1e-6


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable] = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


@dataclass
class MemoryState:
    """State of the neural long-term memory."""

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        return MemoryState(
            weights=[w.detach().clone() for w in self.weights],
            momentum=[m.detach().clone() for m in self.momentum],
        )


class MemoryMLP(nn.Module):
    """MLP that stores information in its weights."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim
        self.activation = get_activation(config.activation)

        layers: list[nn.Linear] = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        self.layers = nn.ModuleList(layers)
        self._init_weights(config.init_std)

    def _init_weights(self, std: float) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def forward_with_weights(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.data.clone() for layer in self.layers]


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module (PyTorch Implementation).

    Learns to memorize at test time using gradient descent with momentum
    and weight decay. Gradient computation is analytical (not autograd).
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim, config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        self.memory = MemoryMLP(config)

        self.gate_decay_proj = nn.Linear(config.dim, 1)
        self.gate_lr_proj = nn.Linear(config.dim, 1)
        self.gate_momentum_proj = nn.Linear(config.dim, 1)

        self.memory_objective = config.memory_objective
        if self.memory_objective == "huber":
            self.gate_delta_proj = nn.Linear(config.dim, 1)

        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

        nn.init.constant_(self.gate_decay_proj.bias, self.config.gate_decay_bias_init)

        if self.memory_objective == "huber":
            nn.init.constant_(self.gate_delta_proj.bias, self.config.huber_delta_init)

    def _apply_conv(
        self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_conv:
            return k, v, q
        seq_len = k.shape[1]
        # PyTorch Conv1d: (B, C, L) — transpose from (B, L, C)
        k = self.conv_k(k.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        v = self.conv_v(v.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        q = self.conv_q(q.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        return k, v, q

    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        num_layers = len(weights)
        if num_layers == 1:
            return self._compute_gradients_linear(keys, values, weights[0], delta=delta)
        return self._compute_gradients_deep(keys, values, weights, delta=delta)

    def _compute_gradients_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weight: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        predictions = F.linear(keys, weight)
        err_clip = self.config.memory_error_clip
        raw_error = torch.clamp(predictions - values, -err_clip, err_clip)

        if self.memory_objective == "huber" and delta is not None:
            abs_error = torch.abs(raw_error)
            error = torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))
        else:
            error = raw_error

        scale = 2.0 / float(error.numel())
        batch_seq = error.shape[0] * error.shape[1]
        error_flat = error.reshape(batch_seq, -1)
        keys_flat = keys.reshape(batch_seq, -1)
        grad_w = scale * (error_flat.T @ keys_flat)

        grad_clip = self.config.memory_grad_clip
        return [torch.clamp(grad_w, -grad_clip, grad_clip)]

    def _compute_gradients_deep(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
        delta: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        num_layers = len(weights)
        batch_size, seq_len = keys.shape[0], keys.shape[1]
        batch_seq = batch_size * seq_len

        activations = [keys]
        pre_activations = []
        h = keys

        for i in range(num_layers):
            h_pre = F.linear(h, weights[i])
            pre_activations.append(h_pre)
            if i < num_layers - 1:
                h = self.memory.activation(h_pre)
                activations.append(h)
            else:
                h = h_pre

        err_clip = self.config.memory_error_clip
        raw_error = torch.clamp(h - values, -err_clip, err_clip)

        if self.memory_objective == "huber" and delta is not None:
            abs_error = torch.abs(raw_error)
            error = torch.where(abs_error <= delta, raw_error, delta * torch.sign(raw_error))
        else:
            error = raw_error

        scale = 2.0 / float(error.numel())
        delta_bp = scale * error

        grad_clip = self.config.memory_grad_clip
        grads: list[torch.Tensor | None] = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            act = activations[i]
            delta_bp_flat = delta_bp.reshape(batch_seq, -1)
            act_flat = act.reshape(batch_seq, -1)
            grad_w = delta_bp_flat.T @ act_flat
            grads[i] = torch.clamp(grad_w, -grad_clip, grad_clip)

            if i > 0:
                delta_bp = F.linear(delta_bp, weights[i].T)
                x = pre_activations[i - 1]
                delta_bp = delta_bp * self._activation_derivative(x)

        return grads  # type: ignore[return-value]

    def _activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.activation == "silu":
            sig = torch.sigmoid(x)
            return sig * (1.0 + x * (1.0 - sig))
        elif self.config.activation == "gelu":
            x_f32 = x.float()
            sqrt2 = 1.4142135623730951
            cdf = 0.5 * (1.0 + torch.erf(x_f32 / sqrt2))
            pdf = torch.exp(-0.5 * x_f32 * x_f32) * 0.3989422804014327
            return (cdf + x_f32 * pdf).to(x.dtype)
        elif self.config.activation == "relu":
            return (x > 0).to(x.dtype)
        else:
            raise ValueError(f"No derivative for activation: {self.config.activation}")

    def init_state(self, batch_size: int) -> MemoryState:  # noqa: ARG002
        weights = [w.detach().clone() for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        return_state: bool = True,
        lr_scale: float | torch.Tensor = 1.0,
        memory_gate: torch.Tensor | None = None,
        return_keys: bool = False,
    ) -> (
        tuple[torch.Tensor, MemoryState | None]
        | tuple[torch.Tensor, MemoryState | None, torch.Tensor]
    ):
        batch_size = x.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        if self.config.quantize_memory_state:
            raise NotImplementedError(
                "Quantized memory state not yet ported. "
                "See archive/titans_mlx/quantize_state.py"
            )

        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        k, v, q = self._apply_conv(k, v, q)

        k = F.silu(k)
        v = F.silu(v)
        q = F.silu(q)

        # L2-normalize in float32
        q_f32 = q.float()
        k_f32 = k.float()
        q = (q_f32 / torch.sqrt(
            torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(q.dtype)
        k = (k_f32 / torch.sqrt(
            torch.sum(k_f32 * k_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(k.dtype)

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

        if return_state:
            detached = new_state.detach()
            if return_keys:
                return output, detached, k
            return output, detached

        if return_keys:
            return output, None, k
        return output, None

    def _parallel_memory_update_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        state: MemoryState,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
    ) -> MemoryState:
        """Tensorized parallel memory update for linear memory (Section 3.2)."""
        B, S, D = keys.shape
        W_0 = state.weights[0]
        S_prev = state.momentum[0]

        err_clip = self.config.memory_error_clip
        grad_clip = self.config.memory_grad_clip
        decay = 1.0 - alpha
        S_f = float(S)

        preds = F.linear(keys, W_0)
        errors = torch.clamp(preds - values, -err_clip, err_clip)

        if self.memory_objective == "huber":
            hub_delta = getattr(self, "_current_delta", None)
            if hub_delta is not None:
                abs_errors = torch.abs(errors)
                errors = torch.where(abs_errors <= hub_delta, errors, hub_delta * torch.sign(errors))

        scale = 2.0 / float(B * S * D)
        errors_scaled = errors * scale

        positions = torch.arange(S, dtype=torch.float32, device=keys.device)

        diff = decay - eta
        abs_diff = torch.abs(diff)
        is_degenerate = abs_diff < _DEGENERATE_THRESHOLD
        safe_diff = torch.where(
            is_degenerate,
            torch.tensor(1.0, device=keys.device),
            torch.maximum(abs_diff, torch.tensor(_L2_NORM_EPS, device=keys.device)) * torch.sign(diff),
        )

        # New momentum
        eta_powers = torch.pow(eta, S_f - 1.0 - positions)
        eta_w = eta_powers.reshape(1, S, 1)

        weighted_eta = errors_scaled * eta_w
        grad_eta_sum = torch.mean(
            weighted_eta.permute(0, 2, 1) @ keys,
            dim=0,
        )
        grad_eta_sum = torch.clamp(grad_eta_sum, -grad_clip, grad_clip)
        new_momentum = torch.pow(eta, S_f) * S_prev - theta * grad_eta_sum

        # New weights
        decay_S = torch.pow(decay, S_f)
        eta_S = torch.pow(eta, S_f)
        c_S0_general = eta * (decay_S - eta_S) / safe_diff
        c_S0_degen = S_f * eta_S
        c_S0 = torch.where(is_degenerate, c_S0_degen, c_S0_general)

        n_vals = S_f - 1.0 - positions
        w_general = (
            torch.pow(decay, n_vals + 1.0) - torch.pow(eta, n_vals + 1.0)
        ) / safe_diff
        w_degen = (n_vals + 1.0) * torch.pow((decay + eta) / 2.0, n_vals)
        w_weights = torch.where(is_degenerate, w_degen, w_general)
        w_w = w_weights.reshape(1, S, 1)

        weighted_w = errors_scaled * w_w
        grad_combined = torch.mean(
            weighted_w.permute(0, 2, 1) @ keys,
            dim=0,
        )
        grad_combined = torch.clamp(grad_combined, -grad_clip, grad_clip)

        new_weights = decay_S * W_0 + c_S0 * S_prev - theta * grad_combined

        return MemoryState(weights=[new_weights], momentum=[new_momentum])

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        q = self.proj_q(queries)

        if self.use_conv:
            seq_len = q.shape[1]
            q = self.conv_q(q.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]

        q = F.silu(q)
        q_f32 = q.float()
        q = (q_f32 / torch.sqrt(
            torch.sum(q_f32 * q_f32, dim=-1, keepdim=True) + _L2_NORM_EPS
        )).to(q.dtype)

        retrieved = self.memory.forward_with_weights(q, state.weights)
        return self.proj_out(retrieved)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/titans/memory.py tests/test_memory.py
git commit -m "feat: add NeuralLongTermMemory with analytical gradient updates"
```

---

### Task 6: Models (RMSNorm, FeedForward, MACBlock, TitansMAC, Stubs)

**Files:**
- Create: `src/titans/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
"""Tests for Titans model architectures."""

import pytest
import torch

from titans.config import TitansConfig
from titans.models import (
    FeedForward,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    process_chunk,
)


class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_preserves_dtype(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32


class TestFeedForward:
    def test_output_shape(self, default_config, device):
        ffn = FeedForward(default_config).to(device)
        x = torch.randn(2, 8, default_config.dim, device=device)
        out = ffn(x)
        assert out.shape == x.shape


class TestTitansMAC:
    def test_forward_shape(self, default_config, device):
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, default_config.vocab_size)
        assert len(states) == default_config.num_layers

    def test_multi_chunk(self, default_config, device):
        """Sequence longer than chunk_size triggers chunked processing."""
        model = TitansMAC(default_config).to(device)
        seq_len = default_config.chunk_size * 2 + 5
        x = torch.randint(0, default_config.vocab_size, (2, seq_len), device=device)
        logits, states = model(x)
        assert logits.shape == (2, seq_len, default_config.vocab_size)

    def test_state_carryover(self, default_config, device):
        """Memory state from first call can be passed to second."""
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        _, states1 = model(x)
        logits2, states2 = model(x, states=states1)
        assert logits2.shape == (2, 16, default_config.vocab_size)
        # States should differ (memory updated further)
        assert not torch.allclose(states1[0].weights[0], states2[0].weights[0])

    def test_weight_tying(self, default_config, device):
        model = TitansMAC(default_config).to(device)
        assert model.head.weight is model.embed.weight

    def test_backward_pass(self, default_config, device):
        """Full training step should produce gradients."""
        model = TitansMAC(default_config).to(device)
        x = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        labels = torch.randint(0, default_config.vocab_size, (2, 16), device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, default_config.vocab_size), labels.view(-1)
        )
        loss.backward()
        # Check some parameters have gradients
        assert model.embed.weight.grad is not None


class TestStubs:
    def test_mag_not_implemented(self):
        config = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=256)
        with pytest.raises(NotImplementedError, match="MAG"):
            TitansMAG(config)

    def test_mal_not_implemented(self):
        config = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=256)
        with pytest.raises(NotImplementedError, match="MAL"):
            TitansMAL(config)

    def test_lmm_not_implemented(self):
        config = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=256)
        with pytest.raises(NotImplementedError, match="LMM"):
            TitansLMM(config)


class TestDeferredFeatures:
    def test_tnt_raises(self):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256, use_tnt=True
        )
        with pytest.raises(NotImplementedError, match="TNT"):
            TitansMAC(config)

    def test_attn_res_raises(self):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256, use_attn_res=True
        )
        with pytest.raises(NotImplementedError, match="AttnRes"):
            TitansMAC(config)

    def test_mca_raises(self):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            use_mca=True, mca_insertion_layers=[0],
        )
        with pytest.raises(NotImplementedError, match="MCA"):
            TitansMAC(config)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.models'`

- [ ] **Step 3: Write models.py**

Create `src/titans/models.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Titans Model Architectures (PyTorch Implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.attention import SegmentedAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.persistent import PersistentMemory


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight).to(orig_dtype)


class FeedForward(nn.Module):
    """Feed-forward network with SiLU gating."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.down_proj(hidden)


def process_chunk(
    blocks: nn.ModuleList,
    chunk: torch.Tensor,
    states: list,
    config: TitansConfig,
    step_count: int = 0,
) -> tuple[torch.Tensor, list]:
    """Process a single chunk through all blocks with standard residuals."""
    if config.use_attn_res:
        raise NotImplementedError(
            "AttnRes not yet ported. See archive/titans_mlx/attn_res.py"
        )

    new_states = []
    x = chunk
    for i, block in enumerate(blocks):
        core_out, new_state = block.core_forward(x, state=states[i])
        x = x + core_out

        if hasattr(block, "has_mca") and block.has_mca:
            raise NotImplementedError(
                "MCA not yet ported. See archive/titans_mlx/mca.py"
            )

        ffn_out = block.ffn_forward(x)
        x = x + ffn_out
        new_states.append(new_state)
    return x, new_states


class MACBlock(nn.Module):
    """Memory as Context Block."""

    def __init__(self, config: TitansConfig, layer_idx: int = -1) -> None:
        super().__init__()
        self.config = config

        if config.use_tnt:
            raise NotImplementedError(
                "TNT hierarchical memory not yet ported. "
                "See archive/titans_mlx/tnt_memory.py"
            )
        if config.use_attn_res:
            raise NotImplementedError(
                "AttnRes not yet ported. See archive/titans_mlx/attn_res.py"
            )
        if config.use_mca and layer_idx in config.mca_active_insertion_layers:
            raise NotImplementedError(
                "MCA not yet ported. See archive/titans_mlx/mca.py"
            )

        self.memory = NeuralLongTermMemory(config)
        self.memory_query = nn.Parameter(
            torch.randn(1, 1, config.dim) * config.init_std
        )
        self.persistent = PersistentMemory(config)
        self.attention = SegmentedAttention(config)
        self.ffn = FeedForward(config)

        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        self.gate_norm_attn = RMSNorm(config.dim)
        self.gate_norm_mem = RMSNorm(config.dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.has_mca = False

    def core_forward(
        self,
        h: torch.Tensor,
        state: MemoryState | None = None,
        memory_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        batch_size = h.shape[0]

        if state is None:
            state = self.memory.init_state(batch_size)

        query = self.memory_query.expand(batch_size, -1, -1)
        memory_retrieved = self.memory.retrieve(query, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        persistent = self.persistent(batch_size)
        normed = self.norm1(h)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        y_t = h + attn_out

        mem_out, new_state = self.memory(y_t, state=state, memory_gate=memory_gate)

        gated = torch.sigmoid(self.gate_norm_attn(y_t)) * torch.sigmoid(
            self.gate_norm_mem(mem_out)
        )

        core_out = attn_out + gated
        return core_out, new_state

    def ffn_forward(self, h: torch.Tensor) -> torch.Tensor:
        normed = self.norm2(h)
        ffn_out = self.ffn(normed)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        return ffn_out

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        core_out, new_state = self.core_forward(x, state=state)
        x = x + core_out
        ffn_out = self.ffn_forward(x)
        x = x + ffn_out
        return x, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()
        self.head.weight = self.embed.weight
        self._step_count = 0

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        if states is None:
            states = [None] * len(self.blocks)

        x = self.embed(input_ids)

        if seq_len <= chunk_size:
            x, new_states = process_chunk(
                self.blocks, x, states, self.config, self._step_count
            )
        else:
            outputs = []
            new_states = list(states)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = x[:, chunk_start:chunk_end]
                chunk, new_states = process_chunk(
                    self.blocks, chunk, new_states, self.config, self._step_count
                )
                outputs.append(chunk)
            x = torch.cat(outputs, dim=1)

        x = self.norm(x)
        logits = self.head(x)
        self._step_count += 1
        return logits, new_states


class TitansMAG(nn.Module):
    """Titans with Memory as Gate — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansMAG not yet ported. See archive/titans_mlx/models.py"
        )


class TitansMAL(nn.Module):
    """Titans with Memory as Layer — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansMAL not yet ported. See archive/titans_mlx/models.py"
        )


class TitansLMM(nn.Module):
    """Titans Long-term Memory Module (standalone) — not yet ported."""

    def __init__(self, config: TitansConfig) -> None:
        raise NotImplementedError(
            "TitansLMM not yet ported. See archive/titans_mlx/models.py"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/titans/models.py tests/test_models.py
git commit -m "feat: add TitansMAC model with MACBlock, RMSNorm, FeedForward"
```

---

### Task 7: Memory Dump (State Serialization)

**Files:**
- Create: `src/titans/memory_dump.py`
- Create: `tests/test_memory_dump.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_dump.py`:

```python
"""Tests for memory state serialization."""

import torch

from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.memory_dump import load_memory_states, save_memory_states


class TestMemoryDump:
    def test_round_trip(self, default_config, tmp_path, device):
        mem = NeuralLongTermMemory(default_config).to(device)
        states = [mem.init_state(batch_size=2) for _ in range(default_config.num_layers)]
        path = tmp_path / "states.npz"
        save_memory_states(states, path)
        loaded = load_memory_states(path, device=device)

        assert len(loaded) == len(states)
        for orig, restored in zip(states, loaded):
            for w_orig, w_loaded in zip(orig.weights, restored.weights):
                torch.testing.assert_close(w_orig, w_loaded)
            for m_orig, m_loaded in zip(orig.momentum, restored.momentum):
                torch.testing.assert_close(m_orig, m_loaded)

    def test_load_to_cpu(self, default_config, tmp_path):
        mem = NeuralLongTermMemory(default_config)
        states = [mem.init_state(batch_size=2)]
        path = tmp_path / "states.npz"
        save_memory_states(states, path)
        loaded = load_memory_states(path, device=torch.device("cpu"))
        assert loaded[0].weights[0].device.type == "cpu"

    def test_file_not_found(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_memory_states(tmp_path / "nonexistent.npz")

    def test_npz_suffix_auto(self, default_config, tmp_path):
        """Loading without .npz suffix should still work."""
        mem = NeuralLongTermMemory(default_config)
        states = [mem.init_state(batch_size=2)]
        path = tmp_path / "states"
        save_memory_states(states, path)
        # numpy saves as states.npz — load without suffix
        loaded = load_memory_states(path)
        assert len(loaded) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_dump.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'titans.memory_dump'`

- [ ] **Step 3: Write memory_dump.py**

Create `src/titans/memory_dump.py`:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Memory state serialization for Titans (PyTorch Implementation).

Uses .npz format (NumPy) for cross-framework compatibility.
Memory dumps saved by the MLX version can be loaded here and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from titans.memory import MemoryState


def save_memory_states(states: list[MemoryState], path: Path) -> None:
    """Serialize memory states to a single .npz file.

    Args:
        states: List of MemoryState, one per model layer.
        path: Output file path.
    """
    arrays: dict[str, np.ndarray] = {}
    arrays["num_layers"] = np.array([len(states)])

    for i, state in enumerate(states):
        arrays[f"num_memory_layers_{i}"] = np.array([len(state.weights)])
        for j, w in enumerate(state.weights):
            arrays[f"layer_{i}_weight_{j}"] = w.detach().cpu().numpy()
        for j, m in enumerate(state.momentum):
            arrays[f"layer_{i}_momentum_{j}"] = m.detach().cpu().numpy()

    path = Path(path)
    np.savez(path, **arrays)


def load_memory_states(
    path: Path, device: torch.device | None = None
) -> list[MemoryState]:
    """Deserialize memory states from a .npz file.

    Args:
        path: Path to .npz file.
        device: Target device for tensors. Defaults to CPU.

    Returns:
        List of MemoryState, one per model layer.
    """
    if device is None:
        device = torch.device("cpu")

    path = Path(path)
    if not path.exists():
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Memory state file not found: {path}")
        path = path.with_suffix(".npz")

    data = np.load(str(path))

    if "num_layers" not in data:
        raise ValueError("Invalid memory state file: missing 'num_layers' metadata")

    num_layers = int(data["num_layers"][0])
    states: list[MemoryState] = []

    for i in range(num_layers):
        key = f"num_memory_layers_{i}"
        if key not in data:
            raise ValueError(f"Invalid memory state file: missing '{key}'")
        num_memory_layers = int(data[key][0])

        weights: list[torch.Tensor] = []
        momentum: list[torch.Tensor] = []
        for j in range(num_memory_layers):
            wk = f"layer_{i}_weight_{j}"
            mk = f"layer_{i}_momentum_{j}"
            if wk not in data:
                raise ValueError(f"Invalid memory state file: missing '{wk}'")
            if mk not in data:
                raise ValueError(f"Invalid memory state file: missing '{mk}'")
            weights.append(torch.from_numpy(data[wk].copy()).to(device))
            momentum.append(torch.from_numpy(data[mk].copy()).to(device))

        states.append(MemoryState(weights=weights, momentum=momentum))

    return states
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_dump.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/titans/memory_dump.py tests/test_memory_dump.py
git commit -m "feat: add memory state serialization (cross-compatible .npz)"
```

---

### Task 8: Public API (__init__.py)

**Files:**
- Modify: `src/titans/__init__.py`

- [ ] **Step 1: Update __init__.py with public exports**

Replace `src/titans/__init__.py` with:

```python
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time — PyTorch Implementation.

Usage:
    import torch
    from titans import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    x = torch.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

from titans.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.memory_dump import load_memory_states, save_memory_states
from titans.models import (
    FeedForward,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    process_chunk,
)
from titans.persistent import PersistentMemory

__version__ = "0.2.0"

__all__ = [
    # Config
    "TitansConfig",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    "save_memory_states",
    "load_memory_states",
    # Attention
    "RotaryPositionEmbedding",
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Persistent Memory
    "PersistentMemory",
    # Models
    "RMSNorm",
    "FeedForward",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    "process_chunk",
]
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS (config + persistent + attention + memory + models + dump).

- [ ] **Step 3: Run linting**

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Fix any issues.

- [ ] **Step 4: Commit**

```bash
git add src/titans/__init__.py
git commit -m "feat: add public API exports"
```

---

### Task 9: Pretraining Script

**Files:**
- Create: `scripts/pretrain.py`

- [ ] **Step 1: Write pretrain.py**

Create `scripts/pretrain.py`:

```python
#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans PyTorch models.

Supports:
- HuggingFace tokenizers and datasets (streaming)
- HuggingFace Accelerate (single/multi-GPU, mixed precision)
- Cosine annealing with warmup
- Gradient accumulation
- WandB logging (optional)

Usage:
    # Demo with synthetic data (CPU/GPU)
    python scripts/pretrain.py --model mac --dim 256 --epochs 10

    # Train with FineWeb-Edu on GPU
    python scripts/pretrain.py --model mac --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer meta-llama/Llama-2-7b-hf --dim 512 --num-layers 12

    # Multi-GPU via accelerate
    accelerate launch scripts/pretrain.py --model mac --dim 1024 --num-layers 24
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from titans import TitansConfig, TitansMAC

# Optional imports
try:
    from accelerate import Accelerator

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

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


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2
    memory_objective: str = "l2"
    huber_delta_init: float = 0.0

    # Data
    dataset: str | None = None
    dataset_subset: str | None = None
    data_path: str | None = None
    tokenizer: str = "gpt2"
    seq_len: int = 4096

    # Training
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    lr: float = 4e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Mixed precision
    mixed_precision: str = "no"  # "no", "fp16", "bf16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10000
    eval_every: int = 500
    resume: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-pytorch"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    synthetic_samples: int = 10000


# =============================================================================
# Datasets
# =============================================================================


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing/demo."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        np.random.seed(seed)
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.data[idx, :-1].copy()).long(),
            "labels": torch.from_numpy(self.data[idx, 1:].copy()).long(),
        }


class TextFileDataset(Dataset):
    """Dataset from a local text file."""

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizerBase, seq_len: int):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.tokens = np.array(
            tokenizer.encode(text, add_special_tokens=False), dtype=np.int32
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {
            "input_ids": torch.from_numpy(x.copy()).long(),
            "labels": torch.from_numpy(y.copy()).long(),
        }


# =============================================================================
# Training
# =============================================================================


def build_model(config: TrainingConfig) -> TitansMAC:
    """Build Titans model from training config."""
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
        memory_objective=config.memory_objective,
        huber_delta_init=config.huber_delta_init,
    )
    if config.model_type != "mac":
        raise NotImplementedError(
            f"Model type '{config.model_type}' not yet ported. Only 'mac' is available."
        )
    return TitansMAC(model_config)


def build_dataset(config: TrainingConfig) -> Dataset:
    """Build training dataset."""
    if config.data_path is not None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for text file datasets")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        return TextFileDataset(Path(config.data_path), tokenizer, config.seq_len)

    if config.dataset is not None:
        raise NotImplementedError(
            "Streaming HuggingFace dataset support coming soon. "
            "Use --data-path with a local text file, or omit for synthetic data."
        )

    logger.info("No dataset specified — using synthetic data for demo")
    return SyntheticDataset(
        config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
    )


def train(config: TrainingConfig) -> None:
    """Main training loop."""
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate is required for training. Install with: pip install accelerate"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb" if config.wandb and HAS_WANDB else None,
    )

    if accelerator.is_main_process:
        logger.info(f"Training config: {config}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")

    # Seed
    torch.manual_seed(config.seed)

    # Model
    model = build_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Dataset & DataLoader
    dataset = build_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Scheduler
    total_steps = config.max_steps if config.max_steps > 0 else len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )

    # Prepare with accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # WandB
    if config.wandb and HAS_WANDB and accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb_project,
            config=vars(config),
            init_kwargs={"wandb": {"name": config.wandb_run_name}},
        )

    # Checkpoint dir
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    memory_states = None

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_main_process)

        for batch in pbar:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            with accelerator.accumulate(model):
                logits, memory_states = model(batch["input_ids"], states=memory_states)

                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    batch["labels"].view(-1),
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Detach memory states at step boundary
            if memory_states is not None:
                memory_states = [s.detach() for s in memory_states]

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                if config.wandb and HAS_WANDB:
                    accelerator.log(
                        {"loss": loss_val, "avg_loss": avg_loss, "lr": lr},
                        step=global_step,
                    )

            # Save checkpoint
            if global_step % config.save_every == 0 and accelerator.is_main_process:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    # Final save
    if accelerator.is_main_process:
        final_path = checkpoint_dir / "final.pt"
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), final_path)
        logger.info(f"Training complete. Final checkpoint: {final_path}")

    if config.wandb and HAS_WANDB:
        accelerator.end_training()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Pretrain Titans PyTorch models")
    parser.add_argument("--model", type=str, default="mac", choices=["mac"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--num-persistent-tokens", type=int, default=16)
    parser.add_argument("--num-memory-layers", type=int, default=2)
    parser.add_argument("--memory-objective", type=str, default="l2", choices=["l2", "huber"])

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-subset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="titans-pytorch")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=10000)

    args = parser.parse_args()

    return TrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        num_persistent_tokens=args.num_persistent_tokens,
        num_memory_layers=args.num_memory_layers,
        memory_objective=args.memory_objective,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        data_path=args.data_path,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        resume=args.resume,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
```

- [ ] **Step 2: Verify the script parses arguments**

```bash
python scripts/pretrain.py --help
```

Expected: Help text showing all arguments.

- [ ] **Step 3: Verify synthetic data training runs for a few steps**

```bash
python scripts/pretrain.py --model mac --dim 64 --num-heads 4 --num-layers 2 --vocab-size 256 --chunk-size 32 --seq-len 64 --batch-size 2 --gradient-accumulation-steps 1 --max-steps 5 --log-every 1 --synthetic-samples 100
```

Expected: 5 training steps complete with decreasing loss logged.

- [ ] **Step 4: Commit**

```bash
git add scripts/pretrain.py
git commit -m "feat: add PyTorch pretraining script with HuggingFace Accelerate"
```

---

### Task 10: Inference Script

**Files:**
- Create: `scripts/inference.py`

- [ ] **Step 1: Write inference.py**

Create `scripts/inference.py`:

```python
#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans PyTorch models.

Usage:
    python scripts/inference.py --checkpoint checkpoints/final.pt \
        --tokenizer gpt2 --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from titans import TitansConfig, TitansMAC
from titans.memory_dump import load_memory_states, save_memory_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TitansMAC, TitansConfig]:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "config" in ckpt:
        config = TitansConfig.from_dict(ckpt["config"])
        state_dict = ckpt["model"]
    else:
        # Bare state_dict — need config from CLI
        raise ValueError(
            "Checkpoint does not contain config. "
            "Pass model config via CLI args (--dim, --num-layers, etc.)"
        )

    model = TitansMAC(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def generate(
    model: TitansMAC,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    memory_states: list | None = None,
) -> tuple[torch.Tensor, list]:
    """Generate tokens autoregressively."""
    generated = input_ids
    states = memory_states

    for _ in range(max_new_tokens):
        logits, states = model(generated, states=states)
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        # Detach states to prevent graph growth
        if states is not None:
            states = [s.detach() for s in states]

    return generated, states


def main() -> None:
    parser = argparse.ArgumentParser(description="Titans inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--memory-state", type=str, default=None, help="Path to memory state .npz")
    parser.add_argument("--save-memory", type=str, default=None, help="Save memory state after inference")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    logger.info(f"Model loaded: dim={config.dim}, layers={config.num_layers}")

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required for tokenization")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    memory_states = None
    if args.memory_state:
        memory_states = load_memory_states(args.memory_state, device=device)
        logger.info(f"Loaded memory state from {args.memory_state}")

    generated, final_states = generate(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        memory_states=memory_states,
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output_text)

    if args.save_memory and final_states:
        save_memory_states(final_states, Path(args.save_memory))
        logger.info(f"Saved memory state to {args.save_memory}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses arguments**

```bash
python scripts/inference.py --help
```

Expected: Help text showing all arguments.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference.py
git commit -m "feat: add inference script with memory persistence"
```

---

### Task 11: Final Validation and Cleanup

**Files:**
- All files from Tasks 1-10

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 2: Run linting**

```bash
ruff check src/ tests/ scripts/
```

Fix any issues, then:

```bash
ruff format src/ tests/ scripts/
```

- [ ] **Step 3: Verify package installs**

```bash
pip install -e ".[dev]"
```

Expected: Clean install with no errors.

- [ ] **Step 4: Run a quick smoke test**

```bash
python -c "
from titans import TitansConfig, TitansMAC
import torch

config = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=256, chunk_size=32)
model = TitansMAC(config)
x = torch.randint(0, 256, (1, 64))
logits, states = model(x)
print(f'Output shape: {logits.shape}')
print(f'States: {len(states)} layers')
print('Smoke test passed!')
"
```

Expected: Output shape (1, 64, 256), 2 layers, smoke test passed.

- [ ] **Step 5: Commit any lint fixes**

```bash
git add -u
git commit -m "chore: lint fixes and formatting"
```

- [ ] **Step 6: Run the training smoke test**

```bash
python scripts/pretrain.py --model mac --dim 64 --num-heads 4 --num-layers 2 --vocab-size 256 --chunk-size 32 --seq-len 64 --batch-size 2 --gradient-accumulation-steps 1 --max-steps 10 --log-every 1 --synthetic-samples 100
```

Expected: 10 steps complete, loss decreasing.
