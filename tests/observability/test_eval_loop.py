"""Tests for eval_loop: EvalConfig, holdout partitioning, run_eval."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from titans.observability.eval_loop import (
    EvalConfig,
    is_eval_example,
    run_eval,
)

# --------------------------------------------------------------------------
# Stable partitioning
# --------------------------------------------------------------------------


def test_partition_is_deterministic() -> None:
    """The same key always lands in the same partition."""
    cfg = EvalConfig(holdout_fraction=0.1, holdout_seed=42)
    results = [is_eval_example(key=f"doc-{i}", cfg=cfg) for i in range(1000)]
    # Re-run, expect identical.
    results_again = [is_eval_example(key=f"doc-{i}", cfg=cfg) for i in range(1000)]
    assert results == results_again


def test_partition_fraction_is_approximately_correct() -> None:
    """Across a large sample, the eval fraction is close to the target."""
    cfg = EvalConfig(holdout_fraction=0.1, holdout_seed=42)
    hits = sum(is_eval_example(key=f"doc-{i}", cfg=cfg) for i in range(10_000))
    # Expect ~1000 ± tolerance. Tolerate ± 3%.
    assert 700 <= hits <= 1300


def test_different_seeds_give_different_partitions() -> None:
    """Two different holdout_seed values produce different partitions."""
    cfg_a = EvalConfig(holdout_fraction=0.1, holdout_seed=1)
    cfg_b = EvalConfig(holdout_fraction=0.1, holdout_seed=2)
    keys = [f"doc-{i}" for i in range(1000)]
    a = [is_eval_example(k, cfg_a) for k in keys]
    b = [is_eval_example(k, cfg_b) for k in keys]
    assert a != b  # overwhelmingly likely false-negative risk is vanishing


def test_train_eval_disjoint_by_construction() -> None:
    """is_eval_example partitions keys — inverse filter gives the training set."""
    cfg = EvalConfig(holdout_fraction=0.1, holdout_seed=7)
    keys = [f"doc-{i}" for i in range(5000)]
    eval_keys = {k for k in keys if is_eval_example(k, cfg)}
    train_keys = {k for k in keys if not is_eval_example(k, cfg)}
    assert eval_keys.isdisjoint(train_keys)
    assert len(eval_keys) + len(train_keys) == len(keys)


# --------------------------------------------------------------------------
# run_eval
# --------------------------------------------------------------------------


class _ToyDataset(Dataset):
    def __init__(self, n: int, seq_len: int, vocab: int = 8) -> None:
        torch.manual_seed(0)
        self.data = torch.randint(0, vocab, (n, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


class _ToyModel(nn.Module):
    """Minimal stateless LM used by eval test."""

    def __init__(self, vocab: int = 8, dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.emb(input_ids))


def _toy_loss_fn(model: _ToyModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Callable adapter matching the loss_fn contract for run_eval."""
    logits = model.forward_logits(batch["input_ids"])
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1)
    )


class _StubAccelerator:
    """Minimal accelerator stand-in supporting gather_for_metrics + prepare."""

    is_main_process = True

    def prepare(self, obj: object) -> object:  # noqa: ARG002
        return obj

    def gather_for_metrics(self, t: torch.Tensor) -> torch.Tensor:
        return t


def test_run_eval_returns_finite_loss() -> None:
    ds = _ToyDataset(n=64, seq_len=16)
    loader = DataLoader(ds, batch_size=8)
    model = _ToyModel()
    cfg = EvalConfig(num_batches=4, reset_memory_state=False)

    out = run_eval(
        model=model,
        eval_loader=loader,
        accelerator=_StubAccelerator(),
        cfg=cfg,
        loss_fn=_toy_loss_fn,
    )

    assert "eval/loss" in out
    assert "eval/ppl" in out
    assert "eval/num_batches" in out
    assert math.isfinite(out["eval/loss"])
    assert math.isfinite(out["eval/ppl"])
    assert out["eval/num_batches"] == 4


def test_run_eval_restores_train_mode() -> None:
    """After run_eval, model.training must be True."""
    ds = _ToyDataset(n=32, seq_len=8)
    loader = DataLoader(ds, batch_size=4)
    model = _ToyModel()
    model.train()

    run_eval(
        model=model,
        eval_loader=loader,
        accelerator=_StubAccelerator(),
        cfg=EvalConfig(num_batches=2, reset_memory_state=False),
        loss_fn=_toy_loss_fn,
    )

    assert model.training is True


def test_run_eval_respects_max_batches() -> None:
    """max_batches overrides cfg.num_batches when smaller."""
    ds = _ToyDataset(n=64, seq_len=16)
    loader = DataLoader(ds, batch_size=8)
    out = run_eval(
        model=_ToyModel(),
        eval_loader=loader,
        accelerator=_StubAccelerator(),
        cfg=EvalConfig(num_batches=10, reset_memory_state=False),
        loss_fn=_toy_loss_fn,
        max_batches=3,
    )
    assert out["eval/num_batches"] == 3


def test_ppl_equals_exp_loss() -> None:
    out = run_eval(
        model=_ToyModel(),
        eval_loader=DataLoader(_ToyDataset(n=32, seq_len=8), batch_size=4),
        accelerator=_StubAccelerator(),
        cfg=EvalConfig(num_batches=2, reset_memory_state=False),
        loss_fn=_toy_loss_fn,
    )
    assert out["eval/ppl"] == pytest.approx(math.exp(out["eval/loss"]), rel=1e-5)
