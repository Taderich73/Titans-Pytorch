"""End-to-end smoke test for LoRA periodic evaluation.

Confirms that ``config.eval_every`` is actually honored by ``train()`` —
specifically that the eval branch fires at least once during a short run
and that no eval dataset configured simply skips without crashing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("accelerate")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lora  # noqa: E402
from lora import (  # noqa: E402
    LoRATrainingConfig,
    SyntheticChatDataset,
    build_eval_dataset,
    evaluate,
    train,
)


def _tiny_config(tmp_path: Path, **overrides) -> LoRATrainingConfig:
    """Minimal LoRATrainingConfig that trains end-to-end on CPU in seconds."""
    base = dict(
        model_type="mac",
        dim=32,
        num_heads=4,
        num_layers=2,
        vocab_size=128,
        chunk_size=16,
        window_size=16,
        num_persistent_tokens=2,
        num_memory_layers=1,
        seq_len=16,
        max_seq_len=16,
        epochs=1,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        lr=1e-3,
        warmup_ratio=0.0,
        mixed_precision="no",
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_targets="attn",
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_every=10_000,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        wandb=False,
        synthetic_samples=8,
        seed=0,
    )
    base.update(overrides)
    return LoRATrainingConfig(**base)


def test_evaluate_helper_runs_on_synthetic_data():
    """evaluate() should return a finite float when supervised tokens exist."""
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    config = LoRATrainingConfig(
        dim=32,
        num_heads=4,
        num_layers=2,
        vocab_size=128,
        chunk_size=16,
        window_size=16,
        num_persistent_tokens=2,
        num_memory_layers=1,
        seq_len=16,
        max_seq_len=16,
        lora_rank=2,
        lora_alpha=4.0,
    )
    model = lora.build_model(config)
    dataset = SyntheticChatDataset(
        config.vocab_size, config.seq_len, num_samples=4, seed=0
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    loss = evaluate(
        model, dataloader, accelerator, config.vocab_size, max_batches=1
    )
    assert isinstance(loss, float)
    assert loss > 0.0 and loss != float("inf")


def test_build_eval_dataset_returns_none_without_path():
    """Missing --eval-data-path must skip eval cleanly, not raise."""
    config = LoRATrainingConfig(eval_data_path=None)
    assert build_eval_dataset(config) is None


def test_train_fires_eval_branch_end_to_end(tmp_path, caplog, monkeypatch):
    """With eval_every=1 the eval branch must fire at least once.

    Monkey-patches ``build_eval_dataset`` to return a small synthetic set so
    we exercise the full ``train()`` path (accelerator prepare, eval branch,
    evaluate() call, logger output) without needing a tokenizer or JSONL
    file.
    """

    def _synthetic_eval(cfg: LoRATrainingConfig):
        return SyntheticChatDataset(
            cfg.vocab_size, cfg.seq_len, num_samples=4, seed=cfg.seed + 1
        )

    monkeypatch.setattr(lora, "build_eval_dataset", _synthetic_eval)

    config = _tiny_config(tmp_path)

    with caplog.at_level(logging.INFO, logger="lora"):
        train(config)

    val_messages = [
        r.message for r in caplog.records if "val loss" in r.message
    ]
    assert val_messages, (
        "Expected at least one 'val loss' log line; got none. "
        f"All messages: {[r.message for r in caplog.records]}"
    )


def test_train_without_eval_dataset_does_not_crash(tmp_path, caplog):
    """No --eval-data-path should skip eval silently without crashing."""
    config = _tiny_config(tmp_path)
    assert config.eval_data_path is None

    with caplog.at_level(logging.INFO, logger="lora"):
        train(config)

    skip_messages = [
        r.message
        for r in caplog.records
        if "skipping periodic evaluation" in r.message
    ]
    assert skip_messages, "Expected skip-eval log line when no eval data set."
