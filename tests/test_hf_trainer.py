"""Tests for TitansTrainer with per-chunk truncated BPTT."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("transformers")

from transformers import TrainingArguments

from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM
from titans.hf.trainer import TitansChunkMixin, TitansTrainer


class DummyDataset(Dataset):
    """Minimal dataset returning input_ids and labels."""

    def __init__(self, vocab_size: int, seq_len: int, size: int = 20):
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens, "labels": tokens.clone()}


@pytest.fixture
def small_hf_config():
    return TitansMACConfig(
        dim=64, num_heads=4, num_layers=2, vocab_size=256,
        chunk_size=32, window_size=32, max_seq_len=256,
        num_memory_layers=2, num_persistent_tokens=4,
    )


@pytest.fixture
def small_model(small_hf_config):
    return TitansMACForCausalLM(small_hf_config)


class TestTitansChunkMixin:
    """TitansChunkMixin provides the per-chunk BPTT loop."""

    def test_mixin_has_compute_loss(self):
        assert hasattr(TitansChunkMixin, "compute_loss")

    def test_mixin_has_memory_state_attrs(self):
        assert hasattr(TitansChunkMixin, "_init_titans_memory")


class TestTitansTrainer:
    """TitansTrainer runs a training step with chunked BPTT."""

    def test_trainer_construction(self, small_model, tmp_path):
        dataset = DummyDataset(vocab_size=256, seq_len=64)
        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            use_cpu=True,
            report_to="none",
        )
        trainer = TitansTrainer(
            model=small_model, args=args, train_dataset=dataset,
        )
        assert trainer._memory_states is None
        assert trainer.reset_memory_per_batch is True

    def test_training_step_runs(self, small_model, tmp_path):
        dataset = DummyDataset(vocab_size=256, seq_len=64)
        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=2,
            use_cpu=True,
            report_to="none",
            logging_steps=1,
        )
        trainer = TitansTrainer(
            model=small_model, args=args, train_dataset=dataset,
        )
        result = trainer.train()
        assert result.training_loss > 0

    def test_memory_reset_per_batch(self, small_model, tmp_path):
        """With reset_memory_per_batch=True, states reset after each batch."""
        dataset = DummyDataset(vocab_size=256, seq_len=64)
        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=3,
            use_cpu=True,
            report_to="none",
        )
        trainer = TitansTrainer(
            model=small_model, args=args, train_dataset=dataset,
            reset_memory_per_batch=True,
        )
        trainer.train()
        assert trainer._memory_states is None
