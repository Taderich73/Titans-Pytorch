# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for streaming eval buffering."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class FakeStreamingDataset:
    """Minimal fake that yields dicts like StreamingDataset.__iter__."""

    def __init__(self, num_samples: int = 200, seq_len: int = 32) -> None:
        self.seq_len = seq_len
        self._num_samples = num_samples

    def __iter__(self):
        for i in range(self._num_samples):
            ids = np.arange(i, i + self.seq_len, dtype=np.int32)
            yield {
                "input_ids": mx.array(ids),
                "labels": mx.array(ids + 1),
            }


class TestBufferedEvalDataset:
    """Tests for BufferedEvalDataset."""

    def test_buffers_requested_count(self) -> None:
        """Buffer should contain exactly num_sequences samples."""
        from pretrain import BufferedEvalDataset

        fake = FakeStreamingDataset(num_samples=200, seq_len=32)
        buf = BufferedEvalDataset(fake, num_sequences=50)
        assert len(buf) == 50

    def test_buffers_fewer_if_stream_short(self) -> None:
        """If stream has fewer samples, buffer takes what's available."""
        from pretrain import BufferedEvalDataset

        fake = FakeStreamingDataset(num_samples=10, seq_len=32)
        buf = BufferedEvalDataset(fake, num_sequences=100)
        assert len(buf) == 10

    def test_get_batch_returns_correct_shape(self) -> None:
        """get_batch(indices) returns stacked arrays with correct batch dim."""
        from pretrain import BufferedEvalDataset

        fake = FakeStreamingDataset(num_samples=50, seq_len=16)
        buf = BufferedEvalDataset(fake, num_sequences=50)

        batch = buf.get_batch([0, 3, 7])
        assert batch["input_ids"].shape == (3, 16)
        assert batch["labels"].shape == (3, 16)

    def test_get_batch_returns_correct_data(self) -> None:
        """get_batch should return the exact buffered sequences by index."""
        from pretrain import BufferedEvalDataset

        fake = FakeStreamingDataset(num_samples=20, seq_len=8)
        buf = BufferedEvalDataset(fake, num_sequences=20)

        batch = buf.get_batch([5])
        expected = np.arange(5, 5 + 8, dtype=np.int32)
        np.testing.assert_array_equal(np.array(batch["input_ids"][0]), expected)

    def test_len_matches_buffer(self) -> None:
        """__len__ must match actual buffered count for evaluate() indexing."""
        from pretrain import BufferedEvalDataset

        fake = FakeStreamingDataset(num_samples=100, seq_len=32)
        buf = BufferedEvalDataset(fake, num_sequences=75)
        assert len(buf) == 75
        batch = buf.get_batch(list(range(75)))
        assert batch["input_ids"].shape[0] == 75


class TestTrainingConfigEvalFields:
    """Tests for eval-related TrainingConfig fields."""

    def test_default_eval_fields(self) -> None:
        """TrainingConfig should have eval_dataset, eval_split, eval_buffer_size."""
        from pretrain import TrainingConfig

        config = TrainingConfig()
        assert config.eval_dataset is None
        assert config.eval_split == "train"
        assert config.eval_buffer_size == 100
