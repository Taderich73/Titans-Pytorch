# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Persistent Memory module (MLX)."""

import mlx.core as mx
import numpy as np

from titans_mlx.config import TitansConfig
from titans_mlx.persistent import PersistentMemory


class TestPersistentMemory:
    """Tests for PersistentMemory module."""

    def test_forward(
        self, default_config: TitansConfig, batch_size: int
    ) -> None:
        """Test forward returns batch-expanded tokens."""
        persistent = PersistentMemory(default_config)
        tokens = persistent(batch_size)
        mx.eval(tokens)

        assert tokens is not None
        assert tokens.shape == (
            batch_size,
            default_config.num_persistent_tokens,
            default_config.dim,
        )

    def test_forward_batch_expansion(
        self, default_config: TitansConfig, batch_size: int
    ) -> None:
        """Test tokens are properly expanded for batch."""
        persistent = PersistentMemory(default_config)

        tokens1 = persistent(batch_size)
        tokens2 = persistent(batch_size * 2)
        mx.eval(tokens1, tokens2)

        assert tokens1.shape[0] == batch_size
        assert tokens2.shape[0] == batch_size * 2

        np.testing.assert_allclose(
            np.array(tokens1[0]),
            np.array(tokens2[0]),
            rtol=1e-6,
        )

    def test_zero_tokens(self) -> None:
        """Test with zero persistent tokens."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_persistent_tokens=0,
        )
        persistent = PersistentMemory(config)

        tokens = persistent(2)

        assert tokens is None
        assert persistent.tokens is None

    def test_get_tokens(self, default_config: TitansConfig) -> None:
        """Test get_tokens returns raw embeddings."""
        persistent = PersistentMemory(default_config)
        tokens = persistent.get_tokens()

        assert tokens is not None
        assert tokens.shape == (
            default_config.num_persistent_tokens,
            default_config.dim,
        )

    def test_get_tokens_zero(self) -> None:
        """Test get_tokens with zero tokens."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_persistent_tokens=0,
        )
        persistent = PersistentMemory(config)

        tokens = persistent.get_tokens()

        assert tokens is None

    def test_tokens_initialized_with_std(self) -> None:
        """Test tokens are initialized with approximately correct std."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_persistent_tokens=100,
            init_std=0.02,
        )
        persistent = PersistentMemory(config)

        tokens = persistent.get_tokens()
        mx.eval(tokens)
        assert tokens is not None

        std = float(mx.std(tokens))
        assert 0.01 < std < 0.04

    def test_different_num_tokens(self) -> None:
        """Test different numbers of persistent tokens."""
        for num_tokens in [1, 8, 32, 64]:
            config = TitansConfig(
                dim=64,
                num_heads=4,
                num_persistent_tokens=num_tokens,
            )
            persistent = PersistentMemory(config)

            tokens = persistent(4)
            mx.eval(tokens)

            assert tokens is not None
            assert tokens.shape[1] == num_tokens
