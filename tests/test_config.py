# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for TitansConfig."""

from titans_mlx.config import TitansConfig


class TestTitansConfig:
    """Tests for TitansConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TitansConfig()

        assert config.dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 12
        assert config.ffn_mult == 4.0
        assert config.num_memory_layers == 2
        assert config.num_persistent_tokens == 16
        assert config.chunk_size == 512
        assert config.window_size == 512
        assert config.dropout == 0.0
        assert config.use_conv is True
        assert config.use_rope is True
        assert config.vocab_size == 32000

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TitansConfig(
            dim=256,
            num_heads=4,
            num_layers=6,
            vocab_size=1000,
        )

        assert config.dim == 256
        assert config.num_heads == 4
        assert config.num_layers == 6
        assert config.vocab_size == 1000

    def test_head_dim_property(self) -> None:
        """Test head_dim computed property."""
        config = TitansConfig(dim=512, num_heads=8)
        assert config.head_dim == 64

        config = TitansConfig(dim=256, num_heads=4)
        assert config.head_dim == 64

    def test_ffn_dim_property(self) -> None:
        """Test ffn_dim computed property."""
        config = TitansConfig(dim=512, ffn_mult=4.0)
        assert config.ffn_dim == 2048

        config = TitansConfig(dim=256, ffn_mult=2.0)
        assert config.ffn_dim == 512

    def test_memory_hidden_dim_property(self) -> None:
        """Test memory_hidden_dim computed property."""
        config = TitansConfig(dim=512, memory_hidden_mult=4.0)
        assert config.memory_hidden_dim == 2048

    def test_to_dict(self) -> None:
        """Test config serialization to dict."""
        config = TitansConfig(dim=256, num_heads=4)
        d = config.to_dict()

        assert d["dim"] == 256
        assert d["num_heads"] == 4
        assert isinstance(d, dict)

    def test_from_dict(self) -> None:
        """Test config creation from dict."""
        original = TitansConfig(dim=128, num_heads=2, num_layers=4)
        d = original.to_dict()
        restored = TitansConfig.from_dict(d)

        assert restored.dim == original.dim
        assert restored.num_heads == original.num_heads
        assert restored.num_layers == original.num_layers

    def test_roundtrip_to_dict_from_dict(self) -> None:
        """Test that to_dict -> from_dict produces identical config."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=3,
            vocab_size=500,
            memory_lr=0.05,
            memory_momentum=0.8,
        )
        restored = TitansConfig.from_dict(config.to_dict())

        assert config.to_dict() == restored.to_dict()


def test_attnres_sub_layer_block_size():
    """Sub-layer block size accounts for 2 sub-layers per block."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=12, vocab_size=100,
        use_attn_res=True, num_attnres_blocks=4,
    )
    # 12 layers * 2 sub-layers / 4 blocks = 6 sub-layers per block
    assert config.attnres_sub_layer_block_size == 6


def test_attnres_sub_layer_block_size_default():
    """Default 8 blocks with 12 layers."""
    config = TitansConfig(
        dim=32, num_heads=4, num_layers=12, vocab_size=100,
        use_attn_res=True, num_attnres_blocks=8,
    )
    # 12 * 2 / 8 = 3
    assert config.attnres_sub_layer_block_size == 3
