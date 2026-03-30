# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for TitansConfig."""

import pytest

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


class TestMemoryStateQuantizationConfig:
    """Tests for memory state quantization config fields."""

    def test_defaults(self) -> None:
        """Memory state quantization is off by default."""
        config = TitansConfig()
        assert config.quantize_memory_state is False
        assert config.memory_state_weight_bits == 4
        assert config.memory_state_momentum_bits == 8

    def test_to_dict_includes_fields(self) -> None:
        """to_dict includes memory state quantization fields."""
        config = TitansConfig(quantize_memory_state=True, memory_state_weight_bits=8)
        d = config.to_dict()
        assert d["quantize_memory_state"] is True
        assert d["memory_state_weight_bits"] == 8
        assert d["memory_state_momentum_bits"] == 8

    def test_from_dict_round_trip(self) -> None:
        """from_dict restores memory state quantization fields."""
        config = TitansConfig(quantize_memory_state=True, memory_state_weight_bits=8)
        restored = TitansConfig.from_dict(config.to_dict())
        assert restored.quantize_memory_state is True
        assert restored.memory_state_weight_bits == 8


class TestMCAConfig:
    """Tests for MCA configuration fields."""

    def test_mca_defaults(self) -> None:
        """MCA is disabled by default with sensible defaults."""
        config = TitansConfig()
        assert config.use_mca is False
        assert config.mca_insertion_layers is None
        assert config.mca_num_heads == 8
        assert config.mca_gate_type == "scalar"
        assert config.mca_gate_bias_init == -3.0

    def test_mca_active_insertion_layers_disabled(self) -> None:
        """When MCA is disabled, no insertion layers."""
        config = TitansConfig(use_mca=False)
        assert config.mca_active_insertion_layers == []

    def test_mca_active_insertion_layers_auto(self) -> None:
        """Auto insertion defaults to midpoint."""
        config = TitansConfig(use_mca=True, num_layers=12)
        assert config.mca_active_insertion_layers == [6]

    def test_mca_active_insertion_layers_explicit(self) -> None:
        """Explicit insertion layers override auto."""
        config = TitansConfig(
            use_mca=True, num_layers=12, mca_insertion_layers=[2, 6, 10]
        )
        assert config.mca_active_insertion_layers == [2, 6, 10]

    def test_mca_insertion_layer_validation(self) -> None:
        """Insertion layer >= num_layers raises ValueError."""
        with pytest.raises(ValueError, match="MCA insertion layer"):
            TitansConfig(use_mca=True, num_layers=6, mca_insertion_layers=[6])

    def test_mca_dump_defaults(self) -> None:
        """Dump config has sensible defaults."""
        config = TitansConfig()
        assert config.mca_auto_dump is False
        assert config.mca_dump_trigger == "session_end"
        assert config.mca_dump_path == "./memory_dumps/"
        assert config.mca_dump_keep_last_n == 10

    def test_attnres_sublayer_count_with_mca(self) -> None:
        """AttnRes sub-layer block size accounts for MCA layers."""
        base = TitansConfig(num_layers=6, num_attnres_blocks=4)
        mca = TitansConfig(
            num_layers=6, num_attnres_blocks=4,
            use_mca=True, mca_insertion_layers=[3],
        )
        assert base.attnres_sub_layer_block_size == 3
        assert mca.attnres_sub_layer_block_size == 3

    def test_mca_in_to_dict(self) -> None:
        """MCA fields round-trip through to_dict/from_dict."""
        config = TitansConfig(
            use_mca=True, mca_insertion_layers=[4], mca_num_heads=4,
        )
        d = config.to_dict()
        assert d["use_mca"] is True
        assert d["mca_insertion_layers"] == [4]
        restored = TitansConfig.from_dict(d)
        assert restored.use_mca is True
        assert restored.mca_insertion_layers == [4]
        assert restored.mca_num_heads == 4
