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


class TestRopeProportionConfig:
    def test_default_rope_proportion(self):
        config = TitansConfig()
        assert config.rope_proportion == 1.0

    def test_custom_rope_proportion(self):
        config = TitansConfig(rope_proportion=0.25)
        assert config.rope_proportion == 0.25

    def test_rope_proportion_in_to_dict(self):
        config = TitansConfig(rope_proportion=0.5)
        d = config.to_dict()
        assert d["rope_proportion"] == 0.5

    def test_rope_proportion_round_trip(self):
        config = TitansConfig(rope_proportion=0.25)
        config2 = TitansConfig.from_dict(config.to_dict())
        assert config2.rope_proportion == 0.25

    def test_rope_proportion_invalid_above_one(self):
        with pytest.raises(ValueError, match="rope_proportion"):
            TitansConfig(rope_proportion=1.5)

    def test_rope_proportion_invalid_negative(self):
        with pytest.raises(ValueError, match="rope_proportion"):
            TitansConfig(rope_proportion=-0.1)
