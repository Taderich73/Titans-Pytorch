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

    def test_tnt_stage2_rejects_non_stage1_input(self):
        """Calling tnt_stage2 on a stage-2 config must raise."""
        s1 = TitansConfig.tnt_stage1()
        s2 = TitansConfig.tnt_stage2(s1)
        with pytest.raises(ValueError, match="stage 1"):
            TitansConfig.tnt_stage2(s2)

    def test_tnt_stage2_halves_chunk_sizes_once(self):
        """Stage-2 chunk sizes are exactly stage-1 sizes // 2."""
        s1 = TitansConfig.tnt_stage1(local_chunk_sizes=[8, 16])
        s2 = TitansConfig.tnt_stage2(s1)
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


class TestAutoCheckpointConfig:
    def test_default_off(self):
        config = TitansConfig()
        assert config.auto_checkpoint is False

    def test_round_trip(self):
        config = TitansConfig(auto_checkpoint=True)
        d = config.to_dict()
        assert d["auto_checkpoint"] is True
        config2 = TitansConfig.from_dict(d)
        assert config2.auto_checkpoint is True

    def test_checkpoint_config_in_dict(self):
        from titans.checkpointing.types import MemoryCheckpointConfig

        config = TitansConfig(
            auto_checkpoint=True,
            checkpoint_config=MemoryCheckpointConfig(sigma_threshold=3.0),
        )
        d = config.to_dict()
        assert d["checkpoint_config"]["sigma_threshold"] == 3.0
        config2 = TitansConfig.from_dict(d)
        assert config2.checkpoint_config.sigma_threshold == 3.0

    def test_from_dict_ignores_unknown_keys(self):
        d = TitansConfig().to_dict()
        d["some_future_field"] = "value"
        config = TitansConfig.from_dict(d)  # should not raise
        assert config.dim == 512


class TestMemoryInnerSteps:
    """Config field num_memory_inner_steps with validation."""

    def test_default_is_one(self):
        from titans.config import TitansConfig

        config = TitansConfig()
        assert config.num_memory_inner_steps == 1

    def test_explicit_value(self):
        from titans.config import TitansConfig

        config = TitansConfig(num_memory_inner_steps=8)
        assert config.num_memory_inner_steps == 8

    def test_rejects_zero(self):
        import pytest

        from titans.config import TitansConfig

        with pytest.raises(ValueError, match="num_memory_inner_steps"):
            TitansConfig(num_memory_inner_steps=0)

    def test_rejects_negative(self):
        import pytest

        from titans.config import TitansConfig

        with pytest.raises(ValueError, match="num_memory_inner_steps"):
            TitansConfig(num_memory_inner_steps=-2)

    def test_to_dict_roundtrip(self):
        from titans.config import TitansConfig

        c1 = TitansConfig(num_memory_inner_steps=4)
        d = c1.to_dict()
        assert d["num_memory_inner_steps"] == 4
        c2 = TitansConfig.from_dict(d)
        assert c2.num_memory_inner_steps == 4

    def test_hf_config_roundtrip(self):
        from titans.config import TitansConfig
        from titans.hf.configuration import TitansMACConfig

        titans_cfg = TitansConfig(num_memory_inner_steps=8)
        hf_cfg = TitansMACConfig.from_titans_config(titans_cfg)
        assert hf_cfg.num_memory_inner_steps == 8
        round_tripped = hf_cfg.to_titans_config()
        assert round_tripped.num_memory_inner_steps == 8


def test_freeze_inner_loop_default_and_roundtrip():
    """freeze_inner_loop defaults False and round-trips through to_dict/from_dict."""
    cfg = TitansConfig()
    assert cfg.freeze_inner_loop is False

    cfg.freeze_inner_loop = True
    d = cfg.to_dict()
    assert d["freeze_inner_loop"] is True

    restored = TitansConfig.from_dict(d)
    assert restored.freeze_inner_loop is True
