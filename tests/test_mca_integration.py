# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Integration tests for Memory Cross-Attention with model variants."""

import mlx.core as mx
import numpy as np
import pytest
from pathlib import Path

from titans_mlx.config import TitansConfig
from titans_mlx.models import MACBlock, MAGBlock, MALBlock, TitansLMM, TitansMAC, TitansMAG, TitansMAL, process_chunk


def _mca_config(**kwargs) -> TitansConfig:
    defaults = dict(
        dim=64, num_heads=4, num_layers=6, vocab_size=256,
        use_mca=True, mca_num_heads=4, mca_insertion_layers=[3],
        num_memory_layers=2, memory_hidden_mult=2.0,
        num_persistent_tokens=4, chunk_size=32, window_size=16,
        max_seq_len=256,
    )
    defaults.update(kwargs)
    return TitansConfig(**defaults)


class TestBlockMCA:
    def test_mac_block_has_mca_at_insertion_layer(self) -> None:
        config = _mca_config(mca_insertion_layers=[3])
        block = MACBlock(config, layer_idx=3)
        assert block.has_mca is True
        assert hasattr(block, "mca")

    def test_mac_block_no_mca_at_other_layer(self) -> None:
        config = _mca_config(mca_insertion_layers=[3])
        block = MACBlock(config, layer_idx=0)
        assert block.has_mca is False
        assert not hasattr(block, "mca")

    def test_mag_block_has_mca(self) -> None:
        config = _mca_config(mca_insertion_layers=[3])
        block = MAGBlock(config, layer_idx=3)
        assert block.has_mca is True

    def test_mal_block_has_mca(self) -> None:
        config = _mca_config(mca_insertion_layers=[3])
        block = MALBlock(config, layer_idx=3)
        assert block.has_mca is True

    def test_mca_forward_returns_correct_shape(self) -> None:
        config = _mca_config(mca_insertion_layers=[0])
        block = MACBlock(config, layer_idx=0)
        x = mx.random.normal((2, 16, 64))
        core_out, state = block.core_forward(x)
        h = x + core_out
        mca_out = block.mca_forward(h, state)
        mx.eval(mca_out)
        assert mca_out.shape == (2, 16, 64)

    def test_block_without_mca_no_mca_forward(self) -> None:
        config = _mca_config(use_mca=False)
        block = MACBlock(config, layer_idx=0)
        assert block.has_mca is False

    def test_mca_with_attn_res(self) -> None:
        config = _mca_config(mca_insertion_layers=[3], use_attn_res=True)
        block = MACBlock(config, layer_idx=3)
        assert hasattr(block, "attn_res_mca")

    def test_backward_compat_layer_idx_default(self) -> None:
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=6, vocab_size=256,
            num_persistent_tokens=4, chunk_size=32, window_size=16, max_seq_len=256,
        )
        block = MACBlock(config)
        assert block.has_mca is False


class TestProcessChunkMCA:
    def test_process_chunk_standard_with_mca(self) -> None:
        config = _mca_config(mca_insertion_layers=[1])
        blocks = [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
        chunk = mx.random.normal((2, 16, 64))
        states = [None] * config.num_layers
        out, new_states = process_chunk(blocks, chunk, states, config)
        mx.eval(out)
        assert out.shape == (2, 16, 64)
        assert not np.any(np.isnan(np.array(out)))
        assert len(new_states) == config.num_layers

    def test_process_chunk_attnres_with_mca(self) -> None:
        config = _mca_config(
            mca_insertion_layers=[1],
            use_attn_res=True, num_attnres_blocks=4,
        )
        blocks = [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
        chunk = mx.random.normal((2, 16, 64))
        states = [None] * config.num_layers
        out, new_states = process_chunk(blocks, chunk, states, config)
        mx.eval(out)
        assert out.shape == (2, 16, 64)
        assert not np.any(np.isnan(np.array(out)))

    def test_process_chunk_no_mca_unchanged(self) -> None:
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            num_persistent_tokens=4, chunk_size=32, window_size=16, max_seq_len=256,
        )
        blocks = [MACBlock(config, layer_idx=i) for i in range(config.num_layers)]
        chunk = mx.random.normal((2, 16, 64))
        states = [None] * config.num_layers
        out, new_states = process_chunk(blocks, chunk, states, config)
        mx.eval(out)
        assert out.shape == (2, 16, 64)
        assert len(new_states) == config.num_layers


class TestModelMCA:
    def test_mac_forward(self) -> None:
        config = _mca_config()
        model = TitansMAC(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)
        assert not np.any(np.isnan(np.array(logits)))

    def test_mag_forward(self) -> None:
        config = _mca_config()
        model = TitansMAG(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)

    def test_mal_forward(self) -> None:
        config = _mca_config()
        model = TitansMAL(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)

    def test_lmm_ignores_mca(self) -> None:
        config = _mca_config()
        model = TitansLMM(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)

    def test_mca_state_evolves_across_chunks(self) -> None:
        config = _mca_config(chunk_size=16)
        model = TitansMAC(config)
        input_ids_1 = mx.random.randint(0, 256, (2, 16))
        input_ids_2 = mx.random.randint(0, 256, (2, 16))
        logits_1, states_1 = model(input_ids_1)
        logits_2, states_2 = model(input_ids_2, states=states_1)
        mx.eval(logits_1, logits_2)
        assert states_2 is not None

    def test_mca_with_tnt(self) -> None:
        config = _mca_config(
            use_tnt=True, global_chunk_size=32,
            local_chunk_sizes=[4, 8], local_shard_length=32,
        )
        model = TitansMAC(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)
        assert not np.any(np.isnan(np.array(logits)))

    def test_mca_with_attn_res(self) -> None:
        config = _mca_config(use_attn_res=True, num_attnres_blocks=4)
        model = TitansMAC(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)
        assert not np.any(np.isnan(np.array(logits)))

    def test_mca_with_tnt_and_attn_res(self) -> None:
        config = _mca_config(
            use_tnt=True, global_chunk_size=32,
            local_chunk_sizes=[4, 8], local_shard_length=32,
            use_attn_res=True, num_attnres_blocks=4,
        )
        model = TitansMAC(config)
        input_ids = mx.random.randint(0, 256, (2, 32))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 32, 256)
        assert not np.any(np.isnan(np.array(logits)))

    def test_mca_no_regression_without(self) -> None:
        config_base = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            num_persistent_tokens=4, chunk_size=32, window_size=16, max_seq_len=256,
        )
        config_mca_off = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            num_persistent_tokens=4, chunk_size=32, window_size=16, max_seq_len=256,
            use_mca=False,
        )
        mx.random.seed(42)
        model_base = TitansMAC(config_base)
        mx.random.seed(42)
        model_mca_off = TitansMAC(config_mca_off)
        input_ids = mx.random.randint(0, 256, (2, 32))
        mx.random.seed(99)
        logits_base, _ = model_base(input_ids)
        mx.random.seed(99)
        logits_off, _ = model_mca_off(input_ids)
        mx.eval(logits_base, logits_off)
        np.testing.assert_allclose(np.array(logits_base), np.array(logits_off), atol=1e-5)


from titans_mlx.memory_dump import MemoryDumpManager


class TestPersistenceE2E:
    """End-to-end tests for state persistence across sessions."""

    def test_dump_load_resume(self, tmp_path: Path) -> None:
        """Dump states, load into fresh model, outputs match."""
        config = _mca_config(mca_dump_path=str(tmp_path / "dumps"))
        model = TitansMAC(config)
        input_ids = mx.random.randint(0, 256, (2, 32))

        logits_1, states_1 = model(input_ids)
        mx.eval(logits_1)

        mgr = MemoryDumpManager(config)
        dump_path = mgr.dump(states_1, step_count=1)

        loaded_states = mgr.load(dump_path)

        logits_2, _ = model(input_ids, states=loaded_states)
        mx.eval(logits_2)

        logits_3, _ = model(input_ids, states=states_1)
        mx.eval(logits_3)

        np.testing.assert_allclose(
            np.array(logits_2), np.array(logits_3), atol=1e-4
        )

    def test_fork_and_diverge(self, tmp_path: Path) -> None:
        """Fork state, process different data, states diverge."""
        config = _mca_config(mca_dump_path=str(tmp_path / "dumps"))
        model = TitansMAC(config)

        input_ids = mx.random.randint(0, 256, (2, 32))
        _, states = model(input_ids)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])

        mgr = MemoryDumpManager(config)
        fork_path = mgr.fork(states, description="before diverge")

        input_ids_a = mx.zeros((2, 32), dtype=mx.int32)
        _, states_a = model(input_ids_a, states=states)
        mx.eval(*[w for s in states_a for w in s.weights + s.momentum])

        forked_states = mgr.load(fork_path)
        input_ids_b = mx.ones((2, 32), dtype=mx.int32) * 200
        _, states_b = model(input_ids_b, states=forked_states)
        mx.eval(*[w for s in states_b for w in s.weights + s.momentum])

        w_a = np.array(states_a[0].weights[0])
        w_b = np.array(states_b[0].weights[0])
        # Different inputs produce different weight updates — states must diverge
        assert not np.array_equal(w_a, w_b)
