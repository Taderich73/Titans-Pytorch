# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for MemoryDumpManager."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory, TNTMemoryState


def _make_states(config: TitansConfig, num_layers: int = 2) -> list[MemoryState]:
    memory = NeuralLongTermMemory(config)
    return [memory.init_state(batch_size=2) for _ in range(num_layers)]


def _make_tnt_states(config: TitansConfig, num_layers: int = 2) -> list[TNTMemoryState]:
    from titans_mlx.tnt_memory import HierarchicalMemory

    memory = HierarchicalMemory(config)
    return [memory.init_state(batch_size=2) for _ in range(num_layers)]


def _tnt_config(dump_dir) -> TitansConfig:
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
        use_tnt=True,
        global_chunk_size=32,
        local_chunk_sizes=[4, 8],
        local_shard_length=32,
        mca_dump_path=str(dump_dir),
    )


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    return tmp_path / "memory_dumps"


@pytest.fixture
def dump_config(dump_dir: Path) -> TitansConfig:
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
        mca_dump_path=str(dump_dir),
    )


class TestDumpRoundTrip:
    def test_dump_load_memory_state(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        dump_path = mgr.dump(states, step_count=100, description="test dump")
        assert dump_path.exists()
        assert (dump_path / "metadata.json").exists()
        loaded = mgr.load(dump_path)
        assert len(loaded) == len(states)
        for orig, rest in zip(states, loaded):
            for ow, rw in zip(orig.weights, rest.weights):
                np.testing.assert_allclose(np.array(ow), np.array(rw), atol=1e-6)
            for om, rm in zip(orig.momentum, rest.momentum):
                np.testing.assert_allclose(np.array(om), np.array(rm), atol=1e-6)

    def test_metadata_content(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        dump_path = mgr.dump(states, step_count=42, description="my description")
        meta = json.loads((dump_path / "metadata.json").read_text())
        assert meta["version"] == "1.0"
        assert meta["model_dim"] == 64
        assert meta["num_layers"] == 2
        assert meta["step_count"] == 42
        assert meta["description"] == "my description"
        assert "created_at" in meta

    def test_dump_strict_mismatch(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        dump_path = mgr.dump(states)
        different_config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=2,
            vocab_size=256,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            mca_dump_path=str(dump_dir),
        )
        mgr2 = MemoryDumpManager(different_config)
        with pytest.raises(ValueError, match="dimension mismatch"):
            mgr2.load(dump_path, strict=True)


class TestInspect:
    def test_inspect_returns_stats(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        dump_path = mgr.dump(states)
        report = mgr.inspect(dump_path)
        assert "per_layer_stats" in report
        assert len(report["per_layer_stats"]) == 2
        for layer_stats in report["per_layer_stats"].values():
            assert "weight_norm" in layer_stats
            assert "momentum_norm" in layer_stats


class TestDiff:
    def test_diff_identical(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        path_a = mgr.dump(states, description="a")
        path_b = mgr.dump(states, description="b")
        result = mgr.diff(path_a, path_b)
        for layer_diff in result["per_layer"].values():
            assert layer_diff["frobenius_distance"] < 1e-6

    def test_diff_after_update(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        memory = NeuralLongTermMemory(dump_config)
        states = [memory.init_state(2) for _ in range(2)]
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        path_a = mgr.dump(states, description="before")
        modified = []
        for s in states:
            new_weights = [w + mx.random.normal(w.shape) * 0.1 for w in s.weights]
            mx.eval(*new_weights)
            modified.append(MemoryState(weights=new_weights, momentum=s.momentum))
        path_b = mgr.dump(modified, description="after")
        result = mgr.diff(path_a, path_b)
        for layer_diff in result["per_layer"].values():
            assert layer_diff["frobenius_distance"] > 0.01


class TestMergeResetFork:
    def test_reset_full(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        reset_states = mgr.reset(states)
        for s in reset_states:
            for w in s.weights:
                mx.eval(w)
                assert mx.max(mx.abs(w)).item() < 1e-10
            for m in s.momentum:
                mx.eval(m)
                assert mx.max(mx.abs(m)).item() < 1e-10

    def test_reset_partial(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        reset_states = mgr.reset(states, layers=[0])
        for w in reset_states[0].weights:
            mx.eval(w)
            assert mx.max(mx.abs(w)).item() < 1e-10
        for ow, rw in zip(states[1].weights, reset_states[1].weights):
            np.testing.assert_allclose(np.array(ow), np.array(rw), atol=1e-6)

    def test_fork_no_mutation(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states = _make_states(dump_config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        original_weights = [np.array(w) for s in states for w in s.weights]
        fork_path = mgr.fork(states, description="snapshot")
        current_weights = [np.array(w) for s in states for w in s.weights]
        for orig, curr in zip(original_weights, current_weights):
            np.testing.assert_array_equal(orig, curr)
        assert fork_path.exists()
        loaded = mgr.load(fork_path)
        assert len(loaded) == len(states)

    def test_merge_weighted_mean(self, dump_config, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        mgr = MemoryDumpManager(dump_config)
        states_a = _make_states(dump_config)
        states_b = _make_states(dump_config)
        states_b = [
            MemoryState(weights=[w + 1.0 for w in s.weights], momentum=s.momentum)
            for s in states_b
        ]
        mx.eval(*[w for s in states_a + states_b for w in s.weights + s.momentum])
        path_a = mgr.dump(states_a, step_count=100)
        path_b = mgr.dump(states_b, step_count=100)
        merged = mgr.merge([path_a, path_b], strategy="weighted_mean")
        for sa, sb, sm in zip(states_a, states_b, merged):
            for wa, wb, wm in zip(sa.weights, sb.weights, sm.weights):
                expected = (np.array(wa) + np.array(wb)) / 2.0
                np.testing.assert_allclose(np.array(wm), expected, atol=1e-5)


class TestTNTState:
    """Tests for TNTMemoryState dump/load/reset/merge."""

    def test_dump_load_tnt_roundtrip(self, dump_dir) -> None:
        """Dump then load produces identical TNTMemoryState."""
        from titans_mlx.memory_dump import MemoryDumpManager

        config = _tnt_config(dump_dir)
        mgr = MemoryDumpManager(config)
        states = _make_tnt_states(config)
        # Eval all tensors
        for s in states:
            mx.eval(*s.global_state.weights, *s.global_state.momentum)
            for ls in s.local_states:
                mx.eval(*ls.weights, *ls.momentum)
            for init_list in s.local_inits:
                mx.eval(*init_list)
            mx.eval(*s.qk_projections)

        dump_path = mgr.dump(states, step_count=50, description="tnt test")
        assert dump_path.exists()

        meta = json.loads((dump_path / "metadata.json").read_text())
        assert meta["use_tnt"] is True

        loaded = mgr.load(dump_path)
        assert len(loaded) == len(states)

        for orig, rest in zip(states, loaded):
            # Global state
            for ow, rw in zip(orig.global_state.weights, rest.global_state.weights):
                np.testing.assert_allclose(np.array(ow), np.array(rw), atol=1e-6)
            for om, rm in zip(orig.global_state.momentum, rest.global_state.momentum):
                np.testing.assert_allclose(np.array(om), np.array(rm), atol=1e-6)

    def test_reset_tnt_full(self, dump_dir) -> None:
        """Full reset zeros TNT global and local states."""
        from titans_mlx.memory_dump import MemoryDumpManager

        config = _tnt_config(dump_dir)
        mgr = MemoryDumpManager(config)
        states = _make_tnt_states(config)
        for s in states:
            mx.eval(*s.global_state.weights, *s.global_state.momentum)
            for ls in s.local_states:
                mx.eval(*ls.weights, *ls.momentum)

        reset_states = mgr.reset(states)
        for s in reset_states:
            assert isinstance(s, TNTMemoryState)
            for w in s.global_state.weights:
                mx.eval(w)
                assert mx.max(mx.abs(w)).item() < 1e-10
            for ls in s.local_states:
                for w in ls.weights:
                    mx.eval(w)
                    assert mx.max(mx.abs(w)).item() < 1e-10

    def test_reset_tnt_partial(self, dump_dir) -> None:
        """Partial reset only resets specified TNT layers."""
        from titans_mlx.memory_dump import MemoryDumpManager

        config = _tnt_config(dump_dir)
        mgr = MemoryDumpManager(config)
        states = _make_tnt_states(config)
        for s in states:
            mx.eval(*s.global_state.weights, *s.global_state.momentum)
            for ls in s.local_states:
                mx.eval(*ls.weights, *ls.momentum)

        reset_states = mgr.reset(states, layers=[0])
        # Layer 0 global should be zeroed
        for w in reset_states[0].global_state.weights:
            mx.eval(w)
            assert mx.max(mx.abs(w)).item() < 1e-10
        # Layer 1 should be unchanged
        for ow, rw in zip(
            states[1].global_state.weights, reset_states[1].global_state.weights
        ):
            np.testing.assert_allclose(np.array(ow), np.array(rw), atol=1e-6)


class TestPruning:
    def test_prunes_old_dumps(self, dump_dir) -> None:
        from titans_mlx.memory_dump import MemoryDumpManager

        config = TitansConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=256,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            mca_dump_path=str(dump_dir),
            mca_dump_keep_last_n=3,
        )
        mgr = MemoryDumpManager(config)
        states = _make_states(config)
        mx.eval(*[w for s in states for w in s.weights + s.momentum])
        for i in range(5):
            mgr.dump(states, step_count=i)
        existing = sorted(dump_dir.iterdir())
        assert len(existing) == 3
