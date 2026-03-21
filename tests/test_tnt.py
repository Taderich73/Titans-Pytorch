# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for TNT: config, state, Q-K projection, hierarchical memory, and models."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from titans_mlx.config import TitansConfig
from titans_mlx.memory import (
    MemoryState,
    TNTMemoryState,
    load_tnt_memory_states,
    save_tnt_memory_states,
)
from titans_mlx.qk_projection import QKProjection, update_projection_state
from titans_mlx.tnt_memory import GlobalMemory, HierarchicalMemory, LocalMemory
from titans_mlx.tnt_models import (
    TNTMACBlock,
    TNTMAGBlock,
    TNTMALBlock,
    TitansTNT,
)


# ============================================================================
# Phase 2.1: TNT Config
# ============================================================================


class TestTNTConfig:
    """Tests for TNT-specific configuration fields."""

    def test_tnt_defaults(self) -> None:
        """TNT fields have correct defaults."""
        config = TitansConfig()
        assert config.use_tnt is False
        assert config.global_chunk_size == 2048
        assert config.local_chunk_sizes == [8, 16]
        assert config.local_shard_length == 2048
        assert config.use_qk_projection is True
        assert config.tnt_stage == 1
        assert config.finetune_local_chunk_sizes is None

    def test_num_local_memories_property(self) -> None:
        """num_local_memories derived from local_chunk_sizes length."""
        config = TitansConfig(local_chunk_sizes=[4, 8, 16])
        assert config.num_local_memories == 3

        config = TitansConfig(local_chunk_sizes=[8])
        assert config.num_local_memories == 1

    def test_active_local_chunk_sizes_stage1(self) -> None:
        """Stage 1 returns base local_chunk_sizes."""
        config = TitansConfig(
            tnt_stage=1,
            local_chunk_sizes=[8, 16],
            finetune_local_chunk_sizes=[4, 8],
        )
        assert config.active_local_chunk_sizes == [8, 16]

    def test_active_local_chunk_sizes_stage2(self) -> None:
        """Stage 2 returns finetune sizes when provided."""
        config = TitansConfig(
            tnt_stage=2,
            local_chunk_sizes=[8, 16],
            finetune_local_chunk_sizes=[4, 8],
        )
        assert config.active_local_chunk_sizes == [4, 8]

    def test_active_local_chunk_sizes_stage2_no_finetune(self) -> None:
        """Stage 2 falls back to base sizes when finetune sizes not set."""
        config = TitansConfig(tnt_stage=2, local_chunk_sizes=[8, 16])
        assert config.active_local_chunk_sizes == [8, 16]

    def test_tnt_roundtrip(self) -> None:
        """TNT fields survive to_dict/from_dict round-trip."""
        config = TitansConfig(
            use_tnt=True,
            global_chunk_size=1024,
            local_chunk_sizes=[4, 8, 16],
            local_shard_length=512,
            use_qk_projection=False,
            tnt_stage=2,
            finetune_local_chunk_sizes=[2, 4, 8],
        )
        restored = TitansConfig.from_dict(config.to_dict())

        assert restored.use_tnt is True
        assert restored.global_chunk_size == 1024
        assert restored.local_chunk_sizes == [4, 8, 16]
        assert restored.local_shard_length == 512
        assert restored.use_qk_projection is False
        assert restored.tnt_stage == 2
        assert restored.finetune_local_chunk_sizes == [2, 4, 8]

    def test_tnt_fields_in_to_dict(self) -> None:
        """All TNT fields present in to_dict output."""
        d = TitansConfig(use_tnt=True).to_dict()
        for key in [
            "use_tnt",
            "global_chunk_size",
            "local_chunk_sizes",
            "local_shard_length",
            "use_qk_projection",
            "tnt_stage",
            "finetune_local_chunk_sizes",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_local_chunk_sizes_independent_instances(self) -> None:
        """Mutable default (list) should not be shared between instances."""
        c1 = TitansConfig()
        c2 = TitansConfig()
        c1.local_chunk_sizes.append(32)
        assert 32 not in c2.local_chunk_sizes


# ============================================================================
# Phase 2.2: TNTMemoryState
# ============================================================================


class TestTNTMemoryState:
    """Tests for TNTMemoryState dataclass."""

    @pytest.fixture
    def dim(self) -> int:
        return 32

    @pytest.fixture
    def num_locals(self) -> int:
        return 2

    @pytest.fixture
    def tnt_state(self, dim: int, num_locals: int) -> TNTMemoryState:
        """Create a sample TNTMemoryState."""
        global_state = MemoryState(
            weights=[mx.random.normal((dim, dim))],
            momentum=[mx.zeros((dim, dim))],
        )
        local_states = [
            MemoryState(
                weights=[mx.random.normal((dim, dim))],
                momentum=[mx.zeros((dim, dim))],
            )
            for _ in range(num_locals)
        ]
        local_inits = [
            [mx.random.normal((dim, dim)) * 0.02] for _ in range(num_locals)
        ]
        qk_projections = [
            mx.random.normal((dim, dim)) for _ in range(num_locals)
        ]
        return TNTMemoryState(
            global_state=global_state,
            local_states=local_states,
            local_inits=local_inits,
            qk_projections=qk_projections,
            local_step_counters=[10, 20],
        )

    def test_creation(self, tnt_state: TNTMemoryState, num_locals: int) -> None:
        """TNTMemoryState can be created with correct structure."""
        assert len(tnt_state.local_states) == num_locals
        assert len(tnt_state.local_inits) == num_locals
        assert len(tnt_state.qk_projections) == num_locals
        assert tnt_state.local_step_counters == [10, 20]

    def test_detach(self, tnt_state: TNTMemoryState, dim: int) -> None:
        """detach() returns a new state with stopped gradients."""
        detached = tnt_state.detach()
        mx.eval(
            detached.global_state.weights[0],
            detached.local_states[0].weights[0],
            detached.local_inits[0][0],
            detached.qk_projections[0],
        )
        # Shapes preserved
        assert detached.global_state.weights[0].shape == (dim, dim)
        assert detached.local_states[0].weights[0].shape == (dim, dim)
        assert detached.local_step_counters == [10, 20]
        # Structure preserved
        assert len(detached.local_states) == len(tnt_state.local_states)

    def test_reset_local_zeros_qk_and_counter(
        self, tnt_state: TNTMemoryState, dim: int
    ) -> None:
        """reset_local() zeros the Q-K projection and counter for the target memory."""
        reset = tnt_state.reset_local(0)
        mx.eval(reset.qk_projections[0])

        # Counter reset
        assert reset.local_step_counters[0] == 0
        # Other counter untouched
        assert reset.local_step_counters[1] == 20
        # Q-K projection zeroed
        assert float(mx.sum(mx.abs(reset.qk_projections[0]))) == 0.0
        # Other Q-K untouched
        assert float(mx.sum(mx.abs(reset.qk_projections[1]))) > 0.0

    def test_reset_local_restores_init_weights(
        self, tnt_state: TNTMemoryState
    ) -> None:
        """reset_local() restores weights from local_inits."""
        reset = tnt_state.reset_local(0)
        mx.eval(reset.local_states[0].weights[0], tnt_state.local_inits[0][0])

        np.testing.assert_allclose(
            np.array(reset.local_states[0].weights[0]),
            np.array(tnt_state.local_inits[0][0]),
            atol=1e-6,
        )

    def test_reset_local_zeros_momentum(self, tnt_state: TNTMemoryState) -> None:
        """reset_local() zeros momentum for the reset memory."""
        reset = tnt_state.reset_local(1)
        mx.eval(reset.local_states[1].momentum[0])
        assert float(mx.sum(mx.abs(reset.local_states[1].momentum[0]))) == 0.0

    def test_reset_local_does_not_mutate_original(
        self, tnt_state: TNTMemoryState
    ) -> None:
        """reset_local() returns a new state, original is unchanged."""
        original_counter = tnt_state.local_step_counters[0]
        _ = tnt_state.reset_local(0)
        assert tnt_state.local_step_counters[0] == original_counter

    def test_global_state_unchanged_after_reset(
        self, tnt_state: TNTMemoryState
    ) -> None:
        """reset_local() does not touch the global state."""
        reset = tnt_state.reset_local(0)
        # Same object reference — global state should be shared
        assert reset.global_state is tnt_state.global_state


# ============================================================================
# Phase 3.1: QKProjection
# ============================================================================


class TestQKProjection:
    """Tests for QKProjection module."""

    @pytest.fixture
    def dim(self) -> int:
        return 16

    @pytest.fixture
    def proj(self, dim: int) -> QKProjection:
        return QKProjection(dim)

    @staticmethod
    def _normalize(x: mx.array) -> mx.array:
        return x / (mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + 1e-8)

    def test_projection_matrix_shape(self, proj: QKProjection, dim: int) -> None:
        """compute_projection_matrix returns correct shape."""
        B, C = 2, 8
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        result = proj.compute_projection_matrix(keys, carry)
        mx.eval(result)
        assert result.shape == (B, C, dim, dim)

    def test_projection_matrix_symmetry(self, proj: QKProjection, dim: int) -> None:
        """Projection matrices are symmetric (sum of k*k^T)."""
        B, C = 2, 8
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        result = proj.compute_projection_matrix(keys, carry)
        mx.eval(result)

        # Check last position for symmetry
        mat = np.array(result[0, -1])
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_projection_matrix_positive_semidefinite(
        self, proj: QKProjection, dim: int
    ) -> None:
        """Projection matrices are PSD (sum of outer products)."""
        B, C = 1, 4
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        result = proj.compute_projection_matrix(keys, carry)
        mx.eval(result)

        mat = np.array(result[0, -1])
        eigenvalues = np.linalg.eigvalsh(mat)
        assert np.all(eigenvalues >= -1e-5)

    def test_projection_matrix_cumulative(
        self, proj: QKProjection, dim: int
    ) -> None:
        """Later positions accumulate more outer products than earlier ones."""
        B, C = 1, 8
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        result = proj.compute_projection_matrix(keys, carry)
        mx.eval(result)

        # Frobenius norm should be non-decreasing
        norms = [float(mx.sum(result[0, i] ** 2)) for i in range(C)]
        for i in range(1, C):
            assert norms[i] >= norms[i - 1] - 1e-5

    def test_carry_over_adds_to_projection(
        self, proj: QKProjection, dim: int
    ) -> None:
        """Non-zero carry-over shifts all projection matrices."""
        B, C = 2, 4
        keys = self._normalize(mx.random.normal((B, C, dim)))

        zero_carry = mx.zeros((dim, dim))
        nonzero_carry = mx.eye(dim) * 0.5

        result_zero = proj.compute_projection_matrix(keys, zero_carry)
        result_nonzero = proj.compute_projection_matrix(keys, nonzero_carry)
        mx.eval(result_zero, result_nonzero)

        diff = np.array(result_nonzero[0, 0] - result_zero[0, 0])
        np.testing.assert_allclose(diff, 0.5 * np.eye(dim), atol=1e-5)

    def test_project_queries_shape(self, proj: QKProjection, dim: int) -> None:
        """project_queries returns correct shape."""
        B, C = 2, 8
        queries = mx.random.normal((B, C, dim))
        proj_matrices = mx.random.normal((B, C, dim, dim))

        result = proj.project_queries(queries, proj_matrices)
        mx.eval(result)
        assert result.shape == (B, C, dim)

    def test_project_queries_identity(self, proj: QKProjection, dim: int) -> None:
        """Identity projection matrices leave queries unchanged."""
        B, C = 2, 4
        queries = mx.random.normal((B, C, dim))
        identity = mx.broadcast_to(
            mx.reshape(mx.eye(dim), (1, 1, dim, dim)), (B, C, dim, dim)
        )

        result = proj.project_queries(queries, identity)
        mx.eval(result)

        np.testing.assert_allclose(np.array(result), np.array(queries), atol=1e-5)

    def test_call_output_shapes(self, proj: QKProjection, dim: int) -> None:
        """__call__ returns (projected_queries, new_carry) with correct shapes."""
        B, C = 2, 8
        queries = mx.random.normal((B, C, dim))
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        projected, new_carry = proj(queries, keys, carry)
        mx.eval(projected, new_carry)

        assert projected.shape == (B, C, dim)
        assert new_carry.shape == (dim, dim)

    def test_call_carry_accumulates(self, proj: QKProjection, dim: int) -> None:
        """Successive calls accumulate carry-over state."""
        B, C = 2, 4
        keys = self._normalize(mx.random.normal((B, C, dim)))
        queries = mx.random.normal((B, C, dim))
        carry = mx.zeros((dim, dim))

        _, carry1 = proj(queries, keys, carry)
        _, carry2 = proj(queries, keys, carry1)
        mx.eval(carry1, carry2)

        norm1 = float(mx.sum(carry1 ** 2))
        norm2 = float(mx.sum(carry2 ** 2))
        assert norm2 > norm1

    def test_no_nan(self, proj: QKProjection, dim: int) -> None:
        """Output contains no NaN values."""
        B, C = 2, 8
        keys = self._normalize(mx.random.normal((B, C, dim)))
        queries = mx.random.normal((B, C, dim))
        carry = mx.zeros((dim, dim))

        projected, new_carry = proj(queries, keys, carry)
        mx.eval(projected, new_carry)

        assert not np.any(np.isnan(np.array(projected)))
        assert not np.any(np.isnan(np.array(new_carry)))


# ============================================================================
# Phase 3.2: update_projection_state with reset
# ============================================================================


class TestUpdateProjectionState:
    """Tests for update_projection_state helper."""

    @pytest.fixture
    def dim(self) -> int:
        return 16

    @staticmethod
    def _normalize(x: mx.array) -> mx.array:
        return x / (mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + 1e-8)

    def test_output_shapes(self, dim: int) -> None:
        """Returns (carry, projections) with correct shapes."""
        B, C = 2, 8
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        new_carry, projections = update_projection_state(carry, keys, reset=False)
        mx.eval(new_carry, projections)

        assert new_carry.shape == (dim, dim)
        assert projections.shape == (B, C, dim, dim)

    def test_reset_zeros_carry(self, dim: int) -> None:
        """With reset=True, carry-over is zeroed before processing."""
        B, C = 1, 4
        keys = self._normalize(mx.random.normal((B, C, dim)))
        nonzero_carry = mx.eye(dim) * 10.0

        carry_reset, proj_reset = update_projection_state(
            nonzero_carry, keys, reset=True
        )
        carry_no_reset, proj_no_reset = update_projection_state(
            mx.zeros((dim, dim)), keys, reset=False
        )
        mx.eval(carry_reset, carry_no_reset, proj_reset, proj_no_reset)

        # Reset should give same result as starting from zero
        np.testing.assert_allclose(
            np.array(carry_reset), np.array(carry_no_reset), atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(proj_reset), np.array(proj_no_reset), atol=1e-5
        )

    def test_no_reset_preserves_carry(self, dim: int) -> None:
        """Without reset, existing carry contributes to projections."""
        B, C = 1, 4
        keys = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.eye(dim) * 5.0

        new_carry, _ = update_projection_state(carry, keys, reset=False)
        mx.eval(new_carry)

        # Result should include carry contribution (larger Frobenius norm)
        zero_carry, _ = update_projection_state(
            mx.zeros((dim, dim)), keys, reset=False
        )
        mx.eval(zero_carry)

        norm_with = float(mx.sum(new_carry ** 2))
        norm_without = float(mx.sum(zero_carry ** 2))
        assert norm_with > norm_without

    def test_sequential_chunks(self, dim: int) -> None:
        """Carry-over correctly chains across multiple chunks."""
        B, C = 1, 4
        keys1 = self._normalize(mx.random.normal((B, C, dim)))
        keys2 = self._normalize(mx.random.normal((B, C, dim)))
        carry = mx.zeros((dim, dim))

        carry1, _ = update_projection_state(carry, keys1, reset=False)
        carry2, _ = update_projection_state(carry1, keys2, reset=False)
        mx.eval(carry, carry1, carry2)

        # Each step should increase the norm (accumulating outer products)
        n0 = float(mx.sum(carry ** 2))
        n1 = float(mx.sum(carry1 ** 2))
        n2 = float(mx.sum(carry2 ** 2))
        assert n1 > n0
        assert n2 > n1


# ============================================================================
# Phase 4.1: GlobalMemory
# ============================================================================


class TestGlobalMemory:
    """Tests for GlobalMemory module."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        return TitansConfig(dim=32, num_memory_layers=1, use_conv=False)

    def test_forward_shape(self, config: TitansConfig) -> None:
        """Forward pass produces correct output shape."""
        gm = GlobalMemory(config)
        x = mx.random.normal((2, 16, 32))
        out, state = gm(x)
        mx.eval(out)
        assert out.shape == (2, 16, 32)

    def test_state_returned(self, config: TitansConfig) -> None:
        """Forward pass returns a valid MemoryState."""
        gm = GlobalMemory(config)
        x = mx.random.normal((2, 16, 32))
        _, state = gm(x)
        mx.eval(state.weights[0])
        assert isinstance(state, MemoryState)
        assert len(state.weights) == 1
        assert state.weights[0].shape == (32, 32)

    def test_retrieve_shape(self, config: TitansConfig) -> None:
        """Retrieval produces correct output shape."""
        gm = GlobalMemory(config)
        x = mx.random.normal((2, 16, 32))
        _, state = gm(x)
        queries = mx.random.normal((2, 4, 32))
        retrieved = gm.retrieve(queries, state)
        mx.eval(retrieved)
        assert retrieved.shape == (2, 4, 32)

    def test_state_evolves(self, config: TitansConfig) -> None:
        """Memory state changes after processing input."""
        gm = GlobalMemory(config)
        state0 = gm.init_state(1)
        x = mx.random.normal((1, 16, 32))
        _, state1 = gm(x, state=state0)
        mx.eval(state0.weights[0], state1.weights[0])

        diff = float(mx.sum(mx.abs(state1.weights[0] - state0.weights[0])))
        assert diff > 0.0

    def test_no_nan(self, config: TitansConfig) -> None:
        """Output contains no NaN values."""
        gm = GlobalMemory(config)
        x = mx.random.normal((2, 8, 32))
        out, state = gm(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))


# ============================================================================
# Phase 4.2: LocalMemory
# ============================================================================


class TestLocalMemory:
    """Tests for LocalMemory module with periodic reset."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        return TitansConfig(
            dim=32, num_memory_layers=1, use_conv=False, use_qk_projection=True
        )

    @pytest.fixture
    def local_mem(self, config: TitansConfig) -> LocalMemory:
        return LocalMemory(config, chunk_size=8, shard_length=64)

    def test_forward_shape(self, local_mem: LocalMemory) -> None:
        """Forward pass produces correct output shape."""
        x = mx.random.normal((2, 8, 32))
        out, state = local_mem(x)
        mx.eval(out)
        assert out.shape == (2, 8, 32)

    def test_init_state_uses_w_init(self, local_mem: LocalMemory) -> None:
        """init_state() uses the learnable W_init, not memory's default."""
        state = local_mem.init_state(1)
        mx.eval(state.weights[0], local_mem.w_init[0])

        np.testing.assert_allclose(
            np.array(state.weights[0]),
            np.array(local_mem.w_init[0]),
            atol=1e-6,
        )

    def test_init_state_zeros_momentum(self, local_mem: LocalMemory) -> None:
        """init_state() initializes momentum to zero."""
        state = local_mem.init_state(1)
        mx.eval(state.momentum[0])
        assert float(mx.sum(mx.abs(state.momentum[0]))) == 0.0

    def test_maybe_reset_at_boundary(self, local_mem: LocalMemory) -> None:
        """maybe_reset() resets state at shard boundaries."""
        state = local_mem.init_state(1)
        x = mx.random.normal((1, 8, 32))
        _, modified_state = local_mem(x, state=state)

        # At shard boundary (step 64), state should reset
        reset_state, counter = local_mem.maybe_reset(modified_state, 64)
        mx.eval(reset_state.weights[0], local_mem.w_init[0])

        assert counter == 0
        np.testing.assert_allclose(
            np.array(reset_state.weights[0]),
            np.array(local_mem.w_init[0]),
            atol=1e-6,
        )

    def test_maybe_reset_not_at_boundary(self, local_mem: LocalMemory) -> None:
        """maybe_reset() preserves state when not at shard boundary."""
        state = local_mem.init_state(1)
        x = mx.random.normal((1, 8, 32))
        _, modified_state = local_mem(x, state=state)

        preserved_state, counter = local_mem.maybe_reset(modified_state, 32)
        assert counter == 32
        # State should be the same object (not reset)
        assert preserved_state is modified_state

    def test_maybe_reset_at_zero_does_not_reset(
        self, local_mem: LocalMemory
    ) -> None:
        """maybe_reset() at step 0 does not reset (only at multiples > 0)."""
        state = local_mem.init_state(1)
        x = mx.random.normal((1, 8, 32))
        _, modified_state = local_mem(x, state=state)

        preserved, counter = local_mem.maybe_reset(modified_state, 0)
        assert counter == 0
        assert preserved is modified_state

    def test_has_qk_projection(self, local_mem: LocalMemory) -> None:
        """QK projection is initialized when use_qk_projection=True."""
        assert local_mem.qk_proj is not None
        assert isinstance(local_mem.qk_proj, QKProjection)

    def test_no_qk_projection(self, config: TitansConfig) -> None:
        """QK projection is None when use_qk_projection=False."""
        config_no_qk = TitansConfig(
            dim=32, num_memory_layers=1, use_conv=False, use_qk_projection=False
        )
        lm = LocalMemory(config_no_qk, chunk_size=8, shard_length=64)
        assert lm.qk_proj is None

    def test_retrieve_shape(self, local_mem: LocalMemory) -> None:
        """Retrieve returns correct shape."""
        x = mx.random.normal((2, 8, 32))
        _, state = local_mem(x)
        queries = mx.random.normal((2, 4, 32))
        out = local_mem.retrieve(queries, state)
        mx.eval(out)
        assert out.shape == (2, 4, 32)

    def test_no_nan(self, local_mem: LocalMemory) -> None:
        """Output contains no NaN values."""
        x = mx.random.normal((2, 8, 32))
        out, _ = local_mem(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))


# ============================================================================
# Phase 4.3: HierarchicalMemory
# ============================================================================


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory combining global + N local memories."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        return TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            use_tnt=True,
            local_chunk_sizes=[4, 8],
            local_shard_length=64,
            use_qk_projection=True,
        )

    @pytest.fixture
    def hier_mem(self, config: TitansConfig) -> HierarchicalMemory:
        return HierarchicalMemory(config)

    def test_forward_shape(self, hier_mem: HierarchicalMemory) -> None:
        """Forward pass produces correct output shape."""
        x = mx.random.normal((2, 16, 32))
        out, state = hier_mem(x)
        mx.eval(out)
        assert out.shape == (2, 16, 32)

    def test_returns_tnt_state(self, hier_mem: HierarchicalMemory) -> None:
        """Forward pass returns a TNTMemoryState."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)
        assert isinstance(state, TNTMemoryState)

    def test_state_structure(self, hier_mem: HierarchicalMemory) -> None:
        """TNTMemoryState has correct number of local memories."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)
        assert len(state.local_states) == 2
        assert len(state.qk_projections) == 2
        assert len(state.local_step_counters) == 2

    def test_step_counters_advance(self, hier_mem: HierarchicalMemory) -> None:
        """Step counters increase by sequence length after each call."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)
        assert state.local_step_counters == [8, 8]

        _, state2 = hier_mem(x, state=state)
        assert state2.local_step_counters == [16, 16]

    def test_retrieve_shape(self, hier_mem: HierarchicalMemory) -> None:
        """Hierarchical retrieval produces correct shape."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)
        queries = mx.random.normal((2, 4, 32))
        out = hier_mem.retrieve(queries, state)
        mx.eval(out)
        assert out.shape == (2, 4, 32)

    def test_global_state_evolves(self, hier_mem: HierarchicalMemory) -> None:
        """Global memory state changes after processing."""
        state0 = hier_mem.init_state(1)
        x = mx.random.normal((1, 8, 32))
        _, state1 = hier_mem(x, state=state0)
        mx.eval(state0.global_state.weights[0], state1.global_state.weights[0])

        diff = float(
            mx.sum(
                mx.abs(
                    state1.global_state.weights[0] - state0.global_state.weights[0]
                )
            )
        )
        assert diff > 0.0

    def test_local_states_evolve(self, hier_mem: HierarchicalMemory) -> None:
        """Local memory states change after processing."""
        state0 = hier_mem.init_state(1)
        x = mx.random.normal((1, 8, 32))
        _, state1 = hier_mem(x, state=state0)

        for i in range(2):
            mx.eval(
                state0.local_states[i].weights[0],
                state1.local_states[i].weights[0],
            )
            diff = float(
                mx.sum(
                    mx.abs(
                        state1.local_states[i].weights[0]
                        - state0.local_states[i].weights[0]
                    )
                )
            )
            assert diff > 0.0, f"Local memory {i} did not evolve"

    def test_qk_projections_nonzero(self, hier_mem: HierarchicalMemory) -> None:
        """Q-K projection matrices become non-zero after processing."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)

        for i in range(2):
            mx.eval(state.qk_projections[i])
            norm = float(mx.sum(mx.abs(state.qk_projections[i])))
            assert norm > 0.0, f"Q-K projection {i} is zero"

    def test_no_qk_projection(self) -> None:
        """Works correctly with Q-K projection disabled."""
        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            use_tnt=True,
            local_chunk_sizes=[4],
            use_qk_projection=False,
        )
        hm = HierarchicalMemory(config)
        x = mx.random.normal((1, 8, 32))
        out, state = hm(x)
        mx.eval(out)
        assert out.shape == (1, 8, 32)

    def test_multi_resolution_local(self) -> None:
        """Three local memories at different resolutions."""
        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            use_tnt=True,
            local_chunk_sizes=[4, 8, 16],
        )
        hm = HierarchicalMemory(config)
        assert len(hm.local_memories) == 3

        x = mx.random.normal((1, 16, 32))
        out, state = hm(x)
        mx.eval(out)
        assert out.shape == (1, 16, 32)
        assert len(state.local_states) == 3

    def test_no_nan(self, hier_mem: HierarchicalMemory) -> None:
        """Output contains no NaN values."""
        x = mx.random.normal((2, 16, 32))
        out, _ = hier_mem(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))

    def test_retrieval_no_nan(self, hier_mem: HierarchicalMemory) -> None:
        """Retrieval output contains no NaN values."""
        x = mx.random.normal((2, 8, 32))
        _, state = hier_mem(x)
        queries = mx.random.normal((2, 4, 32))
        out = hier_mem.retrieve(queries, state)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out)))

    def test_init_state_zeros_qk(self, hier_mem: HierarchicalMemory) -> None:
        """init_state() starts with zeroed Q-K projections."""
        state = hier_mem.init_state(1)
        for qk in state.qk_projections:
            mx.eval(qk)
            assert float(mx.sum(mx.abs(qk))) == 0.0

    def test_init_state_zero_counters(self, hier_mem: HierarchicalMemory) -> None:
        """init_state() starts with zero step counters."""
        state = hier_mem.init_state(1)
        assert state.local_step_counters == [0, 0]


# ============================================================================
# Phase 4.4: TNT Memory State Serialization
# ============================================================================


class TestTNTMemorySerialization:
    """Tests for save/load of TNTMemoryState."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        return TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            use_tnt=True,
            local_chunk_sizes=[4, 8],
        )

    @pytest.fixture
    def tnt_state(self, config: TitansConfig) -> TNTMemoryState:
        hm = HierarchicalMemory(config)
        x = mx.random.normal((1, 8, 32))
        _, state = hm(x)
        # Force evaluation
        mx.eval(
            state.global_state.weights[0],
            state.local_states[0].weights[0],
            state.local_states[1].weights[0],
            state.qk_projections[0],
            state.qk_projections[1],
        )
        return state

    def test_roundtrip_global_weights(self, tnt_state: TNTMemoryState) -> None:
        """Global weights survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        s = loaded[0]
        mx.eval(s.global_state.weights[0])
        np.testing.assert_allclose(
            np.array(s.global_state.weights[0]),
            np.array(tnt_state.global_state.weights[0]),
            atol=1e-6,
        )

    def test_roundtrip_local_weights(self, tnt_state: TNTMemoryState) -> None:
        """Local weights survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        s = loaded[0]
        for i in range(2):
            mx.eval(s.local_states[i].weights[0])
            np.testing.assert_allclose(
                np.array(s.local_states[i].weights[0]),
                np.array(tnt_state.local_states[i].weights[0]),
                atol=1e-6,
            )

    def test_roundtrip_qk_projections(self, tnt_state: TNTMemoryState) -> None:
        """Q-K projection matrices survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        s = loaded[0]
        for i in range(2):
            mx.eval(s.qk_projections[i])
            np.testing.assert_allclose(
                np.array(s.qk_projections[i]),
                np.array(tnt_state.qk_projections[i]),
                atol=1e-6,
            )

    def test_roundtrip_step_counters(self, tnt_state: TNTMemoryState) -> None:
        """Step counters survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        assert loaded[0].local_step_counters == tnt_state.local_step_counters

    def test_roundtrip_local_inits(self, tnt_state: TNTMemoryState) -> None:
        """Local init weights survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        s = loaded[0]
        for i in range(2):
            mx.eval(s.local_inits[i][0])
            np.testing.assert_allclose(
                np.array(s.local_inits[i][0]),
                np.array(tnt_state.local_inits[i][0]),
                atol=1e-6,
            )

    def test_roundtrip_momentum(self, tnt_state: TNTMemoryState) -> None:
        """Momentum arrays survive serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([tnt_state], path)
            loaded = load_tnt_memory_states(path)

        s = loaded[0]
        mx.eval(s.global_state.momentum[0])
        np.testing.assert_allclose(
            np.array(s.global_state.momentum[0]),
            np.array(tnt_state.global_state.momentum[0]),
            atol=1e-6,
        )

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_tnt_memory_states(Path("/tmp/nonexistent_tnt_state"))

    def test_multiple_layers(self, config: TitansConfig) -> None:
        """Serialization works with multiple model layers."""
        hm1 = HierarchicalMemory(config)
        hm2 = HierarchicalMemory(config)
        x = mx.random.normal((1, 8, 32))
        _, s1 = hm1(x)
        _, s2 = hm2(x)
        mx.eval(s1.global_state.weights[0], s2.global_state.weights[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            save_tnt_memory_states([s1, s2], path)
            loaded = load_tnt_memory_states(path)

        assert len(loaded) == 2
        for i, (orig, load) in enumerate(zip([s1, s2], loaded)):
            mx.eval(load.global_state.weights[0])
            np.testing.assert_allclose(
                np.array(load.global_state.weights[0]),
                np.array(orig.global_state.weights[0]),
                atol=1e-6,
            )


# ============================================================================
# Phase 5.1: TNT Blocks
# ============================================================================


@pytest.fixture
def tnt_config() -> TitansConfig:
    """Small TNT config for block and model tests."""
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        ffn_mult=2.0,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        num_persistent_tokens=2,
        chunk_size=16,
        window_size=8,
        dropout=0.0,
        use_conv=False,
        use_rope=False,
        max_seq_len=64,
        vocab_size=50,
        use_tnt=True,
        local_chunk_sizes=[4, 8],
        local_shard_length=64,
    )


class TestTNTMACBlock:
    """Tests for TNTMACBlock."""

    def test_forward_without_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass produces correct shape without initial state."""
        block = TNTMACBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, state = block(x)
        mx.eval(output)
        assert output.shape == (2, 16, 32)
        assert isinstance(state, TNTMemoryState)

    def test_forward_with_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass works with existing state."""
        block = TNTMACBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        _, state1 = block(x)
        output, state2 = block(x, state=state1)
        mx.eval(output)
        assert output.shape == (2, 16, 32)

    def test_no_nan(self, tnt_config: TitansConfig) -> None:
        """Output contains no NaN values."""
        block = TNTMACBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, _ = block(x)
        mx.eval(output)
        assert not np.any(np.isnan(np.array(output)))


class TestTNTMAGBlock:
    """Tests for TNTMAGBlock."""

    def test_forward_without_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass produces correct shape."""
        block = TNTMAGBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, state = block(x)
        mx.eval(output)
        assert output.shape == (2, 16, 32)
        assert isinstance(state, TNTMemoryState)

    def test_forward_with_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass works with existing state."""
        block = TNTMAGBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        _, state1 = block(x)
        output, _ = block(x, state=state1)
        mx.eval(output)
        assert output.shape == (2, 16, 32)

    def test_no_nan(self, tnt_config: TitansConfig) -> None:
        """Output contains no NaN values."""
        block = TNTMAGBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, _ = block(x)
        mx.eval(output)
        assert not np.any(np.isnan(np.array(output)))


class TestTNTMALBlock:
    """Tests for TNTMALBlock."""

    def test_forward_without_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass produces correct shape."""
        block = TNTMALBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, state = block(x)
        mx.eval(output)
        assert output.shape == (2, 16, 32)
        assert isinstance(state, TNTMemoryState)

    def test_forward_with_state(self, tnt_config: TitansConfig) -> None:
        """Forward pass works with existing state."""
        block = TNTMALBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        _, state1 = block(x)
        output, _ = block(x, state=state1)
        mx.eval(output)
        assert output.shape == (2, 16, 32)

    def test_no_nan(self, tnt_config: TitansConfig) -> None:
        """Output contains no NaN values."""
        block = TNTMALBlock(tnt_config)
        x = mx.random.normal((2, 16, 32))
        output, _ = block(x)
        mx.eval(output)
        assert not np.any(np.isnan(np.array(output)))


# ============================================================================
# Phase 5.2: TitansTNT Model
# ============================================================================


class TestTitansTNT:
    """Tests for TitansTNT model."""

    @pytest.mark.parametrize("variant", ["mac", "mag", "mal"])
    def test_forward_shape(self, tnt_config: TitansConfig, variant: str) -> None:
        """Forward pass produces correct logits shape for all variants."""
        model = TitansTNT(tnt_config, variant=variant)
        input_ids = mx.random.randint(0, 50, (2, 16))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 16, 50)
        assert len(states) == 1

    @pytest.mark.parametrize("variant", ["mac", "mag", "mal"])
    def test_state_threading(self, tnt_config: TitansConfig, variant: str) -> None:
        """State threads correctly across consecutive calls."""
        model = TitansTNT(tnt_config, variant=variant)
        input_ids = mx.random.randint(0, 50, (2, 16))

        _, states1 = model(input_ids)
        logits2, states2 = model(input_ids, states=states1)
        mx.eval(logits2)

        assert logits2.shape == (2, 16, 50)
        # Step counters should increase — exact value depends on variant
        # (MAG/MAL prepend persistent tokens to memory input, increasing
        # the effective seq_len per call)
        for c in states2[0].local_step_counters:
            assert c > states1[0].local_step_counters[0]

    def test_multi_chunk(self, tnt_config: TitansConfig) -> None:
        """Sequences longer than chunk_size are processed in chunks."""
        model = TitansTNT(tnt_config, variant="mac")
        # seq_len=48 > chunk_size=16 → 3 chunks
        input_ids = mx.random.randint(0, 50, (2, 48))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 48, 50)
        assert states[0].local_step_counters == [48, 48]

    def test_single_chunk_fast_path(self, tnt_config: TitansConfig) -> None:
        """Sequences <= chunk_size use the fast path."""
        model = TitansTNT(tnt_config, variant="mac")
        input_ids = mx.random.randint(0, 50, (2, 8))
        logits, states = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (2, 8, 50)

    def test_weight_tying(self, tnt_config: TitansConfig) -> None:
        """Embedding and output head weights are tied."""
        model = TitansTNT(tnt_config)
        head_w = np.array(model.head.weight)
        embed_w = np.array(model.embed.weight)
        np.testing.assert_array_equal(head_w, embed_w)

    def test_invalid_variant_raises(self, tnt_config: TitansConfig) -> None:
        """Invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown TNT variant"):
            TitansTNT(tnt_config, variant="bad")

    @pytest.mark.parametrize("variant", ["mac", "mag", "mal"])
    def test_no_nan(self, tnt_config: TitansConfig, variant: str) -> None:
        """All variants produce valid (no NaN/Inf) logits."""
        model = TitansTNT(tnt_config, variant=variant)
        input_ids = mx.random.randint(0, 50, (2, 16))
        logits, _ = model(input_ids)
        mx.eval(logits)
        logits_np = np.array(logits)
        assert not np.any(np.isnan(logits_np)), f"{variant} produced NaN"
        assert not np.any(np.isinf(logits_np)), f"{variant} produced Inf"

    def test_tnt_differs_from_standard_mac(self, tnt_config: TitansConfig) -> None:
        """TitansTNT produces different output than TitansMAC (different architecture)."""
        from titans_mlx.models import TitansMAC

        # Use same config for both
        mac_config = TitansConfig(
            dim=32, num_heads=2, num_layers=1, ffn_mult=2.0,
            num_memory_layers=1, memory_hidden_mult=2.0, num_persistent_tokens=2,
            chunk_size=16, window_size=8, dropout=0.0, use_conv=False,
            use_rope=False, max_seq_len=64, vocab_size=50,
        )

        mx.random.seed(42)
        mac_model = TitansMAC(mac_config)
        mx.random.seed(42)
        tnt_model = TitansTNT(tnt_config, variant="mac")

        input_ids = mx.random.randint(0, 50, (1, 16))

        mac_logits, _ = mac_model(input_ids)
        tnt_logits, _ = tnt_model(input_ids)
        mx.eval(mac_logits, tnt_logits)

        # Both should be valid but different (different memory architectures)
        assert not np.any(np.isnan(np.array(mac_logits)))
        assert not np.any(np.isnan(np.array(tnt_logits)))
        # They should differ (different number of memory modules, different init)
        diff = float(mx.sum(mx.abs(mac_logits - tnt_logits)))
        assert diff > 0.0

    def test_gradient_flow(self, tnt_config: TitansConfig) -> None:
        """Gradients flow through TitansTNT via value_and_grad."""
        model = TitansTNT(tnt_config, variant="mac")
        input_ids = mx.random.randint(0, 50, (2, 16))
        targets = mx.random.randint(0, 50, (2, 16))

        def loss_fn(model: nn.Module) -> mx.array:
            logits, _ = model(input_ids)
            logits_flat = logits.reshape(-1, 50)
            targets_flat = targets.reshape(-1)
            return nn.losses.cross_entropy(logits_flat, targets_flat).mean()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss)

        assert float(loss) > 0
        flat_grads = tree_flatten(grads)
        has_nonzero = any(float(mx.abs(g).sum()) > 0 for _, g in flat_grads)
        assert has_nonzero, "Expected at least some non-zero gradients"


# ============================================================================
# Phase 5.3: Two-stage training configuration
# ============================================================================


class TestTwoStageConfig:
    """Tests for TitansConfig.tnt_stage1 / tnt_stage2 helpers."""

    def test_stage1_defaults(self) -> None:
        """tnt_stage1() sets correct defaults."""
        config = TitansConfig.tnt_stage1()
        assert config.use_tnt is True
        assert config.global_chunk_size == 2048
        assert config.local_chunk_sizes == [8, 16]
        assert config.local_shard_length == 2048
        assert config.tnt_stage == 1

    def test_stage1_override(self) -> None:
        """tnt_stage1() accepts overrides."""
        config = TitansConfig.tnt_stage1(
            dim=128, local_chunk_sizes=[4, 8, 16]
        )
        assert config.dim == 128
        assert config.local_chunk_sizes == [4, 8, 16]
        assert config.use_tnt is True

    def test_stage2_from_stage1(self) -> None:
        """tnt_stage2() derives correct config from stage 1."""
        s1 = TitansConfig.tnt_stage1(local_chunk_sizes=[8, 16, 32])
        s2 = TitansConfig.tnt_stage2(s1)

        assert s2.tnt_stage == 2
        assert s2.finetune_local_chunk_sizes == [4, 8, 16]
        assert s2.active_local_chunk_sizes == [4, 8, 16]
        # Base local_chunk_sizes unchanged
        assert s2.local_chunk_sizes == [8, 16, 32]

    def test_stage2_minimum_chunk_size(self) -> None:
        """tnt_stage2() clamps chunk sizes to minimum of 1."""
        s1 = TitansConfig.tnt_stage1(local_chunk_sizes=[1, 2])
        s2 = TitansConfig.tnt_stage2(s1)
        # 1 // 2 = 0, clamped to 1
        assert s2.finetune_local_chunk_sizes == [1, 1]

    def test_stage2_preserves_other_fields(self) -> None:
        """tnt_stage2() preserves non-TNT fields from stage 1."""
        s1 = TitansConfig.tnt_stage1(dim=256, num_heads=8, num_layers=6)
        s2 = TitansConfig.tnt_stage2(s1)
        assert s2.dim == 256
        assert s2.num_heads == 8
        assert s2.num_layers == 6

    def test_stage2_model_instantiation(self) -> None:
        """TitansTNT can be instantiated with stage 2 config."""
        s1 = TitansConfig.tnt_stage1(
            dim=32, num_heads=2, num_layers=1, vocab_size=50,
            chunk_size=16, window_size=8, use_conv=False, use_rope=False,
            num_memory_layers=1,
        )
        s2 = TitansConfig.tnt_stage2(s1)
        model = TitansTNT(s2, variant="mac")
        input_ids = mx.random.randint(0, 50, (1, 16))
        logits, _ = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (1, 16, 50)


# =============================================================================
# AttnRes Config Tests
# =============================================================================


class TestAttnResConfig(unittest.TestCase):
    """Test AttnRes configuration fields."""

    def test_attnres_defaults(self):
        """AttnRes is disabled by default."""
        config = TitansConfig()
        self.assertFalse(config.use_attn_res)
        self.assertEqual(config.num_attnres_blocks, 8)
        self.assertEqual(config.attnres_warmup_steps, 0)
        self.assertTrue(config.attnres_modulate_global_memory)
        self.assertFalse(config.attnres_modulate_local_memory)

    def test_attnres_base_block_size_even(self):
        """Block size derived correctly when evenly divisible."""
        config = TitansConfig(num_layers=16, num_attnres_blocks=8)
        self.assertEqual(config.attnres_base_block_size, 2)

    def test_attnres_base_block_size_uneven(self):
        """Block size with remainder — last block absorbs extra."""
        config = TitansConfig(num_layers=12, num_attnres_blocks=8)
        self.assertEqual(config.attnres_base_block_size, 1)

    def test_attnres_serialization(self):
        """AttnRes fields survive to_dict/from_dict round-trip."""
        config = TitansConfig(
            use_attn_res=True,
            num_attnres_blocks=4,
            attnres_warmup_steps=1000,
            attnres_modulate_global_memory=False,
            attnres_modulate_local_memory=True,
        )
        d = config.to_dict()
        restored = TitansConfig.from_dict(d)
        self.assertTrue(restored.use_attn_res)
        self.assertEqual(restored.num_attnres_blocks, 4)
        self.assertEqual(restored.attnres_warmup_steps, 1000)
        self.assertFalse(restored.attnres_modulate_global_memory)
        self.assertTrue(restored.attnres_modulate_local_memory)
        self.assertEqual(restored.attnres_base_block_size, config.attnres_base_block_size)
