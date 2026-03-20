# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for TNT configuration, state, Q-K projection, and hierarchical memory."""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from titans_mlx.config import TitansConfig
from titans_mlx.memory import (
    MemoryState,
    TNTMemoryState,
    load_tnt_memory_states,
    save_tnt_memory_states,
)
from titans_mlx.qk_projection import QKProjection, update_projection_state
from titans_mlx.tnt_memory import GlobalMemory, HierarchicalMemory, LocalMemory


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
