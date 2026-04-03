# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for memory state quantization."""

import mlx.core as mx
import pytest


class TestQuantizedTensor:
    """Tests for QuantizedTensor quantize/dequantize."""

    def test_quantize_8bit_round_trip(self) -> None:
        """8-bit quantize -> dequantize preserves values within tolerance."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=8)
        restored = qt.dequantize()

        assert restored.shape == original.shape
        assert restored.dtype == mx.float32

        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.001, f"8-bit round-trip MSE too high: {mse}"

    def test_quantize_8bit_metadata(self) -> None:
        """8-bit QuantizedTensor stores correct metadata."""
        mx.random.seed(42)
        original = mx.random.normal((32, 128))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=8)

        assert qt.bits == 8
        assert qt.original_shape == (32, 128)
        assert qt.data.dtype == mx.uint8
        assert qt.data.shape == (32, 128)
        assert qt.scale.dtype == mx.float32
        assert qt.zero_point.dtype == mx.float32

    def test_quantize_4bit_round_trip(self) -> None:
        """4-bit quantize -> dequantize preserves values within tolerance."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)
        restored = qt.dequantize()

        assert restored.shape == original.shape
        assert restored.dtype == mx.float32

        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.05, f"4-bit round-trip MSE too high: {mse}"

    def test_quantize_4bit_packing(self) -> None:
        """4-bit packing stores two values per byte."""
        mx.random.seed(42)
        original = mx.random.normal((64, 64))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)

        assert qt.bits == 4
        assert qt.data.dtype == mx.uint8
        # 64*64 = 4096 values, packed into 2048 bytes
        assert qt.data.shape == (2048,)

    def test_quantize_4bit_odd_elements(self) -> None:
        """4-bit packing handles odd number of elements."""
        mx.random.seed(42)
        original = mx.random.normal((3, 5))  # 15 elements (odd)
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        qt = quantize_tensor(original, bits=4)
        restored = qt.dequantize()

        assert restored.shape == (3, 5)
        mse = mx.mean((original - restored) ** 2).item()
        assert mse < 0.05

    def test_quantize_constant_tensor(self) -> None:
        """Quantization handles constant tensors (zero range) without error."""
        original = mx.ones((16, 16)) * 3.14
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        for bits in (4, 8):
            qt = quantize_tensor(original, bits=bits)
            restored = qt.dequantize()
            mse = mx.mean((original - restored) ** 2).item()
            assert mse < 1e-6, f"Constant tensor MSE too high at {bits}-bit: {mse}"

    def test_quantize_invalid_bits(self) -> None:
        """Quantization rejects invalid bit-widths."""
        original = mx.random.normal((8, 8))
        mx.eval(original)

        from titans_mlx.quantize_state import quantize_tensor

        with pytest.raises(ValueError, match="Unsupported bit-width"):
            quantize_tensor(original, bits=3)


class TestQuantizedMemoryState:
    """Tests for QuantizedMemoryState quantize/dequantize."""

    def test_quantize_linear_memory_state(self) -> None:
        """Linear memory (1 layer): weights quantized, momentum cast to float16."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 64))],
            momentum=[mx.random.normal((64, 64))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import (
            QuantizedMemoryState,
            QuantizedTensor,
            quantize_memory_state,
        )

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)

        assert isinstance(qstate, QuantizedMemoryState)
        assert isinstance(qstate.weights[0], QuantizedTensor)
        assert qstate.weights[0].bits == 4
        # Linear path: momentum is float16, not QuantizedTensor
        assert isinstance(qstate.momentum[0], mx.array)
        assert qstate.momentum[0].dtype == mx.float16

    def test_quantize_deep_memory_state(self) -> None:
        """Deep memory (2 layers): weights 4-bit, momentum 8-bit."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 128)), mx.random.normal((128, 64))],
            momentum=[mx.random.normal((64, 128)), mx.random.normal((128, 64))],
        )
        mx.eval(*state.weights, *state.momentum)

        from titans_mlx.quantize_state import (
            QuantizedMemoryState,
            QuantizedTensor,
            quantize_memory_state,
        )

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=8)

        assert isinstance(qstate, QuantizedMemoryState)
        assert len(qstate.weights) == 2
        assert all(isinstance(w, QuantizedTensor) for w in qstate.weights)
        assert all(w.bits == 4 for w in qstate.weights)
        assert len(qstate.momentum) == 2
        assert all(isinstance(m, QuantizedTensor) for m in qstate.momentum)
        assert all(m.bits == 8 for m in qstate.momentum)

    def test_dequantize_round_trip(self) -> None:
        """QuantizedMemoryState.dequantize() returns MemoryState with correct shapes."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((64, 64))],
            momentum=[mx.random.normal((64, 64))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)
        restored = qstate.dequantize()

        assert isinstance(restored, MemoryState)
        assert restored.weights[0].shape == (64, 64)
        assert restored.weights[0].dtype == mx.float32
        assert restored.momentum[0].shape == (64, 64)
        assert restored.momentum[0].dtype == mx.float32

    def test_detach_interface(self) -> None:
        """QuantizedMemoryState.detach() returns QuantizedMemoryState."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import (
            QuantizedMemoryState,
            quantize_memory_state,
        )

        qstate = quantize_memory_state(state, weight_bits=8, momentum_bits=None)
        detached = qstate.detach()

        assert isinstance(detached, QuantizedMemoryState)


class TestStateAccessors:
    """Tests for get_weights / get_momentum helpers."""

    def test_get_weights_from_memory_state(self) -> None:
        """get_weights passes through MemoryState weights unchanged."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        w = mx.random.normal((32, 32))
        mx.eval(w)
        state = MemoryState(weights=[w], momentum=[mx.zeros_like(w)])

        from titans_mlx.quantize_state import get_weights

        result = get_weights(state)
        assert len(result) == 1
        assert mx.array_equal(result[0], w)

    def test_get_weights_from_quantized_state(self) -> None:
        """get_weights dequantizes QuantizedMemoryState weights."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        w = mx.random.normal((32, 32))
        mx.eval(w)
        state = MemoryState(weights=[w], momentum=[mx.zeros_like(w)])

        from titans_mlx.quantize_state import get_weights, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=8, momentum_bits=None)
        result = get_weights(qstate)

        assert len(result) == 1
        assert result[0].shape == (32, 32)
        assert result[0].dtype == mx.float32

    def test_get_momentum_from_memory_state(self) -> None:
        """get_momentum passes through MemoryState momentum unchanged."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        m = mx.random.normal((32, 32))
        mx.eval(m)
        state = MemoryState(weights=[mx.zeros_like(m)], momentum=[m])

        from titans_mlx.quantize_state import get_momentum

        result = get_momentum(state)
        assert len(result) == 1
        assert mx.array_equal(result[0], m)

    def test_get_momentum_from_quantized_float16(self) -> None:
        """get_momentum casts float16 momentum back to float32."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import get_momentum, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)
        result = get_momentum(qstate)

        assert len(result) == 1
        assert result[0].dtype == mx.float32
        assert result[0].shape == (32, 32)

    def test_get_momentum_from_quantized_8bit(self) -> None:
        """get_momentum dequantizes 8-bit QuantizedTensor momentum."""
        mx.random.seed(42)
        from titans_mlx.memory import MemoryState

        state = MemoryState(
            weights=[mx.random.normal((32, 64)), mx.random.normal((64, 32))],
            momentum=[mx.random.normal((32, 64)), mx.random.normal((64, 32))],
        )
        mx.eval(*state.weights, *state.momentum)

        from titans_mlx.quantize_state import get_momentum, quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=8)
        result = get_momentum(qstate)

        assert len(result) == 2
        assert all(r.dtype == mx.float32 for r in result)


class TestNeuralLongTermMemoryQuantization:
    """Tests for quantization integration in NeuralLongTermMemory."""

    def test_quantized_state_returned_when_enabled(self) -> None:
        """With quantize_memory_state=True, __call__ returns QuantizedMemoryState."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        mem = NeuralLongTermMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        output, state = mem(x)
        mx.eval(output)

        from titans_mlx.quantize_state import QuantizedMemoryState

        assert isinstance(state, QuantizedMemoryState)

    def test_unquantized_state_when_disabled(self) -> None:
        """With quantize_memory_state=False, __call__ returns MemoryState."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import MemoryState, NeuralLongTermMemory

        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=False,
        )
        mem = NeuralLongTermMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        output, state = mem(x)
        mx.eval(output)

        assert isinstance(state, MemoryState)

    def test_retrieve_works_with_quantized_state(self) -> None:
        """retrieve() works with a QuantizedMemoryState."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        mem = NeuralLongTermMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        _, state = mem(x)
        queries = mx.random.normal((1, 4, 32))
        mx.eval(queries)

        result = mem.retrieve(queries, state)
        mx.eval(result)

        assert result.shape == (1, 4, 32)

    def test_chained_updates_with_quantized_state(self) -> None:
        """Multiple forward passes chaining quantized state work without error."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=32,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        mem = NeuralLongTermMemory(config)

        state = None
        for _ in range(5):
            x = mx.random.normal((1, 8, 32))
            mx.eval(x)
            output, state = mem(x, state=state)
            mx.eval(output)

        from titans_mlx.quantize_state import QuantizedMemoryState

        assert isinstance(state, QuantizedMemoryState)

    def test_deep_memory_quantized_state(self) -> None:
        """Deep memory (L=2) returns QuantizedMemoryState with 8-bit momentum."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=32,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
            memory_state_momentum_bits=8,
        )
        mem = NeuralLongTermMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        output, state = mem(x)
        mx.eval(output)

        from titans_mlx.quantize_state import QuantizedMemoryState, QuantizedTensor

        assert isinstance(state, QuantizedMemoryState)
        assert all(
            isinstance(w, QuantizedTensor) and w.bits == 4 for w in state.weights
        )
        assert all(
            isinstance(m, QuantizedTensor) and m.bits == 8 for m in state.momentum
        )


class TestHierarchicalMemoryQuantization:
    """Tests for Q-K projection quantization in HierarchicalMemory."""

    def test_qk_projections_quantized_when_enabled(self) -> None:
        """Q-K projections are QuantizedTensors when quantization is on."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.tnt_memory import HierarchicalMemory

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=1,
            num_memory_layers=1,
            use_conv=False,
            use_rope=False,
            use_tnt=True,
            global_chunk_size=16,
            local_chunk_sizes=[4],
            local_shard_length=64,
            use_qk_projection=True,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        hmem = HierarchicalMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        output, state = hmem(x)
        mx.eval(output)

        from titans_mlx.quantize_state import QuantizedTensor

        # Q-K projections should be quantized
        assert len(state.qk_projections) == 1
        assert isinstance(state.qk_projections[0], QuantizedTensor)


class TestQuantizedSerialization:
    """Tests for saving/loading quantized memory states."""

    def test_save_load_quantized_state(self) -> None:
        """Quantized state round-trips through save/load."""
        import tempfile
        from pathlib import Path

        mx.random.seed(42)
        from titans_mlx.memory import (
            MemoryState,
            load_memory_states,
            save_memory_states,
        )

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        from titans_mlx.quantize_state import quantize_memory_state

        qstate = quantize_memory_state(state, weight_bits=4, momentum_bits=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.npz"
            save_memory_states([qstate], path)
            loaded = load_memory_states(path)

        assert len(loaded) == 1
        from titans_mlx.quantize_state import QuantizedMemoryState

        assert isinstance(loaded[0], QuantizedMemoryState)

        # Dequantize and check shapes
        restored = loaded[0].dequantize()
        assert restored.weights[0].shape == (32, 32)
        assert restored.momentum[0].shape == (32, 32)

    def test_save_load_unquantized_backward_compat(self) -> None:
        """Unquantized save/load still works (backward compatibility)."""
        import tempfile
        from pathlib import Path

        mx.random.seed(42)
        from titans_mlx.memory import (
            MemoryState,
            load_memory_states,
            save_memory_states,
        )

        state = MemoryState(
            weights=[mx.random.normal((32, 32))],
            momentum=[mx.random.normal((32, 32))],
        )
        mx.eval(state.weights[0], state.momentum[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.npz"
            save_memory_states([state], path)
            loaded = load_memory_states(path)

        assert len(loaded) == 1
        assert isinstance(loaded[0], MemoryState)
        assert loaded[0].weights[0].shape == (32, 32)

    def test_retrieve_works_with_quantized_qk(self) -> None:
        """Retrieval works when Q-K projections are quantized."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.tnt_memory import HierarchicalMemory

        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=1,
            num_memory_layers=1,
            use_conv=False,
            use_rope=False,
            use_tnt=True,
            global_chunk_size=16,
            local_chunk_sizes=[4],
            local_shard_length=64,
            use_qk_projection=True,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        hmem = HierarchicalMemory(config)
        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        _, state = hmem(x)
        queries = mx.random.normal((1, 4, 32))
        mx.eval(queries)

        result = hmem.retrieve(queries, state)
        mx.eval(result)

        assert result.shape == (1, 4, 32)


class TestRetrievalDistortion:
    """Validate that quantization doesn't meaningfully distort retrieval."""

    def test_retrieval_cosine_similarity_linear(self) -> None:
        """4-bit quantized linear memory retrieval has high cosine similarity to original."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=64,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=False,
        )
        mem = NeuralLongTermMemory(config)

        # Build up a state with some content
        x = mx.random.normal((1, 16, 64))
        mx.eval(x)
        _, full_state = mem(x)
        mx.eval(full_state.weights[0])

        # Quantize the state
        from titans_mlx.quantize_state import quantize_memory_state

        qstate = quantize_memory_state(full_state, weight_bits=4, momentum_bits=None)

        # Compare retrieval outputs across many queries
        queries = mx.random.normal((1, 100, 64))
        mx.eval(queries)

        out_full = mem.retrieve(queries, full_state)
        out_quant = mem.retrieve(queries, qstate)
        mx.eval(out_full, out_quant)

        # Cosine similarity per query
        dot = mx.sum(out_full * out_quant, axis=-1)
        norm_full = mx.sqrt(mx.sum(out_full**2, axis=-1) + 1e-8)
        norm_quant = mx.sqrt(mx.sum(out_quant**2, axis=-1) + 1e-8)
        cosine = dot / (norm_full * norm_quant)
        mx.eval(cosine)

        mean_cosine = mx.mean(cosine).item()
        print(f"\nLinear memory mean cosine similarity: {mean_cosine:.6f}")
        assert mean_cosine > 0.97, f"Mean cosine similarity too low: {mean_cosine}"

    def test_retrieval_cosine_similarity_deep(self) -> None:
        """4-bit quantized deep memory retrieval has high cosine similarity."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config = TitansConfig(
            dim=64,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            use_conv=False,
            quantize_memory_state=False,
        )
        mem = NeuralLongTermMemory(config)

        x = mx.random.normal((1, 16, 64))
        mx.eval(x)
        _, full_state = mem(x)
        mx.eval(*full_state.weights)

        from titans_mlx.quantize_state import quantize_memory_state

        qstate = quantize_memory_state(full_state, weight_bits=4, momentum_bits=8)

        queries = mx.random.normal((1, 100, 64))
        mx.eval(queries)

        out_full = mem.retrieve(queries, full_state)
        out_quant = mem.retrieve(queries, qstate)
        mx.eval(out_full, out_quant)

        dot = mx.sum(out_full * out_quant, axis=-1)
        norm_full = mx.sqrt(mx.sum(out_full**2, axis=-1) + 1e-8)
        norm_quant = mx.sqrt(mx.sum(out_quant**2, axis=-1) + 1e-8)
        cosine = dot / (norm_full * norm_quant)
        mx.eval(cosine)

        mean_cosine = mx.mean(cosine).item()
        print(f"\nDeep memory mean cosine similarity: {mean_cosine:.6f}")
        assert mean_cosine > 0.97, f"Mean cosine similarity too low: {mean_cosine}"


class TestMomentumStability:
    """Validate that quantized momentum doesn't diverge over many chunks."""

    def test_linear_memory_stability_50_chunks(self) -> None:
        """Linear memory with float16 momentum stays close to f32 baseline over 50 chunks."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        # Full-precision baseline
        config_fp = TitansConfig(
            dim=64,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=False,
        )
        mem_fp = NeuralLongTermMemory(config_fp)

        # Quantized
        config_q = TitansConfig(
            dim=64,
            num_memory_layers=1,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
        )
        mem_q = NeuralLongTermMemory(config_q)

        # Copy weights so both models are identical
        mem_q.proj_k.weight = mem_fp.proj_k.weight
        mem_q.proj_v.weight = mem_fp.proj_v.weight
        mem_q.proj_q.weight = mem_fp.proj_q.weight
        mem_q.proj_out.weight = mem_fp.proj_out.weight
        mem_q.memory.layers[0].weight = mem_fp.memory.layers[0].weight
        mem_q.gate_decay_proj.weight = mem_fp.gate_decay_proj.weight
        mem_q.gate_decay_proj.bias = mem_fp.gate_decay_proj.bias
        mem_q.gate_lr_proj.weight = mem_fp.gate_lr_proj.weight
        mem_q.gate_lr_proj.bias = mem_fp.gate_lr_proj.bias
        mem_q.gate_momentum_proj.weight = mem_fp.gate_momentum_proj.weight
        mem_q.gate_momentum_proj.bias = mem_fp.gate_momentum_proj.bias

        state_fp = None
        state_q = None

        for _chunk in range(50):
            x = mx.random.normal((1, 8, 64))
            mx.eval(x)
            _, state_fp = mem_fp(x, state=state_fp)
            _, state_q = mem_q(x, state=state_q)
            mx.eval(state_fp.weights[0], state_fp.momentum[0])

        # Compare final weights
        from titans_mlx.quantize_state import get_weights

        w_fp = state_fp.weights[0]
        w_q = get_weights(state_q)[0]
        mx.eval(w_fp, w_q)

        # Relative error
        diff = mx.mean(mx.abs(w_fp - w_q)).item()
        baseline = mx.mean(mx.abs(w_fp)).item()
        relative_error = diff / (baseline + 1e-8)

        print(f"\nLinear memory relative_error after 50 chunks: {relative_error:.4f}")
        # float16 momentum should keep relative error small after 50 chunks
        assert relative_error < 0.2, (
            f"Linear memory diverged too much after 50 chunks: "
            f"relative_error={relative_error:.4f}"
        )

    def test_deep_memory_stability_50_chunks(self) -> None:
        """Deep memory with 8-bit momentum stays close to f32 baseline over 50 chunks."""
        mx.random.seed(42)
        from titans_mlx.config import TitansConfig
        from titans_mlx.memory import NeuralLongTermMemory

        config_fp = TitansConfig(
            dim=64,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            use_conv=False,
            quantize_memory_state=False,
        )
        mem_fp = NeuralLongTermMemory(config_fp)

        config_q = TitansConfig(
            dim=64,
            num_memory_layers=2,
            memory_hidden_mult=2.0,
            use_conv=False,
            quantize_memory_state=True,
            memory_state_weight_bits=4,
            memory_state_momentum_bits=8,
        )
        mem_q = NeuralLongTermMemory(config_q)

        # Copy all weights
        for attr in ("proj_k", "proj_v", "proj_q", "proj_out"):
            getattr(mem_q, attr).weight = getattr(mem_fp, attr).weight
        for gate in ("gate_decay_proj", "gate_lr_proj", "gate_momentum_proj"):
            getattr(mem_q, gate).weight = getattr(mem_fp, gate).weight
            getattr(mem_q, gate).bias = getattr(mem_fp, gate).bias
        for j, layer in enumerate(mem_fp.memory.layers):
            mem_q.memory.layers[j].weight = layer.weight

        state_fp = None
        state_q = None

        for _chunk in range(50):
            x = mx.random.normal((1, 8, 64))
            mx.eval(x)
            _, state_fp = mem_fp(x, state=state_fp)
            _, state_q = mem_q(x, state=state_q)
            mx.eval(*state_fp.weights, *state_fp.momentum)

        from titans_mlx.quantize_state import get_weights

        w_fp = state_fp.weights[0]
        w_q = get_weights(state_q)[0]
        mx.eval(w_fp, w_q)

        diff = mx.mean(mx.abs(w_fp - w_q)).item()
        baseline = mx.mean(mx.abs(w_fp)).item()
        relative_error = diff / (baseline + 1e-8)

        print(f"\nDeep memory relative_error after 50 chunks: {relative_error:.4f}")
        assert relative_error < 0.2, (
            f"Deep memory diverged too much after 50 chunks: "
            f"relative_error={relative_error:.4f}"
        )
