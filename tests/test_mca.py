"""Tests for Memory Cross-Attention."""

import pytest
import torch

from titans.config import TitansConfig


@pytest.fixture
def mca_config():
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        use_mca=True,
        mca_insertion_layers=[0],
        mca_num_heads=4,
        mca_gate_type="scalar",
        mca_gate_bias_init=-3.0,
    )


class TestMemoryCrossAttention:
    def test_output_shape(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_gate_init_near_zero(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.abs().mean() < 1.0

    def test_vector_gate(self, device):
        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            use_mca=True, mca_insertion_layers=[0],
            mca_gate_type="vector",
        )
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(config).to(device)
        x = torch.randn(2, 8, config.dim, device=device)
        memory_weights = torch.randn(config.dim, config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_deep_memory_weights(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device)
        num_rows = mca_config.memory_hidden_dim
        memory_weights = torch.randn(num_rows, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        assert out.shape == x.shape

    def test_backward(self, mca_config, device):
        from titans.mca import MemoryCrossAttention

        mca = MemoryCrossAttention(mca_config).to(device)
        x = torch.randn(2, 8, mca_config.dim, device=device, requires_grad=True)
        memory_weights = torch.randn(mca_config.dim, mca_config.dim, device=device)
        out = mca(x, memory_weights)
        out.sum().backward()
        assert x.grad is not None
