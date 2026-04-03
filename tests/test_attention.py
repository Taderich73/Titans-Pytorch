"""Tests for attention modules."""

import torch

from titans.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans.config import TitansConfig


class TestRotaryPositionEmbedding:
    def test_output_shape(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q, q_rot)

    def test_cache_rebuild(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=8).to(device)
        q = torch.randn(1, 1, 16, 16, device=device)
        k = torch.randn(1, 1, 16, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape

    def test_seq_offset(self, device):
        rope = RotaryPositionEmbedding(dim=16, max_seq_len=64).to(device)
        q = torch.randn(1, 2, 4, 16, device=device)
        k = torch.randn(1, 2, 4, 16, device=device)
        q_rot1, _ = rope(q, k, seq_offset=0)
        q_rot2, _ = rope(q, k, seq_offset=10)
        assert not torch.allclose(q_rot1, q_rot2)


class TestSlidingWindowAttention:
    def test_output_shape(self, default_config, device):
        attn = SlidingWindowAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_prefix(self, default_config, device):
        attn = SlidingWindowAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        prefix = torch.randn(2, 4, default_config.dim, device=device)
        out = attn(x, prefix=prefix)
        assert out.shape == x.shape

    def test_no_rope(self, device):
        config = TitansConfig(dim=64, num_heads=4, use_rope=False, window_size=32)
        attn = SlidingWindowAttention(config).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = attn(x)
        assert out.shape == x.shape


class TestSegmentedAttention:
    def test_output_shape(self, default_config, device):
        attn = SegmentedAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_persistent_and_memory(self, default_config, device):
        attn = SegmentedAttention(default_config).to(device)
        x = torch.randn(2, 16, default_config.dim, device=device)
        persistent = torch.randn(2, 4, default_config.dim, device=device)
        memory = torch.randn(2, 1, default_config.dim, device=device)
        out = attn(x, persistent=persistent, memory=memory)
        assert out.shape == x.shape

    def test_causal_property(self, default_config, device):
        """Early positions should not be affected by later positions."""
        attn = SegmentedAttention(default_config).to(device)
        attn.eval()
        x = torch.randn(1, 8, default_config.dim, device=device)
        out1 = attn(x)
        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(1, default_config.dim, device=device)
        out2 = attn(x_mod)
        torch.testing.assert_close(out1[:, 0, :], out2[:, 0, :])
