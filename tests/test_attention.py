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


class TestProportionalRoPE:
    def test_shape_preserved(self, device):
        """p-RoPE output shape matches input for various proportions."""
        for proportion in [0.25, 0.5, 0.75, 1.0]:
            rope = RotaryPositionEmbedding(
                dim=16, max_seq_len=64, rope_proportion=proportion
            ).to(device)
            q = torch.randn(2, 4, 8, 16, device=device)
            k = torch.randn(2, 4, 8, 16, device=device)
            q_rot, k_rot = rope(q, k)
            assert q_rot.shape == q.shape, f"Failed for proportion={proportion}"
            assert k_rot.shape == k.shape, f"Failed for proportion={proportion}"

    def test_passthrough_unchanged(self, device):
        """Dimensions beyond rotate_dim are not modified."""
        rope = RotaryPositionEmbedding(
            dim=16, max_seq_len=64, rope_proportion=0.5
        ).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        # rotate_dim = 2 * (int(16 * 0.5) // 2) = 8
        torch.testing.assert_close(q_rot[..., 8:], q[..., 8:])
        torch.testing.assert_close(k_rot[..., 8:], k[..., 8:])

    def test_rotated_dims_changed(self, device):
        """Dimensions within rotate_dim are modified."""
        rope = RotaryPositionEmbedding(
            dim=16, max_seq_len=64, rope_proportion=0.5
        ).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q_rot[..., :8], q[..., :8])

    def test_full_proportion_matches_standard(self, device):
        """rope_proportion=1.0 is bit-identical to standard RoPE."""
        torch.manual_seed(42)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)

        rope_full = RotaryPositionEmbedding(
            dim=16, max_seq_len=64, rope_proportion=1.0
        ).to(device)
        q_full, k_full = rope_full(q.clone(), k.clone())

        rope_std = RotaryPositionEmbedding(
            dim=16, max_seq_len=64
        ).to(device)
        q_std, k_std = rope_std(q.clone(), k.clone())

        torch.testing.assert_close(q_full, q_std)
        torch.testing.assert_close(k_full, k_std)

    def test_zero_proportion_is_identity(self, device):
        """rope_proportion=0.0 returns input unchanged."""
        rope = RotaryPositionEmbedding(
            dim=16, max_seq_len=64, rope_proportion=0.0
        ).to(device)
        q = torch.randn(2, 4, 8, 16, device=device)
        k = torch.randn(2, 4, 8, 16, device=device)
        q_rot, k_rot = rope(q, k)
        torch.testing.assert_close(q_rot, q)
        torch.testing.assert_close(k_rot, k)


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

    def test_with_rope_proportion(self, device):
        config = TitansConfig(dim=64, num_heads=4, use_rope=True, rope_proportion=0.5)
        attn = SlidingWindowAttention(config).to(device)
        assert attn.rope is not None
        assert attn.rope.rotate_dim == 8  # head_dim=16, 0.5 * 16 = 8
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

    def test_with_rope_proportion(self, default_config, device):
        config = TitansConfig(
            dim=default_config.dim,
            num_heads=default_config.num_heads,
            num_layers=default_config.num_layers,
            rope_proportion=0.25,
        )
        attn = SegmentedAttention(config).to(device)
        assert attn.rope is not None
        # head_dim=16, rotate_dim = 2*(int(16*0.25)//2) = 2*(4//2) = 4
        assert attn.rope.rotate_dim == 4
        x = torch.randn(2, 16, config.dim, device=device)
        out = attn(x)
        assert out.shape == x.shape
