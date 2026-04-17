"""Tests for Attention Residuals."""

import torch


class TestBlockAttnRes:
    def test_single_source(self, device):
        from titans.attn_res import BlockAttnRes

        ar = BlockAttnRes(dim=64).to(device)
        block = torch.randn(2, 8, 64, device=device)
        h, weights = ar([block], partial_block=None)
        assert h.shape == (2, 8, 64)
        assert weights.shape == (2, 8, 1)
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_multiple_sources(self, device):
        from titans.attn_res import BlockAttnRes

        ar = BlockAttnRes(dim=64).to(device)
        blocks = [torch.randn(2, 8, 64, device=device) for _ in range(3)]
        h, weights = ar(blocks, partial_block=None)
        assert h.shape == (2, 8, 64)
        assert weights.shape == (2, 8, 3)
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, 8, device=device),
        )

    def test_with_partial_block(self, device):
        from titans.attn_res import BlockAttnRes

        ar = BlockAttnRes(dim=64).to(device)
        blocks = [torch.randn(2, 8, 64, device=device)]
        partial = torch.randn(2, 8, 64, device=device)
        h, weights = ar(blocks, partial_block=partial)
        assert h.shape == (2, 8, 64)
        assert weights.shape == (2, 8, 2)

    def test_zero_init_uniform_weights(self, device):
        from titans.attn_res import BlockAttnRes

        ar = BlockAttnRes(dim=64).to(device)
        blocks = [torch.randn(2, 8, 64, device=device) for _ in range(4)]
        _, weights = ar(blocks, partial_block=None)
        expected = torch.full((2, 8, 4), 0.25, device=device)
        torch.testing.assert_close(weights, expected, atol=0.15, rtol=0.0)

    def test_backward(self, device):
        from titans.attn_res import BlockAttnRes

        ar = BlockAttnRes(dim=64).to(device)
        blocks = [torch.randn(2, 8, 64, device=device, requires_grad=True) for _ in range(3)]
        h, _ = ar(blocks, partial_block=None)
        h.sum().backward()
        for b in blocks:
            assert b.grad is not None


class TestAttnResMemoryGate:
    def test_scalar_output(self, device):
        from titans.attn_res import AttnResMemoryGate

        gate = AttnResMemoryGate()
        weights = torch.rand(2, 8, 4, device=device)
        importance = gate(weights)
        assert importance.shape == ()
        assert 0.0 <= importance.item() <= 1.0
