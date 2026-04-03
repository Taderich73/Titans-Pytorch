"""Tests for Adaptive Window Sizing."""

import pytest
import torch


class TestAdaptiveWindowPredictor:
    def test_output_shapes(self, device):
        from titans.adaptive_window import AdaptiveWindowPredictor
        from titans.config import TitansConfig

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            window_size=32, adaptive_window=True,
            adaptive_window_min=8, adaptive_window_max=32,
        )
        predictor = AdaptiveWindowPredictor(config).to(device)
        x = torch.randn(2, 16, 64, device=device)
        soft_mask, falloff_centers = predictor(x)
        assert soft_mask.shape == (2, 1, 16, 16)
        assert falloff_centers.shape == (2, 16, 1)

    def test_mask_is_causal(self, device):
        from titans.adaptive_window import AdaptiveWindowPredictor
        from titans.config import TitansConfig

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            window_size=32, adaptive_window=True,
            adaptive_window_min=8, adaptive_window_max=32,
        )
        predictor = AdaptiveWindowPredictor(config).to(device)
        x = torch.randn(2, 16, 64, device=device)
        soft_mask, _ = predictor(x)
        for i in range(16):
            for j in range(i + 1, 16):
                assert soft_mask[0, 0, i, j].item() == 0.0

    def test_falloff_centers_in_range(self, device):
        from titans.adaptive_window import AdaptiveWindowPredictor
        from titans.config import TitansConfig

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            window_size=32, adaptive_window=True,
            adaptive_window_min=8, adaptive_window_max=32,
        )
        predictor = AdaptiveWindowPredictor(config).to(device)
        x = torch.randn(2, 16, 64, device=device)
        _, falloff_centers = predictor(x)
        assert falloff_centers.min() >= 8.0
        assert falloff_centers.max() <= 32.0

    def test_backward(self, device):
        from titans.adaptive_window import AdaptiveWindowPredictor
        from titans.config import TitansConfig

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            window_size=32, adaptive_window=True,
        )
        predictor = AdaptiveWindowPredictor(config).to(device)
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        soft_mask, _ = predictor(x)
        soft_mask.sum().backward()
        assert x.grad is not None


class TestWindowRegularization:
    def test_regularization_value(self, device):
        from titans.adaptive_window import compute_window_regularization

        fc = [torch.full((2, 8, 1), 16.0, device=device)]
        reg = compute_window_regularization(fc, max_window=32)
        torch.testing.assert_close(reg, torch.tensor(0.5, device=device))

    def test_empty_list(self, device):
        from titans.adaptive_window import compute_window_regularization

        reg = compute_window_regularization([], max_window=32)
        assert reg.item() == 0.0


class TestAdaptiveWindowIntegration:
    def test_mag_with_adaptive_window(self, device):
        from titans.config import TitansConfig
        from titans.models import TitansMAG

        config = TitansConfig(
            dim=64, num_heads=4, num_layers=2, vocab_size=256,
            chunk_size=32, window_size=32, max_seq_len=256,
            num_memory_layers=2, num_persistent_tokens=4,
            adaptive_window=True, adaptive_window_min=8,
        )
        model = TitansMAG(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 16, config.vocab_size)
