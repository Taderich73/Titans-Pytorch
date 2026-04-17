"""Tests for Adaptive Window Sizing."""

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
        logits, states, _ = model(x)
        assert logits.shape == (2, 16, config.vocab_size)


def test_adaptive_window_grid_is_cached() -> None:
    from titans.adaptive_window import _adaptive_window_grid

    _adaptive_window_grid.cache_clear()
    g1 = _adaptive_window_grid(seq_len=32, device_str="cpu")
    g2 = _adaptive_window_grid(seq_len=32, device_str="cpu")
    assert g1 is g2  # identity => cache hit
    info = _adaptive_window_grid.cache_info()
    assert info.hits >= 1


def test_adaptive_window_forward_numerical_parity() -> None:
    from titans.adaptive_window import AdaptiveWindowPredictor
    from titans.config import TitansConfig

    torch.manual_seed(0)
    cfg = TitansConfig(
        dim=32,
        num_heads=4,
        adaptive_window_min=4,
        adaptive_window_max=64,
        adaptive_window_temperature=1.0,
    )
    pred = AdaptiveWindowPredictor(cfg).eval()
    x = torch.randn(2, 16, 32)
    mask_a, centers_a = pred(x)
    mask_b, centers_b = pred(x)
    assert torch.allclose(mask_a, mask_b)
    assert torch.allclose(centers_a, centers_b)
