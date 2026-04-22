"""Shared test fixtures for Titans PyTorch tests."""

import pytest
import torch

from titans.config import TitansConfig


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so unknown-marker warnings do not surface."""
    config.addinivalue_line(
        "markers",
        (
            "compile_cpu: torch.compile tests runnable on CPU. May be slow "
            "(10-30s per test on first compile). Skip via "
            "TITANS_SKIP_COMPILE_TESTS=1 when torch.compile is flaky."
        ),
    )


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Test on CPU always, CUDA when available."""
    return torch.device(request.param)


@pytest.fixture
def default_config():
    """Small config for fast tests."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        max_seq_len=256,
        num_memory_layers=2,
        num_persistent_tokens=4,
    )


@pytest.fixture
def linear_memory_config():
    """Config with single-layer (linear) memory."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        vocab_size=256,
        chunk_size=32,
        window_size=32,
        max_seq_len=256,
        num_memory_layers=1,
        num_persistent_tokens=4,
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 16
