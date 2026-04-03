"""Shared test fixtures for Titans PyTorch tests."""

import pytest
import torch

from titans.config import TitansConfig


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
