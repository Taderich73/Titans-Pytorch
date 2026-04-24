"""Shared test fixtures for Titans PyTorch tests."""

import pytest
import torch

from titans.config import TitansConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--runslow`` so slow/subprocess-heavy tests are opt-in."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.slow (subprocess-heavy smoke / "
            "integration tests). Skipped by default."
        ),
    )


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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``@pytest.mark.slow`` tests unless ``--runslow`` is passed.

    The ``slow`` marker is registered in pyproject.toml for
    subprocess-heavy smoke / integration tests that spawn full training
    runs. Running them in default CI would blow the runner's time budget
    even when the run itself would eventually succeed, so we require an
    explicit opt-in.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test -- pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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
