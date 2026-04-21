"""Shared test fixtures for Titans PyTorch tests."""

import sys
from pathlib import Path

# scripts/ is not a package. Some tests (DPO, RLVR memory plumbing)
# load scripts/dpo.py and scripts/rlvr.py via importlib.util; those
# modules in turn try both `from scripts._common import ...` and the
# sibling fallback `from _common import ...`. Putting scripts/ on
# sys.path here makes the sibling fallback succeed under the
# spec_from_file_location loader without otherwise changing test
# collection behaviour.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

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
