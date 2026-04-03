"""Tests for PersistentMemory module."""

import torch

from titans.config import TitansConfig
from titans.persistent import PersistentMemory


class TestPersistentMemory:
    def test_forward_shape(self, default_config, device):
        mem = PersistentMemory(default_config).to(device)
        result = mem(batch_size=2)
        assert result.shape == (2, default_config.num_persistent_tokens, default_config.dim)
        assert result.device.type == device.type

    def test_returns_none_when_zero_tokens(self, device):
        config = TitansConfig(dim=64, num_persistent_tokens=0)
        mem = PersistentMemory(config).to(device)
        assert mem(batch_size=2) is None

    def test_tokens_are_parameters(self, default_config):
        mem = PersistentMemory(default_config)
        assert isinstance(mem.tokens, torch.nn.Parameter)

    def test_batch_expansion_shares_data(self, default_config, device):
        mem = PersistentMemory(default_config).to(device)
        result = mem(batch_size=3)
        assert result.shape[0] == 3
