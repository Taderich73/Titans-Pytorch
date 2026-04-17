"""Verify the memory-detach list comprehension pattern handles None entries.

Training scripts iterate `[s.detach() for s in memory_states]` at chunk
boundaries; if any layer returns None (configurable), the unguarded version
crashes with AttributeError.
"""
from __future__ import annotations

import pytest
import torch


def _detach_guarded(memory_states):
    """Reference implementation matching pretrain.py:597-600."""
    if memory_states is None:
        return None
    return [s.detach() if s is not None else None for s in memory_states]


def test_detach_guarded_passes_through_none():
    fake = torch.ones(2, 3, requires_grad=True)

    class FakeState:
        def detach(self):
            return "detached"

    mixed = [FakeState(), None, FakeState()]
    out = _detach_guarded(mixed)
    assert out[0] == "detached"
    assert out[1] is None
    assert out[2] == "detached"


def test_sft_and_lora_use_guarded_pattern():
    """Static check: the target lines must contain the guard `if s is not None`."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "scripts" / "sft.py",
        root / "scripts" / "lora.py",
    ]
    pat = re.compile(r"\[s\.detach\(\) for s in memory_states\]")
    for p in targets:
        txt = p.read_text()
        assert not pat.search(txt), (
            f"{p} contains an unguarded detach comprehension — add "
            f"'if s is not None'"
        )
