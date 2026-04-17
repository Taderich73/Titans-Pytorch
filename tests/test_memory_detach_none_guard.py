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
    real = torch.ones(2, 3, requires_grad=True)

    class FakeState:
        def detach(self):
            return "detached"

    mixed = [real, None, FakeState()]
    out = _detach_guarded(mixed)
    assert torch.equal(out[0], real) and not out[0].requires_grad
    assert out[1] is None
    assert out[2] == "detached"


def test_sft_and_lora_use_guarded_pattern():
    """Static check: the target lines must contain the guard `if s is not None`
    AND must NOT contain the unguarded form."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "scripts" / "sft.py",
        root / "scripts" / "lora.py",
    ]
    unguarded_pat = re.compile(r"\[s\.detach\(\) for s in memory_states\]")
    guarded_pat = re.compile(
        r"s\.detach\(\)\s+if\s+s\s+is\s+not\s+None\s+else\s+None"
    )
    for p in targets:
        txt = p.read_text()
        assert not unguarded_pat.search(txt), (
            f"{p} contains an unguarded detach comprehension — add "
            f"'if s is not None'"
        )
        assert guarded_pat.search(txt), (
            f"{p} is missing the guarded detach pattern — regression?"
        )
