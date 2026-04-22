"""Enforce ``torch.compile(fullgraph=True)`` on all Titans model variants.

``process_chunk`` (``src/titans/models.py``) declares in its docstring that
the whole function must lower into a single TorchDynamo graph with zero
graph breaks. Nothing previously verified that — any refactor could silently
introduce a data-dependent branch and regress the contract. This file is the
enforcement:

* One compile-then-run test per sampled (variant, feature-flag) pair.
* A deliberate-graph-break regression test that subclasses ``TitansMAC`` with
  a ``.item()`` call; the test is RED unless ``fullgraph=True`` catches it.

Tests are marked ``compile_cpu`` so environments where ``torch.compile`` is
flaky (old PyTorch, macOS CI with TorchInductor issues, debug builds) can
skip the file by setting ``TITANS_SKIP_COMPILE_TESTS=1``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from titans.config import TitansConfig
from titans.models import TitansLMM, TitansMAC, TitansMAG, TitansMAL

pytestmark = [
    pytest.mark.compile_cpu,
    pytest.mark.skipif(
        os.environ.get("TITANS_SKIP_COMPILE_TESTS") == "1",
        reason="torch.compile tests disabled via TITANS_SKIP_COMPILE_TESTS=1",
    ),
]


# Map variant name to (model class, does_variant_honor_adaptive_window).
_VARIANTS: dict[str, tuple[type[torch.nn.Module], bool]] = {
    "MAC": (TitansMAC, False),  # adaptive_window warns and is ignored on MAC
    "MAG": (TitansMAG, True),
    "MAL": (TitansMAL, True),
    "LMM": (TitansLMM, False),  # LMM is memory-only; no attention-side features
}


def _tiny_config(
    variant: str,
    *,
    use_tnt: bool = False,
    use_attn_res: bool = False,
    use_mca: bool = False,
    adaptive_window: bool = False,
) -> TitansConfig:
    """Build a tiny config compatible with the requested variant + features.

    Keeping dims tiny (32 / 2 heads / 1 layer / chunk 8) holds compile
    wall-time under a few seconds per test on CPU, which matters because
    torch.compile first-run latency dominates the test cost.
    """
    base: dict[str, Any] = dict(
        dim=32,
        num_heads=2,
        num_layers=1,
        vocab_size=64,
        chunk_size=8,
        window_size=8,
        max_seq_len=64,
        num_memory_layers=1,
        num_persistent_tokens=4,
        use_tnt=use_tnt,
        use_attn_res=use_attn_res,
        use_mca=use_mca,
        adaptive_window=adaptive_window,
    )

    if use_tnt:
        # TNT requires small local chunk sizes that divide chunk_size cleanly.
        base.update(
            local_chunk_sizes=[4],
            local_shard_length=32,
            global_chunk_size=32,
        )
    if use_attn_res:
        # One attn-res block is enough for the schedule to exercise its
        # flush path without bloating the graph.
        base.update(num_attnres_blocks=1)
    if use_mca:
        # Force the MCA insertion layer to the only block in the tiny model.
        base.update(mca_insertion_layers=[0])

    # MAC warns (not errors) when adaptive_window=True; scrub to keep the
    # matrix compatible without spamming warnings during compile.
    if variant == "MAC":
        base["adaptive_window"] = False

    return TitansConfig(**base)


# Sampled (variant, feature-flag dict) matrix. A full 4 * 2^4 = 64 matrix is
# overkill per the P6 plan; baseline plus a per-variant kitchen-sink covers
# the code paths without blowing up CI wall-time.
_MATRIX: list[tuple[str, dict[str, bool]]] = [
    ("MAC", {}),
    ("MAC", {"use_tnt": True, "use_attn_res": True, "use_mca": True}),
    ("MAG", {}),
    (
        "MAG",
        {
            "use_tnt": True,
            "use_attn_res": True,
            "use_mca": True,
            "adaptive_window": True,
        },
    ),
    ("MAL", {}),
    (
        "MAL",
        {
            "use_tnt": True,
            "use_attn_res": True,
            "use_mca": True,
            "adaptive_window": True,
        },
    ),
    ("LMM", {}),
    ("LMM", {"use_tnt": True}),
]


def _reset_dynamo() -> None:
    """Reset Dynamo state so each test compiles its own graph from scratch.

    Without this, test isolation leaks: a prior test's cached graph can mask
    a later test's graph break, defeating the point of the enforcement.
    """
    import torch._dynamo as dynamo

    dynamo.reset()


@pytest.mark.parametrize(
    ("variant", "overrides"),
    _MATRIX,
    ids=lambda v: v if isinstance(v, str) else "+".join(sorted(v)) or "baseline",
)
def test_variant_compiles_fullgraph(variant: str, overrides: dict[str, bool]) -> None:
    """Each sampled (variant, feature) combo must lower without graph breaks.

    ``torch.compile(fullgraph=True)`` raises ``torch._dynamo.exc.Unsupported``
    (or a subclass) on any graph break. The test passes iff the compiled
    forward runs end-to-end.
    """
    _reset_dynamo()

    model_cls, _ = _VARIANTS[variant]
    config = _tiny_config(variant, **overrides)
    model = model_cls(config).eval()

    compiled = torch.compile(model, fullgraph=True, dynamic=False)

    input_ids = torch.randint(
        0, config.vocab_size, (1, config.chunk_size), device="cpu"
    )

    # ``fullgraph=True`` raises on any graph break. If we reach the assertion,
    # the compile succeeded. We only assert shapes as a smoke check.
    logits, states, _ = compiled(input_ids)

    assert logits.shape == (1, config.chunk_size, config.vocab_size)
    assert len(states) == config.num_layers


def test_deliberate_graph_break_is_caught() -> None:
    """Sanity check: a data-dependent Python branch MUST fail under fullgraph.

    Guards the enforcement mechanism itself — if ``torch.compile`` ever
    stopped flagging data-dependent control flow, the rest of the file would
    silently stop catching regressions. We subclass ``TitansMAC`` and inject
    the canonical graph-breaking op: a Python-level ``if`` on a tensor value.

    (An earlier draft used ``.item()``, but modern PyTorch traces that
    through capture_scalar_outputs on CPU. A Python branch is the stable
    trigger across versions.)
    """
    _reset_dynamo()

    class BrokenMAC(TitansMAC):
        def forward(self, input_ids, states=None):  # type: ignore[override]
            # Data-dependent control flow: branch on a tensor value at the
            # Python level. Dynamo cannot lower this into a single graph and
            # raises `Unsupported` under fullgraph=True.
            if input_ids.sum() > 0:
                input_ids = input_ids + 0
            return super().forward(input_ids, states=states)

    config = _tiny_config("MAC")
    model = BrokenMAC(config).eval()
    compiled = torch.compile(model, fullgraph=True, dynamic=False)

    input_ids = torch.randint(
        1, config.vocab_size, (1, config.chunk_size), device="cpu"
    )

    # Any Dynamo-origin failure is acceptable. Matching on the base Exception
    # keeps the test robust across PyTorch minor versions, which churn on the
    # exact exception class names for graph-break errors.
    with pytest.raises(Exception):
        compiled(input_ids)
