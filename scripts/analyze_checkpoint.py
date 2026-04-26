#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""Static training-health analysis from a Titans checkpoint pair.

Loads a ``final.pt`` (or any step checkpoint) plus an optional
``memory_*.npz`` and prints a structured report covering:

  [1] Per-block weight norms grouped by component
  [2] Adam ``exp_avg_sq`` per block (gradient signal across depth)
  [3] NeuralLTM memory state norms per block (inner-loop weights)
  [4] Saturation / dead-param check (std<1e-6 or |max|>50)
  [5] Optimizer step-counter consistency

All five sections are static — they read the saved tensors and
summarize. For dynamic train-vs-eval-gap diagnosis run
``scripts/eval_memory_ablation.py`` instead, which forwards the model
on real data with three different memory-state initializations.

Typical usage::

    uv run python scripts/analyze_checkpoint.py \\
        --checkpoint checkpoints/final.pt \\
        --memory-dump checkpoints/memory_final.npz

Both ``--checkpoint`` and ``--memory-dump`` are independent. Pass only
``--checkpoint`` to skip the per-block memory section; pass only
``--memory-dump`` to run the inner-loop analysis without optimizer
state.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from titans._logging import setup_logging
from titans.checkpoint import load_checkpoint

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

_BANNER = "=" * 78
_RULE = "-" * 78


def _section(title: str) -> None:
    """Print a section banner."""
    print(f"\n[{title}]")
    print(_RULE)


def _component_of(param_name: str) -> str:
    """Map a fully-qualified param name to a coarse architectural bucket.

    The model state_dict groups by block — we further bucket inside each
    block so a single line of the report covers attention vs ffn vs
    norm vs memory etc.
    """
    sub = param_name.split(".", 2)[-1]
    head = sub.split(".")[0]
    if head.startswith("norm"):
        return "norm"
    if head in {"attention", "memory", "ffn", "persistent"}:
        return head
    if head.startswith("attn_res") or head.startswith("mca"):
        return "attn_res/mca"
    return head


def analyze_weights(model_state_dict: dict, n_blocks: int) -> None:
    """[1] Per-block weight norms by component."""
    _section("1] WEIGHT NORMS BY BLOCK / COMPONENT")
    print("    Stable across depth = healthy. Dramatic drift = potential issue.\n")
    print(
        f"    {'block':<6} {'attention':>11} {'memory':>11} {'ffn':>11} "
        f"{'norm':>11} {'persistent':>11}"
    )

    def _avg(d: dict[str, list[float]], c: str) -> str:
        vals = d.get(c, [])
        if not vals:
            return f"{'-':>11}"
        return f"{sum(vals) / len(vals):>11.3f}"

    for blk_idx in range(n_blocks):
        prefix = f"blocks.{blk_idx}."
        by_comp: dict[str, list[float]] = defaultdict(list)
        for k, v in model_state_dict.items():
            if not k.startswith(prefix):
                continue
            if not isinstance(v, torch.Tensor) or not v.is_floating_point():
                continue
            by_comp[_component_of(k)].append(v.norm().item())

        print(
            f"    {blk_idx:<6} {_avg(by_comp, 'attention')} "
            f"{_avg(by_comp, 'memory')} {_avg(by_comp, 'ffn')} "
            f"{_avg(by_comp, 'norm')} {_avg(by_comp, 'persistent')}"
        )


def analyze_optimizer_state(osd: dict, names: list[str] | None) -> None:
    """[2] Per-block exp_avg_sq norm — gradient signal across depth.

    [5] Step-counter consistency check.
    """
    state = osd.get("state", {})
    if not state:
        print("\n  (no optimizer state in checkpoint — skipping sections 2 and 5)")
        return
    group_positions = osd.get("param_groups", [{}])[0].get("params", [])
    if names and len(names) != len(group_positions):
        names = None  # mismatch => fall back to position-only

    name_to_pos = (
        dict(zip(names, group_positions))
        if names
        else {f"_{i}": pos for i, pos in enumerate(group_positions)}
    )

    _section("2] EXP_AVG_SQ NORM BY BLOCK")
    print(
        "    Adam's 2nd-moment ≈ recent gradient magnitude.\n"
        "    Vanishing across depth ⇒ gradient flow problem.\n"
    )
    block_signal: dict[int, list[float]] = defaultdict(list)
    for name, pos in name_to_pos.items():
        if pos not in state:
            continue
        ea_sq = state[pos].get("exp_avg_sq")
        if ea_sq is None:
            continue
        if name.startswith("blocks."):
            blk = int(name.split(".")[1])
            block_signal[blk].append(ea_sq.norm().item())

    print(f"    {'block':<6} {'mean':>13} {'min':>13} {'max':>13} {'n_params':>9}")
    for blk in sorted(block_signal):
        vals = block_signal[blk]
        print(
            f"    {blk:<6} {sum(vals) / len(vals):>13.4e} "
            f"{min(vals):>13.4e} {max(vals):>13.4e} {len(vals):>9}"
        )

    _section("5] OPTIMIZER STEP-COUNTER CONSISTENCY")
    print(
        "    Two values expected post-resume from a legacy checkpoint:\n"
        "      * the high value = number of optimizer.step() calls in this run\n"
        "      * 0 = inner-loop NeuralLTM params (detached from outer-loop grad)\n"
    )
    steps = []
    for v in state.values():
        s = v.get("step")
        if s is not None:
            steps.append(s.item() if torch.is_tensor(s) else s)
    unique = sorted(set(steps))
    print(f"    unique step values across {len(steps)} state entries: {unique}")


def analyze_memory_dump(npz_path: Path) -> None:
    """[3] NeuralLTM memory state norms per block.

    Identifies blocks where the inner-loop memory is barely engaged —
    near-zero weight norms suggest the model learned to skip memory
    in those blocks.
    """
    _section("3] NEURALLTM MEMORY STATE NORMS")
    print(
        "    End-of-training inner-loop weights, per block.\n"
        "    Norms ~2-3 are typical; <1.0 means the block barely uses\n"
        "    its memory mechanism.\n"
    )
    mem = np.load(npz_path, allow_pickle=False)
    n_layers = int(mem["num_layers"][0])

    print(
        f"    {'block':<6} {'type':<6} {'#mem':<6} "
        f"{'weight_norms':<28} {'momentum_norms':<28}"
    )
    low_blocks: list[tuple[int, list[float]]] = []
    for li in range(n_layers):
        typ = int(mem[f"layer_{li}_type"][0])
        n_ml = int(mem[f"layer_{li}_num_memory_layers"][0])
        w_norms: list[float] = []
        m_norms: list[float] = []
        for k in range(n_ml):
            wk = f"layer_{li}_weight_{k}"
            mk = f"layer_{li}_momentum_{k}"
            if wk in mem.files:
                w_norms.append(float(np.linalg.norm(mem[wk])))
            if mk in mem.files:
                m_norms.append(float(np.linalg.norm(mem[mk])))
        if any(w < 1.0 for w in w_norms):
            low_blocks.append((li, w_norms))
        w_str = "[" + ", ".join(f"{x:.2f}" for x in w_norms) + "]"
        m_str = "[" + ", ".join(f"{x:.4f}" for x in m_norms) + "]"
        print(f"    {li:<6} {typ:<6} {n_ml:<6} {w_str:<28} {m_str:<28}")

    if low_blocks:
        print()
        print(f"    ⚠ {len(low_blocks)} block(s) have a memory weight with norm < 1.0:")
        for blk, ws in low_blocks:
            print(f"      block {blk}: {[f'{w:.2f}' for w in ws]}")
        print(
            "    Could be beneficial sparsity (model learned to skip\n"
            "    memory there) OR undertrained engagement. Confirm via\n"
            "    scripts/eval_memory_ablation.py."
        )


def analyze_dead_or_saturated(model_state_dict: dict) -> None:
    """[4] Tensors with std<1e-6 or |max|>50."""
    _section("4] DEAD / SATURATED PARAM CHECK")
    print(
        "    Flags tensors with very low variance (std<1e-6) or extreme\n"
        "    magnitudes (|max|>50). Empty list = healthy.\n"
    )
    suspicious: list[tuple[str, str, float]] = []
    for k, v in model_state_dict.items():
        if (
            not isinstance(v, torch.Tensor)
            or not v.is_floating_point()
            or v.numel() < 64
        ):
            continue
        std = v.std().item()
        if std < 1e-6:
            suspicious.append((k, "near-zero std", std))
        elif v.abs().max().item() > 50:
            suspicious.append((k, "large magnitude", v.abs().max().item()))
    print(f"    Tensors flagged: {len(suspicious)}")
    for k, kind, val in suspicious[:10]:
        print(f"      {k}: {kind} ({val:.3e})")
    if len(suspicious) > 10:
        print(f"      … {len(suspicious) - 10} more not shown")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a Titans checkpoint (.pt or .safetensors).",
    )
    parser.add_argument(
        "--memory-dump",
        type=Path,
        help=(
            "Path to a memory_*.npz produced by save_memory_states. "
            "Optional — section [3] runs only when this is provided."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.memory_dump:
        parser.error("Pass --checkpoint or --memory-dump (or both)")

    if args.checkpoint:
        ck = load_checkpoint(str(args.checkpoint), weights_only=False)
        print(_BANNER)
        print(
            f"CHECKPOINT  {args.checkpoint}  step={ck.get('step', '?')}  "
            f"schema={ck.get('titans_schema_version', '?')}"
        )
        msd = ck.get("model", {})
        osd = ck.get("optimizer", {})
        names = ck.get("optimizer_param_names")
        print(
            f"  model entries: {len(msd)},  optimizer state entries: "
            f"{len(osd.get('state', {}))},  param_names: "
            f"{'present' if names else 'absent (legacy)'}"
        )
        print(_BANNER)

        # Detect block count from the model state dict.
        block_indices = {
            int(k.split(".")[1])
            for k in msd
            if k.startswith("blocks.") and k.split(".")[1].isdigit()
        }
        n_blocks = max(block_indices) + 1 if block_indices else 0

        analyze_weights(msd, n_blocks)
        analyze_optimizer_state(osd, names)
        analyze_dead_or_saturated(msd)

    if args.memory_dump:
        if not args.memory_dump.exists():
            logger.error(f"memory dump not found: {args.memory_dump}")
            sys.exit(1)
        analyze_memory_dump(args.memory_dump)

    print()


if __name__ == "__main__":
    main()
