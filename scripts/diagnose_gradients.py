#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Gradient diagnostic tool for Titans PyTorch models.

Runs a small forward+backward pass and logs per-layer gradient norms,
memory state norms, and gate values. This is a diagnostic tool, not a
training script.

Usage:
    uv run python scripts/diagnose_gradients.py
    uv run python scripts/diagnose_gradients.py --model mag --dim 128 --num-layers 4
    uv run python scripts/diagnose_gradients.py --json
    uv run python scripts/diagnose_gradients.py --use-tnt --num-steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.memory import MemoryState


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

def classify_param(name: str) -> str:
    """Classify a named parameter into a diagnostic group.

    Args:
        name: Fully qualified parameter name from model.named_parameters().

    Returns:
        Human-readable group label.
    """
    if name.startswith("embed.") or name.startswith("head."):
        return "embed/head"

    parts = name.split(".")

    if "blocks" not in parts:
        if "norm" in name:
            return "output_norm"
        return "other"

    try:
        block_idx = parts.index("blocks") + 1
        # parts[block_idx] is the numeric index; parts[block_idx+1] is component
        component = parts[block_idx + 1] if len(parts) > block_idx + 1 else "unknown"
    except (ValueError, IndexError):
        return "other"

    if component == "memory":
        sub = parts[block_idx + 2] if len(parts) > block_idx + 2 else ""
        if sub == "memory":
            return "memory.mlp_layers"
        elif sub.startswith("proj_k") or sub.startswith("proj_v"):
            return "memory.proj_kv"
        elif sub.startswith("proj_q") or sub.startswith("proj_out"):
            return "memory.proj_q_out"
        elif sub.startswith("gate_"):
            return "memory.gates"
        elif sub.startswith("conv_"):
            return "memory.conv"
        elif sub.startswith("alpha") or sub.startswith("decay"):
            return "memory.gate_alpha"
        elif sub.startswith("theta") or sub.startswith("lr"):
            return "memory.gate_theta"
        elif sub.startswith("eta") or sub.startswith("momentum"):
            return "memory.gate_eta"
        return f"memory.{sub}"
    elif component == "attention":
        return "attention"
    elif component == "ffn":
        return "ffn"
    elif component == "persistent":
        return "persistent"
    elif component.startswith("norm"):
        return "block_norms"
    elif component == "mca":
        return "mca"
    elif component == "attn_res":
        return "attn_res"

    return f"block.{component}"


# ---------------------------------------------------------------------------
# Diagnostics collection
# ---------------------------------------------------------------------------

def _collect_memory_gate_values(model: torch.nn.Module) -> dict[str, list[float]]:
    """Extract gate parameter values (alpha/theta/eta) from memory modules.

    Args:
        model: The Titans model to inspect.

    Returns:
        Dict mapping gate name to list of scalar values per block.
    """
    gate_values: dict[str, list[float]] = defaultdict(list)

    for name, param in model.named_parameters():
        parts = name.split(".")
        if "memory" not in parts:
            continue

        # Look for named gate tensors
        leaf = parts[-1]
        if any(key in leaf for key in ("alpha", "decay", "gate_alpha", "gate_decay")):
            gate_values["alpha/decay"].append(float(param.data.mean().item()))
        elif any(key in leaf for key in ("theta", "lr", "gate_theta", "gate_lr")):
            gate_values["theta/lr"].append(float(param.data.mean().item()))
        elif any(key in leaf for key in ("eta", "momentum", "gate_eta", "gate_momentum")):
            gate_values["eta/momentum"].append(float(param.data.mean().item()))

    return dict(gate_values)


def _collect_memory_state_norms(states: list[MemoryState]) -> dict[str, list[float]]:
    """Compute per-layer weight and momentum norms from memory states.

    Args:
        states: List of MemoryState objects, one per memory-carrying block.

    Returns:
        Dict with 'weight_norms' and 'momentum_norms' lists.
    """
    weight_norms: list[float] = []
    momentum_norms: list[float] = []

    for state in states:
        # Handle TNTMemoryState which has a global_state attribute
        if hasattr(state, "global_state"):
            weights = state.global_state.weights
            momentum = state.global_state.momentum
        else:
            weights = state.weights
            momentum = state.momentum

        if weights:
            w_norm = float(
                torch.stack([w.float().norm() for w in weights]).mean().item()
            )
            weight_norms.append(w_norm)

        if momentum:
            m_norm = float(
                torch.stack([m.float().norm() for m in momentum]).mean().item()
            )
            momentum_norms.append(m_norm)

    return {"weight_norms": weight_norms, "momentum_norms": momentum_norms}


def diagnose(
    config_kwargs: dict,
    model_type: str = "mac",
    num_steps: int = 5,
    seq_len: int = 128,
    batch_size: int = 2,
) -> dict:
    """Run forward+backward passes and collect gradient diagnostics.

    Args:
        config_kwargs: Keyword arguments forwarded to TitansConfig.
        model_type: One of 'mac', 'mag', 'mal', 'lmm'.
        num_steps: Number of synthetic batches to run.
        seq_len: Sequence length for synthetic inputs.
        batch_size: Batch size for synthetic inputs.

    Returns:
        Diagnostics dict with keys: config, steps, grad_norms,
        memory_state_norms, gate_values, weight_changes.
    """
    config = TitansConfig(**config_kwargs)

    model_cls_map = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in model_cls_map:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose from {list(model_cls_map)}")

    model_cls = model_cls_map[model_type]
    model = model_cls(config)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Snapshot initial parameters for weight-change tracking
    initial_params: dict[str, torch.Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    step_records: list[dict] = []
    final_states: list[MemoryState] | None = None

    for step in range(num_steps):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        model.zero_grad()
        logits, states, _ = model(input_ids)

        # Cross-entropy loss: logits is (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        loss.backward()

        # Collect per-group gradient norms
        group_grad_sq: dict[str, float] = defaultdict(float)
        group_param_count: dict[str, int] = defaultdict(int)
        total_nan = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_f32 = param.grad.float()
            nan_count = int(torch.isnan(grad_f32).sum().item())
            total_nan += nan_count

            group = classify_param(name)
            group_grad_sq[group] += float((grad_f32 ** 2).sum().item())
            group_param_count[group] += param.grad.numel()

        group_grad_norms = {
            g: float(sq ** 0.5) for g, sq in group_grad_sq.items()
        }

        step_records.append({
            "step": step,
            "loss": float(loss.item()),
            "grad_norms_by_group": group_grad_norms,
            "total_grad_norm": float(sum(group_grad_sq.values()) ** 0.5),
            "nan_grads": total_nan,
            "param_counts": dict(group_param_count),
        })

        # Keep final states for memory inspection
        if states is not None:
            final_states = states if isinstance(states, list) else [states]

    # Weight-change analysis
    changes_by_group: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in initial_params:
            continue
        group = classify_param(name)
        if group not in changes_by_group:
            changes_by_group[group] = {"max_change": 0.0, "total_params": 0, "changed_params": 0}

        delta = (param.data.float() - initial_params[name].float()).abs()
        max_change = float(delta.max().item())
        changes_by_group[group]["max_change"] = max(
            changes_by_group[group]["max_change"], max_change
        )
        changes_by_group[group]["total_params"] += param.numel()
        if max_change > 0:
            changes_by_group[group]["changed_params"] += param.numel()

    # Memory state norms
    memory_state_norms: dict = {}
    if final_states:
        memory_state_norms = _collect_memory_state_norms(final_states)

    # Gate values
    gate_values = _collect_memory_gate_values(model)

    return {
        "config": {
            "model_type": model_type,
            "dim": config.dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            "use_tnt": config.use_tnt,
            "use_attn_res": config.use_attn_res,
            "use_mca": config.use_mca,
            "memory_objective": config.memory_objective,
            "device": str(device),
        },
        "steps": step_records,
        "weight_changes": changes_by_group,
        "memory_state_norms": memory_state_norms,
        "gate_values": gate_values,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_table(diagnostics: dict) -> str:
    """Render diagnostics as a formatted plaintext table.

    Args:
        diagnostics: Output dict from :func:`diagnose`.

    Returns:
        Multi-line string suitable for printing to stdout.
    """
    lines: list[str] = []
    cfg = diagnostics["config"]

    lines.append("=" * 75)
    lines.append("TITANS GRADIENT DIAGNOSTIC")
    lines.append("=" * 75)
    lines.append(
        f"Model: {cfg['model_type'].upper()}  dim={cfg['dim']}  "
        f"layers={cfg['num_layers']}  heads={cfg['num_heads']}  "
        f"vocab={cfg['vocab_size']}"
    )
    lines.append(
        f"Flags: use_tnt={cfg['use_tnt']}  use_attn_res={cfg['use_attn_res']}  "
        f"use_mca={cfg['use_mca']}  memory_objective={cfg['memory_objective']}"
    )
    lines.append(f"Device: {cfg['device']}")
    lines.append("")

    # Per-step summary
    lines.append("--- Training Steps ---")
    lines.append(f"{'Step':>4s}  {'Loss':>10s}  {'Grad Norm':>12s}  {'NaN Grads':>10s}")
    lines.append("-" * 45)
    for rec in diagnostics["steps"]:
        lines.append(
            f"{rec['step']:>4d}  {rec['loss']:>10.6f}  "
            f"{rec['total_grad_norm']:>12.6f}  {rec['nan_grads']:>10d}"
        )
    lines.append("")

    # Grad norms by group (from last step)
    if diagnostics["steps"]:
        last_step = diagnostics["steps"][-1]
        lines.append("--- Gradient Norms by Group (final step) ---")
        lines.append(f"{'Group':30s}  {'Grad Norm':>12s}  {'Num Params':>12s}")
        lines.append("-" * 60)
        group_norms = last_step["grad_norms_by_group"]
        param_counts = last_step["param_counts"]
        for group in sorted(group_norms.keys()):
            count = param_counts.get(group, 0)
            lines.append(f"{group:30s}  {group_norms[group]:>12.6e}  {count:>12,d}")
        lines.append("")

    # Weight changes
    changes = diagnostics["weight_changes"]
    if changes:
        lines.append(f"--- Weight Changes After {len(diagnostics['steps'])} Steps ---")
        lines.append(
            f"{'Group':30s}  {'Max |Δw|':>12s}  {'Changed/Total':>22s}  Status"
        )
        lines.append("-" * 85)
        for group in sorted(changes.keys()):
            info = changes[group]
            max_c = info["max_change"]
            changed = info["changed_params"]
            total = info["total_params"]
            status = "UPDATED" if max_c > 0 else "FROZEN"
            lines.append(
                f"{group:30s}  {max_c:>12.6e}  "
                f"{changed:>10,}/{total:>10,}  {status}"
            )
        lines.append("")

    # Memory state norms
    ms = diagnostics.get("memory_state_norms", {})
    if ms:
        lines.append("--- Memory State Norms (final step) ---")
        w_norms = ms.get("weight_norms", [])
        m_norms = ms.get("momentum_norms", [])
        lines.append(f"{'Layer':>6s}  {'Weight Norm':>14s}  {'Momentum Norm':>14s}")
        lines.append("-" * 42)
        for i, (wn, mn) in enumerate(zip(w_norms, m_norms)):
            lines.append(f"{i:>6d}  {wn:>14.6e}  {mn:>14.6e}")
        lines.append("")

    # Gate values
    gates = diagnostics.get("gate_values", {})
    if gates:
        lines.append("--- Gate Parameter Values (mean across blocks) ---")
        for gate_name, values in sorted(gates.items()):
            mean_val = sum(values) / len(values) if values else float("nan")
            per_block = "  ".join(f"{v:.4f}" for v in values)
            lines.append(f"  {gate_name:20s}: mean={mean_val:.4f}  per-block=[{per_block}]")
        lines.append("")

    lines.append("=" * 75)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient diagnostic tool for Titans PyTorch models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["mac", "mag", "mal", "lmm"],
        default="mac",
        help="Model variant to diagnose.",
    )

    # Architecture dimensions
    parser.add_argument("--dim", type=int, default=64, help="Model dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size.")
    parser.add_argument(
        "--num-memory-layers", type=int, default=2, help="Layers inside NeuralLTM MLP."
    )

    # Architecture flags
    parser.add_argument("--use-tnt", action="store_true", help="Enable TNT hierarchical memory.")
    parser.add_argument("--use-attn-res", action="store_true", help="Enable AttnRes gating.")
    parser.add_argument("--use-mca", action="store_true", help="Enable Memory Cross-Attention.")
    parser.add_argument(
        "--memory-objective",
        choices=["l2", "huber"],
        default="l2",
        help="Memory loss objective.",
    )
    parser.add_argument(
        "--rope-proportion", type=float, default=1.0,
        help="Fraction of head_dim pairs to apply RoPE to (0.0-1.0, default 1.0)",
    )

    # Diagnostic parameters
    parser.add_argument("--num-steps", type=int, default=5, help="Number of synthetic steps.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")

    # Output
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output diagnostics as JSON instead of formatted table.",
    )

    args = parser.parse_args()

    config_kwargs: dict = {
        "dim": args.dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "vocab_size": args.vocab_size,
        "num_memory_layers": args.num_memory_layers,
        "memory_objective": args.memory_objective,
        "use_tnt": args.use_tnt,
        "use_attn_res": args.use_attn_res,
        "use_mca": args.use_mca,
        "rope_proportion": args.rope_proportion,
    }

    try:
        diagnostics = diagnose(
            config_kwargs=config_kwargs,
            model_type=args.model,
            num_steps=args.num_steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

    if args.output_json:
        print(json.dumps(diagnostics, indent=2))
    else:
        print(format_table(diagnostics))


if __name__ == "__main__":
    main()
