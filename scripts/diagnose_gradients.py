#!/usr/bin/env python3
"""Gradient flow diagnostic for TitansMAC.

Tests whether optimizer actually updates weights and whether loss changes.
Uses the actual training config (conv OFF, as hardcoded in pretrain.py:1303).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from titans_mlx import TitansConfig, TitansMAC


def classify_param(key: str) -> str:
    """Classify a parameter path into a human-readable group."""
    if key.startswith("head.") or key.startswith("embed."):
        return "embed/head"
    parts = key.split(".")
    if "blocks" not in parts:
        if "norm." in key:
            return "output_norm"
        return "other"
    try:
        block_idx = parts.index("blocks") + 1
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
        return f"memory.{sub}"
    elif component == "attention":
        return "attention"
    elif component == "ffn":
        return "ffn"
    elif component == "persistent":
        return "persistent"
    elif component.startswith("norm"):
        return "block_norms"
    return f"block.{component}"


def main() -> None:
    # Match ACTUAL training config (pretrain.py:1303 hardcodes use_conv=False)
    config = TitansConfig(
        dim=512,
        num_layers=16,
        num_heads=8,
        vocab_size=32000,
        chunk_size=512,
        num_memory_layers=2,
        use_conv=False,
    )

    print("=" * 70)
    print("WEIGHT UPDATE DIAGNOSTIC")
    print("=" * 70)
    print(f"Config: dim={config.dim}, layers={config.num_layers}, "
          f"use_conv={config.use_conv}")

    model = TitansMAC(config)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    # Create optimizer matching training config
    optimizer = optim.AdamW(
        learning_rate=4e-4,
        weight_decay=0.1,
        betas=[0.9, 0.95],
    )

    seq_len = 1024
    batch_size = 2

    # Snapshot initial weights
    print("\nSnapshotting initial weights...")
    initial_weights: dict[str, mx.array] = {}
    for key, val in tree_flatten(model.trainable_parameters()):
        initial_weights[key] = mx.array(val)
    mx.eval(initial_weights)

    # Loss function
    def loss_fn(m: nn.Module, ids: mx.array, labs: mx.array) -> mx.array:
        logits, _ = m(ids)
        return nn.losses.cross_entropy(
            logits.reshape(-1, config.vocab_size), labs.reshape(-1), reduction="mean"
        )

    # Run 5 optimizer steps
    print("\n--- Running 5 optimizer steps ---\n")
    for step in range(5):
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

        loss_and_grad_fn = nn.value_and_grad(
            model, lambda m: loss_fn(m, input_ids, labels)
        )
        loss, grads = loss_and_grad_fn(model)
        mx.eval(loss, grads)

        # Check gradient stats before applying
        total_grad_norm = 0.0
        total_nan = 0
        for _, g in tree_flatten(grads):
            g_f32 = g.astype(mx.float32)
            total_grad_norm += float(mx.sum(g_f32 * g_f32))
            total_nan += int(mx.sum(mx.isnan(g_f32)))
        total_grad_norm = total_grad_norm ** 0.5

        # Clip and apply (same as training)
        clipped = {}
        for k, g in grads.items():
            if isinstance(g, mx.array):
                g = mx.where(mx.isnan(g), mx.zeros_like(g), g)
                g = mx.clip(g, -10.0, 10.0)
                clipped[k] = g
            elif isinstance(g, dict):
                def sanitize_dict(d):
                    out = {}
                    for kk, vv in d.items():
                        if isinstance(vv, mx.array):
                            vv = mx.where(mx.isnan(vv), mx.zeros_like(vv), vv)
                            vv = mx.clip(vv, -10.0, 10.0)
                        elif isinstance(vv, dict):
                            vv = sanitize_dict(vv)
                        elif isinstance(vv, list):
                            vv = [mx.where(mx.isnan(x), mx.zeros_like(x), x)
                                  if isinstance(x, mx.array) else x for x in vv]
                            vv = [mx.clip(x, -10.0, 10.0)
                                  if isinstance(x, mx.array) else x for x in vv]
                        out[kk] = vv
                    return out
                clipped[k] = sanitize_dict(g)
            else:
                clipped[k] = g

        optimizer.update(model, clipped)
        mx.eval(model.parameters(), optimizer.state)

        print(f"  Step {step}: loss={float(loss):.6f}, grad_norm={total_grad_norm:.6f}, "
              f"nan_grads={total_nan}")

    # Compare weights after 5 steps
    print("\n--- Weight changes after 5 optimizer steps ---\n")
    changes_by_group: dict[str, dict] = {}
    for key, val in tree_flatten(model.trainable_parameters()):
        group = classify_param(key)
        if group not in changes_by_group:
            changes_by_group[group] = {"max_change": 0.0, "params": 0, "changed": 0}

        if key in initial_weights:
            diff = (val.astype(mx.float32) - initial_weights[key].astype(mx.float32))
            max_change = float(mx.max(mx.abs(diff)))
            changes_by_group[group]["max_change"] = max(
                changes_by_group[group]["max_change"], max_change
            )
            changes_by_group[group]["params"] += val.size
            if max_change > 0:
                changes_by_group[group]["changed"] += val.size

    print(f"{'Group':25s} | {'Max |Δw|':>12s} | {'Changed/Total':>20s} | Status")
    print("-" * 80)
    for group in sorted(changes_by_group.keys()):
        info = changes_by_group[group]
        changed = info["changed"]
        total = info["params"]
        max_c = info["max_change"]
        status = "UPDATED" if max_c > 0 else "FROZEN"
        print(f"{group:25s} | {max_c:12.6e} | {changed:>8,}/{total:>8,} | {status}")

    # Final loss on fresh data
    print("\n--- Final loss on new data ---")
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, _ = model(input_ids)
    mx.eval(logits)
    final_loss = nn.losses.cross_entropy(
        logits.reshape(-1, config.vocab_size), labels.reshape(-1), reduction="mean"
    )
    mx.eval(final_loss)
    print(f"  Loss after 5 steps: {float(final_loss):.6f}")
    print(f"  Expected random:    10.373300")
    if abs(float(final_loss) - 10.3733) < 0.01:
        print("  --> Loss unchanged from random. Model is NOT learning.")
    else:
        print(f"  --> Loss changed by {float(final_loss) - 10.3733:.4f}. "
              "Model IS updating.")


def test_conv_forward() -> None:
    """Verify conv ON produces correct shapes (no slicing bug)."""
    from titans_mlx.memory import NeuralLongTermMemory

    print("\n" + "=" * 70)
    print("CONV FORWARD PASS TEST")
    print("=" * 70)

    for num_mem_layers in [1, 2]:
        config = TitansConfig(
            dim=64,
            num_layers=1,
            num_heads=2,
            vocab_size=100,
            use_conv=True,
            conv_kernel_size=4,
            num_memory_layers=num_mem_layers,
        )
        mem = NeuralLongTermMemory(config)
        mx.eval(mem.parameters())

        x = mx.random.normal((2, 16, 64))
        out, state = mem(x)
        mx.eval(out, state.weights, state.momentum)

        assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"

        # Also test retrieve path
        retrieved = mem.retrieve(x, state)
        mx.eval(retrieved)
        assert retrieved.shape == (2, 16, 64), (
            f"Expected (2, 16, 64), got {retrieved.shape}"
        )

        print(f"  mem_layers={num_mem_layers}: __call__ shape={out.shape}, "
              f"retrieve shape={retrieved.shape} -- OK")

    print("  Conv forward pass tests PASSED")


if __name__ == "__main__":
    main()
    test_conv_forward()
