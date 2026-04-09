#!/usr/bin/env python3
"""
Launch a Titans pretraining job on HuggingFace Jobs.

Usage:
    # Small test run (A10G, 100 steps, ~$1)
    uv run python scripts/launch_hf_job.py --test

    # Full training run (A100, 10K steps)
    uv run python scripts/launch_hf_job.py --flavor a100-large --timeout 8h

    # Resume a previous run and extend to 50K steps
    uv run python scripts/launch_hf_job.py --resume checkpoints/final.pt --max-steps 50000

    # Custom model with proportional RoPE
    uv run python scripts/launch_hf_job.py --dim 512 --num-layers 12 --rope-proportion 0.25
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token


def _apply_override(script: str, const: str, value: object) -> str:
    """Replace a constant assignment in the script via regex.

    Handles int, float, str, bool, and None values.
    """
    if isinstance(value, bool):
        pattern = rf"{const} = (?:True|False)"
        replacement = f"{const} = {value}"
    elif isinstance(value, str):
        pattern = rf'{const} = "[^"]*"'
        replacement = f'{const} = "{value}"'
    elif isinstance(value, (int, float)):
        pattern = rf"{const} = [\d.]+"
        replacement = f"{const} = {value}"
    else:
        return script
    return re.sub(pattern, replacement, script)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Titans training on HF Jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Job control ---
    job = parser.add_argument_group("Job control")
    job.add_argument(
        "--flavor", type=str, default="a10g-large",
        help="Hardware flavor (e.g. a10g-large, a10g-small, a100-large)",
    )
    job.add_argument("--timeout", type=str, default="4h", help="Job timeout")
    job.add_argument(
        "--test", action="store_true",
        help="Quick test run: 100 steps, smaller model, cheaper hardware",
    )

    # --- Model architecture ---
    model = parser.add_argument_group("Model architecture")
    model.add_argument("--dim", type=int, default=None, help="Override DIM")
    model.add_argument("--num-heads", type=int, default=None, help="Override NUM_HEADS")
    model.add_argument("--num-layers", type=int, default=None, help="Override NUM_LAYERS")
    model.add_argument("--vocab-size", type=int, default=None, help="Override VOCAB_SIZE")
    model.add_argument("--chunk-size", type=int, default=None, help="Override CHUNK_SIZE")
    model.add_argument(
        "--num-memory-layers", type=int, default=None,
        help="Override NUM_MEMORY_LAYERS",
    )
    model.add_argument(
        "--num-persistent-tokens", type=int, default=None,
        help="Override NUM_PERSISTENT_TOKENS",
    )
    model.add_argument(
        "--rope-proportion", type=float, default=None,
        help="Override ROPE_PROPORTION (0.0-1.0)",
    )

    # --- Data ---
    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", type=str, default=None, help="Override DATASET_NAME")
    data.add_argument(
        "--dataset-subset", type=str, default=None, help="Override DATASET_SUBSET",
    )
    data.add_argument("--tokenizer", type=str, default=None, help="Override TOKENIZER_NAME")
    data.add_argument("--seq-len", type=int, default=None, help="Override SEQ_LEN")

    # --- Training ---
    train = parser.add_argument_group("Training")
    train.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    train.add_argument(
        "--gradient-accumulation-steps", type=int, default=None,
        help="Override GRADIENT_ACCUMULATION_STEPS",
    )
    train.add_argument("--lr", type=float, default=None, help="Override LR")
    train.add_argument("--weight-decay", type=float, default=None, help="Override WEIGHT_DECAY")
    train.add_argument("--grad-clip", type=float, default=None, help="Override GRAD_CLIP")
    train.add_argument("--warmup-ratio", type=float, default=None, help="Override WARMUP_RATIO")
    train.add_argument("--max-steps", type=int, default=None, help="Override MAX_STEPS")
    train.add_argument("--log-every", type=int, default=None, help="Override LOG_EVERY")
    train.add_argument("--save-every", type=int, default=None, help="Override SAVE_EVERY")
    train.add_argument(
        "--save-format", type=str, default=None, choices=["pt", "safetensors"],
        help="Override SAVE_FORMAT",
    )
    train.add_argument(
        "--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Override MIXED_PRECISION",
    )
    train.add_argument(
        "--reset-global-state", type=str, default=None,
        choices=["true", "false"],
        help="Override RESET_GLOBAL_STATE_PER_BATCH (default: true)",
    )

    # --- Hub / checkpointing ---
    hub = parser.add_argument_group("Hub / checkpointing")
    hub.add_argument("--hub-repo", type=str, default=None, help="Override HUB_REPO")
    hub.add_argument(
        "--no-push", action="store_true",
        help="Disable pushing checkpoints to Hub",
    )
    hub.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help="Resume from a Hub checkpoint, e.g. 'checkpoints/final.pt'",
    )

    # --- Diagnostics ---
    diag = parser.add_argument_group("Diagnostics")
    diag.add_argument(
        "--titans-sha",
        type=str,
        default="51db2fd",
        help=(
            "Pin the titans package to a specific git commit SHA, branch, or "
            "tag. The launcher injects this into the script's `titans @ "
            "git+https://...` dependency line so uv re-resolves and "
            "reinstalls when the SHA changes (uv caches environments by "
            "content hash, and an unpinned dependency line hashes the same "
            "across runs even when origin/main has moved). Pass a short SHA "
            "to force a fresh install of a specific commit, or 'main' to "
            "track the branch tip. Default is the latest known good commit "
            "(e309d70, tip of the chunk-activation-checkpointing fix)."
        ),
    )
    diag.add_argument(
        "--profile-memory",
        action="store_true",
        help=(
            "Enable high-level CUDA memory profiling in hf_pretrain.py. Logs "
            "torch.cuda.max_memory_allocated() at key points (model build, "
            "first batch load, before/after forward, after backward, after "
            "optimizer step). Adds minor overhead from synchronize() calls; "
            "disable for production training runs. Use "
            "--profile-memory-per-block to also get the per-block trace "
            "inside each forward pass (much noisier, only useful for "
            "localizing O(num_chunks) blowups and similar diagnostic work)."
        ),
    )
    diag.add_argument(
        "--profile-memory-per-block",
        action="store_true",
        help=(
            "Enable the per-block memory trace inside each forward pass "
            "(emits num_blocks * num_chunks lines per step). OFF by default "
            "even under --profile-memory because it's noisy. Requires "
            "--profile-memory to also be passed. Use only when diagnosing "
            "per-block memory growth."
        ),
    )
    # --- Misc ---
    parser.add_argument("--seed", type=int, default=None, help="Override SEED")

    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError(
            "Not authenticated with HuggingFace. Run: huggingface-cli login"
        )

    # Read the training script
    script_path = Path(__file__).parent / "hf_pretrain.py"
    script = script_path.read_text()

    # Inject the titans package SHA pin into the uv script header. The default
    # dependency line is unpinned ("titans @ git+...Titans-Pytorch.git"), which
    # causes uv to cache the resolved environment by a hash that does not
    # change when origin/main moves — meaning successive runs may silently
    # reuse a stale install. Substituting in a SHA forces uv to recognize a
    # new dependency hash and reinstall on the first run that uses it.
    sha_pattern = r'"titans @ git\+https://github\.com/Taderich73/Titans-Pytorch\.git(?:@[^"]+)?"'
    sha_replacement = (
        f'"titans @ git+https://github.com/Taderich73/Titans-Pytorch.git@{args.titans_sha}"'
    )
    new_script, sub_count = re.subn(sha_pattern, sha_replacement, script)
    if sub_count == 0:
        raise RuntimeError(
            "Could not find titans dependency line in hf_pretrain.py to "
            "inject SHA pin. Has the dependency string format changed?"
        )
    if sub_count > 1:
        raise RuntimeError(
            f"Expected exactly one titans dependency line, found {sub_count}."
        )
    script = new_script
    print(f"  titans pin: {args.titans_sha}")

    # Inject memory profiling flag if requested. Toggles a PROFILE_MEMORY
    # constant in the training script that gates the cuda memory tracking
    # codepaths. See --profile-memory in the diagnostics arg group. Uses a
    # regex-anchored MULTILINE substitution (rather than plain str.replace)
    # so it does not accidentally flip PROFILE_MEMORY_PER_BLOCK = False on
    # the way past.
    if args.profile_memory:
        new_script, n = re.subn(
            r"^PROFILE_MEMORY\s*=\s*False",
            "PROFILE_MEMORY = True",
            script,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            raise RuntimeError(
                "Could not find PROFILE_MEMORY = False in hf_pretrain.py to "
                "inject --profile-memory override. Has the constant been "
                "renamed or already set to True upstream?"
            )
        script = new_script
        print("  Memory profiling: enabled")

    # Inject the per-block memory trace flag if requested. Separate from
    # --profile-memory because the per-block trace is noisy (num_blocks *
    # num_chunks lines per step) and only useful for diagnostic work. Requires
    # --profile-memory to also be passed; we enforce this here rather than
    # relying on the script because a silent ignore is the wrong failure mode.
    if args.profile_memory_per_block:
        if not args.profile_memory:
            raise RuntimeError(
                "--profile-memory-per-block requires --profile-memory to "
                "also be passed. The per-block trace is gated on both "
                "constants in hf_pretrain.py."
            )
        new_script, n = re.subn(
            r"^PROFILE_MEMORY_PER_BLOCK\s*=\s*False",
            "PROFILE_MEMORY_PER_BLOCK = True",
            script,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            raise RuntimeError(
                "Could not find PROFILE_MEMORY_PER_BLOCK = False in "
                "hf_pretrain.py to inject --profile-memory-per-block override. "
                "Has the constant been renamed or already set to True upstream?"
            )
        script = new_script
        print("  Per-block memory trace: enabled")

    # For test mode, override the config constants
    if args.test:
        script = script.replace("DIM = 1024", "DIM = 128")
        script = script.replace("NUM_HEADS = 16", "NUM_HEADS = 4")
        script = script.replace("NUM_LAYERS = 16", "NUM_LAYERS = 4")
        script = script.replace("MAX_STEPS = 10000", "MAX_STEPS = 100")
        script = script.replace("SAVE_EVERY = 2500", "SAVE_EVERY = 50")
        script = script.replace("SEQ_LEN = 2048", "SEQ_LEN = 512")
        script = script.replace(
            'HUB_REPO = "FlatFootInternational/titans-mac-1B"',
            'HUB_REPO = "FlatFootInternational/titans-mac-test"',
        )
        # Disable expensive features for test
        script = script.replace("use_tnt=True", "use_tnt=False")
        script = script.replace("use_attn_res=True", "use_attn_res=False")
        script = script.replace("use_mca=True", "use_mca=False")
        script = script.replace("adaptive_window=True", "adaptive_window=False")
        script = script.replace('memory_objective="huber"', 'memory_objective="l2"')
        timeout = "30m"
        flavor = args.flavor if args.flavor != "a10g-large" else "a10g-small"
        print(f"TEST MODE: dim=128, 4 layers, 100 steps, {flavor}")
    else:
        timeout = args.timeout
        flavor = args.flavor
        print(f"FULL RUN: {flavor}")

    # Apply CLI overrides — each maps an argparse flag to a script constant.
    # Only non-None values are applied (so defaults in hf_pretrain.py are kept).
    overrides: list[tuple[str, str, object | None]] = [
        # Model
        ("DIM", "--dim", args.dim),
        ("NUM_HEADS", "--num-heads", args.num_heads),
        ("NUM_LAYERS", "--num-layers", args.num_layers),
        ("VOCAB_SIZE", "--vocab-size", args.vocab_size),
        ("CHUNK_SIZE", "--chunk-size", args.chunk_size),
        ("NUM_MEMORY_LAYERS", "--num-memory-layers", args.num_memory_layers),
        ("NUM_PERSISTENT_TOKENS", "--num-persistent-tokens", args.num_persistent_tokens),
        ("ROPE_PROPORTION", "--rope-proportion", args.rope_proportion),
        # Data
        ("DATASET_NAME", "--dataset", args.dataset),
        ("DATASET_SUBSET", "--dataset-subset", args.dataset_subset),
        ("TOKENIZER_NAME", "--tokenizer", args.tokenizer),
        ("SEQ_LEN", "--seq-len", args.seq_len),
        # Training
        ("BATCH_SIZE", "--batch-size", args.batch_size),
        (
            "GRADIENT_ACCUMULATION_STEPS",
            "--gradient-accumulation-steps",
            args.gradient_accumulation_steps,
        ),
        ("LR", "--lr", args.lr),
        ("WEIGHT_DECAY", "--weight-decay", args.weight_decay),
        ("GRAD_CLIP", "--grad-clip", args.grad_clip),
        ("WARMUP_RATIO", "--warmup-ratio", args.warmup_ratio),
        ("MAX_STEPS", "--max-steps", args.max_steps),
        ("LOG_EVERY", "--log-every", args.log_every),
        ("SAVE_EVERY", "--save-every", args.save_every),
        ("SAVE_FORMAT", "--save-format", args.save_format),
        ("MIXED_PRECISION", "--mixed-precision", args.mixed_precision),
        # Hub
        ("HUB_REPO", "--hub-repo", args.hub_repo),
        # Misc
        ("SEED", "--seed", args.seed),
    ]

    for const, flag, value in overrides:
        if value is not None:
            script = _apply_override(script, const, value)
            print(f"  {flag}: {value}")

    if args.reset_global_state is not None:
        val = args.reset_global_state.lower() == "true"
        script = _apply_override(script, "RESET_GLOBAL_STATE_PER_BATCH", val)
        print(f"  --reset-global-state: {val}")

    # Apply --no-push flag
    if args.no_push:
        script = _apply_override(script, "PUSH_CHECKPOINTS", False)
        print("  Push to Hub: disabled")

    # Apply --resume flag
    if args.resume:
        script = script.replace(
            "RESUME_FROM = None",
            f'RESUME_FROM = "{args.resume}"',
        )
        print(f"  Resuming from: {args.resume}")

    api = HfApi(token=token)

    # Self-verification: print the lines that the launcher's regex/string
    # substitutions touched, so we can VISUALLY confirm they were applied
    # before the job is submitted. After several silent-failure incidents
    # (uv caching the wrong version, --profile-memory not propagating, etc.)
    # the cheap insurance of seeing exactly what's about to ship is well
    # worth a few extra lines of stdout.
    print("\nLauncher substitution check (these will be in the submitted script):")
    saw_dep = False
    saw_profile = False
    saw_per_block = False
    for i, line in enumerate(script.splitlines(), 1):
        stripped = line.strip()
        if "titans @ git" in stripped and "Titans-Pytorch.git" in stripped:
            print(f"  [dep]       line {i:4d}: {stripped}")
            saw_dep = True
        elif stripped.startswith("PROFILE_MEMORY_PER_BLOCK ="):
            print(f"  [per-block] line {i:4d}: {stripped}")
            saw_per_block = True
        elif stripped.startswith("PROFILE_MEMORY ="):
            print(f"  [profile]   line {i:4d}: {stripped}")
            saw_profile = True
    if not saw_dep:
        raise RuntimeError(
            "Self-check FAILED: titans dependency line not found in "
            "submitted script. The SHA pin substitution did not happen "
            "and uv may install the wrong version. Aborting."
        )
    if not saw_profile:
        raise RuntimeError(
            "Self-check FAILED: PROFILE_MEMORY constant not found in "
            "submitted script. The hf_pretrain.py file may have been "
            "modified upstream. Aborting."
        )
    if not saw_per_block:
        raise RuntimeError(
            "Self-check FAILED: PROFILE_MEMORY_PER_BLOCK constant not found "
            "in submitted script. The hf_pretrain.py file may have been "
            "modified upstream or the constant was renamed. Aborting."
        )

    print(f"\nSubmitting job to HF Jobs...")
    print(f"  Hardware: {flavor}")
    print(f"  Timeout: {timeout}")

    # Write modified script to a temp file — run_uv_job expects a file path,
    # not inline content (it tries to stat the string as a path)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(script)
    tmp.close()

    job = api.run_uv_job(
        script=tmp.name,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
    )

    Path(tmp.name).unlink(missing_ok=True)

    print(f"\nJob submitted!")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status.stage}")
    print(f"  URL: https://huggingface.co/jobs/{job.id}")
    print(
        f"\nMonitor with:\n"
        f"  uv run python -c \"from huggingface_hub import HfApi; "
        f"[print(l) for l in HfApi().fetch_job_logs(job_id='{job.id}')]\""
    )


if __name__ == "__main__":
    main()
