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

    # Resume from a specific checkpoint
    uv run python scripts/launch_hf_job.py --resume checkpoints/step_10000.pt --max-steps 50000
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main():
    parser = argparse.ArgumentParser(description="Launch Titans training on HF Jobs")
    parser.add_argument(
        "--flavor",
        type=str,
        default="a10g-large",
        help="Hardware flavor (default: a10g-large = 24GB A10G)",
    )
    parser.add_argument(
        "--timeout",
        type=str,
        default="4h",
        help="Job timeout (default: 4h)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run: 100 steps, synthetic data, smaller model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a Hub checkpoint, e.g. 'checkpoints/final.pt'",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override MAX_STEPS (useful when resuming, e.g. --max-steps 50000)",
    )
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError(
            "Not authenticated with HuggingFace. Run: huggingface-cli login"
        )

    # Read the training script
    script_path = Path(__file__).parent / "hf_pretrain.py"
    script = script_path.read_text()

    # For test mode, override the config constants
    if args.test:
        script = script.replace("DIM = 2048", "DIM = 128")
        script = script.replace("NUM_HEADS = 32", "NUM_HEADS = 4")
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
        print(f"FULL RUN: 1B params, 16 layers, {flavor}")

    # Apply --resume flag
    if args.resume:
        script = script.replace(
            "RESUME_FROM = None",
            f'RESUME_FROM = "{args.resume}"',
        )
        print(f"  Resuming from: {args.resume}")

    # Apply --max-steps override
    if args.max_steps is not None:
        import re
        script = re.sub(
            r"MAX_STEPS = \d+",
            f"MAX_STEPS = {args.max_steps}",
            script,
        )
        print(f"  Max steps: {args.max_steps}")

    api = HfApi(token=token)

    print(f"Submitting job to HF Jobs...")
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
    print(f"\nMonitor with:")
    print(f"  uv run python -c \"from huggingface_hub import HfApi; [print(l) for l in HfApi().fetch_job_logs(job_id='{job.id}')]\"")


if __name__ == "__main__":
    main()
