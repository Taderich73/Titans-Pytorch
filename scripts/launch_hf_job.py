#!/usr/bin/env python3
"""
Launch a Titans pretraining job on HuggingFace Jobs.

Usage:
    # Small test run (A10G, 100 steps, ~$1)
    uv run python scripts/launch_hf_job.py --test

    # Full training run (A10G, 10K steps, ~$15-20)
    uv run python scripts/launch_hf_job.py

    # Custom hardware
    uv run python scripts/launch_hf_job.py --flavor a100-large --timeout 8h
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
        script = script.replace("DIM = 512", "DIM = 128")
        script = script.replace("NUM_HEADS = 8", "NUM_HEADS = 4")
        script = script.replace("NUM_LAYERS = 12", "NUM_LAYERS = 4")
        script = script.replace("MAX_STEPS = 10000", "MAX_STEPS = 100")
        script = script.replace("SAVE_EVERY = 2500", "SAVE_EVERY = 50")
        script = script.replace("SEQ_LEN = 2048", "SEQ_LEN = 512")
        script = script.replace(
            'HUB_REPO = "FlatFootInternational/titans-mac-512"',
            'HUB_REPO = "FlatFootInternational/titans-mac-test"',
        )
        script = script.replace(
            'DATASET_SUBSET = "sample-10BT"',
            'DATASET_SUBSET = "sample-10BT"',
        )
        timeout = "30m"
        flavor = args.flavor if args.flavor != "a10g-large" else "a10g-small"
        print(f"TEST MODE: dim=128, 4 layers, 100 steps, {flavor}")
    else:
        timeout = args.timeout
        flavor = args.flavor
        print(f"FULL RUN: dim=512, 12 layers, 10K steps, {flavor}")

    api = HfApi(token=token)

    print(f"Submitting job to HF Jobs...")
    print(f"  Hardware: {flavor}")
    print(f"  Timeout: {timeout}")

    # Write modified script to a temp file — run_uv_job expects a file path,
    # not inline content (it tries to stat the string as a path)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(script)
    tmp.close()

    # Pass titans as a runtime dependency with token embedded in git URL
    # so UV can clone the private repo during environment setup
    titans_dep = f"titans @ git+https://hf_user:{token}@huggingface.co/FlatFootInternational/titans"

    job = api.run_uv_job(
        script=tmp.name,
        dependencies=[titans_dep],
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
