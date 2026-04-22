"""Regression: --help text mentioned a stale commit SHA (e309d70) that no
longer matched the default. Derive the help text from the default
at runtime."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def test_launch_pretraining_help_sha_matches_default() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "launch_pretraining_job.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout

    # Read the default SHA from the module itself.
    src = script.read_text()
    match = re.search(
        r'DEFAULT_TITANS_SHA\s*=\s*["\']([^"\']+)["\']',
        src,
    )
    assert match, "Expected DEFAULT_TITANS_SHA constant in launcher"
    default_sha = match.group(1)
    assert default_sha in help_text, (
        f"--help output does not reference the actual default SHA "
        f"{default_sha!r}; someone bumped the default and forgot the help "
        f"text."
    )
