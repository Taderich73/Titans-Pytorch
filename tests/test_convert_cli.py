"""End-to-end tests for the unified ``scripts/convert.py`` CLI (Task P13).

Covers the three ``--to`` targets (pt, safetensors, hf) plus the two
deprecated shims (``convert_checkpoint.py``, ``convert_to_hf.py``) which
forward to the new entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from titans.checkpoint import save_checkpoint
from titans.config import TitansConfig
from titans.models import TitansMAC

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "convert.py"
CONVERT_CHECKPOINT_SHIM = REPO_ROOT / "scripts" / "convert_checkpoint.py"
CONVERT_TO_HF_SHIM = REPO_ROOT / "scripts" / "convert_to_hf.py"


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Write a tiny TitansMAC .pt checkpoint with embedded config metadata.

    The HF-emitter path requires a ``config`` or ``titans_config`` key in
    the checkpoint payload, so we save via ``save_checkpoint`` with the
    appropriate metadata rather than a bare state dict.
    """
    config = TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        vocab_size=100,
        chunk_size=16,
        window_size=16,
        num_memory_layers=1,
        num_persistent_tokens=2,
    )
    model = TitansMAC(config)
    ckpt_stem = tmp_path / "tiny_ckpt"
    save_checkpoint(
        model.state_dict(),
        ckpt_stem,
        format="pt",
        metadata={"config": config.to_dict(), "step": 0},
    )
    return tmp_path / "tiny_ckpt.pt"


def _run(
    args: list[str], script: Path = CONVERT_SCRIPT
) -> subprocess.CompletedProcess[str]:
    """Invoke a script via subprocess from the repo root."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Unified CLI -- one test per --to target
# ---------------------------------------------------------------------------


class TestConvertTargets:
    """One end-to-end subprocess run per --to target."""

    def test_to_safetensors(self, tiny_checkpoint: Path, tmp_path: Path) -> None:
        out_stem = tmp_path / "converted"
        r = _run(
            [str(tiny_checkpoint), "--to", "safetensors", "--output", str(out_stem)]
        )
        assert r.returncode == 0, (
            f"convert --to safetensors failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
        assert (tmp_path / "converted.safetensors").exists()

    def test_to_pt(self, tmp_path: Path) -> None:
        """Feed a .safetensors input so --to pt emits a new .pt file.

        ``convert.py`` toggles the output format when --to matches the
        input extension (matching the legacy convert_checkpoint.py
        behavior), so to force a .pt emission we start from
        .safetensors.
        """
        # Create a safetensors input via save_checkpoint.
        config = TitansConfig(
            dim=32,
            num_heads=2,
            num_layers=1,
            vocab_size=100,
            chunk_size=16,
            window_size=16,
            num_memory_layers=1,
            num_persistent_tokens=2,
        )
        model = TitansMAC(config)
        src_stem = tmp_path / "src"
        save_checkpoint(
            model.state_dict(),
            src_stem,
            format="safetensors",
            metadata={"config": config.to_dict()},
        )
        src = tmp_path / "src.safetensors"
        assert src.exists()

        out_stem = tmp_path / "converted"
        r = _run([str(src), "--to", "pt", "--output", str(out_stem)])
        assert r.returncode == 0, (
            f"convert --to pt failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
        assert (tmp_path / "converted.pt").exists()

    def test_to_hf(self, tiny_checkpoint: Path, tmp_path: Path) -> None:
        pytest.importorskip("transformers")
        out_dir = tmp_path / "hf_model"
        r = _run(
            [
                str(tiny_checkpoint),
                "--to",
                "hf",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert r.returncode == 0, (
            f"convert --to hf failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model.safetensors").exists()
        assert (out_dir / "generation_config.json").exists()
        data = json.loads((out_dir / "config.json").read_text())
        assert data["model_type"] == "titans-mac"
        assert data["dim"] == 32

    def test_to_hf_requires_output_dir(self, tiny_checkpoint: Path) -> None:
        r = _run([str(tiny_checkpoint), "--to", "hf"])
        assert r.returncode != 0
        assert "--output-dir" in (r.stderr + r.stdout)


# ---------------------------------------------------------------------------
# Deprecation shims
# ---------------------------------------------------------------------------


class TestDeprecationShims:
    """The legacy scripts must still work and emit a deprecation notice."""

    def test_convert_checkpoint_shim_forwards(
        self, tiny_checkpoint: Path, tmp_path: Path
    ) -> None:
        out_stem = tmp_path / "via_shim"
        r = _run(
            [str(tiny_checkpoint), "--output", str(out_stem)],
            script=CONVERT_CHECKPOINT_SHIM,
        )
        assert r.returncode == 0, (
            f"convert_checkpoint shim failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
        # Shim prints a deprecation notice to stderr.
        assert "deprecat" in r.stderr.lower()
        # Input was .pt -> legacy toggle -> .safetensors output.
        assert (tmp_path / "via_shim.safetensors").exists()

    def test_convert_to_hf_shim_forwards(
        self, tiny_checkpoint: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("transformers")
        out_dir = tmp_path / "hf_via_shim"
        r = _run(
            [
                "--checkpoint",
                str(tiny_checkpoint),
                "--output-dir",
                str(out_dir),
            ],
            script=CONVERT_TO_HF_SHIM,
        )
        assert r.returncode == 0, (
            f"convert_to_hf shim failed: stdout={r.stdout!r} stderr={r.stderr!r}"
        )
        assert "deprecat" in r.stderr.lower()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model.safetensors").exists()
