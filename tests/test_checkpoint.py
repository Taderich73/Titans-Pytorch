"""Tests for the checkpoint save/load module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


class TestSaveCheckpointPt:
    """Tests for save_checkpoint with pt format."""

    def test_save_with_metadata(self, tmp_path: Path) -> None:
        """Saving pt format stores model weights and metadata keys."""
        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        metadata = {"step": 100, "loss": 0.5}
        stem = tmp_path / "ckpt"

        written = save_checkpoint(state_dict, stem, format="pt", metadata=metadata)

        assert len(written) == 1
        pt_path = stem.with_suffix(".pt")
        assert written[0] == pt_path
        assert pt_path.exists()

        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert "model" in loaded
        assert torch.equal(loaded["model"]["weight"], state_dict["weight"])
        assert torch.equal(loaded["model"]["bias"], state_dict["bias"])
        assert loaded["step"] == 100
        assert loaded["loss"] == 0.5

    def test_save_without_metadata(self, tmp_path: Path) -> None:
        """Saving pt format without metadata stores only model key."""
        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(4, 4)}
        stem = tmp_path / "ckpt"

        written = save_checkpoint(state_dict, stem, format="pt")

        pt_path = stem.with_suffix(".pt")
        assert written == [pt_path]
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert set(loaded.keys()) == {"model"}
        assert torch.equal(loaded["model"]["weight"], state_dict["weight"])

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_checkpoint creates intermediate directories."""
        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(2, 2)}
        stem = tmp_path / "deep" / "nested" / "ckpt"

        written = save_checkpoint(state_dict, stem, format="pt")

        assert written[0].exists()
        assert (tmp_path / "deep" / "nested").is_dir()


class TestSaveCheckpointSafetensors:
    """Tests for save_checkpoint with safetensors format."""

    def test_save_with_metadata(self, tmp_path: Path) -> None:
        """Safetensors save writes weights file and metadata sidecar."""
        from safetensors.torch import load_file

        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        metadata = {"step": 200, "config": {"dim": 64}}
        stem = tmp_path / "ckpt"

        written = save_checkpoint(
            state_dict, stem, format="safetensors", metadata=metadata
        )

        sf_path = stem.with_suffix(".safetensors")
        sidecar = stem.with_suffix(".meta.pt")
        assert len(written) == 2
        assert written[0] == sf_path
        assert written[1] == sidecar
        assert sf_path.exists()
        assert sidecar.exists()

        tensors = load_file(sf_path)
        assert torch.equal(tensors["weight"], state_dict["weight"])
        assert torch.equal(tensors["bias"], state_dict["bias"])

        meta = torch.load(sidecar, map_location="cpu", weights_only=False)
        assert meta["step"] == 200
        assert meta["config"]["dim"] == 64

    def test_save_without_metadata(self, tmp_path: Path) -> None:
        """Safetensors save without metadata writes only the weights file."""
        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(4, 4)}
        stem = tmp_path / "ckpt"

        written = save_checkpoint(state_dict, stem, format="safetensors")

        sf_path = stem.with_suffix(".safetensors")
        sidecar = stem.with_suffix(".meta.pt")
        assert written == [sf_path]
        assert sf_path.exists()
        assert not sidecar.exists()

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        """Unsupported format raises ValueError."""
        from titans.checkpoint import save_checkpoint

        state_dict = {"weight": torch.randn(2, 2)}
        stem = tmp_path / "ckpt"

        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            save_checkpoint(state_dict, stem, format="onnx")
