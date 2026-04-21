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


class TestLoadCheckpoint:
    """Tests for load_checkpoint with auto-detection."""

    def test_load_pt(self, tmp_path: Path) -> None:
        """Load a .pt checkpoint by explicit path."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {"weight": torch.randn(4, 4)}
        metadata = {"step": 50}
        stem = tmp_path / "ckpt"
        save_checkpoint(state_dict, stem, format="pt", metadata=metadata)

        result = load_checkpoint(stem.with_suffix(".pt"))

        assert "model" in result
        assert torch.equal(result["model"]["weight"], state_dict["weight"])
        assert result["step"] == 50

    def test_load_safetensors_with_sidecar(self, tmp_path: Path) -> None:
        """Load a .safetensors checkpoint that has a .meta.pt sidecar."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {"weight": torch.randn(4, 4)}
        metadata = {"step": 75, "lr": 1e-4}
        stem = tmp_path / "ckpt"
        save_checkpoint(
            state_dict, stem, format="safetensors", metadata=metadata
        )

        result = load_checkpoint(stem.with_suffix(".safetensors"))

        assert torch.equal(result["model"]["weight"], state_dict["weight"])
        assert result["step"] == 75
        assert result["lr"] == 1e-4

    def test_load_safetensors_no_sidecar(self, tmp_path: Path) -> None:
        """Load a .safetensors checkpoint without metadata sidecar."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {"weight": torch.randn(4, 4)}
        stem = tmp_path / "ckpt"
        save_checkpoint(state_dict, stem, format="safetensors")

        # Remove sidecar if it exists (it shouldn't, but be safe)
        sidecar = stem.with_suffix(".meta.pt")
        if sidecar.exists():
            sidecar.unlink()

        result = load_checkpoint(stem.with_suffix(".safetensors"))

        assert "model" in result
        assert torch.equal(result["model"]["weight"], state_dict["weight"])
        assert set(result.keys()) == {"model"}

    def test_extensionless_prefers_safetensors(self, tmp_path: Path) -> None:
        """Extensionless path loads .safetensors when both formats exist."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        st_dict_sf = {"weight": torch.ones(2, 2)}
        st_dict_pt = {"weight": torch.zeros(2, 2)}
        stem = tmp_path / "ckpt"

        save_checkpoint(st_dict_sf, stem, format="safetensors")
        save_checkpoint(st_dict_pt, stem, format="pt")

        result = load_checkpoint(stem)

        # Should load safetensors (ones), not pt (zeros)
        assert torch.equal(
            result["model"]["weight"], torch.ones(2, 2)
        )

    def test_extensionless_falls_back_to_pt(self, tmp_path: Path) -> None:
        """Extensionless path falls back to .pt when no .safetensors exists."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {"weight": torch.randn(2, 2)}
        stem = tmp_path / "ckpt"
        save_checkpoint(state_dict, stem, format="pt")

        result = load_checkpoint(stem)

        assert torch.equal(result["model"]["weight"], state_dict["weight"])

    def test_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent checkpoint raises FileNotFoundError."""
        from titans.checkpoint import load_checkpoint

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            load_checkpoint(tmp_path / "missing")


class TestRoundTrip:
    """Round-trip save/load tests across formats."""

    @pytest.mark.parametrize("fmt", ["pt", "safetensors"])
    def test_round_trip(self, tmp_path: Path, fmt: str) -> None:
        """State dict survives a save/load round-trip for each format."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {
            "layer.0.weight": torch.randn(8, 8),
            "layer.0.bias": torch.randn(8),
            "layer.1.weight": torch.randn(4, 8),
        }
        metadata = {"step": 999, "config": {"dim": 8}}
        stem = tmp_path / "model"

        written = save_checkpoint(
            state_dict, stem, format=fmt, metadata=metadata
        )
        result = load_checkpoint(written[0])

        assert set(result["model"].keys()) == set(state_dict.keys())
        for key in state_dict:
            assert torch.equal(result["model"][key], state_dict[key]), (
                f"Mismatch on {key}"
            )
        assert result["step"] == 999

    def test_pt_to_safetensors_conversion(self, tmp_path: Path) -> None:
        """Load a pt checkpoint and re-save as safetensors."""
        from titans.checkpoint import load_checkpoint, save_checkpoint

        state_dict = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        metadata = {"step": 10}
        stem_pt = tmp_path / "orig"
        stem_sf = tmp_path / "converted"

        save_checkpoint(state_dict, stem_pt, format="pt", metadata=metadata)
        loaded = load_checkpoint(stem_pt.with_suffix(".pt"))

        save_checkpoint(
            loaded["model"],
            stem_sf,
            format="safetensors",
            metadata={"step": loaded["step"]},
        )
        result = load_checkpoint(stem_sf.with_suffix(".safetensors"))

        assert torch.equal(result["model"]["weight"], state_dict["weight"])
        assert torch.equal(result["model"]["bias"], state_dict["bias"])
        assert result["step"] == 10


def _write_partial_bytes(f, payload: bytes = b"GARBAGE-PARTIAL-BYTES") -> None:
    """Helper: write partial bytes to `f`, which may be a Path, str, or file obj.

    Simulates a crash *after* bytes hit disk but *before* the write completed.
    """
    from pathlib import Path as _Path

    if hasattr(f, "write"):
        f.write(payload)
    elif isinstance(f, (str, _Path)):
        _Path(f).write_bytes(payload)
    else:
        raise TypeError(f"Unexpected target type for partial write: {type(f)!r}")


def test_save_checkpoint_is_atomic_on_crash(tmp_path, monkeypatch):
    """A crash mid-save must leave the final .pt file absent and no .tmp
    file dangling. torch.save may receive either a Path/str or a file object
    depending on the PyTorch version, so the stub handles both shapes."""
    import torch
    from titans.checkpoint import save_checkpoint

    state = {"w": torch.ones(4, 4)}
    final_path = tmp_path / "ckpt"

    def exploding_save(obj, f, *args, **kwargs):
        # Actually write partial bytes to wherever save was told to write,
        # then raise. This is what a real crash / disk-full looks like.
        _write_partial_bytes(f)
        raise RuntimeError("simulated crash mid-save")

    monkeypatch.setattr(torch, "save", exploding_save)

    with pytest.raises(RuntimeError, match="simulated crash"):
        save_checkpoint(state, final_path, format="pt")

    pt_path = final_path.with_suffix(".pt")
    tmp_pt = pt_path.with_suffix(".pt.tmp")

    # Post-fix invariant 1: the final .pt file MUST NOT exist — a partial
    # file at the final path is exactly the bug we're fixing. (Pre-fix code
    # wrote straight to pt_path, so it would contain GARBAGE-PARTIAL-BYTES.)
    assert not pt_path.exists(), (
        f"Final path {pt_path} contains partial file — atomic write "
        "invariant broken"
    )
    # Post-fix invariant 2: no .pt.tmp file should be left dangling.
    assert not tmp_pt.exists(), (
        f"Dangling tmp file {tmp_pt} was not cleaned up after crash"
    )


def test_save_checkpoint_safetensors_is_atomic_on_crash(tmp_path, monkeypatch):
    """Same atomic-write invariant for the safetensors code path: a crash
    mid-save must leave the final .safetensors file absent and no
    .safetensors.tmp file dangling."""
    import safetensors.torch as st_torch
    from titans.checkpoint import save_checkpoint

    state = {"w": torch.ones(4, 4)}
    final_path = tmp_path / "ckpt"

    def exploding_save_file(tensors, f, *args, **kwargs):
        # save_file takes a filename string; write partial bytes to it then raise.
        _write_partial_bytes(f)
        raise RuntimeError("simulated crash mid-safetensors-save")

    monkeypatch.setattr(st_torch, "save_file", exploding_save_file)

    with pytest.raises(RuntimeError, match="simulated crash mid-safetensors-save"):
        save_checkpoint(state, final_path, format="safetensors")

    sf_path = final_path.with_suffix(".safetensors")
    tmp_sf = sf_path.with_suffix(".safetensors.tmp")

    # Post-fix invariant 1: the final .safetensors file MUST NOT exist.
    assert not sf_path.exists(), (
        f"Final path {sf_path} contains partial file — atomic write "
        "invariant broken"
    )
    # Post-fix invariant 2: no .safetensors.tmp file should be left dangling.
    assert not tmp_sf.exists(), (
        f"Dangling tmp file {tmp_sf} was not cleaned up after crash"
    )


def test_atomic_write_tmp_filenames_are_correct(tmp_path):
    """The tmp-file names used during atomic writes must not double-suffix —
    e.g. 'ckpt.meta.pt.tmp', not 'ckpt.meta.meta.pt.tmp'."""
    import torch
    from titans.checkpoint import save_checkpoint

    # Capture the paths torch.save and save_file see during a successful save.
    seen_paths: list[str] = []

    original_torch_save = torch.save
    def spy_torch_save(obj, f, *args, **kwargs):
        seen_paths.append(str(f))
        return original_torch_save(obj, f, *args, **kwargs)

    from safetensors import torch as st
    original_save_file = st.save_file
    def spy_save_file(tensors, filename, *args, **kwargs):
        seen_paths.append(str(filename))
        return original_save_file(tensors, filename, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(torch, "save", spy_torch_save), \
         mock.patch.object(st, "save_file", spy_save_file):
        save_checkpoint({"w": torch.ones(2, 2)}, tmp_path / "ckpt", format="pt")
        save_checkpoint(
            {"w": torch.ones(2, 2)},
            tmp_path / "ckpt_sf",
            format="safetensors",
            metadata={"step": 1},
        )

    # Every intermediate path observed by the serializers must end in exactly
    # one ".tmp" and have no duplicated stem components.
    tmp_paths = [p for p in seen_paths if p.endswith(".tmp")]
    assert tmp_paths, "Expected at least one .tmp write to be observed"
    for p in tmp_paths:
        # No doubled '.meta' anywhere.
        assert ".meta.meta." not in p, f"Malformed tmp path: {p}"
        # Must end in ".tmp" once (not ".tmp.tmp" etc.)
        assert p.count(".tmp") == 1, f"Malformed tmp suffix in: {p}"


def test_save_safetensors_rejects_non_tensor_values(tmp_path):
    """QuantizedMemoryState fields aren't tensors; save_checkpoint in
    safetensors format must raise a clear TypeError instead of a cryptic
    AttributeError on .data_ptr()."""
    from titans.checkpoint import save_checkpoint

    # A non-tensor value masquerading inside a state_dict-like payload.
    bad = {"layer.0.quantized": [1, 2, 3]}
    path = tmp_path / "bad"
    with pytest.raises(TypeError, match="tensor"):
        save_checkpoint(bad, path, format="safetensors")
