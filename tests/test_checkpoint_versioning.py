"""Tests for checkpoint schema versioning (Task P5).

Scenarios covered:

* ``test_same_version_roundtrip`` — save with current version, load, no
  warnings, round-trip values match.
* ``test_unversioned_loads_with_warning`` — save a dict lacking the
  version key; load emits a ``DeprecationWarning`` mentioning
  "unversioned" and still returns usable state.
* ``test_newer_than_code_refuses`` — a file tagged ``CURRENT + 1`` must
  raise ``RuntimeError`` mentioning "upgrade".
* ``test_older_with_migration`` — monkey-patch the migration registry,
  write a file with an older version, verify the migration runs and the
  load succeeds.
* ``test_older_without_migration`` — a file tagged ``CURRENT - 100`` (no
  migration registered) must raise ``RuntimeError`` mentioning "no
  migration".
* ``test_convert_checkpoint_emits_version`` — end-to-end run of
  ``scripts/convert_checkpoint.py`` on a synthetic tiny checkpoint;
  verifies the converted output carries the schema version.
* ``test_hf_config_carries_version`` — instantiating the HF config
  exposes ``titans_schema_version`` matching the package constant.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from titans import TITANS_SCHEMA_VERSION
from titans.checkpoint import load_checkpoint, save_checkpoint
from titans.memory import MemoryState
from titans.memory_dump import (
    _MIGRATIONS,
    _SCHEMA_VERSION_KEY,
    load_memory_states,
    save_memory_states,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_memory_state() -> MemoryState:
    """Minimal MemoryState for fast version-dispatch tests."""
    return MemoryState(
        weights=[torch.ones(2, 2)],
        momentum=[torch.zeros(2, 2)],
    )


def _write_raw_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an npz file directly with the given arrays (no version stamping)."""
    np.savez(str(path), **arrays)


# ---------------------------------------------------------------------------
# memory_dump (.npz) — version dispatch
# ---------------------------------------------------------------------------


class TestMemoryDumpVersioning:
    """Schema-version dispatch in titans.memory_dump."""

    def test_same_version_roundtrip(self, tmp_path: Path) -> None:
        """Round-trip emits and consumes the current schema with no warnings."""
        state = _tiny_memory_state()
        path = tmp_path / "mem.npz"
        save_memory_states([state], path)

        # The file on disk should carry the current schema version.
        with np.load(str(path)) as data:
            assert _SCHEMA_VERSION_KEY in data.files
            assert int(data[_SCHEMA_VERSION_KEY][0]) == TITANS_SCHEMA_VERSION

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_memory_states(path)

        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deps, (
            f"Unexpected DeprecationWarning on same-version load: "
            f"{[str(w.message) for w in deps]}"
        )

        assert len(loaded) == 1
        torch.testing.assert_close(loaded[0].weights[0], state.weights[0])
        torch.testing.assert_close(loaded[0].momentum[0], state.momentum[0])

    def test_unversioned_loads_with_warning(self, tmp_path: Path) -> None:
        """A file lacking ``titans_schema_version`` loads with a warning.

        We emulate a pre-0.7 file by writing the npz directly, omitting
        the schema-version key. The load path must warn via
        ``DeprecationWarning`` and still produce a valid MemoryState.
        """
        arrays: dict[str, np.ndarray] = {
            # Legacy unversioned layout: same keys save_memory_states
            # would emit for a single plain MemoryState, minus
            # ``titans_schema_version``.
            "num_layers": np.array([1]),
            "layer_0_type": np.array([0]),
            "layer_0_num_memory_layers": np.array([1]),
            "layer_0_weight_0": np.ones((2, 2), dtype=np.float32),
            "layer_0_momentum_0": np.zeros((2, 2), dtype=np.float32),
        }
        path = tmp_path / "legacy.npz"
        _write_raw_npz(path, arrays)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_memory_states(path)

        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "Expected a DeprecationWarning for the unversioned load"
        assert any("unversioned" in str(w.message).lower() for w in deps), (
            f"Warning must mention 'unversioned'; got {[str(w.message) for w in deps]}"
        )

        assert len(loaded) == 1
        torch.testing.assert_close(
            loaded[0].weights[0],
            torch.ones(2, 2),
        )

    def test_newer_than_code_refuses(self, tmp_path: Path) -> None:
        """A file stamped with a version newer than the code must refuse."""
        arrays: dict[str, np.ndarray] = {
            _SCHEMA_VERSION_KEY: np.array([TITANS_SCHEMA_VERSION + 1], dtype=np.int64),
            "num_layers": np.array([1]),
            "layer_0_type": np.array([0]),
            "layer_0_num_memory_layers": np.array([1]),
            "layer_0_weight_0": np.ones((2, 2), dtype=np.float32),
            "layer_0_momentum_0": np.zeros((2, 2), dtype=np.float32),
        }
        path = tmp_path / "future.npz"
        _write_raw_npz(path, arrays)

        with pytest.raises(RuntimeError, match="upgrade titans"):
            load_memory_states(path)

    def test_older_with_migration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered migration brings an older file up to current."""

        def _fake_v0_to_current(arrays: dict) -> dict:
            """Pretend the v0 layout used ``legacy_weight_0`` instead of
            ``layer_0_weight_0`` and that the load path expects the
            current key."""
            out = dict(arrays)
            out["layer_0_weight_0"] = out.pop("legacy_weight_0")
            out["layer_0_momentum_0"] = out.pop("legacy_momentum_0")
            # Add the standard layout metadata so the loader recognizes it.
            out["layer_0_num_memory_layers"] = np.array([1])
            return out

        # Register (0 -> CURRENT) only, so the dispatch jumps straight there.
        fake_registry = {(0, TITANS_SCHEMA_VERSION): _fake_v0_to_current}
        monkeypatch.setattr("titans.memory_dump._MIGRATIONS", fake_registry)

        arrays: dict[str, np.ndarray] = {
            _SCHEMA_VERSION_KEY: np.array([0], dtype=np.int64),
            "num_layers": np.array([1]),
            "layer_0_type": np.array([0]),
            "legacy_weight_0": np.ones((2, 2), dtype=np.float32) * 3,
            "legacy_momentum_0": np.zeros((2, 2), dtype=np.float32),
        }
        path = tmp_path / "v0.npz"
        _write_raw_npz(path, arrays)

        loaded = load_memory_states(path)
        assert len(loaded) == 1
        torch.testing.assert_close(
            loaded[0].weights[0],
            torch.full((2, 2), 3.0),
        )

    def test_older_without_migration(self, tmp_path: Path) -> None:
        """Older-than-current with no registered migration must refuse."""
        # A version far older than anything that could ever have a
        # registered migration in this release.
        old_version = TITANS_SCHEMA_VERSION - 100

        arrays: dict[str, np.ndarray] = {
            _SCHEMA_VERSION_KEY: np.array([old_version], dtype=np.int64),
            "num_layers": np.array([1]),
            "layer_0_type": np.array([0]),
            "layer_0_num_memory_layers": np.array([1]),
            "layer_0_weight_0": np.ones((2, 2), dtype=np.float32),
            "layer_0_momentum_0": np.zeros((2, 2), dtype=np.float32),
        }
        path = tmp_path / "ancient.npz"
        _write_raw_npz(path, arrays)

        with pytest.raises(RuntimeError, match="no migration available"):
            load_memory_states(path)


# ---------------------------------------------------------------------------
# checkpoint (.pt / .safetensors) — version dispatch
# ---------------------------------------------------------------------------


class TestCheckpointVersioning:
    """Schema-version dispatch in titans.checkpoint."""

    def test_save_stamps_version_in_pt(self, tmp_path: Path) -> None:
        """Every ``.pt`` written by save_checkpoint carries the version."""
        stem = tmp_path / "ck"
        save_checkpoint({"w": torch.ones(2, 2)}, stem, format="pt")
        loaded = torch.load(
            stem.with_suffix(".pt"), map_location="cpu", weights_only=False
        )
        assert loaded["titans_schema_version"] == TITANS_SCHEMA_VERSION

    def test_save_stamps_version_in_safetensors_sidecar(self, tmp_path: Path) -> None:
        """Every ``.safetensors`` write gets a sidecar carrying the version."""
        stem = tmp_path / "ck"
        save_checkpoint({"w": torch.ones(2, 2)}, stem, format="safetensors")
        sidecar = stem.with_suffix(".meta.pt")
        assert sidecar.exists(), "P5 always emits a sidecar for safetensors"
        meta = torch.load(sidecar, map_location="cpu", weights_only=False)
        assert meta["titans_schema_version"] == TITANS_SCHEMA_VERSION

    def test_same_version_load_is_silent(self, tmp_path: Path) -> None:
        """Round-trip at current version: no DeprecationWarning."""
        stem = tmp_path / "ck"
        save_checkpoint({"w": torch.ones(2, 2)}, stem, format="pt")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_checkpoint(stem.with_suffix(".pt"), weights_only=False)

        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deps, (
            f"Unexpected DeprecationWarning on same-version load: "
            f"{[str(w.message) for w in deps]}"
        )
        assert loaded["titans_schema_version"] == TITANS_SCHEMA_VERSION

    def test_unversioned_pt_loads_with_warning(self, tmp_path: Path) -> None:
        """A pre-P5 ``.pt`` file (no version key) loads with a warning."""
        stem = tmp_path / "legacy"
        pt_path = stem.with_suffix(".pt")
        # Write a legacy-shaped checkpoint directly — no version key.
        torch.save({"model": {"w": torch.ones(2, 2)}}, pt_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_checkpoint(pt_path, weights_only=False)

        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps and any("unversioned" in str(w.message).lower() for w in deps)
        assert torch.equal(loaded["model"]["w"], torch.ones(2, 2))

    def test_newer_version_pt_refuses(self, tmp_path: Path) -> None:
        """A .pt file tagged with a version newer than code must refuse."""
        stem = tmp_path / "future"
        pt_path = stem.with_suffix(".pt")
        torch.save(
            {
                "model": {"w": torch.ones(2, 2)},
                "titans_schema_version": TITANS_SCHEMA_VERSION + 1,
            },
            pt_path,
        )
        with pytest.raises(RuntimeError, match="upgrade titans"):
            load_checkpoint(pt_path, weights_only=False)

    def test_older_version_pt_refuses_without_migration(self, tmp_path: Path) -> None:
        """An older-than-current .pt with no migration registered refuses."""
        stem = tmp_path / "ancient"
        pt_path = stem.with_suffix(".pt")
        torch.save(
            {
                "model": {"w": torch.ones(2, 2)},
                "titans_schema_version": TITANS_SCHEMA_VERSION - 100,
            },
            pt_path,
        )
        with pytest.raises(RuntimeError, match="no migration available"):
            load_checkpoint(pt_path, weights_only=False)


# ---------------------------------------------------------------------------
# Migration dispatch machinery — direct unit tests
# ---------------------------------------------------------------------------


class TestMigrationDispatch:
    """Unit-test the _MIGRATIONS walker independently of npz I/O."""

    def test_empty_registry_refuses(self) -> None:
        """With an empty registry, any older-than-current call errors."""
        from titans.memory_dump import _migrate_arrays_to_current

        with pytest.raises(RuntimeError, match="no migration available"):
            _migrate_arrays_to_current({}, from_version=0, current_version=1)

    def test_single_step_migration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registered (from, to) entry is applied."""
        called: list[str] = []

        def step(arrays: dict) -> dict:
            called.append("step")
            out = dict(arrays)
            out["migrated"] = np.array([1])
            return out

        monkeypatch.setattr("titans.memory_dump._MIGRATIONS", {(0, 1): step})
        from titans.memory_dump import _migrate_arrays_to_current

        out = _migrate_arrays_to_current({}, from_version=0, current_version=1)
        assert called == ["step"]
        assert "migrated" in out

    def test_walk_migrations_symmetry(self) -> None:
        """The memory_dump and checkpoint dispatchers share a single
        walker (``titans._schema_migrations.walk_migrations``) so that
        the two code paths stay symmetric. Breaking this symmetry is
        precisely the P5 foot-gun the shared helper is meant to prevent.
        """
        import titans.checkpoint as ck
        import titans.memory_dump as md
        from titans._schema_migrations import walk_migrations

        # Both modules must expose a wrapper that delegates to the same
        # shared walker. Verifying by calling each with an empty registry
        # and asserting they raise the shared diagnostic is the most
        # robust check (doesn't depend on wrapper internals).
        with pytest.raises(RuntimeError, match="memory_dump schema"):
            md._migrate_arrays_to_current({}, from_version=0, current_version=1)
        with pytest.raises(RuntimeError, match="checkpoint schema"):
            ck._migrate_payload_to_current({}, from_version=0, current_version=1)

        # And the helper itself is importable and callable. Belt-and-
        # braces: if a future refactor drops the wrapper in one module,
        # this direct call still exercises the contract.
        with pytest.raises(RuntimeError, match="no migration available"):
            walk_migrations(
                {},
                from_version=0,
                to_version=1,
                migrations={},
                kind="memory_dump",
            )

    def test_checkpoint_migration_registry_is_dict(self) -> None:
        """The whole-checkpoint migration registry mirrors _MIGRATIONS."""
        from titans.checkpoint import _CHECKPOINT_MIGRATIONS

        assert isinstance(_CHECKPOINT_MIGRATIONS, dict)
        for key in _CHECKPOINT_MIGRATIONS:
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(v, int) for v in key)


# ---------------------------------------------------------------------------
# HF config
# ---------------------------------------------------------------------------


class TestHfConfigVersion:
    """The HF config surfaces ``titans_schema_version`` as a top-level attr."""

    def test_default_matches_current(self) -> None:
        """Default-constructed config picks up the current schema version
        (and warns, because a fresh instance is exactly the "unversioned
        legacy" case — explicit pass required to silence)."""
        from titans.hf.configuration import TitansMACConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = TitansMACConfig()
        assert cfg.titans_schema_version == TITANS_SCHEMA_VERSION

    def test_from_titans_config_carries_version(self) -> None:
        """Config built from a TitansConfig stamps the current schema
        version explicitly (fresh in-memory construction, so no
        unversioned warning)."""
        from titans.config import TitansConfig
        from titans.hf.configuration import TitansMACConfig

        tc = TitansConfig(dim=64, num_heads=4, num_layers=2, vocab_size=128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hf = TitansMACConfig.from_titans_config(tc)
        unversioned = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "titans_schema_version" in str(w.message)
        ]
        assert not unversioned, (
            f"from_titans_config should stamp the version; got "
            f"{[str(w.message) for w in unversioned]}"
        )
        assert hf.titans_schema_version == TITANS_SCHEMA_VERSION

    def test_hf_config_unversioned_warns(self) -> None:
        """Instantiating without ``titans_schema_version`` warns about
        the unversioned legacy layout and falls back to the current
        version (best-effort)."""
        from titans.hf.configuration import TitansMACConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = TitansMACConfig(dim=64)

        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "Expected DeprecationWarning for unversioned HF config"
        assert any("unversioned" in str(w.message).lower() for w in deps), (
            f"Warning must mention 'unversioned'; got {[str(w.message) for w in deps]}"
        )
        # Best-effort fall-through still populates the attribute.
        assert cfg.titans_schema_version == TITANS_SCHEMA_VERSION

    def test_hf_config_newer_raises(self) -> None:
        """A schema version newer than the code must refuse with an
        'upgrade titans' message."""
        from titans.hf.configuration import TitansMACConfig

        with pytest.raises(RuntimeError, match="upgrade"):
            TitansMACConfig(
                dim=64,
                titans_schema_version=TITANS_SCHEMA_VERSION + 5,
            )

    def test_hf_config_older_raises_without_migration(self) -> None:
        """An older version with no migration registered refuses with
        a 'no migration' message. (v1 has no migrations.)"""
        from titans.hf.configuration import TitansMACConfig

        with pytest.raises(RuntimeError, match="no migration"):
            TitansMACConfig(dim=64, titans_schema_version=0)

    def test_json_roundtrip_preserves_version(self, tmp_path: Path) -> None:
        """save_pretrained / from_pretrained round-trips the version
        field with no unversioned-config warning on the equal-version
        happy path."""
        from titans.hf.configuration import TitansMACConfig

        cfg = TitansMACConfig(
            dim=64,
            num_heads=4,
            num_layers=2,
            vocab_size=128,
            titans_schema_version=TITANS_SCHEMA_VERSION,
        )
        cfg.save_pretrained(str(tmp_path))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reloaded = TitansMACConfig.from_pretrained(str(tmp_path))

        unversioned_deps = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "titans_schema_version" in str(w.message)
        ]
        assert not unversioned_deps, (
            f"Equal-version round-trip must not emit the unversioned "
            f"warning; got {[str(w.message) for w in unversioned_deps]}"
        )
        assert reloaded.titans_schema_version == TITANS_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Convert-checkpoint emits the version
# ---------------------------------------------------------------------------


class TestConvertCheckpointVersion:
    """scripts/convert_checkpoint.py emits the version in the output."""

    def test_convert_pt_to_safetensors_emits_version(self, tmp_path: Path) -> None:
        """Convert a tiny pt -> safetensors and verify the sidecar version."""
        # 1. Write a tiny source .pt via the public API so the input
        #    already carries the current schema version.
        src_stem = tmp_path / "src"
        save_checkpoint({"w": torch.ones(2, 2)}, src_stem, format="pt")

        # 2. Run convert_checkpoint.py as a subprocess.
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "convert_checkpoint.py"
        out_stem = tmp_path / "dst"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(src_stem.with_suffix(".pt")),
                "--output",
                str(out_stem),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"convert_checkpoint failed: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

        # 3. Check the converted sidecar carries the schema version.
        sidecar = out_stem.with_suffix(".meta.pt")
        assert sidecar.exists(), f"Expected sidecar at {sidecar}"
        meta = torch.load(sidecar, map_location="cpu", weights_only=False)
        assert meta["titans_schema_version"] == TITANS_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Sanity: the constant is importable and equals 1 in this release
# ---------------------------------------------------------------------------


def test_titans_schema_version_is_one() -> None:
    """First versioned release: schema 1."""
    assert TITANS_SCHEMA_VERSION == 1


def test_migrations_is_dict() -> None:
    """The migration registry is a dict keyed by (from, to) pairs."""
    assert isinstance(_MIGRATIONS, dict)
    for key in _MIGRATIONS:
        assert isinstance(key, tuple) and len(key) == 2
        assert all(isinstance(v, int) for v in key)
