"""Tests for the curated public API of the ``titans`` package.

These tests freeze the stable top-level surface at 12 names and verify
that every legacy name still resolves through the PEP 562 deprecation
shim in :mod:`titans.__init__` with a single ``DeprecationWarning`` per
import site.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import textwrap
import tomllib
import warnings
from typing import Any

import pytest

import titans
from tests._subprocess_helpers import subprocess_python
from titans import _DEPRECATED_EXPORTS

# The exact stable surface for the 0.7.x line. Changing this set is a
# backward-compat event and requires a bump + release note.
EXPECTED_STABLE_API: frozenset[str] = frozenset(
    {
        "TitansConfig",
        "TitansMAC",
        "TitansMAG",
        "TitansMAL",
        "TitansLMM",
        "NeuralLongTermMemory",
        "MemoryState",
        "TNTMemoryState",
        "save_memory_states",
        "load_memory_states",
        "save_checkpoint",
        "load_checkpoint",
        # P5: checkpoint schema versioning constant.
        "TITANS_SCHEMA_VERSION",
    }
)


def _reload_titans() -> Any:
    """Reload ``titans`` so each deprecated-name assertion starts cold.

    The PEP 562 ``__getattr__`` shim caches resolved deprecated names
    onto the module after the first access (the idiomatic
    numpy/scipy/pandas pattern), so a second ``getattr(titans, name)``
    in the same process does NOT re-warn. Reloading evicts that cache.
    Combined with ``simplefilter("always")`` it ensures each
    parametrized case observes exactly one warning on first access.
    """
    return importlib.reload(titans)


class TestStableApi:
    """Freeze the curated surface exposed via ``titans.__all__``."""

    def test_all_size_is_exactly_thirteen(self) -> None:
        """Task P3+P5 budget: ``len(titans.__all__) == 13`` (doc contract).

        ``docs/api.md`` documents the stable surface as frozen at 13 —
        12 from the original P3 curated set plus ``TITANS_SCHEMA_VERSION``
        added by P5. The explicit ``EXPECTED_STABLE_API`` frozenset
        already enforces the exact name set; this test pins the
        cardinality so accidental additions fail loudly in isolation.
        """
        assert len(titans.__all__) == 13, (
            f"titans.__all__ has {len(titans.__all__)} names; the curated "
            "surface is frozen at 13. Update docs/api.md AND "
            "EXPECTED_STABLE_API in lockstep if this changes."
        )

    def test_all_matches_expected_set(self) -> None:
        """The exact names in ``__all__`` are frozen."""
        assert set(titans.__all__) == EXPECTED_STABLE_API, (
            "titans.__all__ drifted from the frozen stable surface. "
            "Update EXPECTED_STABLE_API here AND docs/api.md in lockstep."
        )

    def test_all_has_no_duplicates(self) -> None:
        """``__all__`` should be a list of unique names."""
        assert len(titans.__all__) == len(set(titans.__all__))

    @pytest.mark.parametrize("name", sorted(EXPECTED_STABLE_API))
    def test_stable_name_is_accessible_without_warnings(self, name: str) -> None:
        """Stable names import cleanly with no deprecation warning."""
        _reload_titans()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            attr = getattr(titans, name)
            assert attr is not None
            deprecations = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            assert not deprecations, (
                f"Accessing stable name {name!r} produced a "
                f"DeprecationWarning: {[str(w.message) for w in deprecations]}"
            )

    def test_version_is_a_string(self) -> None:
        """``titans.__version__`` must be a semver-ish string."""
        assert isinstance(titans.__version__, str)
        assert titans.__version__
        # Soft check: looks like MAJOR.MINOR.PATCH
        parts = titans.__version__.split(".")
        assert len(parts) >= 2

    def test_version_matches_pyproject(self) -> None:
        """``titans.__version__`` must agree with ``pyproject.toml``.

        The package reads its version from installed metadata via
        :func:`importlib.metadata.version`. When the package is installed
        (editable or otherwise) with metadata available, the two must
        agree. If the fallback sentinel is active (checkout without any
        metadata), skip — there is nothing to compare against.
        """
        if titans.__version__.startswith("0.0.0+"):
            pytest.skip("version fallback active (not installed with metadata)")
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        with open(repo_root / "pyproject.toml", "rb") as fh:
            pyproject = tomllib.load(fh)
        assert titans.__version__ == pyproject["project"]["version"]


class TestDeprecationShims:
    """The legacy 0.6.x top-level names still import but warn."""

    def test_deprecated_map_is_not_empty(self) -> None:
        assert len(_DEPRECATED_EXPORTS) >= 30, (
            "Expected the deprecation shim to cover every legacy name "
            "that was in __all__ before P3."
        )

    def test_stable_and_deprecated_sets_are_disjoint(self) -> None:
        """A name is either stable or deprecated — never both."""
        stable = set(titans.__all__)
        deprecated = set(_DEPRECATED_EXPORTS)
        overlap = stable & deprecated
        assert not overlap, (
            f"Names appear in both __all__ and _DEPRECATED_EXPORTS: {overlap}"
        )

    @pytest.mark.parametrize(
        "name,submodule",
        sorted(_DEPRECATED_EXPORTS.items()),
    )
    def test_deprecated_name_emits_single_warning(
        self, name: str, submodule: str
    ) -> None:
        """Each deprecated access emits exactly one DeprecationWarning.

        The warning text must name the new submodule path so users know
        where to migrate. Note: the shim caches the resolved value on
        the module after the first access (PEP 562 idiom), so a second
        ``getattr(titans, name)`` in the *same* process would NOT
        re-warn. ``_reload_titans()`` resets that cache so each
        parametrized case starts cold.
        """
        mod = _reload_titans()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(mod, name)

        assert value is not None

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1, (
            f"Accessing {name!r} produced {len(deprecations)} "
            f"DeprecationWarnings (expected 1)."
        )
        message = str(deprecations[0].message)
        assert name in message
        assert submodule in message, (
            f"Deprecation message for {name!r} must mention the new "
            f"submodule path {submodule!r}; got: {message!r}"
        )
        assert "0.8" in message, (
            "Deprecation message should state the removal version (0.8)."
        )

    @pytest.mark.parametrize(
        "name,submodule",
        # Keep the fast path small — subprocess spawning is not free.
        # Pick three representatives covering different submodules.
        [
            ("LoRALinear", "titans.lora"),
            ("SegmentedAttention", "titans.attention"),
            ("PersistentMemory", "titans.persistent"),
        ],
    )
    def test_from_import_syntax_emits_single_warning(
        self, name: str, submodule: str
    ) -> None:
        """``from titans import X`` must emit exactly one DeprecationWarning.

        CPython's ``IMPORT_FROM`` opcode probes ``__getattr__`` twice
        (once for the attribute, once as a potential submodule lookup).
        Without attribute caching, that yields two warnings per import;
        the PEP 562 caching pattern collapses it to one. Each case runs
        in a fresh subprocess to guarantee a cold module state.
        """
        code = textwrap.dedent(
            f"""
            import warnings
            warnings.simplefilter("always")
            with warnings.catch_warnings(record=True) as w:
                from titans import {name}  # noqa: F401
            deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deps) == 1, (
                f"Expected 1 DeprecationWarning, got {{len(deps)}}: "
                f"{{[str(d.message) for d in deps]}}"
            )
            assert "{submodule}" in str(deps[0].message), (
                f"Expected {submodule!r} in: {{deps[0].message}}"
            )
            """
        )
        result = subprocess.run(
            [*subprocess_python(), "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{name}: stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    @pytest.mark.slow
    def test_from_import_syntax_emits_single_warning_for_all_names(self) -> None:
        """Exhaustive variant: verify single-warning behavior for every
        deprecated name. Runs one subprocess per name (~37 spawns), so
        this is marked ``slow`` and skipped by default fast-path runs.
        """
        failures: list[str] = []
        for name, submodule in _DEPRECATED_EXPORTS.items():
            code = textwrap.dedent(
                f"""
                import warnings
                warnings.simplefilter("always")
                with warnings.catch_warnings(record=True) as w:
                    from titans import {name}  # noqa: F401
                deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deps) == 1, (
                    f"Expected 1 DeprecationWarning, got {{len(deps)}}: "
                    f"{{[str(d.message) for d in deps]}}"
                )
                assert "{submodule}" in str(deps[0].message), (
                    f"Expected {submodule!r} in: {{deps[0].message}}"
                )
                """
            )
            result = subprocess.run(
                [*subprocess_python(), "-c", code],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                failures.append(
                    f"{name}: stdout={result.stdout!r} stderr={result.stderr!r}"
                )
        assert not failures, "Failed names:\n" + "\n".join(failures)

    @pytest.mark.parametrize(
        "name,submodule",
        sorted(_DEPRECATED_EXPORTS.items()),
    )
    def test_deprecated_name_matches_submodule(self, name: str, submodule: str) -> None:
        """The shim returns the same object the submodule exposes."""
        _reload_titans()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_value = getattr(titans, name)
        real_module = importlib.import_module(submodule)
        real_value = getattr(real_module, name)
        assert shim_value is real_value

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """Names that are neither stable nor deprecated should 404."""
        _reload_titans()
        with pytest.raises(AttributeError):
            _ = titans.DefinitelyNotAPublicName  # type: ignore[attr-defined]


class TestStarImport:
    """``from titans import *`` must respect the curated surface."""

    def test_star_import_only_exposes_stable_names(self) -> None:
        namespace: dict[str, Any] = {}
        exec("from titans import *", namespace)  # noqa: S102
        exported = {k for k in namespace if not k.startswith("_")}
        assert exported == EXPECTED_STABLE_API, (
            f"`from titans import *` should only expose stable names. "
            f"Unexpected extras: {exported - EXPECTED_STABLE_API}. "
            f"Missing: {EXPECTED_STABLE_API - exported}."
        )


class TestDir:
    """``dir(titans)`` should include both stable and deprecated names."""

    def test_dir_includes_stable_names(self) -> None:
        names = set(dir(titans))
        assert EXPECTED_STABLE_API.issubset(names)

    def test_dir_includes_deprecated_names(self) -> None:
        names = set(dir(titans))
        assert set(_DEPRECATED_EXPORTS).issubset(names)

    def test_dir_includes_version(self) -> None:
        assert "__version__" in dir(titans)
