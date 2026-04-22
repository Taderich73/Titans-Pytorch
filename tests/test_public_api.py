"""Tests for the curated public API of the ``titans`` package.

These tests freeze the stable top-level surface at 12 names and verify
that every legacy name still resolves through the PEP 562 deprecation
shim in :mod:`titans.__init__` with a single ``DeprecationWarning`` per
import site.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import pytest

import titans
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
    }
)


def _reload_titans() -> Any:
    """Reload ``titans`` so each deprecated-name assertion starts cold.

    The ``__getattr__`` shim does not cache resolved attributes on the
    module, but Python's ``warnings`` module de-duplicates identical
    warning text under the default filter. Reloading guarantees a fresh
    module state; combined with ``simplefilter("always")`` it ensures
    each access produces an observable warning.
    """
    return importlib.reload(titans)


class TestStableApi:
    """Freeze the curated surface exposed via ``titans.__all__``."""

    def test_all_size_is_bounded(self) -> None:
        """Task P3 budget: ``len(titans.__all__) <= 15``."""
        assert len(titans.__all__) <= 15, (
            f"titans.__all__ grew to {len(titans.__all__)} names; the "
            "curated budget is 15. Either drop a name or justify the "
            "addition in docs/api.md."
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
        where to migrate.
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
