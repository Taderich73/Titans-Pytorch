# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Shared schema-migration walker used by the P5 versioning dispatchers.

Two persisted-state families each carry a ``titans_schema_version`` tag:

* ``.npz`` memory dumps (:mod:`titans.memory_dump`) — migrations operate
  on ``dict[str, numpy.ndarray]``.
* ``.pt`` / ``.safetensors`` whole-checkpoint payloads
  (:mod:`titans.checkpoint`) — migrations operate on
  ``dict[str, Any]`` (tensors plus metadata).

A third registry for HF ``config.json`` will follow the same pattern when
schema v2 needs one.

The two dispatch paths have identical walk logic, so the shared
implementation lives here. The per-module registries remain private to
their respective modules; only the walker is shared.
"""

from __future__ import annotations

from typing import Any, Callable

# Module-private: both registries key on ``(from_version, to_version)``.
_MigrationMap = dict[tuple[int, int], Callable[[dict[str, Any]], dict[str, Any]]]


def walk_migrations(
    payload: dict[str, Any],
    from_version: int,
    to_version: int,
    migrations: _MigrationMap,
    *,
    kind: str,
) -> dict[str, Any]:
    """Apply registered migrations to upgrade a payload one step at a time.

    At each step we prefer the longest-jump migration registered from the
    current version (``(v, target)`` with the largest ``target <=
    to_version``). When none is registered we raise
    :class:`RuntimeError` with a diagnostic that names which schema gap
    is missing — this is strictly better than silently loading a
    partially-migrated payload.

    Args:
        payload: The mutable dict to migrate. Callers typically pass a
            fresh copy so the on-disk data is untouched.
        from_version: Schema version the file was written with.
        to_version: The code's current schema version.
        migrations: Mapping ``(from, to) -> fn`` where ``fn(payload)``
            returns a new payload in the ``to`` layout.
        kind: Human-readable label (``"memory_dump"`` or
            ``"checkpoint"``) used in error messages so callers can tell
            the two paths apart without catching and re-raising.

    Returns:
        The migrated payload in the ``to_version`` layout.

    Raises:
        RuntimeError: When no migration is registered for some ``(v,
            *)`` on the way to ``to_version``.

    Note on registry design: when you register a longest-jump migration
    ``(1, 5)`` alongside the stepwise ones ``(1, 2) ... (4, 5)``, the
    walker will take the long jump. Only do this when the longest jump
    is *semantically equivalent* to the composed chain — otherwise
    picking the direct path skips intermediate fixes. This is the same
    footgun the original ``memory_dump`` dispatcher had and is preserved
    for symmetry; see ``MIGRATIONS.md``.
    """
    v = from_version
    out = payload
    while v < to_version:
        step = None
        for target in range(to_version, v, -1):
            if (v, target) in migrations:
                step = target
                break
        if step is None:
            raise RuntimeError(
                f"{kind} schema {from_version} is older than code "
                f"schema {to_version}; no migration available "
                f"(stuck at {v}). See MIGRATIONS.md."
            )
        out = migrations[(v, step)](out)
        v = step
    return out
