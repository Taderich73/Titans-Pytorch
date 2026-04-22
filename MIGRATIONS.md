# Migrations

This file tracks schema changes to persisted state written by OpenTitans:

- `.pt` / `.safetensors` checkpoints produced by `titans.save_checkpoint`
  (and therefore every training script under `scripts/`).
- `.npz` memory dumps produced by `titans.save_memory_states`.
- HuggingFace `config.json` produced by `titans.hf.configuration.TitansMACConfig`
  (and `convert_to_hf.py`).

Every file written from 0.7.0 onwards carries a `titans_schema_version`
integer at a well-known location. See
`src/titans/__init__.py::TITANS_SCHEMA_VERSION` for the current value.

## Change log

| Version | Released | Change |
|---------|----------|--------|
| 1       | 0.7.0    | Initial versioned schema. All prior checkpoints are "unversioned" and load with best-effort assuming pre-0.7 layout. |

## Load dispatch

`titans.load_checkpoint` and `titans.load_memory_states` implement the
same four-way dispatch on the schema version carried by the file:

| File says        | Behavior                                                   |
|------------------|------------------------------------------------------------|
| *(key missing)*  | `DeprecationWarning` "unversioned checkpoint, assuming pre-0.7 layout" and best-effort load. |
| `== current`     | Silent load.                                               |
| `> current`      | `RuntimeError` "upgrade titans".                           |
| `< current`      | Run the registered migration chain; if none available, raise `RuntimeError` "no migration available". |

The code-side migration registry for memory dumps lives at
`titans.memory_dump._MIGRATIONS`. An analogous dispatch for
full-checkpoint payloads lives in `titans.checkpoint._check_schema_version`.

## Rules for bumping the schema

When you make a breaking change to any persisted layout:

1. Increment `TITANS_SCHEMA_VERSION` in `src/titans/__init__.py`.
2. Add a row to the change log above describing the change and the
   release it shipped in.
3. If the prior version is loadable with a pure in-memory rewrite
   (renames, reshapes, dropped keys with sensible defaults, …), register
   a migration function:
   - Memory dumps: add a `(from_version, to_version) -> fn` entry to
     `titans.memory_dump._MIGRATIONS`. The function takes a
     `dict[str, np.ndarray]` mapping npz entry names to arrays and
     returns the same structure in the new layout.
   - Full checkpoints: extend the dispatch in
     `titans.checkpoint._check_schema_version` with an analogous
     registry if the change involves top-level key renames or
     tensor-shape migrations.
4. If no backward migration is feasible, document why under the change
   log row. The load path will then refuse old files with a clear
   `RuntimeError` rather than producing silently-wrong outputs.
5. Add a test case to `tests/test_checkpoint_versioning.py` exercising
   the scenario (same-version, migration, refusal).
