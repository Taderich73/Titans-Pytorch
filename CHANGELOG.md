# Changelog

All notable changes to this project will be documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to semantic versioning.

## [Unreleased]

### Added
- GitHub Actions CI (`ci.yml`, `coverage.yml`, issue/PR templates) (#P1 / P11)
- `TITANS_SCHEMA_VERSION` constant and checkpoint-load dispatch for
  missing/newer/older schemas (#P5)
- `MIGRATIONS.md` — protocol for bumping checkpoint schema versions (#P5)
- `torch.compile(fullgraph=True)` enforced tests for MAC/MAG/MAL/LMM variants
  (#P6)
- Curated public API (`titans.__all__` reduced to 13 supported names with
  deprecation shims for the legacy surface) (#P3)
- `docs/api.md`, `docs/README.md` — docs hub + stable surface contract
  (#P2, #P3)
- `docs/paper_alignment.md` — new home for equation-level paper-alignment
  tables (#P2)
- `py.typed` PEP 561 marker so downstream consumers pick up type hints (#P12)
- `CHANGELOG.md` (this file) (#P10)

### Changed
- `Development Status :: 3 - Alpha` → `Development Status :: 4 - Beta` (#P10)
- `src/titans/models.py` refactored: `BaseTitansBlock` extracted to
  de-duplicate MAC/MAG/MAL initialization (#P4)
- `__version__` now derived from `importlib.metadata.version("titans")` with
  a fallback, and deprecated-name lookups are cached (#P3)
- README slimmed to quickstart plus link farm; feature deep-dives moved to
  `docs/` (#P2)
- HF config surface routes unversioned checkpoint loads through the shared
  migration walker with a warning (#P5)
- Unified `scripts/convert_checkpoint.py` and `scripts/convert_to_hf.py`
  into a single `scripts/convert.py` with `--to={pt,safetensors,hf}`
  (#P13). Old scripts retained as deprecated shims; removed in 0.8.

### Deprecated
- Legacy top-level `titans.<name>` imports now emit `DeprecationWarning` on
  access; these shims will be removed in 0.8. See `docs/api.md` for the new
  import paths. (#P3)
- `scripts/convert_checkpoint.py` and `scripts/convert_to_hf.py` are now
  thin shims that forward to `scripts/convert.py` and emit a
  `DeprecationWarning`; they will be removed in 0.8. (#P13)

### Fixed
- Stale README test-count badge replaced with a live CI status badge (#P11)
- README quickstart snippet that would have raised on first run (chunk-size
  mismatch, 3-tuple unpack) (#P2)
- Coverage workflow source discovery for the `src/` layout (#P1)

## [0.7.0] - 2026-04-21

_Historical record begins with the 0.7.0 release. Detailed pre-0.7 history is
not reconstructed — see git log for prior changes._

- Paper Plan 5 and Plan 6 schema revisions (see `docs/paper_alignment.md`).
- Support for TNT hierarchical memory, AttnRes sidecar, MCA sidecar,
  Yaad/Huber bias, adaptive window, p-RoPE.
- HuggingFace integration (MAC-only).
