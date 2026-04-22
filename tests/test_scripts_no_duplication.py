"""Regression guard: no script must re-introduce consolidated helpers.

Plan 3 moved format_chatml / build_loss_mask / tokenize_chat / create_model
/ build_titans_config / base_argparse_parser / init_accelerator_and_logging
/ setup_checkpoint_dir into the shared helper module (now
``titans.scripts._common`` after the HF-Jobs single-file-shipping migration).
If any training script re-defines them at top level, this test fails and
the PR should import from ``titans.scripts`` instead.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = _REPO_ROOT / "scripts"
COMMON_PATH = _REPO_ROOT / "src" / "titans" / "scripts" / "_common.py"

CONSOLIDATED_NAMES = {
    "format_chatml",
    "tokenize_chat",
    "create_model",
    "build_titans_config",
    "base_argparse_parser",
    "init_accelerator_and_logging",
    "setup_checkpoint_dir",
}

# lora.py keeps thin wrappers named build_loss_mask / tokenize_chat that
# delegate to _common (they return the lora-historical label format). We
# whitelist those wrapper definitions explicitly.
ALLOWED_WRAPPERS = {
    ("lora.py", "build_loss_mask"),
    ("lora.py", "tokenize_chat"),
}


def _top_level_defs(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    ]


@pytest.mark.parametrize(
    "script_name",
    ["sft.py", "lora.py", "dpo.py", "rlvr.py", "pretrain.py", "inference.py"],
)
def test_no_duplicate_top_level_definitions(script_name: str) -> None:
    path = SCRIPTS_DIR / script_name
    defs = _top_level_defs(path)
    duplicates = [
        d for d in defs
        if d in CONSOLIDATED_NAMES and (script_name, d) not in ALLOWED_WRAPPERS
    ]
    assert duplicates == [], (
        f"{script_name}: re-defines consolidated helpers at top level: "
        f"{duplicates}. Import from titans.scripts instead."
    )


def test_common_module_exports_consolidated_names() -> None:
    """Consolidated helpers must all be defined in titans.scripts._common."""
    defs = set(_top_level_defs(COMMON_PATH))
    missing = CONSOLIDATED_NAMES - defs
    assert missing == set(), f"{COMMON_PATH} missing: {sorted(missing)}"
