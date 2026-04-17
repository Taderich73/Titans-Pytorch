"""Regression guard: no script must re-introduce consolidated helpers.

Plan 3 moved format_chatml / build_loss_mask / tokenize_chat / create_model
/ build_titans_config / base_argparse_parser / init_accelerator_and_logging
/ setup_checkpoint_dir into scripts/_common.py. If any training script
re-defines them at top level, this test fails and the PR should import
from _common instead.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"

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
        f"{duplicates}. Import from scripts._common instead."
    )


def test_common_module_exports_consolidated_names() -> None:
    """The consolidated helpers must all be defined in scripts/_common.py."""
    common_path = SCRIPTS_DIR / "_common.py"
    defs = set(_top_level_defs(common_path))
    missing = CONSOLIDATED_NAMES - defs
    assert missing == set(), f"scripts/_common.py missing: {sorted(missing)}"
