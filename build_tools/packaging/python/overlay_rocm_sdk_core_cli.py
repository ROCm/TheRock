#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overlay in-tree Python fixes onto published ROCm wheels in the active venv.

Published rocm-sdk-core / rocm wheels may predate the _cli hipcc trampoline fix
and updated test helpers. PyTorch CI installs those wheels from CloudFront while
checking out TheRock at a branch that already has the fixes in templates.

This script copies the template sources into site-packages and clears matching
__pycache__ entries so the updated modules are loaded for ``rocm-sdk test``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_ROOT = THIS_DIR / "templates"

OVERLAY_FILES: tuple[tuple[str, str], ...] = (
    (
        "rocm-sdk-core/src/rocm_sdk_core/_cli.py",
        "rocm_sdk_core._cli",
    ),
    (
        "rocm/src/rocm_sdk/tests/core_test.py",
        "rocm_sdk.tests.core_test",
    ),
)


def _module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"{module_name} is not installed in this environment")
    return Path(spec.origin)


def _invalidate_bytecode(module_file: Path) -> None:
    cache_dir = module_file.parent / "__pycache__"
    if not cache_dir.is_dir():
        return
    stem = module_file.stem
    for cached in cache_dir.glob(f"{stem}.*.pyc"):
        cached.unlink(missing_ok=True)


def overlay_file(relative_template: str, module_name: str, *, quiet: bool = False) -> Path:
    template = TEMPLATE_ROOT.joinpath(*relative_template.split("/"))
    if not template.is_file():
        raise FileNotFoundError(f"Template not found: {template}")

    dest = _module_path(module_name)
    shutil.copy2(template, dest)
    _invalidate_bytecode(dest)
    if not quiet:
        print(f"Overlaid {module_name}:\n  {dest}")
    return dest


def overlay_published_wheel_fixes(*, quiet: bool = False) -> list[Path]:
    return [
        overlay_file(relative_template, module_name, quiet=quiet)
        for relative_template, module_name in OVERLAY_FILES
    ]


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        overlay_published_wheel_fixes()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
