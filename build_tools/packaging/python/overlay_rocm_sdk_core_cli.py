#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overlay in-tree rocm-sdk test helpers onto published wheels in the active venv.

PyTorch CI installs rocm packages from CloudFront before those wheels pick up
template fixes. Copy the test env helpers from the checkout so ``rocm-sdk test``
can run ``hipcc --help`` on Windows runners with stale HIP SDK env vars.
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
        "rocm/src/rocm_sdk/tests/utils.py",
        "rocm_sdk.tests.utils",
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
