#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overlay rocm_sdk_core._cli from the in-tree template onto the active venv.

Published rocm-sdk-core wheels may predate llvm patch 0007 (hipcc path logic)
and an updated console-script trampoline. PyTorch CI installs those wheels from
CloudFront while checking out TheRock at a branch that already has the _cli fix.

This script applies the template _cli.py to the installed package so
``Scripts\\hipcc.exe --help`` ignores stale host ROCM_PATH/HIP_PATH on Windows
runners without rebuilding or republishing rocm-sdk-core.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CLI_TEMPLATE = (
    THIS_DIR / "templates/rocm-sdk-core/src/rocm_sdk_core/_cli.py"
)


def overlay_cli(*, quiet: bool = False) -> Path:
    if not CLI_TEMPLATE.is_file():
        raise FileNotFoundError(f"Template not found: {CLI_TEMPLATE}")

    spec = importlib.util.find_spec("rocm_sdk_core._cli")
    if spec is None or not spec.origin:
        raise RuntimeError("rocm_sdk_core._cli is not installed in this environment")

    dest = Path(spec.origin)
    shutil.copy2(CLI_TEMPLATE, dest)
    if not quiet:
        print(f"Overlaid rocm_sdk_core._cli from template:\n  {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        overlay_cli()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
