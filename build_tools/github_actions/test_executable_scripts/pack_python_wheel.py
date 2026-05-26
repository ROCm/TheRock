#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pack a pre-built hipdnn_frontend package directory into a wheel.

Stages the package next to the pyproject.toml + setup.py adjacent to
this script, then delegates to `pip wheel` so wheel naming, METADATA,
RECORD, and tag selection follow standard packaging tooling.

Usage:
    python pack_python_wheel.py \
        --pkg-dir /path/to/stage/hipdnn_frontend \
        --wheel-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYPROJECT_FILE = SCRIPT_DIR / "pack_python_wheel_pyproject.toml"
SETUP_FILE = SCRIPT_DIR / "pack_python_wheel_setup.py"
EXPECTED_PKG_NAME = "hipdnn_frontend"
NATIVE_EXT_SUFFIXES = (".so", ".pyd")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg-dir",
        required=True,
        type=Path,
        help=f"Directory containing the built {EXPECTED_PKG_NAME} package",
    )
    parser.add_argument(
        "--wheel-dir",
        required=True,
        type=Path,
        help="Output directory for the .whl file",
    )
    args = parser.parse_args()

    pkg_dir: Path = args.pkg_dir.resolve()
    wheel_dir: Path = args.wheel_dir.resolve()

    if not pkg_dir.is_dir():
        raise SystemExit(f"--pkg-dir is not a directory: {pkg_dir}")

    if pkg_dir.name != EXPECTED_PKG_NAME:
        raise SystemExit(
            f"--pkg-dir basename must be {EXPECTED_PKG_NAME!r}; got {pkg_dir.name!r}"
        )

    has_native = next(
        (
            p
            for p in pkg_dir.rglob("*")
            if p.is_file() and p.suffix in NATIVE_EXT_SUFFIXES
        ),
        None,
    )
    if has_native is None:
        raise SystemExit(
            f"No native extension ({'/'.join(NATIVE_EXT_SUFFIXES)}) found under "
            f"{pkg_dir}; refusing to build a platform wheel from pure-Python sources"
        )

    wheel_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        build_dir = Path(td)
        shutil.copytree(pkg_dir, build_dir / EXPECTED_PKG_NAME)
        shutil.copy(PYPROJECT_FILE, build_dir / "pyproject.toml")
        shutil.copy(SETUP_FILE, build_dir / "setup.py")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(build_dir),
            ]
        )

    wheels = list(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise SystemExit(f"pip wheel produced no hipdnn_frontend wheel in {wheel_dir}")

    logging.info(f"Wheel(s) written to {wheel_dir}: {[w.name for w in wheels]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
