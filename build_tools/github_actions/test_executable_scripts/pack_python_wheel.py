#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pack a pre-built hipdnn_frontend package directory into a wheel.

Stages the package next to the pyproject.toml + setup.py.in adjacent to
this script, then delegates to `pip wheel` so wheel naming, METADATA,
RECORD, and tag selection follow standard packaging tooling.

Usage:
    python pack_python_wheel.py \
        --pkg-dir /path/to/stage/hipdnn_frontend \
        --wheel-dir /path/to/output
"""

from __future__ import annotations

import argparse
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYPROJECT_FILE = SCRIPT_DIR / "pack_python_wheel_pyproject.toml"
SETUP_TEMPLATE = SCRIPT_DIR / "pack_python_wheel_setup.py.in"
NATIVE_EXT_SUFFIXES = (".so", ".pyd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg-dir",
        required=True,
        type=Path,
        help="Directory containing the built package files",
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

    pkg_name = pkg_dir.name
    if not pkg_name.isidentifier():
        raise SystemExit(
            f"--pkg-dir basename is not a valid Python package name: {pkg_name!r}"
        )

    has_native = any(
        p.suffix in NATIVE_EXT_SUFFIXES for p in pkg_dir.rglob("*") if p.is_file()
    )
    if not has_native:
        raise SystemExit(
            f"No native extension ({'/'.join(NATIVE_EXT_SUFFIXES)}) found under "
            f"{pkg_dir}; refusing to build a platform wheel from pure-Python sources"
        )

    wheel_dir.mkdir(parents=True, exist_ok=True)

    setup_tmpl = string.Template(SETUP_TEMPLATE.read_text())
    setup_text = setup_tmpl.substitute(pkg_name=pkg_name)

    with tempfile.TemporaryDirectory() as td:
        build_dir = Path(td)
        shutil.copytree(pkg_dir, build_dir / pkg_name)
        shutil.copy(PYPROJECT_FILE, build_dir / "pyproject.toml")
        (build_dir / "setup.py").write_text(setup_text)

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

    print(f"Wheel(s) written to {wheel_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
