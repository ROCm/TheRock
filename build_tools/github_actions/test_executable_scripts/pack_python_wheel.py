#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pack a pre-built Python package directory into a wheel via setuptools.

Stages the package next to a generated pyproject.toml + setup.py rendered
from the .in templates in this directory, then delegates to `pip wheel`
so wheel naming, METADATA, RECORD, and tag selection follow standard
packaging tooling.

Usage:
    python pack_python_wheel.py \
        --pkg-dir  /path/to/stage/hipdnn_frontend \
        --name hipdnn-frontend \
        --version 1.0.0 \
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
PYPROJECT_TEMPLATE = SCRIPT_DIR / "pack_python_wheel_pyproject.toml.in"
SETUP_TEMPLATE = SCRIPT_DIR / "pack_python_wheel_setup.py.in"


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _render_optional_metadata(args: argparse.Namespace) -> str:
    lines: list[str] = []
    if args.summary:
        lines.append(f'description = "{_toml_escape(args.summary)}"')
    if args.license:
        lines.append(f'license = {{text = "{_toml_escape(args.license)}"}}')
    if args.author:
        lines += ["", "[[project.authors]]", f'name = "{_toml_escape(args.author)}"']
    if args.homepage:
        lines += ["", "[project.urls]", f'Homepage = "{_toml_escape(args.homepage)}"']
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg-dir",
        required=True,
        type=Path,
        help="Directory containing the built package files",
    )
    parser.add_argument("--name", required=True, help="Package name")
    parser.add_argument("--version", required=True, help="Package version")
    parser.add_argument(
        "--wheel-dir",
        required=True,
        type=Path,
        help="Output directory for the .whl file",
    )
    parser.add_argument(
        "--requires-python",
        default=">=3.9",
        help="Requires-Python specifier (default: >=3.9)",
    )
    parser.add_argument("--summary", default=None)
    parser.add_argument("--homepage", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--license", default=None)
    args = parser.parse_args()

    pkg_dir: Path = args.pkg_dir.resolve()
    wheel_dir: Path = args.wheel_dir.resolve()

    if not pkg_dir.is_dir():
        raise SystemExit(f"--pkg-dir is not a directory: {pkg_dir}")
    pkg_name = pkg_dir.name

    wheel_dir.mkdir(parents=True, exist_ok=True)

    pyproject_tmpl = string.Template(PYPROJECT_TEMPLATE.read_text())
    setup_tmpl = string.Template(SETUP_TEMPLATE.read_text())

    pyproject_text = pyproject_tmpl.substitute(
        name=_toml_escape(args.name),
        version=_toml_escape(args.version),
        requires_python=_toml_escape(args.requires_python),
        optional_metadata=_render_optional_metadata(args),
    )
    setup_text = setup_tmpl.substitute(pkg_name=pkg_name)

    with tempfile.TemporaryDirectory() as td:
        build_dir = Path(td)
        shutil.copytree(pkg_dir, build_dir / pkg_name)
        (build_dir / "pyproject.toml").write_text(pyproject_text)
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
