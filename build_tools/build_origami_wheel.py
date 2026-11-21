#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build Origami's Python wheel against an installed ROCm prefix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def build_wheel(
    source_dir: Path,
    rocm_dir: Path,
    wheel_dir: Path,
    *,
    isolation: bool = True,
) -> None:
    with tempfile.TemporaryDirectory(prefix="origami-wheel-") as temporary_dir:
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir.resolve()),
            "-Ccmake.define.ORIGAMI_BUILD_FROM_SOURCE=OFF",
            f"-Ccmake.define.CMAKE_PREFIX_PATH={rocm_dir.resolve()}",
            f"-Cbuild-dir={temporary_dir}/build",
        ]
        if not isolation:
            command.append("--no-isolation")
        command.append(str(source_dir.resolve()))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(command, check=True, env=environment)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Origami source directory containing pyproject.toml",
    )
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        required=True,
        help="Installed ROCm prefix with Origami's shared CMake package",
    )
    parser.add_argument(
        "--wheel-dir", type=Path, required=True, help="Wheel output directory"
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Use build dependencies already installed in this interpreter",
    )
    args = parser.parse_args(argv)
    build_wheel(
        args.source_dir, args.rocm_dir, args.wheel_dir, isolation=not args.no_isolation
    )


if __name__ == "__main__":
    main()
