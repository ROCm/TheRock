#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect DASan launcher and CMake diagnostics for the standard S3 uploader."""

import argparse
from pathlib import Path
import shutil


def collect_logs(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    build_dir = output_dir / "build"
    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(output_dir.iterdir()):
        if source.is_file() and source.suffix in (".log", ".json"):
            shutil.copy2(source, log_dir / source.name)

    # Top-level configure failures may occur before TheRock creates its logs.
    # Keep diagnostics from both that configure and nested subproject probes.
    names = {
        "CMakeConfigureLog.yaml",
        "CMakeError.log",
        "CMakeOutput.log",
        "CMakeCache.txt",
    }
    sources = [
        source
        for source in build_dir.rglob("*")
        if source.is_file() and source.name in names and log_dir not in source.parents
    ]
    for source in sorted(sources):
        target = log_dir / "cmake" / source.relative_to(build_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"DASan diagnostics collected in {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    collect_logs(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
