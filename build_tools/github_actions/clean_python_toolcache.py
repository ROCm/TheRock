#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Remove tool cache Python entries whose interpreter no longer runs.

`actions/setup-python` reuses an entry as soon as its directory and the sibling
`<arch>.complete` marker exist, without ever executing the interpreter. A
partially populated entry is therefore reported as a successful setup and only
surfaces later, typically as `pip: cannot execute: required file not found`.
Removing such an entry makes the action provision a fresh one instead.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_interpreter(arch_dir: Path) -> Path:
    # Mirrors the `python-path` output of actions/setup-python, which is the
    # interpreter the later steps run.
    if sys.platform == "win32":
        return arch_dir / "python.exe"
    return arch_dir / "bin" / "python"


def is_usable(arch_dir: Path) -> bool:
    interpreter = find_interpreter(arch_dir)
    if not interpreter.exists():
        return False

    env = dict(os.environ)
    # The published interpreters carry an RPATH pointing at the tool cache root
    # of the image they were built for, which need not exist here.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in [os.fspath(arch_dir / "lib"), env.get("LD_LIBRARY_PATH", "")] if p
    )

    try:
        result = subprocess.run(
            [os.fspath(interpreter), "-m", "pip", "--version"],
            capture_output=True,
            env=env,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def clean_toolcache(toolcache_root: Path, version: str) -> list[Path]:
    version_root = toolcache_root / "Python"
    if not version_root.is_dir():
        return []

    removed = []
    for version_dir in sorted(version_root.glob(f"{version}.*")):
        for arch_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
            if is_usable(arch_dir):
                continue
            print(f"Removing unusable Python tool cache entry {arch_dir}")
            shutil.rmtree(arch_dir, ignore_errors=True)
            arch_dir.with_name(f"{arch_dir.name}.complete").unlink(missing_ok=True)
            removed.append(arch_dir)
    return removed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Python feature version, e.g. 3.12")
    args = parser.parse_args(argv)

    toolcache_root = os.environ.get("AGENT_TOOLSDIRECTORY") or os.environ.get(
        "RUNNER_TOOL_CACHE"
    )
    if not toolcache_root:
        print("No tool cache configured, nothing to check")
        return 0

    removed = clean_toolcache(Path(toolcache_root), args.version)
    if removed:
        runner = os.environ.get("RUNNER_NAME", "unknown")
        print(
            f"::warning::Removed {len(removed)} unusable Python tool cache "
            f"entry/entries on runner '{runner}'; they will be provisioned again."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
