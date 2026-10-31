#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preserve generated Ninja evidence outside the timed build.

Only build files and command data are captured, never environment variables.
Ninja compdb expands response files from the generated rule definitions.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time


def capture(build_dir: Path, output: Path) -> None:
    build_dir = build_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures: list[dict[str, str | int]] = []
    trees = sorted({p.parent for p in build_dir.rglob("build.ninja")})
    with tempfile.TemporaryDirectory(prefix="therock-graph-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(output, "w:gz") as archive:
            for tree in trees:
                relative = tree.relative_to(build_dir)
                for tool, arguments in (
                    ("commands", ["commands"]),
                    ("compdb", ["compdb", "-x"]),
                    ("targets", ["targets", "all"]),
                ):
                    destination = temporary_dir / relative / f"{tool}.txt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as handle:
                        result = subprocess.run(
                            ["ninja", "-C", str(tree), "-t", *arguments],
                            stdout=handle,
                            stderr=subprocess.PIPE,
                            check=False,
                        )
                    if result.returncode:
                        failures.append(
                            {
                                "tree": str(relative),
                                "tool": tool,
                                "returncode": result.returncode,
                                "stderr": result.stderr.decode(errors="replace"),
                            }
                        )
                    archive.add(destination, arcname=str(relative / destination.name))
            for path in sorted(build_dir.rglob("*")):
                # Source trees can contain inaccessible Windows entries. Only
                # inspect files in the diagnostic's declared generated-file set.
                if (
                    path.name == ".ninja_log" or path.suffix == ".ninja"
                ) and path.is_file():
                    archive.add(
                        path,
                        arcname=str(Path("generated") / path.relative_to(build_dir)),
                    )
            manifest = {
                "schema": 1,
                "build_dir": str(build_dir),
                "ninja_tree_count": len(trees),
                "elapsed_seconds": time.monotonic() - started,
                "github_sha": os.environ.get("GITHUB_SHA"),
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "stage": os.environ.get("STAGE_NAME"),
                "amdgpu_family": os.environ.get("AMDGPU_FAMILIES"),
                "failures": failures,
            }
            manifest_path = temporary_dir / "capture-manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            archive.add(manifest_path, arcname=manifest_path.name)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.build_dir, args.output)


if __name__ == "__main__":
    main()
