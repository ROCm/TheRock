#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate or verify artifact_subprojects.json manifest."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "artifact_subprojects.json"

CMAKE_ARGS = [
    "-GNinja",
    "-DTHEROCK_AMDGPU_FAMILIES=gfx1100",  # Required by cmake, but doesn't affect manifest
    "-DTHEROCK_ENABLE_ALL=ON",
    "-DTHEROCK_BUNDLE_SYSDEPS=OFF",
    "-DTHEROCK_ENABLE_LIBHIPCXX=OFF",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="therock-manifest-") as tmp_dir:
        print(f"Running cmake configure in {tmp_dir}...")
        result = subprocess.run(
            ["cmake", "-B", tmp_dir, "-S", str(REPO_ROOT)] + CMAKE_ARGS,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"CMake failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)

        generated = Path(tmp_dir) / "artifact_subprojects.json"
        if not generated.exists():
            print("Manifest not generated", file=sys.stderr)
            sys.exit(1)

        with generated.open() as f:
            new_manifest = json.load(f)

        if args.verify:
            with MANIFEST_PATH.open() as f:
                old_manifest = json.load(f)
            if new_manifest == old_manifest:
                print("Manifest is up-to-date.")
            else:
                print(
                    "Manifest is stale. Run: python build_tools/generate_subproject_manifest.py"
                )
                sys.exit(1)
        else:
            shutil.copy(generated, MANIFEST_PATH)
            print(f"Updated {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
