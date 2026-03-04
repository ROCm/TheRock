#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy prebuilt stage artifacts from a baseline run to the current run.

Parses the linux_variants JSON from configure_ci.py to extract the union of
all dist_amdgpu_families, then calls artifact_manager.py copy for the
specified stages.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_families(linux_variants_json: str) -> str:
    """Extract the union of dist_amdgpu_families from linux_variants JSON.

    Returns a semicolon-separated string of sorted, unique family names.
    """
    variants: list[dict] = json.loads(linux_variants_json)
    families: set[str] = set()
    for variant in variants:
        for family in variant.get("dist_amdgpu_families", "").split(";"):
            if family:
                families.add(family)
    if not families:
        raise ValueError(
            f"No amdgpu families found in linux_variants: {linux_variants_json}"
        )
    return ";".join(sorted(families))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prebuilt-stages",
        required=True,
        help="Comma-separated stage names to copy (e.g. foundation,compiler-runtime)",
    )
    parser.add_argument(
        "--baseline-run-id",
        required=True,
        help="GitHub run ID to copy artifacts from",
    )
    parser.add_argument(
        "--linux-variants",
        required=True,
        help="JSON array from configure_ci.py linux_variants output",
    )
    args = parser.parse_args()

    families = extract_families(args.linux_variants)
    print(f"Extracted families: {families}")
    print(f"Copying stages: {args.prebuilt_stages}")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "build_tools" / "artifact_manager.py"),
        "copy",
        f"--source-run-id={args.baseline_run_id}",
        f"--stage={args.prebuilt_stages}",
        f"--amdgpu-families={families}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
