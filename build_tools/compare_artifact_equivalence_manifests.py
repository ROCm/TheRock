#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare privacy-safe artifact manifests from paired canary runs."""

import argparse
import json
from pathlib import Path
import re


ATTEMPT_SUFFIX = re.compile(r"-\d+$")


def load_manifests(root: Path) -> dict[str, dict]:
    manifests = {}
    for path in sorted(root.rglob("artifact-manifest.json")):
        key = ATTEMPT_SUFFIX.sub("", path.parent.name)
        if key in manifests:
            raise ValueError(f"duplicate manifest key: {key}")
        manifests[key] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def compare(baseline: dict[str, dict], probed: dict[str, dict]) -> dict:
    if not baseline or not probed:
        raise ValueError("both baseline and probed manifest sets must be non-empty")
    keys = sorted(set(baseline) | set(probed))
    comparisons = []
    equivalent = True
    for key in keys:
        left = baseline.get(key)
        right = probed.get(key)
        status = "equivalent" if left == right and left is not None else "different"
        equivalent &= status == "equivalent"
        comparisons.append(
            {
                "artifact": key,
                "status": status,
                "baseline_file_count": len(left.get("files", [])) if left else None,
                "probed_file_count": len(right.get("files", [])) if right else None,
            }
        )
    return {
        "schema": "therock.artifact_equivalence_comparison.v1",
        "equivalent": equivalent,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--probed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load_manifests(args.baseline_dir), load_manifests(args.probed_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
