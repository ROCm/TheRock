# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate the explicit CSAN repair plan and its baseline before copying it."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = {
    "compiler-runtime",
    "runtime-tests",
    "emulation",
    "storage-libs",
    "debug-tools",
    "media-libs",
    "wsl-rocdxg",
    "cv-libs",
    "dctools-core",
    "comm-libs",
    "profiler-apps",
}
REBUILDS = {
    "compiler-runtime": {"amd-llvm", "core-hip", "core-ocl"},
    "runtime-tests": {"rocrtst", "core-hiptests"},
    "cv-libs": {"rpp"},
    "debug-tools": {"rocgdb"},
    "profiler-apps": {"rocprofiler-systems", "rocprofiler-systems-examples"},
    "comm-libs": {"rccl"},
}


def split(value):
    return {part.strip() for part in value.split(",") if part.strip()}


def validate_inputs(prebuilt, baseline, build_stages, family, other_repair=False):
    if split(prebuilt) != STAGES:
        raise ValueError(
            "component_repair requires exactly the eleven published non-math stages"
        )
    if not baseline.isdecimal() or int(baseline) <= 0:
        raise ValueError("component_repair requires a positive baseline_run_id")
    if build_stages.strip() or family.lower().strip() != "gfx94x" or other_repair:
        raise ValueError(
            "component_repair requires gfx94x, no stage allowlist and no other repair mode"
        )


def check_artifacts(backend, keys):
    missing = [
        key
        for archive in keys
        for key in (archive, f"{archive}.sha256sum")
        if not backend.artifact_exists(key)
    ]
    if not keys or missing:
        raise ValueError(f"Incomplete repair baseline: {missing}")


def snapshot_artifacts(backend, targets):
    from artifact_manager import find_available_artifacts, get_topology
    from _therock_utils.artifacts import ArtifactName

    topology = get_topology()
    names = set().union(*(topology.get_produced_artifacts(stage) for stage in STAGES))
    keys = find_available_artifacts(names, targets, set(backend.list_artifacts()))
    present = {
        artifact.name for key in keys if (artifact := ArtifactName.from_filename(key))
    }
    required = set().union(*REBUILDS.values()) | {
        "base",
        "sysdeps",
        "core-runtime",
        "core-amdsmi",
        "core-kpack",
        "rocshmem",
        "rocprofiler-sdk",
        "hipfile",
    }
    if missing := required - present:
        raise ValueError(f"Missing mandatory baseline artifacts: {sorted(missing)}")
    for stage in STAGES:
        if not topology.get_produced_artifacts(stage) & present:
            raise ValueError(f"Baseline stage has no published artifacts: {stage}")
    check_artifacts(backend, keys)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inputs", "snapshot", "verify"])
    parser.add_argument("--manifest", type=Path, default=Path("repair-baseline.json"))
    args = parser.parse_args()
    if args.mode == "inputs":
        validate_inputs(
            os.environ.get("PREBUILT_STAGES", ""),
            os.environ.get("BASELINE_RUN_ID", ""),
            os.environ.get("BUILD_STAGES", ""),
            os.environ.get("LINUX_AMDGPU_FAMILIES", ""),
            any(
                os.environ.get(flag) == "true"
                for flag in ("RUNTIME_REPAIR", "COMM_REPAIR")
            ),
        )
        return

    from artifact_manager import _create_source_backend, parse_target_families
    from _therock_utils.artifact_backend import create_backend_from_env

    if args.mode == "snapshot":
        source = _create_source_backend(
            "linux", os.environ["BASELINE_RUN_ID"], None, None
        )
        targets = parse_target_families(
            argparse.Namespace(
                generic_only=False,
                amdgpu_families=os.environ["DIST_AMDGPU_FAMILIES"],
                expand_family_to_targets=True,
                amdgpu_targets=None,
            )
        )
        keys = snapshot_artifacts(source, targets)
        args.manifest.write_text(json.dumps(keys, indent=2) + "\n")
    else:
        keys = json.loads(args.manifest.read_text())
        check_artifacts(create_backend_from_env(platform="linux"), keys)
    print(f"Verified {len(keys)} archives and SHA256 sidecars")


if __name__ == "__main__":
    main()
