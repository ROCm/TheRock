#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read-only cross-arm parity check; no build, run, normalization or speedup claim."""

import argparse
import json
from pathlib import Path

import comm_stage_evidence as evidence


def read(root: Path, name: str):
    return json.loads((root / name).read_text())


def compare(baseline: Path, candidate: Path) -> dict:
    errors = []
    a, b = read(baseline, "manifest.json"), read(candidate, "manifest.json")
    if (a["mode"], b["mode"]) != (1, 2):
        errors.append("Expected fresh width1 versus width2")
    if not a.get("complete") or not b.get("complete"):
        errors.append("Both full trials and all postchecks must succeed")
    if errors:
        return dict(
            input_and_artifact_parity=False, errors=errors, performance_validated=False
        )
    for field in [
        "source_sha",
        "product_head",
        "reference_head",
        "sdk_run",
        "sdk_repository",
        "version",
        "runner_label",
        "container_digest",
        "cache_controls",
    ]:
        if a.get(field) != b.get(field):
            errors.append("Control mismatch:" + field)
    for name in [
        "source-identity.json",
        "cache-identity.json",
        "python-packages.json",
        "tool-versions.json",
        "link-inputs.json",
        "artifact-files.jsonl",
        "stage-cmake-args.txt",
    ]:
        if (baseline / name).read_bytes() != (candidate / name).read_bytes():
            errors.append("Exact input/artifact mismatch:" + name)
    command_a, command_b = read(baseline, "configure-command.json"), read(
        candidate, "configure-command.json"
    )
    flag_a = "-DTHEROCK_ROCSHMEM_NATIVE_LINK_JOBS=1"
    flag_b = "-DTHEROCK_ROCSHMEM_NATIVE_LINK_JOBS=2"
    if (
        command_a.count(flag_a) != 1
        or command_b.count(flag_b) != 1
        or [flag_a if arg == flag_b else arg for arg in command_b] != command_a
    ):
        errors.append("Root configure changed more than the native link knob")
    edges_a, edges_b = read(baseline, "final-link-edges.json"), read(
        candidate, "final-link-edges.json"
    )
    if set(edges_a) != set(evidence.TARGETS) or set(edges_b) != set(evidence.TARGETS):
        errors.append("Missing final native link edge")
    else:
        for target in evidence.TARGETS:
            try:
                evidence.validate_candidate(edges_a[target], edges_b[target])
            except RuntimeError as error:
                errors.append(f"Final link mismatch:{target}:{error}")
    for project in evidence.PROJECTS:
        prefix = "graphs/" + project + "/"
        for name in ["generated-and-object-manifest.jsonl", "compile_commands.json"]:
            pa, pb = baseline / (prefix + name), candidate / (prefix + name)
            if pa.is_file() != pb.is_file() or (
                pa.is_file() and pa.read_bytes() != pb.read_bytes()
            ):
                errors.append(f"Compile/header/object mismatch:{project}/{name}")

        def object_commands(root):
            # Include RCCL's custom object edges, not merely CMake compile_commands.
            rows = read(root, prefix + "all-edges.json")
            return {
                r["output"]: r
                for r in rows
                if Path(r["output"]).suffix.lower()
                in [".o", ".obj", ".a", ".hsaco", ".co", ".bc"]
            }

        if object_commands(baseline) != object_commands(candidate):
            errors.append("Object/custom-compile edge mismatch:" + project)
    for name in ["rocshmem-native.json", "rccl-native.json"]:
        if read(baseline, name) != read(candidate, name):
            errors.append("Native/per-image byte mismatch:" + name)
    profiles = [read(root, "profile/metadata.json") for root in [baseline, candidate]]
    resource_fields = [
        "visible_logical_cpus",
        "affinity",
        "effective_capacity_visible_upper_bound_cpus",
    ]
    resource_differences = {
        key: [row.get(key) for row in profiles]
        for key in resource_fields
        if profiles[0].get(key) != profiles[1].get(key)
    }
    if resource_differences:
        errors.append("Visible resource/affinity mismatch; inspect full raw cgroups")
    return dict(
        input_and_artifact_parity=not errors,
        errors=errors,
        resource_differences=resource_differences,
        performance_validated=False,
        actual_runners=[a.get("runner_name"), b.get("runner_name")],
        full_stage_wall_s=[
            a.get("full_stage_command_wall_s"),
            b.get("full_stage_command_wall_s"),
        ],
        limitation="Parity pass is not speedup/GPU validation. Compare all four runs, actual quotas/pressure, cache states and order/variability; separate overlapping native/project/stage times.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    evidence.save(args.output, result)
    if not result["input_and_artifact_parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
