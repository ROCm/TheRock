# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve one ROCm runtime revision and a caller-selected native test matrix."""

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    select_weighted_label,
)
from detect_external_repo_config import build_external_repo_config
from fetch_test_configurations import _build_container_options
from github_actions_api import (
    gha_append_step_summary,
    gha_resolve_git_ref,
    gha_set_output,
)


def configure_matrix(matrix_json: str, *, run_hip: bool, runner_families: dict) -> dict:
    entries = json.loads(matrix_json)
    if not isinstance(entries, list):
        raise ValueError("arch_matrix must be a JSON array")
    if not entries:
        if run_hip:
            raise ValueError("HIP execution requires a nonempty arch_matrix")
        return {
            "include": [
                {
                    "profile": "host",
                    "amdgpu_family": "",
                    "amdgpu_targets": "",
                    "test_runs_on": '""',
                    "container_options": "",
                }
            ]
        }
    if len(entries) > 256:
        raise ValueError("arch_matrix exceeds GitHub Actions' 256-job limit")

    targets_by_family = amdgpu_family_map()
    linux_families = {
        config["linux"]["family"]: config["linux"]
        for config in runner_families.values()
        if "linux" in config
    }
    result = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"amdgpu_family": entry}
        if not isinstance(entry, dict) or set(entry) - {
            "amdgpu_family",
            "amdgpu_targets",
            "test_runs_on",
        }:
            raise ValueError(f"Invalid arch_matrix entry: {entry!r}")
        family = entry.get("amdgpu_family")
        if not isinstance(family, str) or family not in targets_by_family:
            raise ValueError(f"Unknown AMDGPU family or target: {family!r}")
        targets = entry.get("amdgpu_targets", targets_by_family[family])
        if isinstance(targets, str):
            targets = targets.split(";")
        if not isinstance(targets, list) or not targets:
            raise ValueError("amdgpu_targets must be a nonempty list or semicolon list")
        for target in targets:
            if (
                not isinstance(target, str)
                or target.split(":")[0] not in targets_by_family[family]
                or any(
                    not re.fullmatch(r"[a-zA-Z0-9_-]+[+-]", feature)
                    for feature in target.split(":")[1:]
                )
            ):
                raise ValueError(f"Target {target!r} does not belong to {family}")

        runner = entry.get("test_runs_on")
        if runner is None:
            config = linux_families.get(family, {})
            if config.get("test-runs-on-labels"):
                runner = select_weighted_label(
                    config["test-runs-on-labels"], f"rocddi-{family}"
                )
            else:
                runner = config.get("test-runs-on", "")
        labels = [runner] if isinstance(runner, str) else runner
        if (
            not isinstance(labels, list)
            or not labels
            or not all(
                isinstance(label, str) and label.strip() and "\n" not in label
                for label in labels
            )
        ):
            raise ValueError(
                f"No Linux test runner for {family}; supply test_runs_on explicitly"
            )
        result.append(
            {
                "profile": f"{index}-{family}",
                "amdgpu_family": family,
                "amdgpu_targets": ";".join(dict.fromkeys(targets)),
                "test_runs_on": json.dumps(runner),
                "container_options": _build_container_options({}, "linux", family)[
                    "container_options"
                ],
            }
        )
    return {"include": result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-systems-ref", default="shared/rocddi-cts-dev")
    parser.add_argument("--arch-matrix", default="[]")
    parser.add_argument("--suites", choices=["both", "cts", "hip"], default="both")
    args = parser.parse_args(argv)
    try:
        matrix = configure_matrix(
            args.arch_matrix,
            run_hip=args.suites != "cts",
            runner_families=get_all_families_for_trigger_types([]),
        )
        commit = gha_resolve_git_ref("ROCm/rocm-systems", args.rocm_systems_ref)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError("ROCm/rocm-systems did not resolve to a full commit SHA")
    except ValueError as error:
        parser.error(str(error))
    source = asdict(build_external_repo_config("ROCm/rocm-systems", ref=commit))
    gha_set_output(
        {
            "matrix": json.dumps(matrix),
            "external_repo_config": json.dumps(source),
            "rocm_systems_sha": commit,
            "run_cts": str(args.suites != "hip").lower(),
            "run_hip": str(args.suites != "cts").lower(),
        }
    )
    gha_append_step_summary(
        "### Runtime source and matrix\n\n```json\n"
        + json.dumps(
            {
                "rocm_systems_ref": args.rocm_systems_ref,
                "rocm_systems_commit": commit,
                "suites": args.suites,
                "matrix": matrix,
            },
            indent=2,
        )
        + "\n```\n"
    )


if __name__ == "__main__":
    main()
