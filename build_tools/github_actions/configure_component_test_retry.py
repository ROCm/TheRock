#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate a test baseline and configure targets for Component Test Retry."""

import argparse
import json
import os
import re
from urllib.parse import urlencode

from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    get_cpu_test_runner,
    load_external_runner_config,
)
from github_actions_api import GitHubAPI, GitHubAPIError, gha_set_output
from monitor_workflow_run import bounded_integer


VALID_PLATFORMS = {"linux", "windows", "wsl"}


def resolve_baseline(api: GitHubAPI, repository: str, run_id: str) -> str:
    """Use an explicit run, or select the latest successful scheduled release on main."""
    run_id = run_id.strip()
    if run_id:
        return run_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Artifact repository must be owner/repository")
    query = urlencode(
        {"branch": "main", "event": "schedule", "status": "success", "per_page": 1}
    )
    response = api.send_request(
        f"https://api.github.com/repos/{repository}/actions/workflows/multi_arch_release.yml/runs?{query}"
    )
    if not isinstance(response, dict) or not isinstance(
        response.get("workflow_runs"), list
    ):
        raise ValueError("GitHub returned an invalid workflow-run list")
    runs = response["workflow_runs"]
    if not runs:
        raise ValueError(
            f"No successful scheduled multi_arch_release.yml run found on {repository} main"
        )
    run = runs[0]
    if (
        not isinstance(run, dict)
        or run.get("head_branch") != "main"
        or run.get("event") != "schedule"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not isinstance(run.get("id"), int)
        or isinstance(run["id"], bool)
        or run["id"] < 1
    ):
        raise ValueError("GitHub returned a run that is not a successful main nightly")
    print(f"Selected latest successful nightly: {repository} run {run['id']}")
    return str(run["id"])


def select_targets(
    families: dict, requested_families: str, platforms: str, test_type: str
) -> list[dict[str, str]]:
    """Resolve family aliases and respect disabled runners and test-tier policies."""
    selected_platforms = {value.strip() for value in platforms.split(",")}
    if not selected_platforms or not selected_platforms <= VALID_PLATFORMS:
        raise ValueError(
            "Platforms must be a comma-separated subset of linux,windows,wsl"
        )
    aliases = {name.lower(): name for name in families}
    for name, entries in families.items():
        for info in entries.values():
            aliases[info["family"].lower()] = name
    requested = {
        value.strip().lower()
        for value in requested_families.split(",")
        if value.strip()
    }
    unknown = requested - aliases.keys()
    if unknown:
        raise ValueError(f"Unknown GPU families: {', '.join(sorted(unknown))}")
    selected = {aliases[value] for value in requested} if requested else set(families)
    cpu_runners = {
        platform: get_cpu_test_runner(platform) for platform in ("linux", "windows")
    }
    targets = []
    for name in sorted(selected):
        for platform in sorted(selected_platforms):
            artifact_platform = "linux" if platform == "wsl" else platform
            info = families[name].get(artifact_platform)
            if info is None:
                print(f"Skipping {name}/{platform}: platform not supported")
                continue
            runner_key = "test-runs-on-wsl" if platform == "wsl" else "test-runs-on"
            runner = info.get(runner_key, "")
            effective_tier = info.get("test_type_for_family") or test_type
            if not runner or not info.get("tests_on_trigger"):
                print(
                    f"Skipping {name}/{platform}: testing disabled or no GPU runner configured"
                )
                continue
            if info.get("run-full-tests-only") and effective_tier == "quick":
                print(
                    f"Skipping {name}/{platform}: quick tests disabled by family policy"
                )
                continue
            if "release" not in info.get("build_variants", []):
                print(f"Skipping {name}/{platform}: release artifacts not supported")
                continue
            if info.get("sanity_check_only_for_family"):
                print(
                    f"Skipping {name}/{platform}: only sanity tests enabled by family policy"
                )
                continue
            targets.append(
                {
                    "name": name,
                    "platform": platform,
                    "family": info["family"],
                    "targets": ",".join(info.get("fetch-gfx-targets", [])),
                    "runner": runner,
                    "cpu_runner": (
                        runner if platform == "wsl" else cpu_runners[platform]
                    ),
                    "test_type": effective_tier,
                }
            )
    if not targets:
        raise ValueError(
            "No requested architecture/platform targets have component test capacity"
        )
    return targets


def validate_baseline(api: GitHubAPI, repository: str, run_id: str) -> None:
    """Validate run identity and conclusion; artifacts are fetched by downstream tests."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Artifact repository must be owner/repository")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("Artifact run ID must be a positive integer")
    run = api.send_request(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
    )
    if (
        not isinstance(run, dict)
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise ValueError("Baseline must be a completed successful workflow run")
    print(f"Baseline: {run['html_url']}")
    print(
        "Baseline artifacts must cover selected release targets; downstream fetch validates availability"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolve-baseline-only",
        action="store_true",
        help="Determine and validate the artifact run without configuring targets",
    )
    parser.add_argument("--artifact-repository", default="ROCm/rockrel")
    parser.add_argument(
        "--artifact-run-id",
        default="",
        help="Known-good artifact run; blank selects the latest successful main nightly",
    )
    parser.add_argument("--families", default="")
    parser.add_argument("--platforms", default="linux,windows")
    parser.add_argument(
        "--test-type",
        choices=["quick", "standard", "comprehensive", "full"],
        default="standard",
    )
    parser.add_argument("--test-labels", default="")
    parser.add_argument("--max-retries", type=bounded_integer(0, 100), default=2)
    parser.add_argument("--poll-interval", type=bounded_integer(30, 900), default=60)
    parser.add_argument("--timeout-minutes", type=bounded_integer(1, 350), default=350)
    args = parser.parse_args(argv)
    try:
        if args.resolve_baseline_only:
            api = GitHubAPI()
            artifact_run_id = resolve_baseline(
                api, args.artifact_repository, args.artifact_run_id
            )
            validate_baseline(api, args.artifact_repository, artifact_run_id)
            gha_set_output({"artifact_run_id": artifact_run_id})
            return 0
        if load_external_runner_config() is None:
            raise ValueError(
                "A readable therock-ci-config checkout is required; refusing stale runner fallbacks"
            )
        repository = os.environ["GITHUB_REPOSITORY"]
        ref = os.environ["GITHUB_SHA"]
        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
        ) or not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError(
                "GITHUB_REPOSITORY and GITHUB_SHA must identify this workflow's checkout"
            )
        api = GitHubAPI()
        artifact_run_id = resolve_baseline(
            api, args.artifact_repository, args.artifact_run_id
        )
        validate_baseline(api, args.artifact_repository, artifact_run_id)
        targets = select_targets(
            get_all_families_for_trigger_types([]),
            args.families,
            args.platforms,
            args.test_type,
        )
        for target in targets:
            # Escape the value for insertion inside the YAML action's JSON string.
            target["test_labels"] = json.dumps(args.test_labels)[1:-1]
        gha_set_output(
            {
                "matrix": json.dumps({"include": targets}),
                "artifact_run_id": artifact_run_id,
            }
        )
        print(f"Configured {len(targets)} architecture/platform targets")
        return 0
    except (ValueError, KeyError, GitHubAPIError) as error:
        print(f"::error::{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
