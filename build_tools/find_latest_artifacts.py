#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Module and CLI script for finding the most recent CI artifacts from a branch.

This script
1. Queries the GitHub API for commits on the chosen branch
2. Invokes find_artifacts_for_commit to find CI artifacts
It skips over commits that are missing artifacts for any reason.

Usage:
    python find_latest_artifacts.py --artifact-group gfx94X-dcgpu
    python find_latest_artifacts.py --artifact-group gfx110X-all gfx120X-all

For script-to-script composition:

    from find_latest_artifacts import find_latest_artifacts

    # Single group
    results = find_latest_artifacts(artifact_groups=["gfx94X-dcgpu"])

    # Multiple groups — finds the most recent commit with ALL groups
    results = find_latest_artifacts(artifact_groups=["gfx110X-all", "gfx120X-all"])
    if results:
        for info in results:
            print(f"Found artifacts at {info.s3_uri}")
"""

import argparse
from contextlib import nullcontext, redirect_stdout
import json
import platform as platform_module
import re
import sys
from typing import Sequence

from botocore.exceptions import BotoCoreError, ClientError

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    find_artifacts_for_commit,
    find_artifacts_for_run,
)
from github_actions.github_actions_api import (
    GitHubAPIError,
    gha_query_recent_branch_commits,
    gha_resolve_git_ref,
)


def find_latest_artifacts(
    artifact_groups: list[str],
    github_repository_name: str = "ROCm/TheRock",
    workflow_file_name: str = "multi_arch_ci.yml",
    platform: str = platform_module.system().lower(),
    branch: str = "main",
    max_commits: int = 50,
    verbose: bool = False,
    *,
    ref: str | None = None,
    run_id: str | None = None,
    amdgpu_targets: Sequence[str] = (),
    required_artifact_patterns: Sequence[str] = (),
    require_single_run: bool = False,
    require_successful_run: bool = False,
) -> list[ArtifactRunInfo] | None:
    """Find artifacts using branch history, an exact ref, or a workflow run ID.

    For branch searches, walks recent commits and checks concrete S3 archives
    until all requested artifact groups are found. With ``ref`` or ``run_id``,
    inspects only the selected commit or workflow run.

    By default, artifact archives may be inspected before a workflow completes or
    after a workflow fails. Set ``require_successful_run`` to inspect only runs
    whose GitHub status is ``completed`` and whose conclusion is ``success``.

    Args:
        artifact_groups: Artifact groups to find, such as
            ``["gfx94X-dcgpu", "gfx120X-all"]``.
        github_repository_name: GitHub repository in ``owner/repo`` format.
        workflow_file_name: Workflow filename, or None to infer from the
            repository.
        platform: Target platform, such as ``linux`` or ``windows``.
        branch: Branch name to search. Defaults to ``main``.
        max_commits: Maximum number of branch commits to search.
        verbose: Print progress information when True.
        ref: Resolve one exact branch, tag, or commit and inspect only that SHA.
        run_id: Inspect one explicit GitHub Actions workflow run.
        amdgpu_targets: Individual GPU targets that may appear instead of an
            artifact family name.
        required_artifact_patterns: Regular-expression patterns that must each
            match at least one filename in every requested group's filtered
            artifact set.
        require_single_run: Require all requested groups to be found under the
            same workflow run.
        require_successful_run: Only inspect runs whose GitHub status is
            ``completed`` and whose conclusion is ``success``.

    Returns:
        Artifact information in requested group order when all requested groups
        are found; otherwise None.

    Raises:
        GitHubAPIError: If a GitHub API request fails.
        BotoCoreError: If the S3 artifact lookup fails.
        ClientError: If the S3 artifact lookup fails.
        ValueError: If both ``ref`` and ``run_id`` are provided, or if explicit
            AMDGPU targets are provided with more than one artifact group.

    """
    if ref and run_id:
        raise ValueError("ref and run_id are mutually exclusive")

    if run_id:
        results = find_artifacts_for_run(
            workflow_run_id=run_id,
            artifact_groups=artifact_groups,
            github_repository_name=github_repository_name,
            workflow_file_name=workflow_file_name,
            platform=platform,
            amdgpu_targets=amdgpu_targets,
            required_artifact_patterns=required_artifact_patterns,
            require_successful_run=require_successful_run,
        )
        return results or None

    if ref:
        resolved_commit = gha_resolve_git_ref(
            github_repository=github_repository_name,
            ref=ref,
        )
        commits = [resolved_commit]
    else:
        commits = gha_query_recent_branch_commits(
            github_repository_name=github_repository_name,
            branch=branch,
            max_count=max_commits,
        )

    if verbose:
        if ref:
            search_location = f"ref {ref}"
        else:
            search_location = f"branch {branch}"

        print(
            f"Searching {len(commits)} commit(s) from {search_location} in "
            f"{github_repository_name} for {len(artifact_groups)} group(s): "
            f"{', '.join(artifact_groups)}...",
            file=sys.stderr,
        )

    for i, commit in enumerate(commits):
        if verbose:
            print(
                f"  [{i + 1}/{len(commits)}] Checking {commit[:8]}...",
                file=sys.stderr,
            )

        results = find_artifacts_for_commit(
            commit=commit,
            artifact_groups=artifact_groups,
            github_repository_name=github_repository_name,
            workflow_file_name=workflow_file_name,
            platform=platform,
            amdgpu_targets=amdgpu_targets,
            required_artifact_patterns=required_artifact_patterns,
            require_single_run=require_single_run,
            require_successful_run=require_successful_run,
        )

        if not results:
            if verbose:
                print("    No artifacts found", file=sys.stderr)
            continue

        if len(results) < len(artifact_groups):
            found_groups = [r.artifact_group for r in results]
            if verbose:
                print(
                    f"    Partial: found {', '.join(found_groups)} "
                    f"(need all {len(artifact_groups)})",
                    file=sys.stderr,
                )
            continue

        if verbose:
            run_ids = sorted(set(r.workflow_run_id for r in results))
            print(
                f"    Found all {len(artifact_groups)} group(s): "
                f"run(s) {', '.join(run_ids)}",
                file=sys.stderr,
            )

        return results

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find CI artifacts by branch history, exact ref, or workflow run ID"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selector = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--repo",
        type=str,
        default="ROCm/TheRock",
        help="Repository in 'owner/repo' format",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="multi_arch_ci.yml",
        help="Workflow filename that produces artifacts",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows"],
        default=platform_module.system().lower(),
        help=f"Platform (default: {platform_module.system().lower()})",
    )
    parser.add_argument(
        "--artifact-group",
        type=str,
        nargs="+",
        required=True,
        help="Artifact group(s) (e.g., gfx94X-dcgpu gfx120X-all)",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=50,
        help="Maximum commits to search (default: 50, max: 100)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--amdgpu-target",
        type=str,
        action="append",
        default=[],
        help=(
            "Individual AMDGPU target that may appear in split artifacts. "
            "May be repeated, for example: --amdgpu-target gfx942"
        ),
    )
    parser.add_argument(
        "--require-artifact",
        type=str,
        action="append",
        default=[],
        help=(
            "Regex that must match at least one concrete artifact filename. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Require all requested artifact groups to exist in one workflow run",
    )
    parser.add_argument(
        "--require-successful-run",
        action="store_true",
        help=("Only accept workflow runs with status=completed and conclusion=success"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    selector.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch whose recent commits should be searched (default: main)",
    )
    selector.add_argument(
        "--ref",
        type=str,
        help="Resolve one exact branch, tag, or commit and inspect only that SHA",
    )
    selector.add_argument(
        "--run-id",
        type=str,
        help="Inspect one explicit GitHub Actions workflow run",
    )

    args = parser.parse_args(argv)

    try:
        lookup_output = redirect_stdout(sys.stderr) if args.json else nullcontext()

        with lookup_output:
            results = find_latest_artifacts(
                artifact_groups=args.artifact_group,
                github_repository_name=args.repo,
                workflow_file_name=args.workflow,
                platform=args.platform,
                branch=args.branch or "main",
                ref=args.ref,
                run_id=args.run_id,
                max_commits=args.max_commits,
                amdgpu_targets=args.amdgpu_target,
                required_artifact_patterns=args.require_artifact,
                require_single_run=args.single_run,
                require_successful_run=args.require_successful_run,
                verbose=args.verbose,
            )
    except (GitHubAPIError, BotoCoreError, ClientError, ValueError, re.error) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if results is None:
        if args.run_id:
            location = f"workflow run {args.run_id}"
        elif args.ref:
            location = f"ref {args.ref}"
        else:
            location = (
                f"last {args.max_commits} commits on "
                f"{args.repo}/{args.branch or 'main'}"
            )

        print(
            f"No matching artifacts found for {location}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "status": "found",
                    "results": [info.to_dict() for info in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for info in results:
            info.print()
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
