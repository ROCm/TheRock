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
import platform as platform_module
import sys

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    find_artifacts_for_commit,
)
from github_actions.github_actions_utils import (
    GitHubAPIError,
    gha_query_recent_branch_commits,
)


def find_latest_artifacts(
    artifact_groups: list[str],
    github_repository_name: str = "ROCm/TheRock",
    workflow_file_name: str = "ci.yml",
    platform: str = platform_module.system().lower(),
    branch: str = "main",
    max_commits: int = 50,
    verbose: bool = False,
) -> list[ArtifactRunInfo] | None:
    """Find the most recent commit on a branch with artifacts for all groups.

    Searches through commits on the branch and checks if artifacts actually
    exist in S3 for ALL requested GPU families. This handles cases where:
    - A workflow is still in progress but artifacts for some families are uploaded
    - A workflow failed for some families but others succeeded

    Args:
        artifact_groups: Artifact groups to find (e.g., ["gfx94X-dcgpu", "gfx120X-all"])
        github_repository_name: GitHub repository in "owner/repo" format
        workflow_file_name: Workflow filename, or None to infer from repo
        branch: Branch name to search (default: "main")
        platform: Target platform ("linux" or "windows"), or None for current
        max_commits: Maximum number of commits to search through
        verbose: If True, print progress information

    Returns:
        List of ArtifactRunInfo (one per group) for the most recent commit
        where ALL groups have artifacts, or None if no such commit found.

    Raises:
        GitHubAPIError: If the GitHub API request fails (rate limit, network
            error, etc.).
    """
    commits = gha_query_recent_branch_commits(
        github_repository_name=github_repository_name,
        branch=branch,
        max_count=max_commits,
    )

    if verbose:
        print(
            f"Searching {len(commits)} commits on {github_repository_name}/{branch} "
            f"for {len(artifact_groups)} group(s): {', '.join(artifact_groups)}...",
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
        description="Find the most recent commit on a branch with CI artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--repo",
        type=str,
        default="ROCm/TheRock",
        help="Repository in 'owner/repo' format (default: detect from git remote)",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="ci.yml",
        help="Workflow filename that produces artifacts (default: infer from repo, e.g. ci.yml in TheRock)",
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
        "--branch",
        type=str,
        default="main",
        help="Branch name to search (default: main)",
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

    args = parser.parse_args(argv)

    try:
        results = find_latest_artifacts(
            artifact_groups=args.artifact_group,
            github_repository_name=args.repo,
            workflow_file_name=args.workflow,
            platform=args.platform,
            branch=args.branch,
            max_commits=args.max_commits,
            verbose=args.verbose,
        )
    except GitHubAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if results is None:
        print(
            f"No artifacts found in last {args.max_commits} commits on {args.repo}/{args.branch}",
            file=sys.stderr,
        )
        return 1

    for info in results:
        info.print()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
