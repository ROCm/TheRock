#!/usr/bin/env python
"""Find the most recent commit on main with CI artifacts.

This script queries the GitHub API for commits on the main branch and finds
the most recent one that has a completed workflow run with artifacts.

Usage:
    python find_latest_artifacts.py --amdgpu-family gfx94X-dcgpu

For script-to-script composition, import and call find_latest_artifacts():

    from find_latest_artifacts import find_latest_artifacts

    info = find_latest_artifacts(
        amdgpu_family="gfx94X-dcgpu",
    )
    if info:
        print(f"Found artifacts at {info.s3_uri}")
"""

import argparse
import platform as platform_module
import sys

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    check_artifacts_exist,
    find_artifacts_for_commit,
    print_artifact_info,
)
from github_actions.github_actions_utils import gha_send_request


def get_branch_commits(
    github_repository_name: str,
    branch: str = "main",
    max_count: int = 50,
) -> list[str]:
    """Get commit SHAs from a branch via GitHub API.

    Args:
        github_repository_name: Repository in "owner/repo" format.
        branch: Branch name (default: "main").
        max_count: Maximum number of commits to retrieve (max 100 per API).

    Returns:
        List of commit SHAs, most recent first.

    Raises:
        Exception: If GitHub API request fails.
    """
    url = f"https://api.github.com/repos/{github_repository_name}/commits?sha={branch}&per_page={max_count}"
    response = gha_send_request(url)

    return [commit["sha"] for commit in response]


def find_latest_artifacts(
    amdgpu_family: str,
    github_repository_name: str = "ROCm/TheRock",
    branch: str = "main",
    platform: str | None = None,
    max_commits: int = 50,
    verbose: bool = False,
) -> ArtifactRunInfo | None:
    """Find the most recent commit on a branch with artifacts.

    Searches through commits on the branch and checks if artifacts actually
    exist in S3 for the requested GPU family. This handles cases where:
    - A workflow is still in progress but artifacts for this family are uploaded
    - A workflow failed for other families but this family succeeded

    Args:
        amdgpu_family: GPU family for S3 index URL (e.g., "gfx94X-dcgpu").
        github_repository_name: GitHub repository in "owner/repo" format.
        branch: Branch name to search (default: "main").
        platform: Target platform ("linux" or "windows"), or None for current.
        max_commits: Maximum number of commits to search through.
        verbose: If True, print progress information.

    Returns:
        ArtifactRunInfo for the most recent commit with artifacts, or None
        if no matching commit found within max_commits.
    """
    if platform is None:
        platform = platform_module.system().lower()

    try:
        commits = get_branch_commits(
            github_repository_name=github_repository_name,
            branch=branch,
            max_count=max_commits,
        )
    except Exception as e:
        print(f"Error getting commits from GitHub: {e}", file=sys.stderr)
        return None

    if verbose:
        print(
            f"Searching {len(commits)} commits on {github_repository_name}/{branch}...",
            file=sys.stderr,
        )

    for i, commit in enumerate(commits):
        if verbose:
            print(
                f"  [{i + 1}/{len(commits)}] Checking {commit[:8]}...",
                file=sys.stderr,
            )

        info = find_artifacts_for_commit(
            commit=commit,
            github_repository_name=github_repository_name,
            amdgpu_family=amdgpu_family,
            platform=platform,
        )

        if info is None:
            if verbose:
                print("    No workflow run found", file=sys.stderr)
            continue

        if verbose:
            print(
                f"    Found artifacts: run {info.workflow_run_id}",
                file=sys.stderr,
            )

        return info

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find the most recent commit on main with CI artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="GPU family (e.g., gfx94X-dcgpu, gfx110X-all)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ROCm/TheRock",
        help="Repository in 'owner/repo' format (default: ROCm/TheRock)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch name to search (default: main)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows"],
        default=platform_module.system().lower(),
        help=f"Platform (default: {platform_module.system().lower()})",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=50,
        help="Maximum commits to search (default: 50)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args(argv)

    info = find_latest_artifacts(
        amdgpu_family=args.amdgpu_family,
        github_repository_name=args.repo,
        branch=args.branch,
        platform=args.platform,
        max_commits=args.max_commits,
        verbose=args.verbose,
    )

    if info is None:
        print(
            f"No artifacts found in last {args.max_commits} commits on {args.repo}/{args.branch}",
            file=sys.stderr,
        )
        return 1

    print_artifact_info(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
