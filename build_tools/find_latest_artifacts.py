#!/usr/bin/env python
"""Module and CLI script for finding the most recent CI artifacts from a branch.

This script
1. Queries the GitHub API for commits on the chosen branch
2. Invokes find_artifacts_for_commit to find CI artifacts
It skips over commits that are missing artifacts for any reason.

Usage:
    python find_latest_artifacts.py --amdgpu-family gfx94X-dcgpu

For script-to-script composition, import and call find_latest_artifacts():

    from find_latest_artifacts import find_latest_artifacts

    # Using the default branch, repository, etc.
    info = find_latest_artifacts(artifact_group="gfx94X-dcgpu")
    if info:
        print(f"Found artifacts at {info.s3_uri}")
"""

import argparse
import platform as platform_module
import sys

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    detect_repo_from_git,
    find_artifacts_for_commit,
)
from github_actions.github_actions_utils import gha_send_request


# TODO: move to github_actions_utils or github_utils for reuse in other files?
#       could also rename 'gha_send_request' to 'gh_send_request'
def get_recent_branch_commits_via_api(
    github_repository_name: str,
    branch: str = "main",
    max_count: int = 50,
) -> list[str]:
    """Gets the list of recent commit SHAs for a branch via the GitHub API.

    Commits could also be enumerated via local `git log` commands, but using
    the API ensures that we get the latest commits regardless of local
    repository state.

    Args:
        github_repository_name: Repository in "owner/repo" format
        branch: Branch name (default: "main")
        max_count: Maximum number of commits to retrieve (max 100 per API)

    Returns:
        List of commit SHAs, most recent first.

    Raises:
        GitHubAPIError: If GitHub API request fails.
    """
    url = f"https://api.github.com/repos/{github_repository_name}/commits?sha={branch}&per_page={max_count}"
    response = gha_send_request(url)

    return [commit["sha"] for commit in response]


def infer_default_branch_for_repo(github_repository_name: str) -> str:
    """Infers the default branch name for a repository.

    We could also look this up with an API call, as needed.

    Args:
        github_repository_name: Repository in "owner/repo" format

    Returns:
        Branch name (e.g., "main")
    """
    _, repo_name = github_repository_name.split("/")

    if repo_name == "TheRock":
        return "main"
    elif repo_name in ("rocm-libraries", "rocm-systems"):
        return "develop"
    else:
        # Default fallback
        return "develop"


def find_latest_artifacts(
    artifact_group: str,
    github_repository_name: str,
    workflow_file_name: str | None = None,
    branch: str | None = None,
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
        artifact_group: Artifact group to find (e.g., "gfx94X-dcgpu", ""gfx950-dcgpu-asan")
        github_repository_name: GitHub repository in "owner/repo" format
        workflow_file_name: Workflow filename, or None to infer from repo
        branch: Branch name to search (default: "main")
        platform: Target platform ("linux" or "windows"), or None for current
        max_commits: Maximum number of commits to search through
        verbose: If True, print progress information

    Returns:
        ArtifactRunInfo for the most recent commit with artifacts, or None
        if no matching commit found within max_commits.
    """
    if branch is None:
        branch = infer_default_branch_for_repo(github_repository_name)

    try:
        commits = get_recent_branch_commits_via_api(
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
            workflow_file_name=workflow_file_name,
            artifact_group=artifact_group,
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
        "--repo",
        type=str,
        help="Repository in 'owner/repo' format (default: detect from git remote)",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        help="Workflow filename that produces artifats (default: infer from repo, e.g. ci.yml in TheRock)",
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
        required=True,
        help="Artifact group (e.g., gfx94X-dcgpu, gfx950-dcgpu-asan)",
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
        help="Maximum commits to search (default: 50)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args(argv)

    repo = args.repo
    if repo is None:
        repo = detect_repo_from_git()

    info = find_latest_artifacts(
        artifact_group=args.artifact_group,
        github_repository_name=repo,
        workflow_file_name=args.workflow,
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

    info.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
