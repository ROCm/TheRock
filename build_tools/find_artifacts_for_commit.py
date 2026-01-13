#!/usr/bin/env python
"""Find CI artifacts for a given commit.

This script queries the GitHub API to find workflow runs for a commit and
returns information about where the artifacts are stored in S3.

Usage:
    python find_artifacts_for_commit.py \
        --commit abc123 \
        --repo ROCm/TheRock \
        --amdgpu-family gfx94X-dcgpu

For script-to-script composition, import and call find_artifacts_for_commit():

    from find_artifacts_for_commit import find_artifacts_for_commit, ArtifactRunInfo

    info = find_artifacts_for_commit(
        commit="abc123",
        repo="ROCm/TheRock",
        amdgpu_family="gfx94X-dcgpu",
    )
    if info:
        print(f"Artifacts at {info.s3_uri}")
"""

import argparse
from dataclasses import dataclass
import platform as platform_module
import re
import subprocess
import sys
import urllib.request
import urllib.error

from github_actions.github_actions_utils import (
    GitHubAPIError,
    gha_query_workflow_runs_for_commit,
    retrieve_bucket_info,
)


# TODO: wrap `ArtifactBackend` (or `S3Backend`) class here? Or use `BucketMetadata`?
#       (we have a few classes tracking similar metadata and reimplementing URL schemes)
@dataclass
class ArtifactRunInfo:
    """Information about a workflow run's artifacts."""

    git_commit_sha: str
    github_repository_name: str
    external_repo: str  # e.g. "ROCm-TheRock" (used for namespacing, may be empty)

    platform: str  # "linux" or "windows"
    amdgpu_family: str  # e.g., "gfx94X-dcgpu"

    workflow_file_name: str  # e.g. "ci.yml"
    workflow_run_id: str  # e.g. "12345678901"
    workflow_run_status: str  # "completed", "in_progress", etc.
    workflow_run_conclusion: str | None  # "success", "failure", None if in_progress
    workflow_run_html_url: str

    s3_bucket: str  # e.g. "therock-ci-artifacts"

    @property
    def git_commit_url(self) -> str:
        return f"https://github.com/{self.github_repository_name}/commit/{self.git_commit_sha}"

    @property
    def s3_path(self) -> str:
        return f"{self.external_repo}{self.workflow_run_id}-{self.platform}/"

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_path}"

    @property
    def s3_index_url(self) -> str:
        return f"https://{self.s3_bucket}.s3.amazonaws.com/{self.s3_path}index-{self.amdgpu_family}.html"


def check_artifacts_exist(info: ArtifactRunInfo) -> bool:
    """Check if artifacts exist at the expected S3 location.

    Performs an HTTP HEAD request to the S3 index URL to verify artifacts
    have been uploaded. This is useful because artifacts for a specific GPU
    family may be available early, before all workflow jobs are complete.

    Args:
        info: ArtifactRunInfo with the S3 location to check.

    Returns:
        True if artifacts exist (HTTP 200), False otherwise.
    """
    try:
        request = urllib.request.Request(info.s3_index_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def infer_workflow_for_repo(github_repository_name: str) -> str:
    """Infers the standard workflow file that produces artifacts for a repository.

    Args:
        github_repository_name: Repository in "owner/repo" format

    Returns:
        Workflow filename (e.g., "ci.yml")
    """
    _, repo_name = github_repository_name.split("/")

    if repo_name == "TheRock":
        return "ci.yml"
    elif repo_name in ("rocm-libraries", "rocm-systems"):
        return "therock-ci.yml"
    else:
        # Default fallback
        return "ci.yml"


def detect_repo_from_git() -> str | None:
    """Detects the github repository name based on git remotes.

    Looks for any remote pointing to github.com/ROCm/*.
    Falls back to ROCm/TheRock if no ROCm remote found.

    Returns:
        Repository in "owner/repo" format, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get-regexp", r"remote\..*\.url"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Look for ROCm repos in any remote URL
        # Matches both SSH (git@github.com:ROCm/X) and HTTPS (github.com/ROCm/X)
        for line in result.stdout.splitlines():
            match = re.search(r"github\.com[:/](ROCm/[^/\s]+)", line)
            if match:
                return match.group(1).removesuffix(".git")

        # No ROCm remote found - default to TheRock
        return "ROCm/TheRock"
    except subprocess.CalledProcessError:
        # Not in a git repo or git not available
        return None


def _build_artifact_run_info(
    commit: str,
    github_repository_name: str,
    amdgpu_family: str,
    workflow_file_name: str,
    platform: str,
    workflow_run: dict,
) -> ArtifactRunInfo:
    """Builds ArtifactRunInfo from a workflow run dict."""
    external_repo, bucket = retrieve_bucket_info(
        github_repository=github_repository_name,
        workflow_run=workflow_run,
    )

    return ArtifactRunInfo(
        git_commit_sha=commit,
        github_repository_name=github_repository_name,
        external_repo=external_repo,
        workflow_file_name=workflow_file_name,
        workflow_run_id=str(workflow_run["id"]),
        workflow_run_status=workflow_run.get("status", "unknown"),
        workflow_run_conclusion=workflow_run.get("conclusion"),
        workflow_run_html_url=workflow_run.get("html_url", ""),
        platform=platform,
        amdgpu_family=amdgpu_family,
        s3_bucket=bucket,
    )


def find_artifacts_for_commit(
    commit: str,
    github_repository_name: str,
    amdgpu_family: str,
    workflow_file_name: str | None = None,
    platform: str | None = None,
) -> ArtifactRunInfo | None:
    """Main entry point: finds artifact info for a commit.

    Queries GitHub for workflow runs on this commit, then checks each run
    (most recent first) for available artifacts. Returns the first run
    where artifacts exist, or None if no artifacts are found.

    A commit may have multiple workflow runs if the workflow was retriggered.
    This function finds the first run with actual artifacts available in S3.

    Args:
        commit: Git commit SHA (full or abbreviated)
        github_repository_name: Repository in "owner/repo" format
        amdgpu_family: GPU family (e.g., "gfx94X-dcgpu")
        workflow_file_name: Workflow filename, or None to infer from repo
        platform: "linux" or "windows", or None for current platform

    Returns:
        ArtifactRunInfo for the first run with artifacts, or None if no
        workflow runs exist or no artifacts are available.
    """
    if workflow_file_name is None:
        workflow_file_name = infer_workflow_for_repo(github_repository_name)

    if platform is None:
        platform = platform_module.system().lower()

    try:
        workflow_runs = gha_query_workflow_runs_for_commit(
            github_repository_name, workflow_file_name, commit
        )
    except GitHubAPIError as e:
        print(f"Error querying GitHub API: {e}", file=sys.stderr)
        return None

    if not workflow_runs:
        return None

    # Find the first workflow run with available artifacts
    for workflow_run in workflow_runs:
        info = _build_artifact_run_info(
            commit=commit,
            github_repository_name=github_repository_name,
            amdgpu_family=amdgpu_family,
            workflow_file_name=workflow_file_name,
            platform=platform,
            workflow_run=workflow_run,
        )

        if check_artifacts_exist(info):
            return info

    # No runs had artifacts available
    return None


# TODO: move into ArtifactRunInfo itself? Make a `class` instead of `dataclass`?
def print_artifact_info(info: ArtifactRunInfo) -> None:
    """Prints artifact info in human-readable format."""
    status_str = info.workflow_run_status
    if info.workflow_run_conclusion:
        status_str = f"{info.workflow_run_status} ({info.workflow_run_conclusion})"

    print(f"Git repository:      {info.github_repository_name}")
    print(f"Git commit:          {info.git_commit_sha}")
    print(f"Git commit URL:      {info.git_commit_url}")
    print(f"Platform:            {info.platform}")
    print(f"GPU Family:          {info.amdgpu_family}")
    print(f"Workflow name:       {info.workflow_file_name}")
    print(f"Workflow run ID:     {info.workflow_run_id}")
    print(f"Workflow run URL:    {info.workflow_run_html_url}")
    print(f"Workflow run status: {status_str}")
    print(f"S3 Bucket:           {info.s3_bucket}")
    print(f"S3 Path:             {info.s3_path}")
    print(f"S3 Index:            {info.s3_index_url}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find CI artifacts for a given commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Git commit SHA to find artifacts for",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository in 'owner/repo' format (default: detect from git remote)",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        help="Workflow filename (default: infer from repo)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows"],
        default=platform_module.system().lower(),
        help=f"Platform (default: {platform_module.system().lower()})",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="GPU family (e.g., gfx94X-dcgpu, gfx110X-all)",
    )

    args = parser.parse_args(argv)

    repo = args.repo
    if repo is None:
        repo = detect_repo_from_git()
        if repo is None:
            print("Error: Could not detect repository. Use --repo.", file=sys.stderr)
            return 2

    info = find_artifacts_for_commit(
        commit=args.commit,
        github_repository_name=repo,
        amdgpu_family=args.amdgpu_family,
        workflow_file_name=args.workflow,
        platform=args.platform,
    )

    if info is None:
        print(
            f"No artifacts found for commit {args.commit} "
            f"(platform={args.platform}, amdgpu_family={args.amdgpu_family})",
            file=sys.stderr,
        )
        return 1

    print_artifact_info(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
