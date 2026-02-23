#!/usr/bin/env python
"""Module and CLI script for finding CI artifacts for a given commit.

This script queries the GitHub API to find workflow runs for a commit and
returns information about where the artifacts are stored in S3.

Usage:
    python find_artifacts_for_commit.py \
        --commit abc123 \
        --repo ROCm/TheRock \
        --artifact-group gfx94X-dcgpu

For script-to-script composition:

    from find_artifacts_for_commit import find_artifacts_for_commit, ArtifactRunInfo

    info = find_artifacts_for_commit(
        commit="abc123",
        repo="ROCm/TheRock",
        artifact_group="gfx94X-dcgpu",
    )
    if info:
        print(f"Artifacts at {info.s3_uri}")
"""

import argparse
from dataclasses import dataclass
import platform as platform_module
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
    artifact_group: str  # e.g., "gfx94X-dcgpu", "gfx950-dcgpu-asan"

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
        return f"https://{self.s3_bucket}.s3.amazonaws.com/{self.s3_path}index-{self.artifact_group}.html"

    def print(self):
        """Prints artifact info in a human-readable format."""
        status_str = self.workflow_run_status
        if self.workflow_run_conclusion:
            status_str = f"{self.workflow_run_status} ({self.workflow_run_conclusion})"

        print(f"Artifact info:")
        print(f"  Git repository:      {self.github_repository_name}")
        print(f"  Git commit:          {self.git_commit_sha}")
        print(f"  Git commit URL:      {self.git_commit_url}")
        print(f"  Platform:            {self.platform}")
        print(f"  Artifact group:      {self.artifact_group}")
        print(f"  Workflow name:       {self.workflow_file_name}")
        print(f"  Workflow run ID:     {self.workflow_run_id}")
        print(f"  Workflow run URL:    {self.workflow_run_html_url}")
        print(f"  Workflow run status: {status_str}")
        print(f"  S3 Bucket:           {self.s3_bucket}")
        print(f"  S3 Path:             {self.s3_path}")
        print(f"  S3 Index:            {self.s3_index_url}")


def _build_artifact_run_info(
    commit: str,
    github_repository_name: str,
    artifact_group: str,
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
        artifact_group=artifact_group,
        s3_bucket=bucket,
    )


def check_if_artifacts_exist(info: ArtifactRunInfo) -> bool:
    """Checks if artifacts exist at the expected S3 location.

    Performs an HTTP HEAD request to the S3 index URL to verify artifacts
    have been uploaded. Note that this does not guarantee that all artifacts
    exist. Artifacts could be partially uploaded.

    TODO(scotttodd): plumb through a list of artifact keys to check for, then
       use `ArtifactBackend::artifact_exists(artifact_key)`

    Args:
        info: ArtifactRunInfo with the S3 location to check

    Returns:
        True if artifacts are likely to exist, False otherwise
    """
    try:
        request = urllib.request.Request(info.s3_index_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def find_artifacts_for_commit(
    commit: str,
    artifact_group: str,
    github_repository_name: str = "ROCm/TheRock",
    workflow_file_name: str = "ci.yml",
    platform: str = platform_module.system().lower(),
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
        artifact_group: GPU family (e.g., "gfx94X-dcgpu", "gfx950-dcgpu-asan")
        workflow_file_name: Workflow filename, or None to infer from repo
        platform: "linux" or "windows", or None for current platform

    Returns:
        ArtifactRunInfo for the first run with artifacts, or None if no
        workflow runs exist or no artifacts are available.

    Raises:
        GitHubAPIError: If the GitHub API request fails (rate limit, network
            error, etc.). Callers should handle this to distinguish between
            "no artifacts found" (None) and "couldn't check" (exception).
    """
    workflow_runs = gha_query_workflow_runs_for_commit(
        github_repository_name, workflow_file_name, commit
    )

    if not workflow_runs:
        return None

    # Find the first workflow run with available artifacts
    for workflow_run in workflow_runs:
        info = _build_artifact_run_info(
            commit=commit,
            github_repository_name=github_repository_name,
            artifact_group=artifact_group,
            workflow_file_name=workflow_file_name,
            platform=platform,
            workflow_run=workflow_run,
        )

        if check_if_artifacts_exist(info):
            return info

    # No runs had artifacts available
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find CI artifacts for a given commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--commit",
        type=str,
        required=True,
        help="Git commit SHA to find artifacts for (full SHA)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ROCm/TheRock",
        help="Repository in 'owner/repo' format (default: ROCm/TheRock)",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="ci.yml",
        help="Workflow filename that produces artifacts (default: ci.yml)",
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

    args = parser.parse_args(argv)

    try:
        info = find_artifacts_for_commit(
            commit=args.commit,
            github_repository_name=args.repo,
            workflow_file_name=args.workflow,
            platform=args.platform,
            artifact_group=args.artifact_group,
        )
    except GitHubAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if info is None:
        print(
            f"No artifacts found for commit {args.commit} "
            f"(platform={args.platform}, artifact_group={args.artifact_group})",
            file=sys.stderr,
        )
        return 1

    info.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
