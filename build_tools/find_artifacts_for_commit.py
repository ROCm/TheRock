#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Module and CLI script for finding CI artifacts for a given commit.

This script queries the GitHub API to find workflow runs for a commit and
returns information about where the artifacts are stored in S3.

Usage:
    python find_artifacts_for_commit.py \
        --commit abc123 \
        --repo ROCm/TheRock \
        --artifact-group gfx94X-dcgpu gfx120X-all

For script-to-script composition:

    from find_artifacts_for_commit import find_artifacts_for_commit, ArtifactRunInfo

    results = find_artifacts_for_commit(
        commit="abc123",
        artifact_groups=["gfx94X-dcgpu", "gfx120X-all"],
    )
    for info in results:
        print(f"Artifacts at {info.s3_uri}")
"""

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass, field
import json
import platform as platform_module
import re
import sys
from typing import Sequence

from botocore.exceptions import BotoCoreError, ClientError

from _therock_utils.artifact_backend import S3Backend
from _therock_utils.artifacts import ArtifactName
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.github_actions_api import (
    GitHubAPIError,
    gha_query_workflow_run_by_id,
    gha_query_workflow_runs_for_commit,
)
from _therock_utils.cmake_amdgpu_targets import (
    amdgpu_family_map,
    expand_families,
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

    # Artifact inspection constraints.
    amdgpu_targets: tuple[str, ...] = ()
    required_artifact_patterns: tuple[str, ...] = ()

    # Populated during artifact inspection.
    artifact_filenames: tuple[str, ...] = ()
    missing_required_artifact_patterns: tuple[str, ...] = ()

    # Internal data used to inspect the run's artifact backend. Excluded from
    # repr/comparison and JSON output.
    _output_root: WorkflowOutputRoot | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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
        return f"https://{self.s3_bucket}.s3.amazonaws.com/{self.s3_path}index.html"

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
        if self.amdgpu_targets:
            print("  AMDGPU targets:      " + ", ".join(self.amdgpu_targets))
        print(f"  Workflow name:       {self.workflow_file_name}")
        print(f"  Workflow run ID:     {self.workflow_run_id}")
        print(f"  Workflow run URL:    {self.workflow_run_html_url}")
        print(f"  Workflow run status: {status_str}")
        print(f"  S3 Bucket:           {self.s3_bucket}")
        print(f"  S3 Path:             {self.s3_path}")
        print(f"  S3 Index:            {self.s3_index_url}")
        print(f"  Artifact count:      {len(self.artifact_filenames)}")

        if self.artifact_filenames:
            print("  Artifacts:")
            for filename in self.artifact_filenames:
                print(f"    {filename}")

        if self.missing_required_artifact_patterns:
            print("  Missing required patterns:")
            for pattern in self.missing_required_artifact_patterns:
                print(f"    {pattern}")

    def to_dict(self) -> dict[str, object]:
        """Return stable machine-readable artifact information."""

        return {
            "git_commit_sha": self.git_commit_sha,
            "git_commit_url": self.git_commit_url,
            "github_repository_name": self.github_repository_name,
            "external_repo": self.external_repo,
            "platform": self.platform,
            "artifact_group": self.artifact_group,
            "amdgpu_targets": list(self.amdgpu_targets),
            "workflow_file_name": self.workflow_file_name,
            "workflow_run_id": self.workflow_run_id,
            "workflow_run_status": self.workflow_run_status,
            "workflow_run_conclusion": self.workflow_run_conclusion,
            "workflow_run_html_url": self.workflow_run_html_url,
            "s3_bucket": self.s3_bucket,
            "s3_path": self.s3_path,
            "s3_uri": self.s3_uri,
            "s3_index_url": self.s3_index_url,
            "artifact_filenames": list(self.artifact_filenames),
            "required_artifact_patterns": list(self.required_artifact_patterns),
            "missing_required_artifact_patterns": list(
                self.missing_required_artifact_patterns
            ),
        }


def check_if_artifacts_exist(info: ArtifactRunInfo) -> bool:
    """Checks whether matching artifacts exist for a workflow run.

    Lists concrete artifact archives from the workflow run's storage prefix,
    verifies that the requested artifact group is represented, and checks any
    required artifact filename patterns.

    Without required artifact patterns, this confirms only that the requested
    artifact group is represented. It does not prove that every artifact required
    by a build stage was uploaded.

    Args:
        info: ArtifactRunInfo describing the artifact location and artifact
            requirements.

    Returns:
        True if the requested artifact group is represented and every required
        filename pattern matches at least one artifact; False otherwise.
    """
    if info._output_root is None:
        return False

    try:
        backend = S3Backend(output_root=info._output_root)
        available_filenames = backend.list_artifacts()
    except (BotoCoreError, ClientError):
        return False

    matching_filenames = _filter_artifact_filenames(
        available_filenames,
        artifact_group=info.artifact_group,
        amdgpu_targets=info.amdgpu_targets,
    )
    missing_patterns = _find_missing_required_patterns(
        matching_filenames,
        info.required_artifact_patterns,
    )

    info.artifact_filenames = matching_filenames
    info.missing_required_artifact_patterns = missing_patterns

    return bool(matching_filenames) and not missing_patterns


def _get_base_arch(target: str) -> str:
    """Strip target variants such as ``:xnack+`` and ``:xnack-``."""

    return target.split(":", maxsplit=1)[0]


def _matches_target(
    artifact_target: str,
    requested_targets: Sequence[str],
) -> bool:
    """Return whether an artifact target matches a requested target."""

    artifact_base = _get_base_arch(artifact_target).lower()
    requested_bases = {
        _get_base_arch(target).lower() for target in requested_targets if target
    }
    return artifact_base in requested_bases


def _filter_artifact_filenames(
    artifact_filenames: Sequence[str],
    *,
    artifact_group: str,
    amdgpu_targets: Sequence[str],
) -> tuple[str, ...]:
    """Filter concrete archives for one requested artifact group.

    Generic archives are included because they are commonly required together
    with target-specific archives.

    For a non-generic artifact group, at least one target-specific archive must
    match. This prevents an S3 prefix containing only early generic artifacts
    from being reported as having the requested GPU artifact group.
    """

    # Artifacts may be uploaded using either the family name
    # (for example, gfx94X-dcgpu) or individual GPU targets
    # (for example, gfx942). Expand known families automatically
    # while still preserving explicitly supplied targets.
    family_targets = expand_families(
        [artifact_group],
        amdgpu_family_map(),
        strict=False,
    )

    requested_targets = tuple(
        dict.fromkeys(
            (
                artifact_group,
                *family_targets,
                *amdgpu_targets,
            )
        )
    )

    matched: list[str] = []
    target_specific_match_found = artifact_group.lower() == "generic"

    for filename in artifact_filenames:
        artifact_name = ArtifactName.from_filename(filename)
        if artifact_name is None:
            continue

        target_family = artifact_name.target_family
        if target_family.lower() == "generic":
            matched.append(filename)
            continue

        if _matches_target(target_family, requested_targets):
            matched.append(filename)
            target_specific_match_found = True

    if not target_specific_match_found:
        return ()

    return tuple(sorted(set(matched)))


def _find_missing_required_patterns(
    artifact_filenames: Sequence[str],
    required_patterns: Sequence[str],
) -> tuple[str, ...]:
    """Return required regex patterns that match no artifact filename."""

    missing: list[str] = []

    for pattern_text in required_patterns:
        pattern = re.compile(pattern_text)
        if not any(pattern.search(filename) for filename in artifact_filenames):
            missing.append(pattern_text)

    return tuple(missing)


def _validate_artifact_request(
    artifact_groups: Sequence[str],
    amdgpu_targets: Sequence[str],
) -> None:
    """Validate artifact-group and explicit-target combinations."""

    unique_groups = tuple(dict.fromkeys(artifact_groups))

    if amdgpu_targets and len(unique_groups) != 1:
        raise ValueError(
            "Explicit AMDGPU targets may be used only with one artifact group"
        )


def _find_artifacts_in_workflow_runs(
    *,
    commit: str,
    artifact_groups: list[str],
    workflow_runs: Sequence[dict],
    github_repository_name: str,
    workflow_file_name: str,
    platform: str,
    amdgpu_targets: Sequence[str] = (),
    required_artifact_patterns: Sequence[str] = (),
    require_single_run: bool = False,
    require_successful_run: bool = False,
) -> list[ArtifactRunInfo]:
    """Inspect workflow runs for the requested artifact groups.

    Workflow runs must be provided newest-first.

    By default, artifact groups may be accumulated across retriggered workflow
    runs. Once a group is found in a newer run, an older run cannot replace it.

    When ``require_single_run`` is enabled, every requested group must be found
    under the same workflow run's artifact prefix.

    When ``require_successful_run`` is enabled, runs are inspected only when their
    GitHub status is ``completed`` and their conclusion is ``success``.

    Results are returned in the same order as ``artifact_groups``.
    """
    # Deduplicate backend checks while preserving first-seen group order.
    requested_groups = list(dict.fromkeys(artifact_groups))

    # Used only for the default cross-run accumulation behavior.
    # Once a group is found in a newer run, an older run must not replace it.
    found: dict[str, ArtifactRunInfo] = {}

    # Do not reverse this sequence. Callers provide workflow runs newest-first.
    for workflow_run in workflow_runs:
        if require_successful_run:
            workflow_status = workflow_run.get("status")
            workflow_conclusion = workflow_run.get("conclusion")

            if workflow_status != "completed" or workflow_conclusion != "success":
                continue

        # Bucket and artifact-prefix metadata depend on the workflow run, not on
        # the individual artifact group, so resolve them once per run.
        output_root = WorkflowOutputRoot.from_workflow_run(
            run_id=str(workflow_run["id"]),
            platform=platform,
            github_repository=github_repository_name,
            workflow_run=workflow_run,
        )

        # Keep per-run results separate so single-run mode never combines artifact
        # groups from different workflow runs.
        found_in_this_run: dict[str, ArtifactRunInfo] = {}

        for group in requested_groups:
            # In normal mode, do not inspect an older run for a group already
            # found in a newer run.
            if not require_single_run and group in found:
                continue

            info = ArtifactRunInfo(
                git_commit_sha=commit,
                github_repository_name=github_repository_name,
                external_repo=output_root.external_repo,
                platform=platform,
                artifact_group=group,
                workflow_file_name=workflow_file_name,
                workflow_run_id=str(workflow_run["id"]),
                workflow_run_status=workflow_run.get(
                    "status",
                    "unknown",
                ),
                workflow_run_conclusion=workflow_run.get("conclusion"),
                workflow_run_html_url=workflow_run.get(
                    "html_url",
                    "",
                ),
                s3_bucket=output_root.bucket,
                amdgpu_targets=tuple(amdgpu_targets),
                required_artifact_patterns=tuple(required_artifact_patterns),
                _output_root=output_root,
            )

            if not check_if_artifacts_exist(info):
                continue

            found_in_this_run[group] = info

            if not require_single_run:
                found[group] = info

        if require_single_run:
            # Only return when one run independently contains every requested
            # group. Never combine found_in_this_run with another run.
            if len(found_in_this_run) == len(requested_groups):
                return [found_in_this_run[group] for group in artifact_groups]
        else:
            # Cross-run accumulation is complete. Because runs are inspected
            # newest-first and existing groups are never overwritten, the
            # newest available run wins for each group.
            if len(found) == len(requested_groups):
                break

    if require_single_run:
        return []

    return [found[group] for group in artifact_groups if group in found]


def find_artifacts_for_commit(
    commit: str,
    artifact_groups: list[str],
    github_repository_name: str = "ROCm/TheRock",
    workflow_file_name: str = "multi_arch_ci.yml",
    platform: str = platform_module.system().lower(),
    *,
    amdgpu_targets: Sequence[str] = (),
    required_artifact_patterns: Sequence[str] = (),
    require_single_run: bool = False,
    require_successful_run: bool = False,
) -> list[ArtifactRunInfo]:
    """Find artifact info for one or more groups from a commit.

    Queries GitHub for workflow runs on this commit, then checks each run
    (most recent first) for the requested groups. Accumulates results across
    runs — if attempt 2 has gfx110X artifacts and attempt 1 has gfx120X
    artifacts, both are returned.

    Args:
        commit: Git commit SHA (full or abbreviated)
        artifact_groups: GPU families (e.g., ["gfx94X-dcgpu", "gfx120X-all"])
        github_repository_name: Repository in "owner/repo" format
        workflow_file_name: Workflow filename, or None to infer from repo
        platform: "linux" or "windows", or None for current platform
        amdgpu_targets: Individual GPU targets that may appear instead of the
            family name in split artifact pipelines.
        required_artifact_patterns: Regex patterns that must each match at least
            one archive in every requested group's filtered artifact set.
        require_single_run: Require all requested groups to be present in one
            workflow run instead of accumulating across workflow runs.
        require_successful_run: Only inspect runs whose GitHub status is
            ``completed`` and whose conclusion is ``success``.

    Returns:
        List of ArtifactRunInfo for groups that have artifacts. May be empty
        if no workflow runs exist or no artifacts are available.

    Raises:
        GitHubAPIError: If the GitHub API request fails (rate limit, network
            error, etc.). Callers should handle this to distinguish between
            "no artifacts found" (empty list) and "couldn't check" (exception).
        ValueError: If explicit AMDGPU targets are provided with more than one
            artifact group.
    """
    _validate_artifact_request(
        artifact_groups,
        amdgpu_targets,
    )
    workflow_runs = gha_query_workflow_runs_for_commit(
        github_repository_name, workflow_file_name, commit
    )

    if not workflow_runs:
        return []

    # Share workflow-run inspection with explicit run-ID lookup while preserving
    # the existing newest-first and cross-run accumulation behavior.
    return _find_artifacts_in_workflow_runs(
        commit=commit,
        artifact_groups=artifact_groups,
        workflow_runs=workflow_runs,
        github_repository_name=github_repository_name,
        workflow_file_name=workflow_file_name,
        platform=platform,
        amdgpu_targets=amdgpu_targets,
        required_artifact_patterns=required_artifact_patterns,
        require_single_run=require_single_run,
        require_successful_run=require_successful_run,
    )


def find_artifacts_for_run(
    workflow_run_id: str,
    artifact_groups: list[str],
    github_repository_name: str = "ROCm/TheRock",
    workflow_file_name: str = "multi_arch_ci.yml",
    platform: str = platform_module.system().lower(),
    *,
    amdgpu_targets: Sequence[str] = (),
    required_artifact_patterns: Sequence[str] = (),
    require_successful_run: bool = False,
) -> list[ArtifactRunInfo]:
    """Find requested artifacts in one explicitly selected workflow run.

    Raises:
        GitHubAPIError: If the workflow run cannot be queried.
        ValueError: If explicit AMDGPU targets are provided with more than one
            artifact group.
    """

    _validate_artifact_request(
        artifact_groups,
        amdgpu_targets,
    )

    workflow_run = gha_query_workflow_run_by_id(
        github_repository=github_repository_name,
        workflow_run_id=workflow_run_id,
    )

    commit = str(workflow_run.get("head_sha") or "")

    return _find_artifacts_in_workflow_runs(
        commit=commit,
        artifact_groups=artifact_groups,
        workflow_runs=[workflow_run],
        github_repository_name=github_repository_name,
        workflow_file_name=workflow_file_name,
        platform=platform,
        amdgpu_targets=amdgpu_targets,
        required_artifact_patterns=required_artifact_patterns,
        require_single_run=True,
        require_successful_run=require_successful_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find CI artifacts for a given commit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    selector = parser.add_mutually_exclusive_group(required=True)
    output_options = parser.add_argument_group("output options")
    selector.add_argument(
        "--commit",
        type=str,
        help="Git commit SHA to find artifacts for (full SHA)",
    )
    selector.add_argument(
        "--run-id",
        type=str,
        help="Inspect one explicit GitHub Actions workflow run",
    )
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
    output_options.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args(argv)

    try:
        lookup_output = redirect_stdout(sys.stderr) if args.json else nullcontext()

        with lookup_output:
            if args.run_id:
                results = find_artifacts_for_run(
                    workflow_run_id=args.run_id,
                    artifact_groups=args.artifact_group,
                    github_repository_name=args.repo,
                    workflow_file_name=args.workflow,
                    platform=args.platform,
                    amdgpu_targets=args.amdgpu_target,
                    required_artifact_patterns=args.require_artifact,
                    require_successful_run=args.require_successful_run,
                )
            else:
                results = find_artifacts_for_commit(
                    commit=args.commit,
                    artifact_groups=args.artifact_group,
                    github_repository_name=args.repo,
                    workflow_file_name=args.workflow,
                    platform=args.platform,
                    amdgpu_targets=args.amdgpu_target,
                    required_artifact_patterns=args.require_artifact,
                    require_single_run=args.single_run,
                    require_successful_run=args.require_successful_run,
                )
    except (GitHubAPIError, ValueError, re.error) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if not results:
        if args.run_id:
            selector_text = f"run {args.run_id}"
        else:
            selector_text = f"commit {args.commit}"

        print(
            f"No matching artifacts found for {selector_text} "
            f"(platform={args.platform}, "
            f"artifact_group={args.artifact_group})",
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
