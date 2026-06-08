#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for selecting baseline workflow runs for prebuilt artifact reuse.

Example:

    baseline = select_baseline_run(
        required_artifacts=[
            RequiredArtifact("blas", "gfx94X-dcgpu"),
            RequiredArtifact("base", "generic"),
        ],
        platform="linux",
        exclude_run_ids=[os.environ["GITHUB_RUN_ID"]],
    )
    if baseline:
        # Pass baseline.run_id as the multi-arch CI baseline_run_id.
        ...
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from urllib.parse import urlencode, quote

# Add parent directory to path for artifact and _therock_utils imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact_manager import ARTIFACT_COMPONENTS
from _therock_utils.artifact_backend import (
    ARTIFACT_EXTENSIONS,
    ArtifactBackend,
    S3Backend,
)
from _therock_utils.workflow_outputs import WorkflowOutputRoot

from github_actions_api import gha_send_request


@dataclass(frozen=True)
class RequiredArtifact:
    """Artifact archive requirement for one target family."""

    name: str
    target_family: str


@dataclass(frozen=True)
class ArtifactAvailability:
    """Result of checking a backend for required artifact archives."""

    required_artifacts: tuple[RequiredArtifact, ...]
    matched_filenames: tuple[str, ...]
    missing_artifacts: tuple[RequiredArtifact, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_artifacts


@dataclass(frozen=True)
class WorkflowJobHealth:
    """Result of checking required workflow jobs in a candidate run."""

    required_name_substrings: tuple[str, ...]
    matched_job_names: tuple[str, ...]
    failed_job_names: tuple[str, ...]
    missing_name_substrings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.failed_job_names and not self.missing_name_substrings


@dataclass(frozen=True)
class BaselineRun:
    """Workflow run with build jobs and artifacts suitable for reuse."""

    run_id: str
    html_url: str
    head_sha: str
    branch: str
    workflow_name: str
    platform: str
    job_health: WorkflowJobHealth
    artifact_availability: ArtifactAvailability


ArtifactBackendFactory = Callable[[dict, str, str], ArtifactBackend]
WorkflowJobsFetcher = Callable[[dict, str], Sequence[dict]]


def _dedupe_required_artifacts(
    required_artifacts: Iterable[RequiredArtifact],
) -> tuple[RequiredArtifact, ...]:
    result: list[RequiredArtifact] = []
    seen: set[RequiredArtifact] = set()
    for artifact in required_artifacts:
        normalized = RequiredArtifact(
            name=artifact.name.strip(),
            target_family=artifact.target_family.strip(),
        )
        if not normalized.name or not normalized.target_family:
            raise ValueError(
                "required_artifacts must have non-empty names and target families"
            )
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    if not result:
        raise ValueError("required_artifacts must contain at least one value")
    return tuple(result)


def _dedupe_nonempty_strings(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must contain non-empty values")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def is_completed_workflow_run(workflow_run: dict) -> bool:
    """Return True when a workflow run has completed."""
    return workflow_run.get("status") == "completed"


def is_successful_workflow_run(workflow_run: dict) -> bool:
    """Return True when a workflow run completed successfully."""
    return (
        workflow_run.get("status") == "completed"
        and workflow_run.get("conclusion") == "success"
    )


def is_successful_workflow_job(workflow_job: dict) -> bool:
    """Return True when a workflow job completed successfully."""
    return (
        workflow_job.get("status") == "completed"
        and workflow_job.get("conclusion") == "success"
    )


def query_completed_workflow_runs(
    *,
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
    max_runs: int = 20,
) -> list[dict]:
    """Query recent completed workflow runs for a workflow and branch."""
    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")

    per_page = min(max_runs, 100)
    workflow_path = quote(workflow_name, safe="")
    query = urlencode(
        {
            "status": "completed",
            "branch": branch,
            "per_page": per_page,
            "sort": "created",
            "direction": "desc",
        }
    )
    url = (
        f"https://api.github.com/repos/{github_repository}"
        f"/actions/workflows/{workflow_path}/runs?{query}"
    )
    response = gha_send_request(url)
    workflow_runs = response.get("workflow_runs", [])
    return workflow_runs[:max_runs]


def query_successful_workflow_runs(
    *,
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
    max_runs: int = 20,
) -> list[dict]:
    """Query recent successful workflow runs for a workflow and branch."""
    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")

    per_page = min(max_runs, 100)
    workflow_path = quote(workflow_name, safe="")
    query = urlencode(
        {
            "status": "success",
            "branch": branch,
            "per_page": per_page,
            "sort": "created",
            "direction": "desc",
        }
    )
    url = (
        f"https://api.github.com/repos/{github_repository}"
        f"/actions/workflows/{workflow_path}/runs?{query}"
    )
    response = gha_send_request(url)
    workflow_runs = response.get("workflow_runs", [])
    return workflow_runs[:max_runs]


def query_workflow_run_jobs(
    *,
    github_repository: str = "ROCm/TheRock",
    run_id: str,
    run_attempt: int | str | None = None,
    max_pages: int = 10,
) -> list[dict]:
    """Query jobs for a workflow run or a specific run attempt."""
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    all_jobs: list[dict] = []
    for page in range(1, max_pages + 1):
        if run_attempt is None:
            url = (
                f"https://api.github.com/repos/{github_repository}"
                f"/actions/runs/{run_id}/jobs"
                f"?filter=latest&per_page=100&page={page}"
            )
        else:
            url = (
                f"https://api.github.com/repos/{github_repository}"
                f"/actions/runs/{run_id}/attempts/{run_attempt}/jobs"
                f"?per_page=100&page={page}"
            )
        response = gha_send_request(url)
        jobs = response.get("jobs", [])
        all_jobs.extend(jobs)
        if len(jobs) < 100:
            break
    return all_jobs


def query_jobs_for_workflow_run(
    workflow_run: dict,
    github_repository: str,
) -> list[dict]:
    """Query jobs for a workflow run, preferring the run attempt when known."""
    run_attempt = workflow_run.get("run_attempt")
    return query_workflow_run_jobs(
        github_repository=github_repository,
        run_id=str(workflow_run["id"]),
        run_attempt=run_attempt,
    )


def create_artifact_backend_for_workflow_run(
    workflow_run: dict,
    github_repository: str,
    platform: str,
) -> ArtifactBackend:
    """Create an artifact backend rooted at a workflow run's output prefix."""
    run_id = str(workflow_run["id"])
    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id,
        platform=platform,
        github_repository=github_repository,
        workflow_run=workflow_run,
    )
    return S3Backend(output_root=output_root)


def _find_matching_artifact_archives(
    required_artifact: RequiredArtifact,
    available: set[str],
) -> list[str]:
    matches: list[str] = []
    for component in ARTIFACT_COMPONENTS:
        for extension in ARTIFACT_EXTENSIONS:
            filename = (
                f"{required_artifact.name}_{component}_"
                f"{required_artifact.target_family}{extension}"
            )
            if filename in available:
                matches.append(filename)
                break
    return matches


def validate_required_artifacts_available(
    *,
    backend: ArtifactBackend,
    required_artifacts: Iterable[RequiredArtifact],
) -> ArtifactAvailability:
    """Validate that a backend has archives for each artifact/family pair.

    This mirrors the artifact filename matching used by ``artifact_manager.py``
    copy/fetch operations. It validates artifact/family presence, not a
    complete per-component manifest.
    """
    requirements = _dedupe_required_artifacts(required_artifacts)

    available = set(backend.list_artifacts())
    matched: list[str] = []
    missing: list[RequiredArtifact] = []
    for required_artifact in requirements:
        artifact_matches = _find_matching_artifact_archives(
            required_artifact, available
        )
        if artifact_matches:
            matched.extend(artifact_matches)
        else:
            missing.append(required_artifact)

    return ArtifactAvailability(
        required_artifacts=requirements,
        matched_filenames=tuple(matched),
        missing_artifacts=tuple(missing),
    )


def _format_job_status(workflow_job: dict) -> str:
    name = workflow_job.get("name", "unknown")
    status = workflow_job.get("status", "unknown")
    conclusion = workflow_job.get("conclusion", "unknown")
    return f"{name} ({status}/{conclusion})"


def validate_required_jobs_successful(
    *,
    workflow_jobs: Sequence[dict],
    required_name_substrings: Iterable[str],
) -> WorkflowJobHealth:
    """Validate that matching workflow jobs completed successfully.

    Candidate workflow runs can have failed test jobs while still containing
    reusable build artifacts. This check lets callers require the relevant
    build jobs to be healthy without requiring the whole workflow conclusion to
    be successful.
    """
    requirements = _dedupe_nonempty_strings(
        required_name_substrings,
        field_name="required_name_substrings",
    )

    matched: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    seen_matched: set[str] = set()
    seen_failed: set[str] = set()

    for requirement in requirements:
        requirement_matches = [
            job
            for job in workflow_jobs
            if requirement.casefold() in job.get("name", "").casefold()
        ]
        if not requirement_matches:
            missing.append(requirement)
            continue

        for job in requirement_matches:
            job_name = job.get("name", "unknown")
            if job_name not in seen_matched:
                seen_matched.add(job_name)
                matched.append(job_name)
            if is_successful_workflow_job(job):
                continue
            job_status = _format_job_status(job)
            if job_status not in seen_failed:
                seen_failed.add(job_status)
                failed.append(job_status)

    return WorkflowJobHealth(
        required_name_substrings=requirements,
        matched_job_names=tuple(matched),
        failed_job_names=tuple(failed),
        missing_name_substrings=tuple(missing),
    )


def select_baseline_run(
    *,
    required_artifacts: Iterable[RequiredArtifact],
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
    platform: str,
    max_runs: int = 20,
    exclude_run_ids: Iterable[str] = (),
    required_successful_job_name_substrings: Iterable[str] = ("Build",),
    workflow_runs: Sequence[dict] | None = None,
    backend_factory: ArtifactBackendFactory = create_artifact_backend_for_workflow_run,
    workflow_jobs_fetcher: WorkflowJobsFetcher = query_jobs_for_workflow_run,
) -> BaselineRun | None:
    """Select the newest workflow run with healthy build jobs and artifacts.

    Args:
        required_artifacts: Artifact/family pairs that must be present in the
            baseline run output.
        github_repository: Repository in ``owner/repo`` format.
        workflow_name: Workflow filename to search.
        branch: Branch to search.
        platform: Artifact platform, e.g. ``linux`` or ``windows``.
        max_runs: Maximum workflow runs to inspect.
        exclude_run_ids: Run IDs that must not be selected, such as the
            current workflow run.
        required_successful_job_name_substrings: Job-name substrings that must
            match at least one completed successful job in the candidate run.
            Defaults to ``("Build",)`` so unrelated test failures do not
            automatically disqualify otherwise reusable build artifacts.
        workflow_runs: Optional pre-fetched candidate runs for testing or for
            callers that already queried GitHub.
        backend_factory: Factory used to create an artifact backend for each
            candidate run.
        workflow_jobs_fetcher: Factory used to query jobs for each candidate
            run.

    Returns:
        The first completed candidate run that has healthy required jobs and
        all required artifact/family pairs, or ``None`` if no valid baseline is
        found.
    """
    # Validate these early so a missing requirement is a caller error instead of
    # being discovered only after GitHub/API work.
    requirements = _dedupe_required_artifacts(required_artifacts)
    required_jobs = _dedupe_nonempty_strings(
        required_successful_job_name_substrings,
        field_name="required_successful_job_name_substrings",
    )
    excluded = {str(run_id) for run_id in exclude_run_ids}

    candidates = (
        list(workflow_runs)
        if workflow_runs is not None
        else query_completed_workflow_runs(
            github_repository=github_repository,
            workflow_name=workflow_name,
            branch=branch,
            max_runs=max_runs,
        )
    )

    for workflow_run in candidates[:max_runs]:
        run_id = str(workflow_run["id"])
        if run_id in excluded:
            continue
        if not is_completed_workflow_run(workflow_run):
            continue

        job_health = validate_required_jobs_successful(
            workflow_jobs=workflow_jobs_fetcher(workflow_run, github_repository),
            required_name_substrings=required_jobs,
        )
        if not job_health.is_valid:
            continue

        backend = backend_factory(workflow_run, github_repository, platform)
        availability = validate_required_artifacts_available(
            backend=backend,
            required_artifacts=requirements,
        )
        if not availability.is_valid:
            continue

        return BaselineRun(
            run_id=run_id,
            html_url=workflow_run.get("html_url", ""),
            head_sha=workflow_run.get("head_sha", ""),
            branch=workflow_run.get("head_branch", branch),
            workflow_name=workflow_name,
            platform=platform,
            job_health=job_health,
            artifact_availability=availability,
        )

    return None
