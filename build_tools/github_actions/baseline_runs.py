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
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from urllib.parse import urlencode, quote

logger = logging.getLogger(__name__)

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
class CommitCompatibility:
    """Result of checking a candidate run's commit against the current commit.

    A baseline run is only safe to reuse when it was built from source that the
    current commit is based on. Reusing artifacts from a divergent or newer-than-current
    commit risks mixing incompatible binaries, so those are rejected.
    """

    current_commit_sha: str
    candidate_head_sha: str
    relationship: str

    @property
    def is_valid(self) -> bool:
        return self.relationship in ("same", "ancestor")


@dataclass(frozen=True)
class RunRecency:
    """Result of checking how old a candidate run is."""

    created_at: str
    age_hours: float | None
    max_age_hours: float | None

    @property
    def is_valid(self) -> bool:
        if self.age_hours is None:
            return False
        if self.age_hours < 0:
            return False
        if self.max_age_hours is None:
            return True
        return self.age_hours <= self.max_age_hours


@dataclass(frozen=True)
class RunTiming:
    """Timing measurements for a workflow run."""

    run_id: str
    created_at: str
    run_started_at: str
    updated_at: str
    queue_seconds: float | None
    run_seconds: float | None
    total_seconds: float | None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "run_started_at": self.run_started_at,
            "updated_at": self.updated_at,
            "queue_seconds": self.queue_seconds,
            "run_seconds": self.run_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class WorkflowRunSummary:
    """Compact source/ref summary for a workflow run."""

    repository: str
    branch: str
    commit: str
    workflow: str
    run_id: str
    status: str
    conclusion: str | None
    timestamp: str | None
    html_url: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "branch": self.branch,
            "commit": self.commit,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "status": self.status,
            "conclusion": self.conclusion,
            "timestamp": self.timestamp,
            "html_url": self.html_url,
        }


@dataclass(frozen=True)
class BaselineRun:
    """Workflow run with build jobs and artifacts suitable for reuse."""

    source_ref: WorkflowRunSummary
    platform: str
    job_health: WorkflowJobHealth
    artifact_availability: ArtifactAvailability
    commit_compatibility: CommitCompatibility | None = None
    run_recency: RunRecency | None = None

    @property
    def run_id(self) -> str:
        return self.source_ref.run_id

    @property
    def html_url(self) -> str:
        return self.source_ref.html_url

    @property
    def head_sha(self) -> str:
        return self.source_ref.commit

    @property
    def branch(self) -> str:
        return self.source_ref.branch

    @property
    def workflow_name(self) -> str:
        return self.source_ref.workflow

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ref": self.source_ref.to_dict(),
            "platform": self.platform,
            "job_health": self.job_health,
            "artifact_availability": self.artifact_availability,
        }


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


def create_workflow_run_summary(
    workflow_run: dict,
    *,
    github_repository: str,
    workflow_name: str,
) -> WorkflowRunSummary:
    """Create a compact source/ref summary for a workflow run."""
    return WorkflowRunSummary(
        repository=github_repository,
        branch=workflow_run.get("head_branch", ""),
        commit=workflow_run.get("head_sha", ""),
        workflow=workflow_name,
        run_id=str(workflow_run["id"]),
        status=workflow_run.get("status", ""),
        conclusion=workflow_run.get("conclusion"),
        timestamp=workflow_run.get("created_at"),
        html_url=workflow_run.get("html_url", ""),
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
    """Find artifact archives matching a requirement.

    First tries an exact match with the requested target family. If not found
    and the target is not already 'generic', falls back to checking if a
    generic version exists. This handles artifacts like sysdeps that are only
    built as generic but are needed by all target builds.
    """
    matches: list[str] = []
    for component in ARTIFACT_COMPONENTS:
        found = False
        for extension in ARTIFACT_EXTENSIONS:
            filename = (
                f"{required_artifact.name}_{component}_"
                f"{required_artifact.target_family}{extension}"
            )
            if filename in available:
                matches.append(filename)
                found = True
                break
        # Fallback to generic if target-specific not found
        if not found and required_artifact.target_family != "generic":
            for extension in ARTIFACT_EXTENSIONS:
                generic_filename = (
                    f"{required_artifact.name}_{component}_generic{extension}"
                )
                if generic_filename in available:
                    matches.append(generic_filename)
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


def _normalize_sha(value: str) -> str:
    return value.strip().lower()


def _parse_iso_timestamp(value: str) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp into an aware UTC datetime."""

    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    """Return (end - start) in seconds, or None if either bound is missing."""
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    if seconds < 0:
        logger.debug(
            "Ignoring negative duration: start=%s end=%s (%.1fs)",
            start.isoformat(),
            end.isoformat(),
            seconds,
        )
        return None
    return seconds


def parse_run_timing(workflow_run: dict) -> RunTiming:
    """Parse queue/run timing from a workflow-run dict."""

    created_at = workflow_run.get("created_at", "") or ""
    run_started_at = workflow_run.get("run_started_at", "") or ""
    updated_at = workflow_run.get("updated_at", "") or ""

    created = _parse_iso_timestamp(created_at)
    started = _parse_iso_timestamp(run_started_at)
    updated = _parse_iso_timestamp(updated_at)

    return RunTiming(
        run_id=str(workflow_run.get("id", "")),
        created_at=created_at,
        run_started_at=run_started_at,
        updated_at=updated_at,
        queue_seconds=_seconds_between(created, started),
        run_seconds=_seconds_between(started, updated),
        total_seconds=_seconds_between(created, updated),
    )


def validate_commit_compatibility(
    *,
    candidate_head_sha: str,
    current_commit_sha: str,
    ordered_commit_shas: Sequence[str],
) -> CommitCompatibility:
    """Validate that a candidate run was built from a compatible commit.

    ``ordered_commit_shas`` is the branch history newest-first (as returned by
    ``gha_query_recent_branch_commits``). The current commit is expected to be
    at or near the front of that list. A candidate is:

    * ``same`` when its head_sha equals the current commit;
    * ``ancestor`` when its head_sha appears *after* the current commit in the
      newest-first history (i.e. the current commit is built on top of it);
    * ``descendant_or_divergent`` when it appears *before* the current commit
      (newer than current, so unsafe to reuse);
    * ``unknown`` when it does not appear in the provided history window.

    Only ``same`` and ``ancestor`` are considered valid. ``unknown`` is treated
    as unsafe rather than assumed-good: a candidate outside the history window
    cannot be confirmed as an ancestor, so widen the window if legitimate
    ancestors are being rejected.
    """
    current = _normalize_sha(current_commit_sha)
    candidate = _normalize_sha(candidate_head_sha)
    if not current:
        raise ValueError("current_commit_sha must be non-empty")
    if not candidate:
        raise ValueError("candidate_head_sha must be non-empty")

    if candidate == current:
        relationship = "same"
    else:
        history = [_normalize_sha(sha) for sha in ordered_commit_shas]
        try:
            current_index = history.index(current)
        except ValueError:
            current_index = None
        try:
            candidate_index = history.index(candidate)
        except ValueError:
            candidate_index = None

        if candidate_index is None:
            relationship = "unknown"
        elif current_index is None:
            # Current commit is outside the window but the candidate is in it;
            # we cannot establish ordering, so treat as unknown.
            relationship = "unknown"
        elif candidate_index > current_index:
            # Newest-first: a larger index is older than the current commit.
            relationship = "ancestor"
        else:
            relationship = "descendant_or_divergent"

    return CommitCompatibility(
        current_commit_sha=current,
        candidate_head_sha=candidate,
        relationship=relationship,
    )


def validate_run_recency(
    *,
    workflow_run: dict,
    max_age_hours: float | None,
    now: datetime | None = None,
) -> RunRecency:
    """Validate that a candidate run is recent enough to reuse."""

    if max_age_hours is not None and max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")

    created_at_raw = workflow_run.get("created_at", "") or ""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    age_hours: float | None = None
    parsed = _parse_iso_timestamp(created_at_raw)
    if parsed is not None:
        age_hours = (reference - parsed).total_seconds() / 3600.0

    return RunRecency(
        created_at=created_at_raw,
        age_hours=age_hours,
        max_age_hours=max_age_hours,
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
    current_commit_sha: str | None = None,
    ordered_commit_shas: Sequence[str] | None = None,
    max_age_hours: float | None = None,
    now: datetime | None = None,
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
        current_commit_sha: When provided, enables the commit-compatibility
            rule: a candidate run is only accepted when its head_sha equals or
            is an ancestor of this commit. Requires ``ordered_commit_shas``.
        ordered_commit_shas: Branch history newest-first (e.g. from
            ``gha_query_recent_branch_commits``) used to establish ancestry for
            the commit-compatibility rule. Ignored unless ``current_commit_sha``
            is set.
        max_age_hours: When provided, enables the recency rule: candidate runs
            older than this many hours (by ``created_at``) are rejected.
        now: Reference time for the recency rule; defaults to the current UTC
            time. Primarily for testing.
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

    check_commit = current_commit_sha is not None
    if check_commit:
        if not current_commit_sha.strip():
            raise ValueError(
                "current_commit_sha must be non-empty when commit checking is enabled"
            )
        if ordered_commit_shas is None:
            raise ValueError(
                "ordered_commit_shas is required when current_commit_sha is set"
            )
    check_recency = max_age_hours is not None

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

        # Cheap, local checks (recency, commit ancestry) run before the
        # job-health and artifact checks, which require extra API/backend calls.
        run_recency: RunRecency | None = None
        if check_recency:
            run_recency = validate_run_recency(
                workflow_run=workflow_run,
                max_age_hours=max_age_hours,
                now=now,
            )
            if not run_recency.is_valid:
                logger.info(
                    "Skipping run %s: too old or not date-parseable "
                    "(age_hours=%s, max_age_hours=%s)",
                    run_id,
                    run_recency.age_hours,
                    run_recency.max_age_hours,
                )
                continue

        commit_compatibility: CommitCompatibility | None = None
        if check_commit:
            candidate_head_sha = (workflow_run.get("head_sha", "") or "").strip()
            if not candidate_head_sha:
                # A run without a head_sha cannot be confirmed compatible.
                logger.info("Skipping run %s: no head_sha to compare", run_id)
                continue
            commit_compatibility = validate_commit_compatibility(
                candidate_head_sha=candidate_head_sha,
                current_commit_sha=current_commit_sha,
                ordered_commit_shas=ordered_commit_shas or (),
            )
            if not commit_compatibility.is_valid:
                logger.info(
                    "Skipping run %s: commit %s is %s relative to current %s",
                    run_id,
                    candidate_head_sha,
                    commit_compatibility.relationship,
                    commit_compatibility.current_commit_sha,
                )
                continue

        job_health = validate_required_jobs_successful(
            workflow_jobs=workflow_jobs_fetcher(workflow_run, github_repository),
            required_name_substrings=required_jobs,
        )
        if not job_health.is_valid:
            logger.info(
                "Skipping run %s: required build jobs not healthy "
                "(failed=%s, missing=%s)",
                run_id,
                job_health.failed_job_names,
                job_health.missing_name_substrings,
            )
            continue

        backend = backend_factory(workflow_run, github_repository, platform)
        availability = validate_required_artifacts_available(
            backend=backend,
            required_artifacts=requirements,
        )
        if not availability.is_valid:
            logger.info(
                "Skipping run %s: missing artifacts %s",
                run_id,
                availability.missing_artifacts,
            )
            continue

        source_ref = create_workflow_run_summary(
            workflow_run,
            github_repository=github_repository,
            workflow_name=workflow_name,
        )

        return BaselineRun(
            source_ref=source_ref,
            platform=platform,
            job_health=job_health,
            artifact_availability=availability,
            commit_compatibility=commit_compatibility,
            run_recency=run_recency,
        )

    return None
