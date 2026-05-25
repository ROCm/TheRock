#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for selecting baseline workflow runs for prebuilt artifact reuse."""

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
class BaselineRun:
    """Successful workflow run with artifacts suitable for reuse."""

    run_id: str
    html_url: str
    head_sha: str
    branch: str
    workflow_name: str
    platform: str
    artifact_availability: ArtifactAvailability


ArtifactBackendFactory = Callable[[dict, str, str], ArtifactBackend]


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


def is_successful_workflow_run(workflow_run: dict) -> bool:
    """Return True when a workflow run completed successfully."""
    return (
        workflow_run.get("status") == "completed"
        and workflow_run.get("conclusion") == "success"
    )


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


def select_baseline_run(
    *,
    required_artifacts: Iterable[RequiredArtifact],
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
    platform: str,
    max_runs: int = 20,
    exclude_run_ids: Iterable[str] = (),
    workflow_runs: Sequence[dict] | None = None,
    backend_factory: ArtifactBackendFactory = create_artifact_backend_for_workflow_run,
) -> BaselineRun | None:
    """Select the newest successful workflow run with required artifacts.

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
        workflow_runs: Optional pre-fetched candidate runs for testing or for
            callers that already queried GitHub.
        backend_factory: Factory used to create an artifact backend for each
            candidate run.

    Returns:
        The first candidate run that completed successfully and has all required
        artifact/family pairs, or ``None`` if no valid baseline is found.
    """
    # Validate these early so a missing requirement is a caller error instead of
    # being discovered only after GitHub/API work.
    requirements = _dedupe_required_artifacts(required_artifacts)
    excluded = {str(run_id) for run_id in exclude_run_ids}

    candidates = (
        list(workflow_runs)
        if workflow_runs is not None
        else query_successful_workflow_runs(
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
        if not is_successful_workflow_run(workflow_run):
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
            artifact_availability=availability,
        )

    return None
