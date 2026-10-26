# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Import CI history as evidence for replay, never as verified local payloads.

Run listings and archive URLs cannot attest to output completeness or digests.
They live in provenance records, separate from artifact_keys/cache_entries.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Callable

from .cache_db import timestamp
from .cache_key import ArtifactIdentity, derive_fallback_fingerprint
from .inventory import COMPONENTS, parse_identity
from .manifest import ArtifactKey, OutputObject, TrustLevel, ValidationState

DEFAULT_COMPONENTS = COMPONENTS
ARTIFACT_EXTENSIONS = (".tar.zst", ".tar.xz", ".tar.gz", ".tar")
ArtifactLister = Callable[[dict], list[OutputObject]]


def parse_artifact_filename(filename, components=DEFAULT_COMPONENTS):
    if "/" in filename or "\\" in filename:
        return None
    for ext in ARTIFACT_EXTENSIONS:
        if filename.endswith(ext):
            parsed = parse_identity(filename[:-len(ext)])
            return parsed if parsed and parsed[1] in components else None
    return None


def trust_for_run(run, repository=None):
    """Trust only push/schedule history on main in the expected repository.

    This classifies supplied run metadata; it does not authenticate a producer.
    """
    branch, event = run.get("head_branch", "") or "", run.get("event", "")
    head_repo = run.get("head_repository") or {}
    if repository and head_repo.get("full_name", repository) != repository:
        return TrustLevel.UNTRUSTED
    trusted = branch == "main" and event in ("push", "schedule")
    return TrustLevel.TRUSTED if trusted else TrustLevel.UNTRUSTED


def validation_for_run(run):
    return (ValidationState.BUILD_PASSED if run.get("conclusion") == "success"
            and run.get("status", "completed") == "completed" else ValidationState.UNVALIDATED)


@dataclass
class IngestStats:
    runs_scanned: int = 0
    runs_indexed: int = 0
    runs_duplicate: int = 0
    runs_skipped: int = 0
    units_indexed: int = 0
    objects_indexed: int = 0

    def to_dict(self):
        return {**asdict(self), "scope": "history_evidence", "cache_entries_published": 0}


def ingest_runs(*, db, runs, artifact_lister, repository="ROCm/TheRock",
                build_variant="release", platform="linux", require_trusted=False):
    stats = IngestStats()
    for run in runs:
        stats.runs_scanned += 1
        if not isinstance(run, dict):
            raise ValueError("each history run must be an object")
        trust = trust_for_run(run, repository)
        # A completed run's updated_at is conservative availability evidence;
        # created_at is never used as completion time.
        available = run.get("completed_at") or run.get("updated_at")
        started = run.get("run_started_at") or run.get("created_at")
        if (not run.get("id") or not run.get("head_sha") or not started or not available
                or run.get("status", "completed") != "completed"
                or (require_trusted and trust is not TrustLevel.TRUSTED)):
            stats.runs_skipped += 1
            continue
        if timestamp(available) < timestamp(started):
            raise ValueError("run completion predates its start")
        objects = artifact_lister(run)
        groups = {}
        for obj in objects:
            parsed = parse_artifact_filename(obj.key)
            if parsed is None:
                continue
            name, component, family = parsed
            identity = f"{name}_{component}_{family}"
            groups.setdefault(identity, []).append(obj.to_dict())
        fingerprints = run.get("source_fingerprints", {})
        if not isinstance(fingerprints, dict):
            raise ValueError("source_fingerprints must map component identity to raw fingerprint")
        units = []
        for identity, outputs in sorted(groups.items()):
            raw = fingerprints.get(identity)
            if raw is not None:
                raw = ArtifactKey(identity, raw).fingerprint
                kind = "cmake"
            else:
                raw = derive_fallback_fingerprint(
                    source_commit=run["head_sha"], artifact=ArtifactIdentity(*parse_identity(identity)),
                    build_variant=build_variant)
                kind = "commit_fallback"
            units.append({"artifact_identity": identity, "fingerprint": raw,
                          "fingerprint_kind": kind, "objects": sorted(outputs, key=lambda o: (o["key"], o["uri"]))})
        evidence = {
            "schema_version": 1, "repository": repository, "run_id": str(run["id"]),
            "run_attempt": int(run.get("run_attempt", 1)), "commit": run["head_sha"],
            "started_at": started, "available_at": available,
            "platform": platform, "build_variant": build_variant,
            "trust": trust.value, "validation": validation_for_run(run).value,
            "units": units,
        }
        _, inserted = db.record_history(evidence)
        stats.runs_indexed += int(inserted)
        stats.runs_duplicate += int(not inserted)
        stats.units_indexed += len(units) if inserted else 0
        stats.objects_indexed += sum(len(u["objects"]) for u in units) if inserted else 0
    return stats


# -- production helpers (lazy TheRock imports) -----------------------------
def _add_therock_paths() -> None:
    """Put build_tools/ and build_tools/github_actions/ on sys.path."""
    here = Path(__file__).resolve()
    github_actions_dir = here.parents[1]
    build_tools_dir = here.parents[2]
    for p in (str(github_actions_dir), str(build_tools_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


def query_recent_runs(
    *,
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
    max_runs: int = 20,
) -> list[dict]:
    """Thin wrapper over ``baseline_runs.query_completed_workflow_runs``."""
    _add_therock_paths()
    from baseline_runs import query_completed_workflow_runs  # type: ignore

    return query_completed_workflow_runs(
        github_repository=github_repository,
        workflow_name=workflow_name,
        branch=branch,
        max_runs=max_runs,
    )


def make_s3_artifact_lister(
    *, repository: str = "ROCm/TheRock", platform: str = "linux"
) -> ArtifactLister:
    """Artifact lister backed by TheRock's S3 artifact backend.

    Requires AWS access to the CI artifact buckets. Objects are returned as
    pointers (``uri``) into the existing bucket; digests are left ``None`` here
    to avoid an extra fetch per object (``.sha256sum`` can be read later).
    """
    _add_therock_paths()
    from baseline_runs import create_artifact_backend_for_workflow_run  # type: ignore

    def lister(run: dict) -> list[OutputObject]:
        backend = create_artifact_backend_for_workflow_run(run, repository, platform)
        base = backend.base_uri.rstrip("/")
        objects: list[OutputObject] = []
        for filename in backend.list_artifacts():
            objects.append(OutputObject(key=filename, uri=f"{base}/{filename}"))
        return objects

    return lister
