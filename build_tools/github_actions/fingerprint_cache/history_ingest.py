# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Populate the cache from real workflow-run history.

This is the cold-start / evidence path from the design: scrape recent workflow
runs, read the artifact archives each produced, and publish a cache entry per
artifact unit. It reuses TheRock's existing helpers
(``github_actions_api``/``baseline_runs``) for enumerating runs and listing
artifacts, so authentication is handled the usual way (``GITHUB_TOKEN`` in CI,
``gh`` CLI locally, otherwise unauthenticated + rate limited).

Because real artifact fingerprints are not yet uploaded, the cache key for each
ingested unit is the fallback derived from the run's head commit and the
artifact identity (see ``cache_key.derive_fallback_fingerprint``). The ingest
pipeline, trust/validation tagging, grouping, and publish ordering are all the
production shape; only the key derivation changes when real fingerprints land.
"""


from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Iterable, Sequence

from .cache_key import ArtifactIdentity, derive_fallback_fingerprint
from .cache_db import CacheDB
from .manifest import (
    OutputObject,
    Provenance,
    TrustLevel,
    ValidationState,
)
from .publish import publish_build_outputs
from .topology_view import BUNDLE_COMPONENT

# Default artifact component tokens (TheRock's artifact_manager.ARTIFACT_COMPONENTS
# is authoritative; this fallback lets the parser run without importing it).
DEFAULT_COMPONENTS = ("lib", "dev", "run", "dbg", "doc", "test")
ARTIFACT_EXTENSIONS = (".tar.zst", ".tar.xz")

# Branches whose push/schedule runs produce trusted entries.
_TRUSTED_BRANCHES_PREFIXES = ("release/",)
_TRUSTED_BRANCHES = ("main",)
_TRUSTED_EVENTS = ("push", "schedule")


@dataclass
class IngestStats:
    runs_scanned: int = 0
    runs_indexed: int = 0
    units_published: int = 0
    objects_indexed: int = 0
    units_skipped_policy: int = 0

    def to_dict(self) -> dict:
        return {
            "runs_scanned": self.runs_scanned,
            "runs_indexed": self.runs_indexed,
            "units_published": self.units_published,
            "objects_indexed": self.objects_indexed,
            "units_skipped_policy": self.units_skipped_policy,
        }


def parse_artifact_filename(
    filename: str, components: Sequence[str] = DEFAULT_COMPONENTS
) -> tuple[str, str, str] | None:
    """Parse ``{name}_{component}_{target_family}{ext}``.

    Returns ``(name, component, target_family)`` or ``None`` if the filename is
    not a recognized artifact archive. Artifact names may contain underscores,
    so the component token (from ``components``) is used as the split point.
    """
    stem = filename
    for ext in ARTIFACT_EXTENSIONS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    else:
        return None  # not an artifact archive
    tokens = stem.split("_")
    # Find the component token (search from the right: target_family uses
    # hyphens, not underscores, so the component is the last matching token
    # before the family).
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] in components:
            name = "_".join(tokens[:i])
            component = tokens[i]
            target_family = "_".join(tokens[i + 1 :])
            if name and target_family:
                return (name, component, target_family)
    return None


def trust_for_run(run: dict) -> TrustLevel:
    """Classify a run's trust from its event and branch."""
    event = run.get("event", "")
    branch = run.get("head_branch", "") or ""
    if event in _TRUSTED_EVENTS and (
        branch in _TRUSTED_BRANCHES
        or any(branch.startswith(p) for p in _TRUSTED_BRANCHES_PREFIXES)
    ):
        return TrustLevel.TRUSTED
    return TrustLevel.UNTRUSTED


def validation_for_run(run: dict) -> ValidationState:
    """Map a run conclusion to a validation state (build-level only here)."""
    if run.get("conclusion") == "success":
        return ValidationState.BUILD_PASSED
    return ValidationState.UNVALIDATED


def _provenance_for_run(run: dict, repository: str) -> Provenance:
    return Provenance(
        repository=repository,
        run_id=str(run.get("id", "")),
        branch=run.get("head_branch", "") or "",
        commit=run.get("head_sha", "") or "",
        workflow=run.get("name", "") or run.get("path", "") or "",
        created_at=run.get("created_at", "") or "",
        html_url=run.get("html_url", "") or "",
    )


# An artifact lister turns a run dict into the output objects it produced.
ArtifactLister = Callable[[dict], list[OutputObject]]


def ingest_runs(
    *,
    db: CacheDB,
    runs: Iterable[dict],
    artifact_lister: ArtifactLister,
    repository: str = "ROCm/TheRock",
    build_variant: str = "release",
    components: Sequence[str] = DEFAULT_COMPONENTS,
    require_trusted: bool = False,
) -> IngestStats:
    """Ingest a sequence of workflow runs into the cache.

    Args:
        db: destination cache.
        runs: workflow-run dicts (e.g. from ``query_completed_workflow_runs``).
        artifact_lister: returns the ``OutputObject``s a run produced. For
            production use ``make_s3_artifact_lister``; tests inject a fake.
        repository: owner/repo for provenance.
        build_variant: variant scoping the keys.
        components: artifact component tokens for filename parsing.
        require_trusted: if True, only index trusted (push/main) runs.
    """
    stats = IngestStats()
    for run in runs:
        stats.runs_scanned += 1
        trust = trust_for_run(run)
        if require_trusted and trust is not TrustLevel.TRUSTED:
            continue
        validation = validation_for_run(run)
        provenance = _provenance_for_run(run, repository)
        objects = artifact_lister(run)
        if not objects:
            continue

        # Group component archives into artifact-granularity bundle units,
        # keyed by (artifact_name, target_family).
        groups: dict[tuple[str, str], list[OutputObject]] = {}
        for obj in objects:
            parsed = parse_artifact_filename(obj.key, components)
            if parsed is None:
                continue
            name, _component, target_family = parsed
            groups.setdefault((name, target_family), []).append(obj)

        indexed_any = False
        for (name, target_family), group_objects in groups.items():
            identity = ArtifactIdentity(name, BUNDLE_COMPONENT, target_family)
            fingerprint = derive_fallback_fingerprint(
                source_commit=provenance.commit,
                artifact=identity,
                build_variant=build_variant,
            )
            manifest = publish_build_outputs(
                db=db,
                artifact=identity,
                outputs=group_objects,
                provenance=provenance,
                source_fingerprint=fingerprint,
                build_variant=build_variant,
                trust=trust,
                validation=validation,
            )
            if manifest is None:
                stats.units_skipped_policy += 1
                continue
            stats.units_published += 1
            stats.objects_indexed += len(group_objects)
            indexed_any = True
        if indexed_any:
            stats.runs_indexed += 1
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
