# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Restore / verify a required unit against object storage (flowchart 2).

The planner decides a unit is a *hit* from the database. Before a run relies on
that, the objects must actually be present and intact in storage -- the cache
points into existing S3 buckets whose retention can delete objects out from
under an otherwise-valid manifest. ``restore_units`` performs the right-hand
side of flowchart 2:

    manifest -> download objects -> verify digests and required outputs
        -> valid        => reusable unit
        -> missing/invalid => unit needs building

It takes an ``ObjectStore`` so the real S3 backend, a local directory, or a
in-memory store (tests / dry runs) can be plugged in interchangeably.
"""


from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from .manifest import Manifest
from .planner import PlanEntry


class ObjectStore(Protocol):
    """Where immutable artifact objects live."""

    def exists(self, uri: str) -> bool: ...

    def fetch_digest(self, uri: str) -> str | None:
        """Return the stored object's sha256, or ``None`` if unknown."""
        ...


class InMemoryObjectStore:
    """In-memory object store for tests / dry runs.

    By default every URI referenced by a manifest is considered present and
    intact. Configure ``missing`` and ``bad_digest`` to simulate retention
    deletion or corruption and exercise the ``needs_build`` edges.
    """

    def __init__(
        self,
        present: Iterable[str] | None = None,
        *,
        missing: Iterable[str] = (),
        digests: dict[str, str] | None = None,
        bad_digest: Iterable[str] = (),
        default_present: bool = True,
    ):
        self._present = set(present) if present is not None else None
        self._missing = set(missing)
        self._digests = dict(digests or {})
        self._bad_digest = set(bad_digest)
        self._default_present = default_present

    def exists(self, uri: str) -> bool:
        if uri in self._missing:
            return False
        if self._present is not None:
            return uri in self._present
        return self._default_present

    def fetch_digest(self, uri: str) -> str | None:
        if uri in self._bad_digest:
            return "0" * 64  # deliberately wrong
        return self._digests.get(uri)


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of resolving one unit against storage."""

    cache_key: str
    artifact: str  # str(ArtifactIdentity)
    outcome: str  # "reusable" | "needs_build"
    reason: str
    verified_objects: tuple[str, ...] = ()

    @property
    def reusable(self) -> bool:
        return self.outcome == "reusable"


def restore_manifest(
    manifest: Manifest,
    store: ObjectStore,
    *,
    verify_digests: bool = True,
) -> RestoreResult:
    """Resolve a single hit manifest: download + verify -> reusable/needs_build."""
    key = manifest.cache_key
    artifact = str(manifest.artifact)
    if not manifest.is_servable():
        return RestoreResult(
            key,
            artifact,
            "needs_build",
            "manifest not servable (incomplete/unpublished)",
        )
    verified: list[str] = []
    for obj in manifest.outputs:
        if not store.exists(obj.uri):
            return RestoreResult(
                key, artifact, "needs_build", f"object missing in storage: {obj.key}"
            )
        if verify_digests and obj.digest is not None:
            actual = store.fetch_digest(obj.uri)
            if actual is not None and actual != obj.digest:
                return RestoreResult(
                    key, artifact, "needs_build", f"digest mismatch: {obj.key}"
                )
        verified.append(obj.uri)
    return RestoreResult(
        key, artifact, "reusable", "all objects present and verified", tuple(verified)
    )


def restore_units(
    entries: Sequence[PlanEntry],
    store: ObjectStore,
    *,
    verify_digests: bool = True,
) -> list[RestoreResult]:
    """Resolve plan entries against storage.

    Misses pass straight through as ``needs_build``. Hits are verified against
    the object store; a hit whose objects are gone or corrupt is downgraded to
    ``needs_build`` -- defending against database/storage drift.
    """
    results: list[RestoreResult] = []
    for entry in entries:
        if not entry.is_hit or entry.manifest is None:
            results.append(
                RestoreResult(
                    entry.unit.cache_key.value,
                    str(entry.unit.artifact),
                    "needs_build",
                    entry.reason,
                )
            )
            continue
        results.append(
            restore_manifest(entry.manifest, store, verify_digests=verify_digests)
        )
    return results
