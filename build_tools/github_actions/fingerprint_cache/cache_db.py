# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite-backed fingerprint -> manifest cache store.

The lookup requirement from the design is "bounded direct lookups, independent
of workflow-history length, preferably batched". SQLite satisfies that:
a single ``WHERE cache_key IN (...)`` query resolves a whole batch in one
round trip, and the query layer is abstracted behind ``CacheDB`` so it can be
repointed at the team's ClickHouse instance later without touching callers.

Design points implemented here:

* **Batched O(1)-ish lookup** (``lookup_batch``): one SQL statement per batch,
  not one GitHub API call per candidate run.
* **Ordered publish** (``publish``): refuses to insert an entry unless the
  manifest is complete and marked ``PUBLISHED`` -- the "publish the fingerprint
  entry last" rule from flowchart 3. Callers upload objects first, then publish.
* **Read policy**: trust/validation/publication filtering so untrusted or
  under-validated entries can be excluded at lookup time (trusted vs untrusted
  cache from the design meeting).
* **Eviction**: ``evict_before`` (retention) and ``evict_where_missing`` (the
  cache points into existing S3 buckets with their own retention, so entries
  must be droppable when their objects disappear).
"""


from dataclasses import dataclass
import sqlite3
from typing import Callable, Iterable, Sequence

from .manifest import (
    Manifest,
    PublicationState,
    TrustLevel,
    ValidationState,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifests (
    cache_key       TEXT PRIMARY KEY,
    artifact_name   TEXT NOT NULL,
    component       TEXT NOT NULL,
    target_family   TEXT NOT NULL,
    build_variant   TEXT NOT NULL,
    trust           TEXT NOT NULL,
    validation      TEXT NOT NULL,
    publication     TEXT NOT NULL,
    repository      TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    branch          TEXT NOT NULL,
    commit_sha      TEXT NOT NULL,
    workflow        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    manifest_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_manifests_run_id ON manifests(run_id);
CREATE INDEX IF NOT EXISTS idx_manifests_artifact
    ON manifests(artifact_name, target_family);
CREATE INDEX IF NOT EXISTS idx_manifests_created_at ON manifests(created_at);
"""


@dataclass(frozen=True)
class ReadPolicy:
    """What a consumer is willing to reuse.

    Defaults are conservative: only published, build-passed, trusted entries.
    Loosen (e.g. ``min_trust=UNTRUSTED``) for same-PR reuse experiments.
    """

    min_trust: TrustLevel = TrustLevel.TRUSTED
    min_validation: ValidationState = ValidationState.BUILD_PASSED
    require_published: bool = True
    # Optional point-in-time bound for replay: ignore entries created at/after
    # this ISO timestamp, so a replay can only see what existed at lookup time.
    created_before: str | None = None

    def allows(self, manifest: Manifest) -> bool:
        if (
            self.require_published
            and manifest.publication is not PublicationState.PUBLISHED
        ):
            return False
        if not manifest.trust.meets(self.min_trust):
            return False
        if not manifest.validation.meets(self.min_validation):
            return False
        if self.created_before is not None and (
            manifest.provenance.created_at >= self.created_before
        ):
            return False
        return True


class CacheDB:
    """A fingerprint -> manifest store backed by SQLite.

    Usable as a context manager::

        with CacheDB("cache.sqlite") as db:
            db.publish(manifest)
            hits = db.lookup_batch([key.value for key in keys])
    """

    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CacheDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def publish(self, manifest: Manifest) -> None:
        """Publish a completed entry (flowchart 3 write ordering).

        Refuses incomplete or non-``PUBLISHED`` manifests: objects are expected
        to be uploaded *before* this call, and the fingerprint entry is written
        *last*. Re-publishing the same key replaces the entry (idempotent).
        """
        if manifest.publication is not PublicationState.PUBLISHED:
            raise ValueError(
                "refusing to publish a manifest that is not PUBLISHED "
                f"(state={manifest.publication.value}); upload objects and mark "
                "PUBLISHED before writing the cache entry"
            )
        if not manifest.is_complete():
            raise ValueError(
                "refusing to publish an incomplete manifest "
                f"(key={manifest.cache_key}); every output needs a URI"
            )
        a = manifest.artifact
        p = manifest.provenance
        self._conn.execute(
            """
            INSERT INTO manifests (
                cache_key, artifact_name, component, target_family,
                build_variant, trust, validation, publication,
                repository, run_id, branch, commit_sha, workflow, created_at,
                manifest_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(cache_key) DO UPDATE SET
                artifact_name=excluded.artifact_name,
                component=excluded.component,
                target_family=excluded.target_family,
                build_variant=excluded.build_variant,
                trust=excluded.trust,
                validation=excluded.validation,
                publication=excluded.publication,
                repository=excluded.repository,
                run_id=excluded.run_id,
                branch=excluded.branch,
                commit_sha=excluded.commit_sha,
                workflow=excluded.workflow,
                created_at=excluded.created_at,
                manifest_json=excluded.manifest_json
            """,
            (
                manifest.cache_key,
                a.name,
                a.component,
                a.target_family,
                manifest.build_variant,
                manifest.trust.value,
                manifest.validation.value,
                manifest.publication.value,
                p.repository,
                p.run_id,
                p.branch,
                p.commit,
                p.workflow,
                p.created_at,
                manifest.to_json(),
            ),
        )
        self._conn.commit()

    # -- reads -------------------------------------------------------------
    def lookup(
        self, cache_key: str, policy: ReadPolicy | None = None
    ) -> Manifest | None:
        """Look up a single key, honoring ``policy`` (defaults applied)."""
        return self.lookup_batch([cache_key], policy).get(cache_key)

    def lookup_batch(
        self, cache_keys: Sequence[str], policy: ReadPolicy | None = None
    ) -> dict[str, Manifest | None]:
        """Resolve a batch of keys in a single query.

        Returns a dict mapping every requested key to its servable ``Manifest``
        or ``None`` (miss, or filtered out by policy). This is the operation
        that replaces the current per-candidate GitHub API traversal.
        """
        policy = policy or ReadPolicy()
        result: dict[str, Manifest | None] = {k: None for k in cache_keys}
        unique = list(dict.fromkeys(cache_keys))  # dedupe, preserve order
        CHUNK = 500  # stay well under SQLite's variable limit
        for start in range(0, len(unique), CHUNK):
            chunk = unique[start : start + CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT cache_key, manifest_json FROM manifests "
                f"WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                manifest = Manifest.from_json(row["manifest_json"])
                if policy.allows(manifest):
                    result[row["cache_key"]] = manifest
        return result

    def get_by_run_id(self, run_id: str) -> list[Manifest]:
        """All entries produced by a run (O(1) via index) -- diagnostics."""
        rows = self._conn.execute(
            "SELECT manifest_json FROM manifests WHERE run_id = ?",
            (str(run_id),),
        ).fetchall()
        return [Manifest.from_json(r["manifest_json"]) for r in rows]

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) AS n FROM manifests").fetchone()["n"]
        )

    def stats(self) -> dict[str, int]:
        """Summary counts by trust/validation for monitoring."""
        out: dict[str, int] = {"total": self.count()}
        for col in ("trust", "validation", "publication"):
            for row in self._conn.execute(
                f"SELECT {col} AS v, COUNT(*) AS n FROM manifests GROUP BY {col}"
            ).fetchall():
                out[f"{col}:{row['v']}"] = int(row["n"])
        return out

    # -- eviction ----------------------------------------------------------
    def evict_before(self, created_before_iso: str) -> int:
        """Drop entries older than a timestamp (retention). Returns #removed."""
        cur = self._conn.execute(
            "DELETE FROM manifests WHERE created_at < ?", (created_before_iso,)
        )
        self._conn.commit()
        return cur.rowcount

    def evict_where_missing(self, object_exists: Callable[[str], bool]) -> int:
        """Drop entries whose backing objects have disappeared from storage.

        The cache points into existing S3 buckets that have their own retention
        policy, so entries must be tolerant of objects vanishing. ``object_exists``
        is called with each output URI; an entry is evicted if any object is gone.
        """
        removed = 0
        rows = self._conn.execute(
            "SELECT cache_key, manifest_json FROM manifests"
        ).fetchall()
        for row in rows:
            manifest = Manifest.from_json(row["manifest_json"])
            if not all(object_exists(o.uri) for o in manifest.outputs):
                self._conn.execute(
                    "DELETE FROM manifests WHERE cache_key = ?", (row["cache_key"],)
                )
                removed += 1
        self._conn.commit()
        return removed

    def all_manifests(self) -> Iterable[Manifest]:
        for row in self._conn.execute("SELECT manifest_json FROM manifests"):
            yield Manifest.from_json(row["manifest_json"])
