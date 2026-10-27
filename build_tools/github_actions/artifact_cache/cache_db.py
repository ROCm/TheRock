# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Access the existing local relational artifact cache.

This module owns connection handling, schema checks, batched candidate lookup,
and observation persistence. Opening a database never creates or migrates it.
All cache matches are metadata candidates; bytes must be verified before reuse.
Optional policy and imported-history metadata live in existing provenance JSON.
No history import creates ready outputs or changes the revision-6 schema.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

from .diagnostics import trace, traced
import time
from typing import Any, Callable, Iterator, Sequence
from .cache_key import current_cache_epoch, validate_cache_epoch

from .manifest import (
    ArtifactFile, ArtifactKey, LocalManifest,
    Manifest, PublicationState, TrustLevel, ValidationState,
)


class SchemaError(RuntimeError):
    """The database does not expose the supported artifact index structure."""


def _encode_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)
    encoded.encode("utf-8")
    return encoded


# Database storage identifiers for producing runs.
# Public interfaces use run_id.
_RUN_TABLE = "prototype_runs"
_RUN_ID = "prototype_run_id"
_REQUIRED_COLUMNS = {
    "artifacts": {"artifact_id", "artifact_identity"},
    "artifact_keys": {
        "artifact_key_id", "artifact_id", "fingerprint", "epoch", "components_json"
    },
    _RUN_TABLE: {
        _RUN_ID, "label", "repository_commit", "configure_state", "source_state",
        "parameters_json", "source_revisions_json", "overlay_identity",
        "fingerprint_collect_ms", "lookup_ms", "plan_ms",
    },
    "artifact_observations": {
        _RUN_ID, "artifact_id", "artifact_key_id", "fingerprint_path",
        "fingerprint_state", "detail", "lookup_result", "selected_entry_id",
    },
    "cache_entries": {
        "cache_entry_id", "artifact_key_id", "producer_run_id", "local_directory",
        "expected_files_json", "state", "validation_state", "validated_at",
        "expires_at", "created_at",
    },
    "artifact_files": {
        "cache_entry_id", "relative_path", "content_sha256", "size_bytes",
        "availability",
    },
}
_REQUIRED_TRIGGERS = {
    "artifacts_no_update", "artifact_keys_no_update",
    "artifact_observations_configure_succeeded_insert",
    "artifact_observations_configure_succeeded_update",
    "cache_entries_start_pending", "cache_entries_identity_immutable",
    "cache_entries_manifest_valid", "artifact_files_expected_only",
    "artifact_files_identity_immutable", "cache_entries_publish_complete",
    "cache_entries_invalid_final",
}


LOOKUP_SQL = """
WITH requested AS (
    SELECT CAST(j.key AS INTEGER) AS request_index,
           json_extract(j.value, '$.artifact_identity') AS artifact_identity,
           json_extract(j.value, '$.fingerprint') AS fingerprint,
           json_extract(j.value, '$.fingerprint_state') AS fingerprint_state,
           json_extract(j.value, '$.epoch') AS epoch,
           COALESCE(json_extract(j.value, '$.components_json'), '[]') AS components_json
    FROM json_each(?) j
), policy AS (SELECT json(?) AS document), matched AS (
    SELECT r.*, k.artifact_key_id
    FROM requested r
    LEFT JOIN artifacts a ON a.artifact_identity = r.artifact_identity
    LEFT JOIN artifact_keys k
      ON k.artifact_id = a.artifact_id
     AND k.fingerprint = r.fingerprint
     AND k.epoch = r.epoch
     AND k.components_json = r.components_json
     AND r.fingerprint_state = 'valid'
), ranked AS (
    SELECT m.request_index, e.cache_entry_id, e.local_directory,
           ROW_NUMBER() OVER (
               PARTITION BY m.request_index
               ORDER BY e.created_at DESC, e.cache_entry_id DESC
           ) AS choice
    FROM matched m
    JOIN reusable_cache_entries e ON e.artifact_key_id = m.artifact_key_id
    JOIN prototype_runs p ON p.prototype_run_id = e.producer_run_id
    CROSS JOIN policy
    WHERE (json_extract(policy.document, '$.trusted') = 0
           OR json_extract(p.parameters_json, '$.cache_provenance.trust') = 'trusted')
      AND (json_extract(policy.document, '$.tests_passed') = 0
           OR e.validation_state = 'tests_passed')
      AND (json_extract(policy.document, '$.object_prefix') IS NULL
           OR substr(e.local_directory, 1,
                     length(json_extract(policy.document, '$.object_prefix')))
              = json_extract(policy.document, '$.object_prefix'))
      AND (json_extract(policy.document, '$.before') IS NULL
           OR (e.created_at < json_extract(policy.document, '$.before')
               AND e.validated_at < json_extract(policy.document, '$.before')))
)
SELECT m.request_index, m.artifact_key_id, r.cache_entry_id, r.local_directory
FROM matched m
LEFT JOIN ranked r ON r.request_index = m.request_index AND r.choice = 1
ORDER BY m.request_index
"""


def _validate_schema(db: sqlite3.Connection, *, without_epoch: bool = False,
                     namespaced: bool = False, component_only: bool = False) -> None:
    """Validate v6 or an explicitly requested migration input shape."""
    objects = {
        row["name"]: row["type"]
        for row in db.execute("SELECT name, type FROM sqlite_master")
    }
    for table, required in _REQUIRED_COLUMNS.items():
        if table == "artifact_keys":
            if without_epoch or namespaced or component_only:
                required = required - {"components_json"}
            if without_epoch or namespaced:
                required = required | {"key_namespace"}
            if without_epoch:
                required = required - {"epoch"}
        if objects.get(table) != "table":
            raise SchemaError(f"incompatible cache schema: missing table {table}")
        # Table identifiers are fixed implementation constants, never user input.
        available = {row["name"] for row in db.execute(f'PRAGMA table_info("{table}")')}
        if table == "artifact_keys" and not (without_epoch or namespaced) and "key_namespace" in available:
            raise SchemaError(
                "cache schema requires migration; run python3 -m "
                "build_tools.github_actions.artifact_cache.migrate_schema "
                "--db <database> --backup <new-backup-file>"
            )
        if missing := required - available:
            if table == "artifact_keys" and "components_json" in missing:
                raise SchemaError("cache schema needs the bundle membership migration; run python3 -m "
                                  "build_tools.github_actions.artifact_cache.migrate_schema "
                                  "--db <database> --backup <new-backup-file>")
            if table == "artifact_keys" and "epoch" in missing:
                raise SchemaError(
                    "cache schema requires migration; run python3 -m "
                    "build_tools.github_actions.artifact_cache.migrate_schema "
                    "--db <database> --backup <new-backup-file>"
                )
            raise SchemaError(
                f"incompatible cache schema: {table} missing {', '.join(sorted(missing))}"
            )
    unique_keys = {
        tuple(r["name"] for r in db.execute(
            "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (index["name"],)
        ))
        for index in db.execute('PRAGMA index_list("artifact_keys")')
        if index["unique"] and not index["partial"]
    }
    expected = (("key_namespace", "artifact_id", "fingerprint") if without_epoch else
                ("key_namespace", "epoch", "artifact_id", "fingerprint") if namespaced else
                ("epoch", "artifact_id", "fingerprint") if component_only else
                ("epoch", "artifact_id", "fingerprint", "components_json"))
    if expected not in unique_keys or (not (without_epoch or namespaced) and
            unique_keys - {expected, ("artifact_key_id", "artifact_id")}):
        raise SchemaError("artifact_keys uniqueness is incompatible; run the schema migration")
    missing_triggers = {name for name in _REQUIRED_TRIGGERS if objects.get(name) != "trigger"}
    if missing_triggers:
        raise SchemaError(
            "incompatible cache schema: missing triggers " + ", ".join(sorted(missing_triggers))
        )
    if objects.get("reusable_cache_entries") != "view":
        raise SchemaError("incompatible cache schema: missing reusable_cache_entries view")
    # Validate the view's dependencies and the query features before writing.
    if without_epoch or namespaced or component_only:
        db.execute("SELECT * FROM reusable_cache_entries LIMIT 0").fetchall()
    else:
        db.execute(LOOKUP_SQL, ("[]", '{"trusted":0,"tests_passed":0}')).fetchall()
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != 6:
            raise SchemaError(
                f"unsupported database schema version: {version}; expected revision 6"
            )


def _connect(database: Path, timeout: float) -> sqlite3.Connection:
    # mode=rw prevents a typo in --db from creating a new empty database.
    db = sqlite3.connect(
        database.expanduser().resolve().as_uri() + "?mode=rw", uri=True, timeout=timeout
    )
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        if db.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise SchemaError("SQLite foreign key enforcement is unavailable")
        return db
    except BaseException:
        db.close()
        raise


def _save(
    db: sqlite3.Connection,
    records: list[dict[str, Any]],
    *,
    label: str,
    commit: str,
    epoch: int,
    configure_state: str,
    parameters: dict[str, Any],
    source_revisions: dict[str, Any],
    overlay_identity: str | None,
    collect_ms: float,
    lookup_ms: float,
    plan_ms: float,
) -> int:
    run_id = db.execute(
        f"""INSERT INTO {_RUN_TABLE} (
            label, repository_commit, configure_state, source_state,
            parameters_json, source_revisions_json, overlay_identity,
            fingerprint_collect_ms, lookup_ms, plan_ms
        ) VALUES (?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?)""",
        (
            label, commit, configure_state, _encode_json(parameters),
            _encode_json(source_revisions), overlay_identity, collect_ms, lookup_ms, plan_ms,
        ),
    ).lastrowid
    encoded = _encode_json(records)
    db.execute(
        """INSERT INTO artifacts (artifact_identity)
        SELECT json_extract(value, '$.artifact_identity') FROM json_each(?)
        WHERE 1
        ON CONFLICT(artifact_identity) DO NOTHING""",
        (encoded,),
    )
    db.execute(
        """INSERT INTO artifact_keys (artifact_id, fingerprint, epoch)
        SELECT a.artifact_id, json_extract(j.value, '$.fingerprint'), ?
        FROM json_each(?) j
        JOIN artifacts a
          ON a.artifact_identity = json_extract(j.value, '$.artifact_identity')
        WHERE json_extract(j.value, '$.fingerprint_state') = 'valid'
        ON CONFLICT(epoch, artifact_id, fingerprint, components_json) DO NOTHING""",
        (epoch, encoded),
    )
    inserted = db.execute(
        f"""INSERT INTO artifact_observations (
            {_RUN_ID}, artifact_id, artifact_key_id, fingerprint_path,
            fingerprint_state, detail, lookup_result, selected_entry_id
        )
        SELECT ?, a.artifact_id, k.artifact_key_id,
               json_extract(j.value, '$.fingerprint_path'),
               json_extract(j.value, '$.fingerprint_state'),
               json_extract(j.value, '$.detail'),
               json_extract(j.value, '$.lookup_result'),
               json_extract(j.value, '$.selected_entry_id')
        FROM json_each(?) j
        JOIN artifacts a
          ON a.artifact_identity = json_extract(j.value, '$.artifact_identity')
        LEFT JOIN artifact_keys k
          ON k.artifact_id = a.artifact_id AND k.epoch = ?
         AND k.fingerprint = json_extract(j.value, '$.fingerprint')
         AND k.components_json = '[]'""",
        (run_id, encoded, epoch),
    )
    if run_id is None or inserted.rowcount != len(records):
        raise SchemaError("incomplete run or observation insertion; transaction rolled back")
    return run_id


@dataclass(frozen=True)
class ReadPolicy:
    """Filter recorded metadata, not credentials or producer authorization.

    Trust comes from explicit producer provenance in parameters_json; absent
    provenance is untrusted. SQLite write access remains the trust boundary.
    Pending/unvalidated entries can never be made servable by relaxing policy.
    """

    min_trust: TrustLevel = TrustLevel.TRUSTED
    min_validation: ValidationState = ValidationState.BUILD_PASSED
    require_published: bool = True
    # Optional point-in-time bound for replay: ignore entries created at/after
    # this ISO timestamp, so a replay can only see what existed at lookup time.
    created_before: str | None = None

    def __post_init__(self):
        if not isinstance(self.min_trust, TrustLevel) or not isinstance(self.min_validation, ValidationState):
            raise ValueError("read policy requires TrustLevel and ValidationState enums")
        if type(self.require_published) is not bool:
            raise ValueError("require_published must be boolean")
        if self.created_before is not None:
            timestamp(self.created_before)

    def query_parameters(self) -> dict:
        return {"trusted": int(self.min_trust is TrustLevel.TRUSTED),
                "tests_passed": int(self.min_validation is ValidationState.TESTS_PASSED),
                "before": timestamp(self.created_before) if self.created_before is not None else None}

    def allows(self, manifest: Manifest | LocalManifest) -> bool:
        if isinstance(manifest, LocalManifest):
            return (manifest.trust.meets(self.min_trust)
                    and ValidationState(manifest.validation_state).meets(self.min_validation)
                    and (self.created_before is None or max(manifest.created_at, manifest.validated_at)
                         < timestamp(self.created_before)))
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


def timestamp(value: str) -> float:
    """Parse an explicitly zoned ISO timestamp; reject ambiguous local times."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string with a timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return parsed.timestamp()


class CacheDB:
    """Access an existing revision-6 SQLite artifact cache.

    Collection uses lookup_observations and record_observations in one short
    write transaction. lookup_batch and get_manifest reconstruct LocalManifest
    candidates for verified restoration. Local publication registers completed
    object generations; it does not run builds or copy files inside SQL locks.

    Lookups require ArtifactKey(output_identity, fprint, epoch=...)
    containing the raw CMake fingerprint and requested component membership.
    """

    def __init__(self, path: str | Path, *, timeout: float = 5.0):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 <= timeout <= 60
        ):
            raise ValueError("timeout must be a finite number between 0 and 60 seconds")
        trace("DB_OPEN", path=Path(path).expanduser().absolute(), timeout_seconds=timeout)
        self._conn = _connect(Path(path), timeout)
        try:
            _validate_schema(self._conn)
            trace("DB_SCHEMA_VALID", revision=6, foreign_keys=True)
        except BaseException:
            self._conn.close()
            raise

    def __enter__(self) -> "CacheDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # sqlite3.close rolls back any unfinished transaction.
        self._conn.close()

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[None]:
        """Commit once on success; roll back all work on failure."""
        if self._conn.in_transaction:
            raise RuntimeError("nested cache transactions are not supported")
        started = time.monotonic()
        trace("DB_TRANSACTION_BEGIN", mode="write" if write else "read")
        self._conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self._conn.commit()
            trace("DB_TRANSACTION_COMMITTED", mode="write" if write else "read",
                  duration_seconds=round(time.monotonic() - started, 3))
        except BaseException as exc:
            self._conn.rollback()
            trace("DB_TRANSACTION_ROLLED_BACK", mode="write" if write else "read",
                  reason=str(exc), duration_seconds=round(time.monotonic() - started, 3))
            raise

    def _lookup(self, requests: list[dict[str, Any]], policy: ReadPolicy | None = None,
                *, cache_root: Path | None = None) -> list[sqlite3.Row]:
        options = policy.query_parameters() if policy else {"trusted": 0, "tests_passed": 0}
        # Storage location filters candidates, never key identity. Apply before
        # ranking so a newer production in another store cannot hide a local one.
        if cache_root is not None:
            options["object_prefix"] = str(Path(cache_root).expanduser().resolve()).rstrip(os.sep) + os.sep
        trace("DB_LOOKUP", requests=len(requests), scope="metadata_only",
              min_trust=policy.min_trust.value if policy else "untrusted",
              min_validation=policy.min_validation.value if policy else "build_passed",
              created_before=policy.created_before if policy else None,
              cache_root=cache_root,
              requirements="ready, validated, unexpired, complete available object metadata")
        rows = self._conn.execute(LOOKUP_SQL, (_encode_json(requests), _encode_json(options))).fetchall()
        if [row["request_index"] for row in rows] != list(range(len(requests))):
            raise SchemaError("lookup did not return exactly one result per expected output")
        trace("DB_LOOKUP_RESULT", requests=len(rows),
              selected=sum(row["cache_entry_id"] is not None for row in rows),
              payloads_verified=False)
        return rows

    def lookup_observations(
        self, records: list[dict[str, Any]], *, epoch: int | None = None
    ) -> list[sqlite3.Row]:
        """Resolve collected inputs, preserving missing/invalid observations."""
        epoch = current_cache_epoch() if epoch is None else validate_cache_epoch(epoch)
        requests = []
        for record in records:
            if record["fingerprint_state"] == "valid":
                ArtifactKey(record["artifact_identity"], record["fingerprint"], epoch=epoch)
            requests.append({
                "epoch": epoch,
                "artifact_identity": record["artifact_identity"],
                "fingerprint": record["fingerprint"],
                "fingerprint_state": record["fingerprint_state"],
            })
        return self._lookup(requests)

    def record_observations(
        self,
        records: list[dict[str, Any]],
        *,
        label: str,
        commit: str,
        configure_state: str,
        parameters: dict[str, Any],
        source_revisions: dict[str, Any],
        overlay_identity: str | None,
        collect_ms: float,
        lookup_ms: float,
        plan_ms: float,
        epoch: int | None = None,
    ) -> int:
        """Persist a collector run inside the caller's transaction.

        No cache_entries or artifact_files are created by observation ingestion.
        The collector validates input files; SQLite enforces relational rules.
        """
        if not self._conn.in_transaction:
            raise RuntimeError("record_observations requires an active transaction")
        epoch = current_cache_epoch() if epoch is None else validate_cache_epoch(epoch)
        return _save(
            self._conn, records, label=label, commit=commit, epoch=epoch,
            configure_state=configure_state, parameters=parameters,
            source_revisions=source_revisions, overlay_identity=overlay_identity,
            collect_ms=collect_ms, lookup_ms=lookup_ms, plan_ms=plan_ms,
        )

    def lookup(
        self, key: ArtifactKey, policy: ReadPolicy | None = None,
        *, cache_root: Path | None = None,
    ) -> LocalManifest | None:
        return self.lookup_batch([key], policy, cache_root=cache_root).get(key)

    @traced("BATCH_LOOKUP")
    def lookup_batch(
        self, keys: Sequence[ArtifactKey], policy: ReadPolicy | None = None,
        *, cache_root: Path | None = None,
    ) -> dict[ArtifactKey, LocalManifest | None]:
        """Resolve full keys in a consistent snapshot, independent of CI history.

        One batched query selects candidates; one retrieves their file metadata.
        Duplicate requests are collapsed. No per-artifact queries or file I/O.
        """
        if policy is not None and not isinstance(policy, ReadPolicy):
            raise TypeError("policy must be ReadPolicy")
        if any(not isinstance(key, ArtifactKey) for key in keys):
            raise TypeError(
                "lookup requires ArtifactKey(artifact_identity, fingerprint, epoch=...); "
                "derived string keys are not supported"
            )
        unique = list(dict.fromkeys(keys))
        if not unique:
            return {}
        requests = [
            {"epoch": key.epoch,
             "artifact_identity": key.artifact_identity,
             "fingerprint": key.fingerprint, "fingerprint_state": "valid",
             "components_json": _encode_json(list(key.components))}
            for key in unique
        ]
        scope = nullcontext() if self._conn.in_transaction else self.transaction(write=False)
        with scope:
            rows = self._lookup(requests, policy, cache_root=cache_root)
            manifests = self._load_manifests([
                row["cache_entry_id"] for row in rows if row["cache_entry_id"] is not None
            ])
            return {
                key: manifests.get(row["cache_entry_id"])
                for key, row in zip(unique, rows)
            }

    def get_manifest(self, cache_entry_id: int) -> LocalManifest | None:
        """Read a currently eligible candidate; retired/expired entries return None."""
        if type(cache_entry_id) is not int or cache_entry_id <= 0:
            raise ValueError("cache_entry_id must be a positive integer")
        return self._load_manifests([cache_entry_id]).get(cache_entry_id)

    def _load_manifests(self, entry_ids: Sequence[int]) -> dict[int, LocalManifest]:
        if not entry_ids:
            return {}
        rows = self._conn.execute(
            """SELECT e.*, k.fingerprint, k.epoch, k.components_json,
                      a.artifact_identity, p.repository_commit, p.label, p.parameters_json,
                      f.relative_path, f.content_sha256, f.size_bytes
               FROM reusable_cache_entries AS e
               JOIN artifact_keys AS k USING (artifact_key_id)
               JOIN artifacts AS a USING (artifact_id)
               JOIN prototype_runs AS p ON p.prototype_run_id = e.producer_run_id
               JOIN artifact_files AS f USING (cache_entry_id)
               WHERE e.cache_entry_id IN (SELECT value FROM json_each(?))
               ORDER BY e.cache_entry_id, f.relative_path""",
            (_encode_json(list(dict.fromkeys(entry_ids))),),
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["cache_entry_id"], []).append(row)
        result = {}
        for entry_id, files in grouped.items():
            row = files[0]
            try:
                result[entry_id] = LocalManifest(
                    cache_entry_id=entry_id,
                    artifact_key_id=row["artifact_key_id"],
                    key=ArtifactKey(row["artifact_identity"],
                                    row["fingerprint"], epoch=row["epoch"],
                                    components=tuple(json.loads(row["components_json"]))),
                    producer_run_id=row["producer_run_id"],
                    producer_commit=row["repository_commit"],
                    producer_label=row["label"],
                    local_directory=row["local_directory"],
                    expected_files=tuple(json.loads(row["expected_files_json"])),
                    files=tuple(ArtifactFile(
                        f["relative_path"], f["content_sha256"], f["size_bytes"]
                    ) for f in files),
                    validation_state=row["validation_state"],
                    validated_at=row["validated_at"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    producer_metadata=json.loads(row["parameters_json"]),
                )
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"invalid manifest metadata for cache entry {entry_id}: {exc}") from exc
        return result

    def stats(self) -> dict[str, int]:
        """Counts from the local schema; never initialize another set of tables."""
        result = {}
        for table in (
            "artifacts", "artifact_keys", "prototype_runs", "artifact_observations",
            "cache_entries", "artifact_files", "reusable_cache_entries",
        ):
            result[table] = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        return result

    @traced("REGISTER_OUTPUTS")
    def register_local_outputs(
        self, *, producer_run_id: int,
        outputs: Sequence[tuple[ArtifactKey, Path, Sequence[ArtifactFile]]],
    ) -> list[LocalManifest]:
        """Register verified generations after the managed publisher built them.

        This is the metadata boundary, not a build attestation API. The caller
        must own the build and byte verification. All entries/inventories and
        their ready transitions commit together, after filesystem work.
        """
        if type(producer_run_id) is not int or producer_run_id <= 0 or not outputs:
            raise ValueError("publication needs a producing run and outputs")
        entry_ids = []
        trace("REGISTER_OUTPUTS", producer_run_id=producer_run_id, outputs=len(outputs))
        with self.transaction():
            producer = self._conn.execute(
                "SELECT configure_state, parameters_json FROM prototype_runs "
                "WHERE prototype_run_id = ?", (producer_run_id,),
            ).fetchone()
            if producer is None or producer["configure_state"] != "succeeded":
                raise ValueError("publication needs a successfully configured producer")
            evidence = json.loads(producer["parameters_json"]).get("managed_build", {})
            if (evidence.get("returncode") != 0 or not evidence.get("command")
                    or evidence.get("inputs_stable") is not True):
                raise ValueError("producer has no successful managed-build record")
            for key, directory, files in outputs:
                if not isinstance(key, ArtifactKey) or not Path(directory).is_absolute():
                    raise ValueError("publication requires full keys and absolute locations")
                files = tuple(files)
                if (any(not isinstance(f, ArtifactFile) for f in files)
                        or sorted(f.relative_path for f in files)
                        != ["inventory.json", "payload.tar"]):
                    raise ValueError("local publication requires both archive and inventory")
                if key.components:
                    self._record_bundle_observation(producer_run_id, key)
                observed = self._conn.execute(
                    """SELECT k.artifact_key_id FROM artifact_observations o
                       JOIN artifact_keys k USING (artifact_key_id)
                       JOIN artifacts a ON a.artifact_id = k.artifact_id
                       WHERE o.prototype_run_id = ? AND o.artifact_id = a.artifact_id
                         AND o.fingerprint_state = 'valid'
                         AND k.epoch = ?
                         AND a.artifact_identity = ? AND k.fingerprint = ? AND k.components_json = ?""",
                    (producer_run_id, key.epoch,
                     key.artifact_identity, key.fingerprint, _encode_json(list(key.components))),
                ).fetchone()
                if observed is None:
                    raise ValueError("producing run did not observe this exact artifact key")
                entry_id = self._conn.execute(
                    """INSERT INTO cache_entries (
                           artifact_key_id, producer_run_id, local_directory, expected_files_json
                       ) VALUES (?, ?, ?, ?)""",
                    (observed["artifact_key_id"], producer_run_id, str(directory),
                     _encode_json([f.relative_path for f in files])),
                ).lastrowid
                self._conn.executemany(
                    """INSERT INTO artifact_files (
                           cache_entry_id, relative_path, content_sha256, size_bytes
                       ) VALUES (?, ?, ?, ?)""",
                    [(entry_id, f.relative_path, f.content_sha256, f.size_bytes) for f in files],
                )
                self._conn.execute(
                    """UPDATE cache_entries SET state = 'ready',
                           validation_state = 'build_passed', validated_at = ?
                       WHERE cache_entry_id = ?""", (int(time.time()), entry_id),
                )
                entry_ids.append(entry_id)
            manifests = self._load_manifests(entry_ids)
            if set(manifests) != set(entry_ids):
                raise SchemaError("published outputs did not become eligible candidates")
        for entry_id in entry_ids:
            manifest = manifests[entry_id]
            trace("ENTRY_COMMITTED", artifact=manifest.key.artifact_identity,
                  epoch=manifest.key.epoch, entry=entry_id, producer_run_id=producer_run_id,
                  directory=manifest.local_directory, state="ready")
        return [manifests[entry_id] for entry_id in entry_ids]

    def _record_bundle_observation(self, run_id: int, key: ArtifactKey) -> None:
        """Bind a bundle to all its actually observed component keys in this run.

        Called inside register_local_outputs' transaction; bundle observation,
        object rows and ready transition commit or roll back together.
        """
        members = _encode_json(list(key.components))
        rows = self._conn.execute(
            """SELECT a.artifact_identity FROM artifact_observations o
               JOIN artifacts a USING(artifact_id)
               JOIN artifact_keys k USING(artifact_key_id)
               WHERE o.prototype_run_id=? AND o.fingerprint_state='valid'
                 AND k.epoch=? AND k.fingerprint=? AND k.components_json='[]'
                 AND a.artifact_identity IN (SELECT value FROM json_each(?))""",
            (run_id, key.epoch, key.fingerprint, members)).fetchall()
        if {r[0] for r in rows} != set(key.components):
            raise ValueError("producing run did not observe every bundle component with the same fingerprint")
        self._conn.execute("INSERT INTO artifacts(artifact_identity) VALUES (?) "
                           "ON CONFLICT DO NOTHING", (key.artifact_identity,))
        artifact_id = self._conn.execute("SELECT artifact_id FROM artifacts WHERE artifact_identity=?",
                                        (key.artifact_identity,)).fetchone()[0]
        self._conn.execute("INSERT INTO artifact_keys(artifact_id,epoch,fingerprint,components_json) "
                           "VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                           (artifact_id, key.epoch, key.fingerprint, members))
        key_id = self._conn.execute("SELECT artifact_key_id FROM artifact_keys WHERE artifact_id=? "
                                   "AND epoch=? AND fingerprint=? AND components_json=?",
                                   (artifact_id, key.epoch, key.fingerprint, members)).fetchone()[0]
        self._conn.execute("""INSERT INTO artifact_observations(prototype_run_id,artifact_id,
                           artifact_key_id,fingerprint_state,detail,lookup_result)
                           VALUES (?,?,?,'valid',?,'not_checked')""",
                           (run_id, artifact_id, key_id, "bundle components: " + members))

    def invalidate_local_output(self, cache_entry_id: int) -> None:
        """Retire only the candidate whose physical objects failed verification."""
        if type(cache_entry_id) is not int or cache_entry_id <= 0:
            raise ValueError("cache_entry_id must be a positive integer")
        with self.transaction():
            self._conn.execute(
                "UPDATE cache_entries SET state = 'invalid' WHERE cache_entry_id = ?",
                (cache_entry_id,),
            )
        trace("ENTRY_INVALIDATED", entry=cache_entry_id)

    def count(self) -> int:
        return self.stats()["reusable_cache_entries"]

    def all_manifests(self, policy: ReadPolicy | None = None) -> list[LocalManifest]:
        with self.transaction(write=False):
            ids = [r[0] for r in self._conn.execute("SELECT cache_entry_id FROM reusable_cache_entries")]
            return [m for m in self._load_manifests(ids).values() if policy is None or policy.allows(m)]

    def evict_before(self, created_before_iso: str) -> int:
        """Retire old entries without breaking historical observation references.

        Payload deletion/garbage collection is deliberately separate.
        """
        cutoff = timestamp(created_before_iso)
        with self.transaction():
            return self._conn.execute(
                "UPDATE cache_entries SET state='invalid' WHERE state <> 'invalid' AND created_at < ?",
                (cutoff,),
            ).rowcount

    def evict_where_missing(self, object_exists: Callable[[str], bool]) -> int:
        bad = [m.cache_entry_id for m in self.all_manifests()
               if any(not object_exists(str(Path(m.local_directory) / f.relative_path)) for f in m.files)]
        with self.transaction():
            return self._conn.execute(
                "UPDATE cache_entries SET state='invalid' WHERE state <> 'invalid' "
                "AND cache_entry_id IN (SELECT value FROM json_each(?))", (_encode_json(bad),),
            ).rowcount

    def record_history(self, evidence: dict) -> tuple[int, bool]:
        """Persist run-history evidence, never fabricated ready productions.

        A typed parameters_json record preserves the existing schema. These
        rows have configure_state=not_run and do not create artifact keys.
        Identical imports are idempotent; corrections are separate snapshots.
        """
        if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
            raise ValueError("history evidence requires schema_version=1")
        for name in ("repository", "run_id", "commit", "platform", "build_variant"):
            if not isinstance(evidence.get(name), str) or not evidence[name]:
                raise ValueError(f"history requires a nonempty {name}")
        if type(evidence.get("run_attempt")) is not int or evidence["run_attempt"] < 1:
            raise ValueError("history requires a positive run_attempt")
        if timestamp(evidence["available_at"]) < timestamp(evidence["started_at"]):
            raise ValueError("history availability predates start")
        TrustLevel(evidence["trust"])
        ValidationState(evidence["validation"])
        if not isinstance(evidence.get("units"), list):
            raise ValueError("history units must be an array")
        seen = set()
        for unit in evidence["units"]:
            name = unit["artifact_identity"]
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("duplicate/invalid history output identity")
            seen.add(name)
            if unit.get("fingerprint_kind") == "cmake":
                ArtifactKey(name, unit["fingerprint"])
            elif (unit.get("fingerprint_kind") != "commit_fallback"
                  or not isinstance(unit.get("fingerprint"), str)
                  or not unit["fingerprint"].startswith("srcfp1:")):
                raise ValueError("invalid history fingerprint kind/value")
            if not isinstance(unit.get("objects"), list) or not unit["objects"]:
                raise ValueError("history unit needs recorded objects")
            if any(not isinstance(o, dict) or not o.get("uri") or not o.get("key") for o in unit["objects"]):
                raise ValueError("history object requires a key and URI")
        encoded = _encode_json(evidence)
        identity = hashlib.sha256(encoded.encode()).hexdigest()
        scope = nullcontext() if self._conn.in_transaction else self.transaction()
        with scope:
            prior = self._conn.execute(
                "SELECT prototype_run_id FROM prototype_runs WHERE "
                "json_extract(parameters_json, '$.history_digest') = ?", (identity,),
            ).fetchone()
            if prior:
                return prior[0], False
            run_id = self._conn.execute(
                """INSERT INTO prototype_runs(label, repository_commit, configure_state,
                       source_state, parameters_json) VALUES (?, ?, 'not_run', 'prepared', ?)""",
                (f"history:{evidence['repository']}:{evidence['run_id']}", evidence["commit"],
                 _encode_json({"cache_history": evidence, "history_digest": identity})),
            ).lastrowid
            return run_id, True

    def history(self, repository: str | None = None) -> list[dict]:
        rows = self._conn.execute(
            "SELECT prototype_run_id, json_extract(parameters_json, '$.cache_history') AS evidence "
            "FROM prototype_runs WHERE json_type(parameters_json, '$.cache_history') = 'object' "
            "ORDER BY prototype_run_id"
        ).fetchall()
        result = []
        for row in rows:
            evidence = json.loads(row["evidence"])
            if repository is None or evidence.get("repository") == repository:
                result.append({**evidence, "observation_run_id": row["prototype_run_id"]})
        return result

    def publish(self, manifest: Manifest) -> None:
        """Require managed-build evidence through register_local_outputs."""
        raise NotImplementedError(
            "local publication must register pending cache_entries and verified "
            "artifact_files before marking an entry ready; publication from a CI manifest "
            "alone is not supported"
        )
