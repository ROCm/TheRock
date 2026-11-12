# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ClickHouse-backed cache database (shared serving index).

A drop-in alternative to the local SQLite ``CacheDB`` for the lookup/publish
surface the planner uses: ``lookup_batch(keys, policy) -> {ArtifactKey:
LocalManifest | None}`` and ``register_local_outputs(...)``. The SQLite store
keeps the relational-integrity model for a single machine; this backend is the
denormalized, append-only index that many runs share.

ClickHouse enforces no foreign keys, uniqueness, triggers, or CHECK
constraints, so the integrity those provide in SQLite is enforced here in
Python: entries are written already-validated, a ``ReplacingMergeTree`` keyed on
the artifact identity keeps the newest production per key, and ``ReadPolicy``
gates trust/validation/time at read time -- identical semantics to the SQLite
path, which is why the planner cannot tell the two apart.

Connection comes from the environment so no credential is ever written to disk:
``CLICKHOUSE_URL`` (https://host:8443), ``CLICKHOUSE_USER``,
``CLICKHOUSE_PASSWORD``, optional ``CLICKHOUSE_DATABASE`` (default ``default``).
"""

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence

from .manifest import (
    ArtifactFile,
    ArtifactKey,
    LocalManifest,
    TrustLevel,
    ValidationState,
)
from .cache_db import ReadPolicy

_TABLE = "cache_entries"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {db}.{table}
(
    epoch                 UInt32,
    artifact_identity     String,
    fingerprint           FixedString(64),
    components_json       String DEFAULT '[]',
    cache_entry_id        UInt64,
    artifact_key_id       UInt64,
    producer_run_id       UInt64,
    producer_commit       String,
    producer_label        String,
    local_directory       String,
    files_json            String,
    validation_state      Enum8('unvalidated'=0,'build_passed'=1,'tests_passed'=2),
    trust                 Enum8('untrusted'=0,'trusted'=1) DEFAULT 'untrusted',
    producer_metadata_json String DEFAULT '{{}}',
    state                 Enum8('pending'=0,'ready'=1,'invalid'=2) DEFAULT 'ready',
    validated_at          UInt32,
    created_at            UInt32,
    expires_at            Nullable(UInt32) DEFAULT NULL,
    version               UInt64
)
ENGINE = ReplacingMergeTree(version)
ORDER BY (epoch, fingerprint, components_json, artifact_identity)
"""


def _stable_id(*parts: str) -> int:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:7], "big") or 1  # 56-bit positive


@dataclass(frozen=True)
class ClickHouseConfig:
    url: str
    user: str
    password: str
    database: str = "default"
    timeout: float = 30.0
    max_retries: int = 6

    @staticmethod
    def from_env() -> "ClickHouseConfig":
        url = os.environ.get("CLICKHOUSE_URL")
        if not url:
            raise RuntimeError("CLICKHOUSE_URL is not set")
        return ClickHouseConfig(
            url=url.rstrip("/"),
            user=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            database=os.environ.get("CLICKHOUSE_DATABASE", "default"),
            timeout=float(os.environ.get("CLICKHOUSE_TIMEOUT", "30")),
            max_retries=int(os.environ.get("CLICKHOUSE_MAX_RETRIES", "6")),
        )


class ClickHouseCacheDB:
    """Shared ClickHouse serving index exposing the planner's DB contract."""

    def __init__(
        self, config: ClickHouseConfig | None = None, *, ensure_schema: bool = True
    ):
        self._cfg = config or ClickHouseConfig.from_env()
        if ensure_schema:
            self._execute(_CREATE_TABLE.format(db=self._cfg.database, table=_TABLE))

    # -- HTTP ---------------------------------------------------------------
    def _post(self, body: bytes, *, params: dict | None = None) -> str:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        token = base64.b64encode(
            f"{self._cfg.user}:{self._cfg.password}".encode("utf-8")
        ).decode("ascii")
        # ClickHouse Cloud is serverless: an idle service returns 503 while it
        # wakes (seconds). CI must tolerate that, so retry transient HTTP/network
        # failures with backoff instead of failing the job on a cold cache.
        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            request = urllib.request.Request(self._cfg.url + "/" + query, data=body)
            request.add_header("Authorization", f"Basic {token}")
            try:
                with urllib.request.urlopen(
                    request, timeout=self._cfg.timeout
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as exc:
                last_error = exc
            time.sleep(min(2**attempt, 15))
        raise RuntimeError(
            f"ClickHouse request failed after {self._cfg.max_retries} attempts: {last_error}"
        )

    def ping(self) -> str:
        """Preflight connectivity check; returns the server version."""
        return self._post(b"SELECT version()").strip()

    def _execute(self, sql: str) -> str:
        return self._post(sql.encode("utf-8"))

    def _query_json(self, sql: str) -> list[dict]:
        text = self._post((sql + "\nFORMAT JSONEachRow").encode("utf-8"))
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _insert_json(self, rows: list[dict]) -> None:
        if not rows:
            return
        body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        self._post(
            body,
            params={
                "query": f"INSERT INTO {self._cfg.database}.{_TABLE} FORMAT JSONEachRow"
            },
        )

    # -- writes -------------------------------------------------------------
    def register_local_outputs(
        self,
        key: ArtifactKey,
        *,
        local_directory: str,
        files: Sequence[ArtifactFile],
        validation_state: str = "build_passed",
        producer_label: str = "local",
        producer_commit: str = "",
        trust: TrustLevel = TrustLevel.UNTRUSTED,
        expires_at: int | None = None,
    ) -> int:
        """Publish one ready artifact unit; returns its cache_entry_id."""
        if validation_state not in ("build_passed", "tests_passed"):
            raise ValueError("only build/tests-passed outputs are publishable")
        if not files:
            raise ValueError("an artifact unit must declare at least one output file")
        now = int(time.time())
        components_json = json.dumps(list(key.components))
        entry_id = _stable_id(
            str(key.epoch), key.artifact_identity, key.fingerprint, components_json
        )
        metadata = {"cache_provenance": {"trust": trust.value}}
        self._insert_json(
            [
                {
                    "epoch": key.epoch,
                    "artifact_identity": key.artifact_identity,
                    "fingerprint": key.fingerprint,
                    "components_json": components_json,
                    "cache_entry_id": entry_id,
                    "artifact_key_id": _stable_id(key.fingerprint, components_json),
                    "producer_run_id": _stable_id(producer_label, producer_commit),
                    "producer_commit": producer_commit,
                    "producer_label": producer_label,
                    "local_directory": local_directory,
                    "files_json": json.dumps(
                        [
                            {
                                "relative_path": f.relative_path,
                                "content_sha256": f.content_sha256,
                                "size_bytes": f.size_bytes,
                            }
                            for f in files
                        ]
                    ),
                    "validation_state": validation_state,
                    "trust": trust.value,
                    "producer_metadata_json": json.dumps(metadata),
                    "state": "ready",
                    "validated_at": now,
                    "created_at": now,
                    "expires_at": expires_at,
                    "version": int(time.time() * 1000),
                }
            ]
        )
        return entry_id

    # -- reads --------------------------------------------------------------
    def _row_to_manifest(self, row: dict) -> LocalManifest:
        files = tuple(
            ArtifactFile(
                relative_path=f["relative_path"],
                content_sha256=f["content_sha256"],
                size_bytes=int(f["size_bytes"]),
            )
            for f in json.loads(row["files_json"])
        )
        key = ArtifactKey(
            row["artifact_identity"],
            row["fingerprint"],
            epoch=int(row["epoch"]),
            components=tuple(json.loads(row["components_json"])),
        )
        expires = row.get("expires_at")
        return LocalManifest(
            cache_entry_id=int(row["cache_entry_id"]),
            artifact_key_id=int(row["artifact_key_id"]),
            key=key,
            producer_run_id=int(row["producer_run_id"]),
            producer_commit=row["producer_commit"],
            producer_label=row["producer_label"],
            local_directory=row["local_directory"],
            expected_files=tuple(f.relative_path for f in files),
            files=files,
            validation_state=row["validation_state"],
            validated_at=int(row["validated_at"]),
            created_at=int(row["created_at"]),
            expires_at=None if expires in (None, "") else int(expires),
            producer_metadata=json.loads(row["producer_metadata_json"]),
        )

    def lookup_batch(
        self, keys: Sequence[ArtifactKey], policy: ReadPolicy | None = None
    ) -> dict[ArtifactKey, LocalManifest | None]:
        """Resolve full keys against the shared index, gated by ReadPolicy.

        Mirrors the SQLite backend: returns one dict entry per unique key, with
        the ready, policy-eligible, unexpired candidate or None. Trust,
        validation and point-in-time gating use the same ReadPolicy.allows the
        SQLite path uses, so the planner's behaviour is backend-independent.
        """
        if policy is not None and not isinstance(policy, ReadPolicy):
            raise TypeError("policy must be ReadPolicy")
        if any(not isinstance(key, ArtifactKey) for key in keys):
            raise TypeError("lookup requires ArtifactKey instances")
        unique = list(dict.fromkeys(keys))
        if not unique:
            return {}
        fingerprints = sorted({k.fingerprint for k in unique})
        epochs = sorted({int(k.epoch) for k in unique})
        fp_list = ",".join("'" + fp + "'" for fp in fingerprints)
        epoch_list = ",".join(str(e) for e in epochs)
        now = int(time.time())
        sql = (
            f"SELECT * FROM {self._cfg.database}.{_TABLE} FINAL "
            f"WHERE epoch IN ({epoch_list}) AND fingerprint IN ({fp_list}) "
            f"AND state = 'ready' "
            f"AND (expires_at IS NULL OR expires_at > {now})"
        )
        by_identity: dict[tuple, LocalManifest] = {}
        for row in self._query_json(sql):
            manifest = self._row_to_manifest(row)
            by_identity[
                (
                    manifest.key.epoch,
                    manifest.key.fingerprint,
                    manifest.key.artifact_identity,
                    manifest.key.components,
                )
            ] = manifest
        result: dict[ArtifactKey, LocalManifest | None] = {}
        for key in unique:
            manifest = by_identity.get(
                (key.epoch, key.fingerprint, key.artifact_identity, key.components)
            )
            if manifest is not None and (policy is None or policy.allows(manifest)):
                result[key] = manifest
            else:
                result[key] = None
        return result

    def count(self) -> int:
        rows = self._query_json(
            f"SELECT count() AS n FROM {self._cfg.database}.{_TABLE} FINAL WHERE state='ready'"
        )
        return int(rows[0]["n"]) if rows else 0
