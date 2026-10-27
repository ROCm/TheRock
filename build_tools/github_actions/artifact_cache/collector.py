#!/usr/bin/env python3
"""Collect fingerprints, build/publish outputs, and restore local artifacts.

The collector records configured build outputs and reports eligible database
candidates. Artifact publication, payload verification, and restoration are
separate operations. A reported hit requires verification before restoration.
``build-publish`` owns the configure/build sequence and stores complete exports
before registering ready entries. ``restore`` reconfigures and materializes
verified exports into a new directory; it does not skip CMake stages.

Run after CMake using a fresh build directory or known-fresh configure outputs.
The caller supplies configure_state; this program cannot prove that an old
.fprint belongs to the claimed commit/configuration. Use one known build
environment until fingerprint input coverage has been audited.

Expected-output JSON (identities must come from the configured output contract):
    {"schema_version": 1, "artifacts": [
        {"artifact_identity": "example_lib_generic",
         "fingerprint_path": "example_lib_generic.fprint"}
    ]}

Paths are relative to --artifacts-dir. Do not generate this list by scanning only
existing .fprint files: that would silently omit missing fingerprints.

Example (from TheRock's root):
    python3 -m build_tools.github_actions.artifact_cache.collector collect \
        --db ~/cache.db --expected /tmp/expected_artifacts.json \
        --artifacts-dir /tmp/therock-build/artifacts \
        --label baseline --commit COMMIT_SHA \
        --configure-state succeeded

Python standard library only. The database must already exist. This tool does
not initialize, migrate, or silently replace a database. JSON is written to
stdout; diagnostics go to stderr. Collection and successful build/publication
record runs; restoration does not create a producing run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import stat
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn, TypedDict


# Support direct execution and python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB, SchemaError
from build_tools.github_actions.artifact_cache.cache_key import (
    current_cache_epoch,
    validate_cache_epoch,
)
from build_tools.github_actions.artifact_cache.diagnostics import (
    add_logging_args,
    run_logged,
    trace,
    traced,
)


MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FINGERPRINT_BYTES = 4096
DEFAULT_DATABASE_TIMEOUT = 5.0
REPORT_VERSION = 3
ConfigureState = Literal["succeeded", "failed", "not_run"]


class ExpectedArtifact(TypedDict):
    artifact_identity: str
    fingerprint_path: str


class FingerprintRecord(ExpectedArtifact):
    fingerprint: str | None
    fingerprint_state: Literal["valid", "missing", "invalid"]
    detail: str | None


class InputError(ValueError):
    """A supplied manifest, path, or configuration value is invalid."""


def _json(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)
        encoded.encode("utf-8")
        return encoded
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError(f"value is not valid UTF-8 JSON: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise InputError(f"non-finite JSON number: {value}")


def _read_regular_file(path: Path, maximum: int) -> bytes:
    """Read bounded input and detect changes to the opened file during reading."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise InputError(f"{path}: expected a regular file")
        if before.st_size > maximum:
            raise InputError(f"{path}: file exceeds {maximum} bytes")
        contents = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if len(contents) > maximum:
        raise InputError(f"{path}: file exceeds {maximum} bytes")
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise InputError(f"{path}: file changed while being read")
    return contents


def _object_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_file(path, MAX_JSON_BYTES).decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise InputError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"{path}: expected a JSON object")
    # Also rejects overflow such as 1e999, which JSON parsing converts to inf.
    _json(value)
    return value


def _nonempty(value: Any, field: str) -> str:
    if (
        not isinstance(value, str) or not value.strip()
        or value != value.strip() or any(ord(char) < 32 for char in value)
        or "\x7f" in value
    ):
        raise InputError(f"{field} must be nonempty, without control or edge whitespace")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise InputError(f"{field} must be valid UTF-8 text") from exc
    return value


def load_expected(path: Path) -> list[ExpectedArtifact]:
    """Validate explicit identities and paths; never infer identities from names."""
    manifest = _object_file(path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise InputError("expected-output manifest must have schema_version: 1")
    if set(manifest) != {"schema_version", "artifacts"}:
        raise InputError("manifest fields must be schema_version and artifacts")
    items = manifest.get("artifacts")
    if not isinstance(items, list):
        raise InputError("expected-output manifest needs an artifacts list")
    identities: set[str] = set()
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "artifact_identity", "fingerprint_path"
        }:
            raise InputError(
                "each expected artifact needs exactly artifact_identity and fingerprint_path"
            )
        identity = _nonempty(item.get("artifact_identity"), "artifact_identity")
        relative = _nonempty(item.get("fingerprint_path"), "fingerprint_path")
        # This local format uses canonical POSIX relative paths on all hosts.
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or "\\" in relative
            or ":" in relative
            or not relative.endswith(".fprint")
        ):
            raise InputError(f"unsafe or non-.fprint relative path: {relative!r}")
        if identity in identities:
            raise InputError(f"duplicate artifact identity: {identity}")
        identities.add(identity)
        result.append({"artifact_identity": identity, "fingerprint_path": relative})
    return result


@traced("FINGERPRINT_READ")
def read_fingerprints(
    expected: list[ExpectedArtifact], root: Path, configure_state: ConfigureState
) -> list[FingerprintRecord]:
    root = Path(root).expanduser().resolve()
    result = []
    for item in expected:
        path = root / item["fingerprint_path"]
        record: FingerprintRecord = {
            "artifact_identity": item["artifact_identity"],
            "fingerprint_path": str(path),
            "fingerprint": None,
            "fingerprint_state": "invalid",
            "detail": None,
        }
        if configure_state != "succeeded":
            record["detail"] = (
                f"configure_state={configure_state}; existing fingerprints ignored"
            )
        else:
            try:
                # Reject paths resolving outside the selected local build root.
                # Configure must have finished; this is not a sandbox for an
                # untrusted process concurrently replacing directories/symlinks.
                resolved = path.resolve()
                resolved.relative_to(root)
                raw = _read_regular_file(resolved, MAX_FINGERPRINT_BYTES)
                fingerprint = raw.decode("ascii").strip()
                if not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
                    raise InputError("expected exactly one SHA-256 hexadecimal value")
                record["fingerprint"] = fingerprint.lower()
                record["fingerprint_state"] = "valid"
            except FileNotFoundError:
                record["fingerprint_state"] = "missing"
                record["detail"] = "expected fingerprint file is missing"
            except (OSError, ValueError, RuntimeError) as exc:
                record["detail"] = str(exc)
        trace("FINGERPRINT", artifact=record["artifact_identity"], path=path,
              state=record["fingerprint_state"], fingerprint=record["fingerprint"],
              reason=record["detail"])
        result.append(record)
    return result


@traced("COLLECT")
def collect(
    *,
    database: Path,
    expected_manifest: Path,
    artifacts_dir: Path,
    label: str,
    commit: str,
    configure_state: ConfigureState,
    parameters: dict[str, Any] | None = None,
    source_revisions: dict[str, Any] | None = None,
    overlay_identity: str | None = None,
    database_timeout: float = DEFAULT_DATABASE_TIMEOUT,
    epoch: int | None = None,
) -> dict[str, Any]:
    """Record one configured build run in an existing database, atomically.

    lookup_ms includes connect, write-lock acquisition, schema validation, one
    batch lookup, and row decoding. Persistence timing includes commit.
    Source preparation, configure, restore and full preflight are not measured;
    their database fields remain NULL.

    Raises InputError for invalid arguments, SchemaError for an incompatible
    database, OSError for inaccessible inputs, or sqlite3.Error on SQL failures.
    All database mutations roll back if selection or persistence fails.
    """
    started = time.perf_counter()
    epoch = current_cache_epoch() if epoch is None else validate_cache_epoch(epoch)
    for field, value in (("label", label), ("commit", commit)):
        _nonempty(value, field)
    if configure_state not in ("succeeded", "failed", "not_run"):
        raise InputError("configure_state must be succeeded, failed, or not_run")
    if overlay_identity is not None:
        _nonempty(overlay_identity, "overlay_identity")
    if (
        isinstance(database_timeout, bool)
        or not isinstance(database_timeout, (int, float))
        or not math.isfinite(database_timeout)
        or not 0 <= database_timeout <= 60
    ):
        raise InputError("database_timeout must be a finite number between 0 and 60 seconds")
    parameters = {} if parameters is None else parameters
    source_revisions = {} if source_revisions is None else source_revisions
    if not isinstance(parameters, dict) or not isinstance(source_revisions, dict):
        raise InputError("parameters and source_revisions must be JSON objects")
    _json(parameters)
    _json(source_revisions)

    collect_started = time.perf_counter()
    expected = load_expected(expected_manifest.expanduser())
    root = artifacts_dir.expanduser().resolve()
    if configure_state == "succeeded" and not root.is_dir():
        raise InputError(f"artifacts directory does not exist: {root}")
    records: list[dict[str, Any]] = [
        dict(record) for record in read_fingerprints(expected, root, configure_state)
    ]
    collect_ms = (time.perf_counter() - collect_started) * 1000

    lookup_started = time.perf_counter()
    with CacheDB(database, timeout=database_timeout) as db:
        with db.transaction():
            # One short transaction keeps selection and recorded observations
            # consistent. No source scanning or payload hashing holds this lock.
            rows = db.lookup_observations(records, epoch=epoch)
            lookup_ms = (time.perf_counter() - lookup_started) * 1000
            plan_started = time.perf_counter()
            counts = {"hit": 0, "miss": 0, "uncacheable": 0}
            seen_count = 0
            for row, record in zip(rows, records):
                seen = row["artifact_key_id"] is not None
                if record["fingerprint_state"] != "valid":
                    status = "uncacheable"
                else:
                    status = "hit" if row["cache_entry_id"] is not None else "miss"
                counts[status] += 1
                seen_count += int(seen)
                record.update(
                    fingerprint_seen_before=seen,
                    lookup_result=status,
                    selected_entry_id=row["cache_entry_id"],
                    local_directory=row["local_directory"],
                    next_step=(
                        "fix_configuration" if configure_state != "succeeded"
                        else "verify_before_restore" if status == "hit"
                        else "build"
                    ),
                )
            plan_ms = (time.perf_counter() - plan_started) * 1000
            persist_started = time.perf_counter()
            run_id = db.record_observations(
                records, label=label, commit=commit, epoch=epoch,
                configure_state=configure_state, parameters=parameters,
                source_revisions=source_revisions, overlay_identity=overlay_identity,
                collect_ms=collect_ms, lookup_ms=lookup_ms, plan_ms=plan_ms,
            )
        persistence_ms = (time.perf_counter() - persist_started) * 1000

    return {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "label": label,
        "repository_commit": commit,
        "epoch": epoch,
        "configure_state": configure_state,
        "lookup_scope": "database_metadata",
        "payloads_verified": False,
        "summary": {
            "expected": len(records), "fingerprints_seen_before": seen_count, **counts
        },
        "timings_ms": {
            "fingerprint_collect": collect_ms,
            "lookup": lookup_ms,
            "plan": plan_ms,
            "persistence": persistence_ms,
            "collector_total": (time.perf_counter() - started) * 1000,
            "source_prepare": None,
            "cmake_configure": None,
            "preflight": None,
            "restore": None,
        },
        "artifacts": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="artifact-cache", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser(
        "collect", help="record fingerprints and report database candidates",
        allow_abbrev=False,
    )
    add_logging_args(scan)
    scan.add_argument("--db", type=Path, required=True, help="existing revision-6 cache.db")
    scan.add_argument("--expected", type=Path, required=True, help="explicit expected-output JSON")
    scan.add_argument("--artifacts-dir", type=Path, required=True)
    scan.add_argument("--label", required=True)
    scan.add_argument(
        "--commit", required=True,
        help="actual configured commit; provenance, not a cache key",
    )
    scan.add_argument(
        "--configure-state", required=True, choices=("succeeded", "failed", "not_run")
    )
    scan.add_argument(
        "--parameters", type=Path,
        help="JSON object recording flags/tools/environment (diagnostic only)",
    )
    scan.add_argument(
        "--source-revisions", type=Path,
        help="JSON object recording submodule/source revisions",
    )
    scan.add_argument("--overlay-identity", help="optional external repository overlay provenance")
    scan.add_argument(
        "--database-timeout", type=float, default=DEFAULT_DATABASE_TIMEOUT,
        help="maximum SQLite lock wait in seconds, from 0 to 60 (default: 5)",
    )
    for name, help_text in (
        ("build-publish", "run configure/build and publish verified local exports"),
        ("restore", "reconfigure and restore verified exports into a new directory"),
    ):
        operation = subparsers.add_parser(name, help=help_text, allow_abbrev=False)
        add_logging_args(operation)
        operation.add_argument("--db", type=Path, required=True)
        operation.add_argument("--expected", type=Path, required=True)
        operation.add_argument("--build-dir", type=Path, required=True)
        operation.add_argument("--cache-root", type=Path, required=True)
        if name == "build-publish":
            operation.add_argument("--label", required=True)
            operation.add_argument("--jobs", type=int)
        else:
            operation.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_logged(args, _execute_logged)


def _execute_logged(args):
    if args.command != "collect":
        return _local_operation(args)
    try:
        report = collect(
            database=args.db, expected_manifest=args.expected,
            artifacts_dir=args.artifacts_dir, label=args.label, commit=args.commit,
            configure_state=args.configure_state,
            parameters=_object_file(args.parameters.expanduser()) if args.parameters else None,
            source_revisions=(
                _object_file(args.source_revisions.expanduser())
                if args.source_revisions else None
            ),
            overlay_identity=args.overlay_identity,
            database_timeout=args.database_timeout,
        )
    except KeyboardInterrupt:
        print("artifact-cache: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        trace("OPERATION_FAILED", error_type=type(exc).__name__, reason=str(exc))
        print(f"artifact-cache: {exc}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        sys.stdout.flush()
    except OSError as exc:
        print(
            f"artifact-cache: run_id={report['run_id']} committed, "
            f"but report output failed: {exc}", file=sys.stderr,
        )
        if isinstance(exc, BrokenPipeError):
            # Avoid a second BrokenPipeError during interpreter shutdown.
            try:
                with open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), sys.stdout.fileno())
            except (OSError, ValueError):
                pass
        return 3
    return 0 if args.configure_state == "succeeded" else 1


def _local_operation(args: argparse.Namespace) -> int:
    # Keep normal collection imports independent of publication/restore/CI code.
    import importlib
    import subprocess
    import tarfile
    prefix = __package__ or "build_tools.github_actions.artifact_cache"
    try:
        common = dict(
            database=args.db, expected_manifest=args.expected, build_dir=args.build_dir,
            cache_root=args.cache_root,
        )
        if args.command == "build-publish":
            publisher = importlib.import_module(prefix + ".publish")
            report = publisher.build_and_publish(**common, label=args.label, jobs=args.jobs)
        else:
            restorer = importlib.import_module(prefix + ".restore")
            report = restorer.restore_local_outputs(**common, destination=args.destination)
    except KeyboardInterrupt:
        print("artifact-cache: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.Error,
            subprocess.CalledProcessError, tarfile.TarError) as exc:
        trace("OPERATION_FAILED", error_type=type(exc).__name__, reason=str(exc))
        print(f"artifact-cache: {exc}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        sys.stdout.flush()
        importlib.import_module(prefix + ".reporting").log_report(report)
    except OSError as exc:
        print(f"artifact-cache: {args.command} completed but reporting failed; "
              f"run_id={report.get('run_id')}, destination={report.get('destination')}: {exc}",
              file=sys.stderr)
        return 3
    return 1 if report.get("needs_build", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
