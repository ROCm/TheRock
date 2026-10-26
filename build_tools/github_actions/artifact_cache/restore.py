# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify and restore relational cache productions into new directories.

restore_local_outputs owns reconfiguration and input checks. restore_local_key
checks immutable object digests and the complete archive inventory, retiring
bad productions and trying older eligible candidates. restore_units accepts a
metadata plan but still verifies every candidate. None of these writes CMake
prebuilt markers or suppresses test selection.

The retained in-memory ObjectStore helper is for isolated model checks only;
it is not the database restoration path and never downloads/extracts bytes.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from .manifest import Manifest
from .diagnostics import trace, traced, ByteProgress


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
        if verify_digests and obj.digest is None:
            return RestoreResult(key, artifact, "needs_build", f"missing recorded digest: {obj.key}")
        if verify_digests and obj.digest is not None:
            actual = store.fetch_digest(obj.uri)
            if actual != obj.digest:
                return RestoreResult(
                    key, artifact, "needs_build", f"digest mismatch: {obj.key}"
                )
        verified.append(obj.uri)
    return RestoreResult(
        key, artifact, "reusable", "object-store checks passed; no files restored", tuple(verified)
    )


def restore_units(entries, *, db, cache_root, destination, policy=None):
    """Materialize a plan into a new directory; caller owns current-input checks.

    The high-level restore_local_outputs API also reconfigures and checks source
    state. No metadata candidate alone becomes a hit or skips a CMake target.
    """
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise ValueError("restore destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = []
    from .local_store import safe_relative
    with tempfile.TemporaryDirectory(prefix=".planned-restore-", dir=destination.parent) as temp:
        staging = Path(temp) / "exports"
        staging.mkdir()
        for entry in entries:
            identity = safe_relative(entry.unit.artifact_identity)
            if "/" in identity and (entry.status == "candidate" and
                                       not (entry.unit.cache_key and entry.unit.cache_key.components)):
                raise ValueError("restore requires component or explicit bundle identities")
            if entry.status != "candidate" or entry.unit.cache_key is None:
                results.append({"artifact_identity": identity, "outcome": "needs_build",
                                "components": list(entry.unit.components), "reason": entry.reason})
            else:
                results.append(_restore_requested(db, entry.unit.cache_key, Path(cache_root),
                                                  staging, policy=policy))
        if any(r["outcome"] == "restored" for r in results):
            if os.path.lexists(destination):
                raise ValueError("restore destination appeared during extraction")
            staging.rename(destination)
    return results


def _restore_requested(db, key, cache, staging, *, policy=None):
    """Keep bundle verification atomic; materialize original component names."""
    if not key.components:
        return restore_local_key(db, key, cache, staging / key.artifact_identity, policy=policy)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=staging.parent) as temp:
        export = Path(temp) / "export"
        result = restore_local_key(db, key, cache, export, policy=policy)
        if result["outcome"] == "restored":
            if any(os.path.lexists(staging / name) for name in key.components):
                raise ValueError("overlapping requested bundle components")
            for name in key.components:
                (export / name).rename(staging / name)
        return result


class CandidateError(ValueError):
    """A particular cache production is physically missing or invalid."""


@traced("OBJECT_VERIFY")
def _copy_verified_objects(manifest, cache_root: Path, temporary: Path) -> None:
    from .local_store import LOCAL_OBJECTS, regular_reader, _object_record
    source = Path(manifest.local_directory)
    root = cache_root.resolve(strict=True)
    # A wrong --cache-root is a caller error, not a reason to retire an entry.
    if not source.is_absolute() or not source.is_relative_to(root) or source == root:
        raise RuntimeError(f"cache entry is outside the supplied cache root: {source}")
    for part in source.relative_to(root).parts:
        root = root / part
        if root.is_symlink():
            raise CandidateError(f"symlink in stored object directory: {root}")
    if sorted(manifest.expected_files) != list(LOCAL_OBJECTS):
        raise CandidateError("unsupported local archive/inventory contract")
    for item in manifest.files:
        path = source / item.relative_path
        trace("OBJECT_CHECK", entry=manifest.cache_entry_id, path=path,
              expected_size=item.size_bytes, expected_sha256=item.content_sha256)
        progress = ByteProgress("OBJECT_PROGRESS", total_bytes=item.size_bytes,
                                entry=manifest.cache_entry_id, path=path)
        digest = hashlib.sha256()
        size = 0
        # Writes are outside the source-read exception handler: ENOSPC or a
        # destination permission error must not retire a good cached object.
        destination = temporary / item.relative_path
        with destination.open("xb") as writer:
            try:
                with regular_reader(path) as reader:
                    while True:
                        try:
                            block = reader.read(1 << 20)
                        except OSError as exc:
                            raise CandidateError(f"cannot read cached object: {path}") from exc
                        if not block:
                            break
                        writer.write(block)
                        digest.update(block)
                        size += len(block)
                        progress.advance(len(block))
            except (FileNotFoundError, ValueError) as exc:
                raise CandidateError(f"missing or unstable cached object: {path}") from exc
        if size != item.size_bytes or digest.hexdigest() != item.content_sha256:
            raise CandidateError(f"cached object size/digest mismatch: {path}")
        if _object_record(destination) != item:
            raise RuntimeError(f"copied object failed verification: {destination}")
        trace("OBJECT_VERIFIED", entry=manifest.cache_entry_id, path=path,
              size_bytes=size, sha256=digest.hexdigest())


@traced("VERIFY_CANDIDATES")
def restore_local_key(db, key, cache_root: Path, destination: Path, *, policy=None) -> dict:
    """Try eligible productions in order, retiring only physically bad ones.

    destination must not exist. The caller may keep it inside a private staging
    directory until all requested outputs are processed. No prebuilt markers
    are written and no CMake targets are skipped by this function.
    """
    from .local_store import unpack_component
    if os.path.lexists(destination):
        raise ValueError(f"restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rejected = []
    while True:
        manifest = db.lookup(key, policy, cache_root=cache_root)
        if manifest is None:
            trace("CANDIDATES_EXHAUSTED", artifact=key.artifact_identity, rejected=len(rejected))
            return {"artifact_identity": key.artifact_identity, "epoch": key.epoch,
                    "outcome": "needs_build", "components": list(key.components),
                    "reason": "no verified cache production", "rejected_entries": rejected}
        trace("CANDIDATE_CHECK", artifact=key.artifact_identity, entry=manifest.cache_entry_id,
              directory=manifest.local_directory, epoch=key.epoch, attempt=len(rejected) + 1)
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination.parent) as temp:
            temporary = Path(temp)
            objects, extracted = temporary / "objects", temporary / "export"
            objects.mkdir()
            extracted.mkdir()
            try:
                _copy_verified_objects(manifest, cache_root, objects)
                try:
                    unpack_component(objects, extracted, components=key.components)
                except (ValueError, EOFError, tarfile.TarError) as exc:
                    raise CandidateError(str(exc)) from exc
            except CandidateError as exc:
                trace("CANDIDATE_REJECTED", artifact=key.artifact_identity,
                      entry=manifest.cache_entry_id, reason=str(exc), action="invalidate_then_retry")
                db.invalidate_local_output(manifest.cache_entry_id)
                rejected.append({"cache_entry_id": manifest.cache_entry_id, "reason": str(exc)})
                continue
            current = db.get_manifest(manifest.cache_entry_id)
            if current is None or (policy is not None and not policy.allows(current)):
                # Retired/expired during verification: query another production.
                trace("CANDIDATE_CHANGED", artifact=key.artifact_identity,
                      entry=manifest.cache_entry_id, reason="no longer eligible", action="retry")
                continue
            if os.path.lexists(destination):
                raise ValueError(f"restore destination appeared during extraction: {destination}")
            extracted.rename(destination)
            trace("CANDIDATE_VERIFIED", artifact=key.artifact_identity, entry=manifest.cache_entry_id,
                  destination=destination, payloads_verified=True)
            return {"artifact_identity": key.artifact_identity, "epoch": key.epoch,
                    "outcome": "restored", "components": list(key.components),
                    "cache_entry_id": manifest.cache_entry_id, "payloads_verified": True,
                    "rejected_entries": rejected}


@traced("RESTORE_OUTPUTS")
def restore_local_outputs(
    *, database: Path, expected_manifest: Path, build_dir: Path,
    cache_root: Path, destination: Path, policy=None, granularity="component",
) -> dict:
    """Reconfigure current inputs, verify cache objects and restore exports.

    Materializes component directories in a new destination. This is a real
    disk restore, but integration with CMake's prebuilt/stage scheduler is a
    separate operation; a restored component alone cannot skip a whole stage.
    """
    if __package__.startswith("build_tools."):
        from . import collector
    else:
        from . import collector
    from .cache_db import CacheDB
    from .inventory import scan_build
    from .manifest import ArtifactKey
    from .local_store import _object_record, regular_reader, safe_relative
    from .publish import _build_lock, _source_revisions, _run

    build = Path(build_dir).expanduser().resolve(strict=True)
    cache = Path(cache_root).expanduser().resolve(strict=True)
    requested_destination = Path(destination).expanduser().absolute()
    if os.path.lexists(requested_destination):
        raise ValueError(f"restore destination must be new: {requested_destination}")
    destination = requested_destination.resolve()
    expected_manifest = Path(expected_manifest).expanduser().resolve(strict=True)
    expected_state = _object_record(expected_manifest)
    expected = collector.load_expected(expected_manifest)
    if not expected:
        raise ValueError("restoration requires at least one expected output")
    for item in expected:
        identity = safe_relative(item["artifact_identity"])
        if "/" in identity or item["fingerprint_path"] != identity + ".fprint":
            raise ValueError("restoration requires the emitted component/sidecar identity")
    with regular_reader(build / "CMakeCache.txt") as stream:
        lines = stream.read().decode("utf-8").splitlines()
    values = {}
    for line in lines:
        if line.startswith(("CMAKE_HOME_DIRECTORY:INTERNAL=", "CMAKE_COMMAND:INTERNAL=")):
            field, value = line.split("=", 1)
            values[field.split(":", 1)[0]] = value
    if set(values) != {"CMAKE_HOME_DIRECTORY", "CMAKE_COMMAND"}:
        raise ValueError("build directory lacks a configured CMake source/tool location")
    source = Path(values["CMAKE_HOME_DIRECTORY"]).resolve(strict=True)
    for tree in (source, build, cache):
        if destination == tree or destination in tree.parents or tree in destination.parents:
            raise ValueError("restore destination must be separate from source, build and cache trees")
    with CacheDB(database) as db, _build_lock(build):
        revisions = _source_revisions(source)
        _run([values["CMAKE_COMMAND"], "-S", str(source), "-B", str(build)])
        config_state = _object_record(build / "CMakeCache.txt")
        trace("RESTORE_REQUEST", expected_file=expected_manifest, outputs=len(expected),
              destination=destination)
        records = scan_build(build, expected_manifest, granularity=granularity)
        if revisions != _source_revisions(source):
            raise ValueError("source changed during restore configuration")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".artifact-restore-", dir=destination.parent) as temp:
            staging = Path(temp) / "exports"
            staging.mkdir()
            results = []
            for record, key in records:
                identity = record["artifact_identity"]
                if key is None:
                    trace("RESTORE_SKIPPED", artifact=identity, reason=record["detail"] or "invalid fingerprint")
                    results.append({"artifact_identity": identity, "outcome": "needs_build",
                                    "components": record.get("components", []), "reason": record["detail"] or "invalid fingerprint"})
                    continue
                results.append(_restore_requested(db, key, cache, staging, policy=policy))
            if (revisions != _source_revisions(source)
                    or records != scan_build(build, expected_manifest, granularity=granularity)
                    or config_state != _object_record(build / "CMakeCache.txt")
                    or expected_state != _object_record(expected_manifest)):
                raise ValueError("source/configuration/fingerprints changed during restoration")
            restored = sum(result["outcome"] == "restored" for result in results)
            if restored:
                if os.path.lexists(destination):
                    raise ValueError(f"restore destination appeared: {destination}")
                staging.rename(destination)
                trace("RESTORE_DESTINATION_READY", destination=destination, artifacts=restored)
            return {"report_version": 2, "operation": "restore", "granularity": granularity, "restored": restored,
                    "needs_build": len(results) - restored,
                    "destination": str(destination) if restored else None,
                    "build_skipped": False, "artifacts": results}
