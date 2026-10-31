# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Retain six fresh dependency archives and every RCCL resource sidecar.

Diagnostic-only post-build collection. No source/product changes or comparison
normalization. Original payload bytes/modes/mtimes are retained in a tar, verified
against a separate content manifest, then published locally only when complete.
"""

import json
from contextlib import closing
from pathlib import Path
import shutil
import stat
import time

import comm_stage_evidence as evidence
from _therock_utils.archive_util import open_archive_for_read

ARCHIVES = (
    "third-party/fmt/build/libfmt.a",
    "third-party/googletest/build/lib/libgmock.a",
    "third-party/googletest/build/lib/libgmock_main.a",
    "third-party/googletest/build/lib/libgtest.a",
    "third-party/googletest/build/lib/libgtest_main.a",
    "third-party/yaml-cpp/build/libyaml-cpp.a",
)


def _identity(path: Path) -> tuple:
    row = path.lstat()
    return (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _reserve(path: Path, required: int = 0) -> None:
    usage = shutil.disk_usage(path)
    floor = max(5 * 1024**3, (usage.total + 9) // 10)
    if usage.free < floor + required:
        raise RuntimeError("Auxiliary evidence collection disk reserve exhausted")


def retain(build: Path, output: Path, deadline: float) -> dict:
    """Capture exact bytes outside the measured build and preserve failures."""
    started = time.monotonic()
    build = build.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "dependency-archives-and-rccl-resources.tar.zst"
    partial_path = output / "dependency-archives-and-rccl-resources.partial.tar.zst"
    manifest_path = output / "dependency-archives-and-rccl-resources.jsonl"
    summary_path = output / "dependency-archives-and-rccl-resources-summary.json"
    if any(
        p.exists() for p in [archive_path, partial_path, manifest_path, summary_path]
    ):
        raise RuntimeError("Auxiliary evidence already exists; preserve it")
    summary = dict(
        complete=False, scope="Post-build exact input bytes; no timing gain claim"
    )
    evidence.save(summary_path, summary)
    try:
        paths = [build / relative for relative in ARCHIVES]
        device = build / "comm-libs/rccl/build/device_build"
        if device.resolve(strict=True) != device.absolute():
            raise RuntimeError("RCCL sidecar directory must not alias another tree")
        resources = sorted(device.rglob("*.resources.json"))
        if not resources:
            raise RuntimeError("No RCCL resource sidecars found")
        paths.extend(resources)
        initial = {}
        total_bytes = 0
        for path in paths:
            evidence.check_deadline(deadline)
            if path.resolve(strict=True) != path.absolute():
                raise RuntimeError("Auxiliary input aliases or escapes its build path")
            if not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError("Auxiliary input must be a regular file")
            name = path.relative_to(build).as_posix()
            if name in initial:
                raise RuntimeError("Duplicate auxiliary input")
            initial[name] = _identity(path)
            total_bytes += path.stat().st_size
        # Conservative storage bound for the original tar payload, headers and
        # codec overhead. The unchanged5GiB/10% floor is also checked live.
        _reserve(output, 2 * total_bytes + 4096 * len(paths) + 1024**2)
        records = []
        with manifest_path.open("x", encoding="utf-8") as manifest:
            # Explicit close also releases the owned codec on an exception;
            # TarFile.__exit__ can skip close() for an external file object.
            with closing(
                evidence.open_archive_for_write(partial_path, "zstd", 3)
            ) as archive:
                for index, path in enumerate(paths):
                    evidence.check_deadline(deadline)
                    if index % 128 == 0:
                        _reserve(output)
                    name = path.relative_to(build).as_posix()
                    if _identity(path) != initial[name]:
                        raise RuntimeError("Auxiliary input changed before capture")
                    row = dict(
                        path=name,
                        bytes=path.stat().st_size,
                        mode=stat.S_IMODE(path.stat().st_mode),
                        mtime_ns=path.stat().st_mtime_ns,
                        sha256=evidence.sha256(path, deadline),
                        kind=(
                            "dependency_archive"
                            if name in ARCHIVES
                            else "rccl_resource_sidecar"
                        ),
                    )
                    with path.open("rb") as source:
                        info = archive.gettarinfo(
                            name=str(path), arcname=name, fileobj=source
                        )
                        seconds, nanos = divmod(row["mtime_ns"], 10**9)
                        info.pax_headers["mtime"] = f"{seconds}.{nanos:09d}"
                        archive.addfile(info, source)
                    if _identity(path) != initial[name]:
                        raise RuntimeError("Auxiliary input changed during capture")
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
                    records.append(row)
        expected = {r["path"]: r for r in records}
        observed = set()
        import hashlib

        with closing(open_archive_for_read(partial_path)) as archive:
            for member in archive:
                evidence.check_deadline(deadline)
                if (
                    not member.isfile()
                    or member.name not in expected
                    or member.name in observed
                ):
                    raise RuntimeError("Unexpected/duplicate auxiliary archive member")
                row = expected[member.name]
                hashed = hashlib.sha256()
                with archive.extractfile(member) as source:
                    while block := source.read(1024**2):
                        evidence.check_deadline(deadline)
                        hashed.update(block)
                if (
                    member.size != row["bytes"]
                    or stat.S_IMODE(member.mode) != row["mode"]
                    or member.pax_headers.get("mtime")
                    != f"{row['mtime_ns'] // 10**9}.{row['mtime_ns'] % 10**9:09d}"
                    or hashed.hexdigest() != row["sha256"]
                ):
                    raise RuntimeError("Auxiliary archived payload/metadata mismatch")
                observed.add(member.name)
        if observed != set(expected):
            raise RuntimeError("Auxiliary archive incomplete")
        for path in paths:
            evidence.check_deadline(deadline)
            if _identity(path) != initial[path.relative_to(build).as_posix()]:
                raise RuntimeError(
                    "Auxiliary input changed before verification completed"
                )
        _reserve(output)
        partial_path.rename(archive_path)
        summary.update(
            complete=True,
            dependency_archives=len(ARCHIVES),
            rccl_resource_sidecars=len(resources),
            sidecar_scope=(
                "All existing regular *.resources.json under device_build. "
                "Reconcile expected/executed coverage separately with retained "
                "Ninja commands and resource aggregates."
            ),
            source_bytes=total_bytes,
            archive_bytes=archive_path.stat().st_size,
            archive_sha256=evidence.sha256(archive_path, deadline),
            manifest_sha256=evidence.sha256(manifest_path, deadline),
            verified_payload_members=len(observed),
            collector_wall_s=time.monotonic() - started,
        )
    except Exception as error:
        summary.update(error=repr(error), collector_wall_s=time.monotonic() - started)
        evidence.save(summary_path, summary)
        raise
    evidence.save(summary_path, summary)
    return summary
