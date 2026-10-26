#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local cache with the existing relational DB and managed build publication.

cache: configure, verify existing productions, build and publish missing keys.
keys: configure and inspect current keys. restore: use the package restore CLI.
A verified cache object does not automatically skip a CMake target or CI job.
"""

import argparse

from .diagnostics import add_logging_args, run_logged, trace
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile

from .cache_db import CacheDB, ReadPolicy
from .cache_key import current_cache_epoch
from .inventory import configured_build, scan_build, write_expected
from .manifest import TrustLevel, ValidationState
from .reporting import log_build_progress, log_report


@contextmanager
def prepared_scan(build, expected=None, variant=None, *, show_configure=False, granularity="component"):
    from .publish import _build_lock, _run, _source_revisions
    from .local_store import _object_record
    build, source, cmake = configured_build(build, variant)
    with _build_lock(build):
        revisions = _source_revisions(source)
        _run([cmake, "-S", str(source), "-B", str(build)],
             configure_log_dir=None if show_configure else build / "logs" / "artifact-cache")
        configured_build(build, variant)
        config = _object_record(build / "CMakeCache.txt")
        records = scan_build(build, expected, granularity=granularity)
        expected_state = _object_record(Path(expected)) if expected else None
        trace("CONFIGURED_SCAN", build=build, source=source, records=len(records))
        if revisions != _source_revisions(source):
            raise ValueError("source changed during configuration")
        yield records
        trace("SCAN_RECHECK_START", build=build)
        if (revisions != _source_revisions(source)
                or config != _object_record(build / "CMakeCache.txt")
                or records != scan_build(build, expected, granularity=granularity)
                or (expected and expected_state != _object_record(Path(expected)))):
            raise ValueError("inputs changed during cache inspection; retry")
        trace("SCAN_RECHECK_DONE", build=build)


def cache_build(*, database, build, cache,
                expected=None, variant=None, jobs=None, policy=None, progress=None,
                show_configure=False, granularity="component"):
    from .publish import build_and_publish
    from .restore import restore_local_key
    from .bundles import component_requests
    epoch = current_cache_epoch()
    notify = progress if progress is not None else lambda *args, **kwargs: None
    cache = Path(cache).expanduser().resolve()
    results, missing = [], []
    with CacheDB(database) as db, tempfile.TemporaryDirectory(prefix="therock-cache-check-") as temp:
        temporary = Path(temp)
        notify("SCANNING", build=str(build), epoch=epoch)
        with prepared_scan(build, expected, variant,
                           show_configure=show_configure, granularity=granularity) as records:
            if not records:
                raise ValueError("no artifact outputs discovered; supply --expected for a cold build")
            initial = {r["artifact_identity"]: key for r, key in records}
            # One metadata lookup classifies the entire request; byte checks follow.
            notify("VERIFYING_CACHE")
            candidates = db.lookup_batch([key for _, key in records if key], policy, cache_root=cache)
            for record, key in records:
                identity = record["artifact_identity"]
                if key is None:
                    results.append({"artifact_identity": identity, "outcome": "uncacheable",
                                    "components": record.get("components", []),
                                    "fingerprint_state": record["fingerprint_state"],
                                    "reason": record.get("detail") or "missing or invalid fingerprint"})
                    trace("CACHE_SKIPPED", artifact=identity, reason=results[-1]["reason"])
                    notify("UNCACHEABLE", identity, reason=results[-1]["reason"])
                    continue
                candidate = candidates.get(key)
                trace("CACHE_DECISION", artifact=identity, epoch=key.epoch,
                      result="candidate" if candidate else "miss",
                      entry=candidate.cache_entry_id if candidate else None,
                      payloads_verified=False)
                if candidate:
                    notify("CHECKING", identity, epoch=key.epoch, entry=candidate.cache_entry_id)
                result = (restore_local_key(db, key, cache, temporary / identity, policy=policy)
                          if candidate else {"outcome": "needs_build"})
                if result["outcome"] == "restored":
                    trace("CACHE_VERIFIED", artifact=identity, entry=result["cache_entry_id"],
                          use="temporary verification; not installed into build")
                    results.append({**result, "outcome": "verified_cached"})
                    notify("VERIFIED_CACHED", identity, epoch=key.epoch,
                           entry=result["cache_entry_id"], payloads_verified=True)
                else:
                    missing.extend(component_requests(record))
                    notify("MISS", identity, epoch=key.epoch,
                           reason=result.get("reason") or "no eligible cached output")
        trace("BUILD_DECISION", cacheable_misses=len(missing),
              verified_cached=sum(r["outcome"] == "verified_cached" for r in results),
              cache_skipped=sum(r["outcome"] == "uncacheable" for r in results),
              action="managed_build" if missing else "no_build")
        publication = None
        if missing:
            manifest = temporary / "missing.json"
            write_expected(manifest, missing)
            publication = build_and_publish(
                database=Path(database), expected_manifest=manifest, build_dir=Path(build),
                cache_root=cache, label="local-cache", jobs=jobs,
                progress=progress, granularity=granularity)
            # The managed publisher has its own source/key checks. Verify that
            # any hits decided before that build still describe these inputs.
            notify("RECHECKING", reason="checking inputs after publication")
            with prepared_scan(build, expected, variant, granularity=granularity) as current:
                current_keys = {r["artifact_identity"]: key for r, key in current}
                if (any(current_keys.get(name) != key for name, key in initial.items())
                        or any(key is not None and name not in initial
                               for name, key in current_keys.items())):
                    raise ValueError("configured output keys changed after planning; retry cache command")
                trace("FINAL_KEYS_VERIFIED", outputs=len(current_keys))
                for record, key in current:
                    if key is None and record["artifact_identity"] not in initial:
                        results.append({"artifact_identity": record["artifact_identity"],
                                        "outcome": "uncacheable", "components": record.get("components", []),
                                        "fingerprint_state": record["fingerprint_state"], "reason": record.get("detail")
                                        or "missing or invalid fingerprint"})
                        notify("UNCACHEABLE", record["artifact_identity"], reason=results[-1]["reason"])
            for published in publication["published"]:
                results.append({"artifact_identity": published["artifact_identity"],
                                "outcome": "stored", "components": published.get("components", []),
                                "cache_entry_id": published["cache_entry_id"]})
        else:
            notify("NO_BUILD")
        return {"report_version": 2, "operation": "cache", "granularity": granularity, "build_executed": publication is not None,
                "epoch": epoch,
                "build_skipped_by_cache": False,
                "inventory_scope": ("explicit_outputs" if expected else
                                    "configured_ninja_and_observed_outputs" if granularity == "bundle"
                                    else "observed_outputs"),
                "summary": {status: sum(r["outcome"] == status for r in results)
                            for status in ("verified_cached", "stored", "uncacheable")},
                "artifacts": sorted(results, key=lambda r: r["artifact_identity"]),
                "run_id": publication["run_id"] if publication else None}


def synth(out, change=None):
    """Create a tiny, isolated real CMake build fixture, never alter a real build.

    Its private Git repo supplies clean source provenance. Changes are committed
    only there so managed publication exercises the same guards as a real build.
    """
    from .publish import _run
    out = Path(out).expanduser().resolve()
    source = out / ".cache-fixture-source"
    marker = source / "fixture-version"
    if out.exists() and not marker.is_file():
        raise ValueError("synth requires a new directory or its own existing fixture")
    names = {"amd-llvm": "amd-llvm_lib_generic", "base": "base_lib_generic",
             "blas": "blas_lib_gfx94X-dcgpu", "core-hip": "core-hip_lib_generic",
             "fft": "fft_lib_gfx94X-dcgpu"}
    if change is not None and change not in names:
        raise ValueError("unknown fixture artifact: " + change)
    fresh = not marker.exists()
    if not fresh:
        if marker.read_text() != "1\n":
            raise ValueError("unknown fixture version")
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
        if status:
            raise ValueError("fixture source has edits; preserve or remove them before synth")
    else:
        source.mkdir(parents=True)
        marker.write_text("1\n")
        (source / "units.json").write_text(json.dumps(names))
        for name in names:
            (source / (name + ".txt")).write_text(name + " version 1\n")
        (source / "export.py").write_text('''from pathlib import Path
import json, shutil, sys
source, build = map(Path, sys.argv[1:])
for name, identity in json.loads((source / "units.json").read_text()).items():
    dest = build / "artifacts" / identity
    (dest / "stage").mkdir(parents=True, exist_ok=True)
    (dest / "artifact_manifest.txt").write_text("stage\\n")
    shutil.copyfile(source / (name + ".txt"), dest / "stage" / "payload.txt")
(build / "artifacts" / "miopen_lib_gfx94X-dcgpu").mkdir(exist_ok=True)
''')
        lines = ["cmake_minimum_required(VERSION 3.25)", "project(cache_fixture NONE)",
                 'file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts")']
        for name, identity in names.items():
            lines += [f'file(SHA256 "${{CMAKE_SOURCE_DIR}}/{name}.txt" fp)',
                      f'file(WRITE "${{CMAKE_BINARY_DIR}}/artifacts/{identity}.fprint" "${{fp}}")']
        byproducts = " ".join('"${CMAKE_BINARY_DIR}/artifacts/' + identity + '/artifact_manifest.txt"'
                              for identity in names.values())
        lines += ['find_package(Python3 COMPONENTS Interpreter REQUIRED)',
                  'add_custom_target(exports ALL COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/export.py" "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" BYPRODUCTS ' + byproducts + ' VERBATIM)']
        (source / "CMakeLists.txt").write_text("\n".join(lines) + "\n")
        _run(["git", "init", str(source)])
    if change:
        with (source / (change + ".txt")).open("a") as stream:
            stream.write("changed\n")
    if fresh or change:
        _run(["git", "-C", str(source), "add", "."])
        _run(["git", "-C", str(source), "-c", "user.name=Cache fixture", "-c",
              "user.email=fixture@localhost", "-c", "commit.gpgsign=false", "commit", "-m", "Update cache fixture"])
    _run(["cmake", "-S", str(source), "-B", str(out), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
          "-DPython3_EXECUTABLE=" + sys.executable])
    return {"operation": "synth", "build": str(out), "fixture_only": True, "build_executed": False}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("cache", "keys"):
        s = sub.add_parser(name)
        add_logging_args(s)
        s.add_argument("--build", type=Path, required=True)
        s.add_argument("--expected", type=Path)
        s.add_argument("--granularity", choices=["bundle", "component"], default="bundle",
                       help="bundle groups components of each artifact and target (default)")
        s.add_argument("--variant", help="must match configured CMAKE_BUILD_TYPE")
        if name == "cache":
            s.add_argument("--db", type=Path, required=True)
            s.add_argument("--cache", type=Path, required=True)
            s.add_argument("--jobs", type=int)
            s.add_argument("--min-trust", choices=[v.value for v in TrustLevel], default="untrusted")
            s.add_argument("--min-validation", choices=[v.value for v in ValidationState], default="build_passed")
    s = sub.add_parser("stats")
    add_logging_args(s)
    s.add_argument("--db", type=Path, required=True)
    s = sub.add_parser("synth")
    add_logging_args(s)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--change")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_logged(args, _execute_logged)


def _execute_logged(args):
    try:
        if args.command == "cache":
            report = cache_build(database=args.db, build=args.build, cache=args.cache,
                                 expected=args.expected, variant=args.variant,
                                 jobs=args.jobs, policy=ReadPolicy(min_trust=TrustLevel(args.min_trust),
                                 min_validation=ValidationState(args.min_validation)),
                                 progress=log_build_progress, show_configure=True, granularity=args.granularity)
        elif args.command == "keys":
            with prepared_scan(args.build, args.expected, args.variant, granularity=args.granularity) as records:
                report = {"operation": "keys", "granularity": args.granularity, "epoch": current_cache_epoch(),
                          "artifacts": [r for r, _ in records]}
        elif args.command == "stats":
            with CacheDB(args.db) as db:
                report = db.stats()
        else:
            report = synth(args.out, args.change)
        print(json.dumps(report, indent=2))
        log_report(report)
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
        trace("OPERATION_FAILED", error_type=type(exc).__name__, reason=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
