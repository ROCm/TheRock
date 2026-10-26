# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish freshly built outputs to the cache (flowchart 3).

Write ordering is the whole point of this module:

    built outputs + fingerprint + provenance
        -> outputs complete?                      (no  -> do NOT publish)
        -> producer & validation policy satisfied? (no  -> do NOT publish)
        -> upload immutable objects FIRST
        -> publish manifest + fingerprint entry LAST

The managed driver owns configure/build and input checks. Immutable objects
are verified first; register_local_outputs records inventory and marks entries
ready in one transaction. Raw component keys preserve the existing DB contract.
"""


from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .cache_db import CacheDB
from .cache_key import current_cache_epoch
from .manifest import ArtifactKey, TrustLevel
from .diagnostics import trace, traced

def publish_build_outputs(**managed_build_options):
    """Public publication entry point: run the managed build before storing.

    Legacy lists of already-existing outputs and caller-supplied success flags
    are no longer publication evidence. See build_and_publish for arguments.
    """
    return build_and_publish(**managed_build_options)


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, encoding="utf-8"
    ).strip()


@traced("SOURCE_CHECK")
def _source_revisions(source: Path) -> dict[str, str]:
    """Record actual HEADs and reject tracked edits, including nested repos.

    The fingerprint contract ignores untracked files. This local wrapper uses
    that same contract; it does not claim to repair missing CMake key inputs.
    """
    result = {}
    def visit(directory):
        if Path(_git(directory, "rev-parse", "--show-toplevel")).resolve() != directory:
            raise ValueError(f"source must be a Git checkout root: {directory}")
        diff = subprocess.run(
            ["git", "-C", str(directory), "diff", "--quiet", "--ignore-submodules", "HEAD"],
            check=False,
        )
        if diff.returncode:
            raise ValueError(f"tracked source changes or Git error in {directory}; commit first")
        result[directory.relative_to(source).as_posix()] = _git(directory, "rev-parse", "HEAD")
        tracked = subprocess.check_output(
            ["git", "-C", str(directory), "ls-files", "--stage", "-z"]
        )
        for record in tracked.split(b"\0"):
            if record.startswith(b"160000 "):
                relative = os.fsdecode(record.split(b"\t", 1)[1])
                child = directory / relative
                if (child / ".git").exists():
                    visit(child.resolve())
    visit(source)
    trace("SOURCE_STATE", root=source, repositories=len(result), root_commit=result["."],
          tracked_files_clean=True, untracked_files_checked=False)
    return result


@contextmanager
def _build_lock(build: Path):
    """Exclude concurrent managed publishers; manual builds must also be stopped."""
    path = build / ".therock-artifact-cache.lock"
    trace("BUILD_LOCK_REQUEST", path=path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"cache build already locked: {path}; check for an active process") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        trace("BUILD_LOCK_ACQUIRED", path=path, pid=os.getpid())
        yield
    finally:
        path.unlink()
        trace("BUILD_LOCK_RELEASED", path=path)


@traced("COMMAND")
def _run(command, *, configure_log_dir=None):
    trace("COMMAND", argv=[str(arg) for arg in command],
          output="terminal_stderr" if configure_log_dir is None else "configure_log")
    # Keep stdout reserved for the CLI's JSON report, including build failures.
    if configure_log_dir is None:
        subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
        return

    # Retain every configure pass, including successful warnings, without
    # repeating CMake's full status output in the terminal. Build output streams
    # normally through the branch above.
    log_dir = Path(configure_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", errors="replace", dir=log_dir,
        prefix="configure-", suffix=".log", delete=False,
    ) as log:
        trace("CONFIGURE_LOG", path=log.name)
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except (OSError, subprocess.SubprocessError):
            print(f"CMake configuration failed. Full log: {log.name}", file=sys.stderr)
            log.flush()
            log.seek(0)
            shutil.copyfileobj(log, sys.stderr)
            raise


@traced("BUILD_PUBLISH")
def build_and_publish(
    *, database: Path, expected_manifest: Path, build_dir: Path,
    cache_root: Path, label: str, jobs: int | None = None,
    producer_trust: TrustLevel = TrustLevel.UNTRUSTED,
    progress=None, granularity="component",
) -> dict:
    """Own configure -> build -> input recheck -> package -> ready publication.

    No caller-supplied success flag or arbitrary producer run is accepted.
    The default CMake build must assemble the requested component exports.
    No prebuilt imports or concurrent manual source/build edits are supported.
    Like any build driver, correctness still relies on the build graph tracking
    its inputs and the CMake fingerprint covering the intended environment.
    """
    if __package__.startswith("build_tools."):
        from . import collector
    else:
        from . import collector
    from .local_store import regular_reader, safe_relative, store_component, store_bundle
    from .inventory import scan_build

    if not isinstance(producer_trust, TrustLevel):
        raise ValueError("producer_trust must be a TrustLevel")
    epoch = current_cache_epoch()
    notify = progress if progress is not None else lambda *args, **kwargs: None
    collector._nonempty(label, "label")
    if jobs is not None and (type(jobs) is not int or jobs < 1):
        raise ValueError("jobs must be a positive integer")
    expected_manifest = Path(expected_manifest).expanduser().resolve(strict=True)
    expected = collector.load_expected(expected_manifest)
    if not expected:
        raise ValueError("build publication requires at least one expected output")
    for item in expected:
        identity = safe_relative(item["artifact_identity"])
        if "/" in identity or item["fingerprint_path"] != identity + ".fprint":
            raise ValueError("publication requires the emitted component directory/sidecar identity")
    build = Path(build_dir).expanduser().resolve(strict=True)
    cache = Path(cache_root).expanduser().resolve()
    cache_file = build / "CMakeCache.txt"
    with regular_reader(cache_file) as stream:
        cache_text = stream.read().decode("utf-8")
    cache_values = {}
    for line in cache_text.splitlines():
        if line.startswith(("CMAKE_HOME_DIRECTORY:INTERNAL=", "CMAKE_COMMAND:INTERNAL=")):
            name, value = line.split("=", 1)
            cache_values[name.split(":", 1)[0]] = value
    if set(cache_values) != {"CMAKE_HOME_DIRECTORY", "CMAKE_COMMAND"}:
        raise ValueError("build directory lacks a configured CMake source/tool location")
    source = Path(cache_values["CMAKE_HOME_DIRECTORY"]).resolve(strict=True)
    for tree in (source, build):
        if cache == tree or cache in tree.parents or tree in cache.parents:
            raise ValueError("cache storage must be separate from the source and build trees")
    with CacheDB(database):
        pass  # Reject a missing/incompatible database before starting a build.

    def digest(path):
        value = hashlib.sha256()
        with regular_reader(path) as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                value.update(block)
        return value.hexdigest()

    expected_digest = digest(expected_manifest)
    artifacts = build / "artifacts"
    def fingerprints():
        records = collector.read_fingerprints(expected, artifacts, "succeeded")
        invalid = [r["artifact_identity"] for r in records if r["fingerprint_state"] != "valid"]
        if invalid:
            raise ValueError("missing/invalid fingerprints: " + ", ".join(invalid))
        return {r["artifact_identity"]: r["fingerprint"] for r in records}

    cmake = cache_values["CMAKE_COMMAND"]
    configure_command = [cmake, "-S", str(source), "-B", str(build)]
    configure_log_dir = build / "logs" / "artifact-cache"
    build_command = [cmake, "--build", str(build)]
    if jobs is not None:
        build_command.extend(["--parallel", str(jobs)])

    with _build_lock(build):
        if next(build.rglob("*.prebuilt"), None) is not None:
            raise ValueError("managed publication requires a source build without .prebuilt imports")
        source_before = _source_revisions(source)
        notify("CONFIGURING", build=str(build), epoch=epoch)
        _run(configure_command, configure_log_dir=configure_log_dir)
        initial = fingerprints()
        units = scan_build(build, expected_manifest, granularity=granularity)
        if any(key is None for _, key in units):
            raise ValueError("publication requires complete units with matching valid fingerprints")
        config_digest = digest(cache_file)
        started = time.perf_counter()
        notify("BUILD_START", build=str(build), reason="default incremental build")
        try:
            _run(build_command)
        except (OSError, subprocess.SubprocessError):
            notify("BUILD_FAILED", reason="outputs will not be published")
            raise
        build_ms = (time.perf_counter() - started) * 1000
        notify("BUILD_PASSED", reason="checking inputs before storing outputs")
        # A configure-only change cannot reach this point without an actual
        # successful build. Reconfigure refreshes source-derived fingerprints.
        _run(configure_command, configure_log_dir=configure_log_dir)

        @traced("PUBLICATION_INPUT_CHECK")
        def check_inputs():
            if (source_before != _source_revisions(source) or initial != fingerprints()
                    or expected_digest != digest(expected_manifest)
                    or config_digest != digest(cache_file)):
                raise ValueError("source/configuration/fingerprints changed during the build; not publishing")

        check_inputs()
        trace("BUILD_INPUTS_VERIFIED", build_ms=round(build_ms, 3), requested_outputs=len(expected))
        notify("SAVING_OUTPUTS")
        outputs = []
        for record, key in units:
            identity = key.artifact_identity
            notify("STORING", identity, epoch=epoch)
            stored = (store_bundle(artifacts, key, cache) if key.components
                      else store_component(artifacts / identity, cache))
            outputs.append((key, stored.directory, stored.files))
        check_inputs()
        trace("PRODUCER_RECORD_START", database=database, outputs=len(outputs))
        report = collector.collect(
            database=database, expected_manifest=expected_manifest, artifacts_dir=artifacts,
            label=label, commit=source_before["."],
            epoch=epoch,
            configure_state="succeeded", source_revisions=source_before,
            parameters={"cache_provenance": {"trust": producer_trust.value}, "managed_build": {
                "command": build_command, "configure_command": configure_command,
                "granularity": granularity,
                "returncode": 0, "duration_ms": build_ms, "inputs_stable": True,
                "build_directory": str(build),
            }},
        )
        trace("PRODUCER_RECORD_COMMITTED", run_id=report["run_id"],
              ready_entries_committed=False)
        check_inputs()
        with CacheDB(database) as db:
            manifests = db.register_local_outputs(producer_run_id=report["run_id"], outputs=outputs)
        # register_local_outputs commits the complete set before returning.
        for manifest in manifests:
            notify("STORED", manifest.key.artifact_identity, epoch=manifest.key.epoch,
                   entry=manifest.cache_entry_id)
        return {
            "report_version": 2, "operation": "build-publish", "granularity": granularity, "run_id": report["run_id"],
            "build_executed": True, "build_ms": build_ms,
            "published": [m.to_dict() for m in manifests],
        }
