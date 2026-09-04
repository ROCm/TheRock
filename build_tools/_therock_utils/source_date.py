# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolves the source timestamp stamped into reproducible archives.

Archives need a timestamp that is identical across two builds of the same
source, so that the same content produces the same archive hash (see
https://github.com/ROCm/TheRock/issues/4202), but that is not so old it makes
freshly installed SDK files look older than a downstream project's existing
build outputs. Extraction restores mtime, so an epoch timestamp would do exactly
that and suppress rebuilds.

This is the expensive half of that: it shells out to git several times, so
orchestrators call it once and pass the result down to the workers that write
archives, rather than every worker recomputing it. See `archive_util` for the
consuming side.
"""

from typing import NamedTuple
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

# The variable orchestrators use to hand the resolved value to workers.
# Deliberately not SOURCE_DATE_EPOCH: see EXPORT_WARNING below.
ENV_VAR = "THEROCK_SOURCE_DATE_EPOCH"

# The reproducible-builds standard name. Read as a deliberate user override, and
# only ever *set* behind an explicit opt-in.
STANDARD_ENV_VAR = "SOURCE_DATE_EPOCH"

EXPORT_WARNING = f"""\
Exporting {STANDARD_ENV_VAR} affects far more than TheRock's archives. It is a
cross-ecosystem convention that many tools honor automatically:

  * GCC and Clang rewrite __DATE__ and __TIME__ in every translation unit.
  * CPython switches .pyc files from timestamp invalidation to checked-hash
    invalidation, changing the bytes that ship inside wheels.
  * setuptools/wheel use it for zip entry timestamps.
  * dpkg-deb clamps tar entry mtimes to it (it never raises older ones).

Those are usually desirable for a reproducible build, but they are a much wider
blast radius than stamping archive metadata, so TheRock does not set it for you.
Use {ENV_VAR} for archives alone, or opt in explicitly when you want the
ecosystem-wide behavior too."""


class SubmoduleEntry(NamedTuple):
    """One line of `git submodule status`."""

    # ' ' matches the pin, '+' checked out at a different commit, '-' not
    # initialized, 'U' has merge conflicts.
    state: str
    sha: str
    path: Path

    @property
    def is_populated(self) -> bool:
        return self.state != "-"

    @property
    def differs_from_pin(self) -> bool:
        return self.state in ("+", "U")


def _git(*args: str, cwd: Path) -> str | None:
    """Runs git, returning raw stdout or None if it could not be run.

    Deliberately unstripped: `git submodule status` encodes state in the first
    character of each line, and a leading space is the "matches the pin" state.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # git is not installed.
        return None
    if result.returncode != 0:
        # Not a git checkout (e.g. building from an exported source archive).
        return None
    return result.stdout


def _head_timestamp(repo_dir: Path) -> int | None:
    output = _git("log", "-1", "--format=%ct", cwd=repo_dir)
    output = output.strip() if output else ""
    return int(output) if output else None


def submodule_entries(repo_dir: Path = THEROCK_DIR) -> list[SubmoduleEntry]:
    """Parses `git submodule status`.

    One git call yields the state, hash and path of every submodule. Note that
    this compares commits only -- a submodule whose working tree is dirty but
    whose commit still matches the pin reports ' ', not '+'.
    """
    output = _git("submodule", "status", cwd=repo_dir)
    if output is None:
        return []
    entries = []
    for line in output.splitlines():
        if not line:
            continue
        # "<state><40-char sha> <path>[ (<describe>)]". The describe suffix is
        # absent for submodules that are not initialized.
        state, sha, rest = line[0], line[1:41], line[42:]
        path = rest.rsplit(" (", 1)[0] if rest.endswith(")") else rest
        entries.append(SubmoduleEntry(state, sha, repo_dir / path))
    return entries


def _parse_porcelain_z(output: str, repo_dir: Path) -> list[Path]:
    """Paths from `git status --porcelain -z`.

    NUL-separated rather than line-based because paths may contain spaces or
    newlines, which the line format quotes and escapes. Each record is two
    status characters, a space, then the path; rename and copy records are
    followed by a second record holding the original path, which is skipped.
    """
    paths = []
    records = output.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            continue
        status, path = record[:2], record[3:]
        if "R" in status or "C" in status:
            index += 1  # the paired original path
        if path:
            paths.append(repo_dir / path)
    return paths


def dirty_paths(
    repo_dir: Path = THEROCK_DIR,
    entries: list[SubmoduleEntry] | None = None,
) -> list[Path]:
    """Every uncommitted path, in the superproject and in each submodule.

    Only genuinely uncommitted content counts. A plain `git status --porcelain`
    on the superproject is not usable here: it reports ` M <path>` for a
    submodule that merely sits at a different commit than its pin, which is
    committed work already accounted for by folding that submodule's commit time
    in. Counting it here would mask that logic.

    So the superproject is asked with `--ignore-submodules=all`, and each
    populated submodule is asked about its own working tree separately.
    """
    output = _git(
        "status", "--porcelain", "-z", "--ignore-submodules=all", cwd=repo_dir
    )
    paths = _parse_porcelain_z(output, repo_dir) if output else []

    if entries is None:
        entries = submodule_entries(repo_dir)
    for entry in entries:
        if not entry.is_populated:
            continue
        submodule_output = _git("status", "--porcelain", "-z", cwd=entry.path)
        if submodule_output:
            paths.extend(_parse_porcelain_z(submodule_output, entry.path))
    return paths


def is_worktree_dirty(
    repo_dir: Path = THEROCK_DIR,
    entries: list[SubmoduleEntry] | None = None,
) -> bool:
    """Whether anything is uncommitted, in the superproject or any submodule."""
    return bool(dirty_paths(repo_dir, entries))


def newest_dirty_mtime(
    repo_dir: Path = THEROCK_DIR,
    entries: list[SubmoduleEntry] | None = None,
) -> int | None:
    """Modification time of the most recently touched uncommitted file.

    Deliberately not the current time. The value ends up in archives and, when
    exported, in build inputs, so it must not move every time it is asked --
    otherwise an untouched dirty tree would produce a different answer on every
    configure and invalidate everything downstream.

    An untracked *directory* is reported by git as a single entry, and only its
    own mtime is read: that moves when entries are added or removed, but not
    when a file inside it is edited. Walking it could mean walking something
    very large, so this accepts the imprecision.
    """
    newest = None
    for path in dirty_paths(repo_dir, entries):
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            # Deleted, or otherwise not stat-able. It has no mtime to offer.
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return newest


# Where therock_manifest.json sits, relative to a directory of artifacts. The
# build installs it to share/therock; the first form is the exploded artifact
# layout, the second is what `artifact_manager fetch --flatten` produces.
MANIFEST_RELPATHS = (
    Path("base_lib_generic/base/aux-overlay/stage/share/therock/therock_manifest.json"),
    Path("share/therock/therock_manifest.json"),
)


def find_manifest(search_dir: Path) -> Path | None:
    """Locates therock_manifest.json under a directory of artifacts."""
    for relpath in MANIFEST_RELPATHS:
        candidate = search_dir / relpath
        if candidate.is_file():
            return candidate
    return None


def timestamp_from_manifest(search_dir: Path) -> int | None:
    """The source timestamp recorded when the artifacts were built.

    This is the only value that survives the jump between jobs. Packaging runs
    on a different runner from the build, and may be pointed at artifacts from
    an entirely different run, so re-deriving from the local checkout would
    describe the packaging checkout rather than the source that was built.

    Returns None when there is no manifest, or when it predates this field, or
    when it was produced from a tree with no git metadata -- all cases where the
    caller should fall back to deriving one.
    """
    manifest_path = find_manifest(search_dir)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    timestamp = manifest.get("source_date_epoch")
    return timestamp if isinstance(timestamp, int) else None


def _read_manifest(search_dir: Path) -> dict | None:
    manifest_path = find_manifest(search_dir)
    if manifest_path is None:
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def describe_source_drift(
    manifest_dir: Path,
    repo_dir: Path = THEROCK_DIR,
) -> list[str]:
    """Ways the artifacts being packaged disagree with this checkout.

    Packaging normally runs against artifacts from its own run, where these
    agree. They diverge when a job is pointed at another run's artifacts, or
    when the build came from a dirty tree -- in both cases the archives are
    stamped with something the local checkout cannot account for. Worth saying
    out loud rather than resolving silently.

    Returns human-readable reasons; empty means no disagreement was detectable.
    """
    manifest = _read_manifest(manifest_dir)
    if manifest is None:
        return [
            f"no therock_manifest.json under {manifest_dir}, so the source "
            "timestamp is derived from this checkout rather than from the "
            "artifacts being packaged"
        ]

    reasons = []

    if manifest.get("source_date_epoch") is None:
        reasons.append(
            "the manifest records no source_date_epoch (built without git "
            "metadata, or predates the field); deriving from this checkout"
        )
    if manifest.get("source_dirty"):
        reasons.append(
            "the artifacts were built from a tree with uncommitted changes, so "
            "they are not reproducible from their recorded commit"
        )

    manifest_commit = manifest.get("the_rock_commit")
    local_commit = _git("rev-parse", "HEAD", cwd=repo_dir)
    local_commit = local_commit.strip() if local_commit else None
    if manifest_commit and local_commit and manifest_commit != local_commit:
        reasons.append(
            f"the artifacts were built from {manifest_commit[:12]} but this "
            f"checkout is at {local_commit[:12]}"
        )
    if local_commit and is_worktree_dirty(repo_dir):
        reasons.append(
            "this checkout has uncommitted changes, which the artifacts cannot "
            "contain"
        )

    return reasons


def resolve(
    *,
    manifest_dir: Path | None = None,
    repo_dir: Path = THEROCK_DIR,
) -> int:
    """The timestamp to stamp into archives, preferring the built artifacts'.

    Falls back to deriving one from `repo_dir` when the artifacts carry none.
    """
    if manifest_dir is not None:
        from_manifest = timestamp_from_manifest(manifest_dir)
        if from_manifest is not None:
            return from_manifest
    return compute_source_date_epoch(repo_dir)


def resolve_checked(
    *,
    manifest_dir: Path | None = None,
    repo_dir: Path = THEROCK_DIR,
    fail_on_drift: bool = False,
    report=print,
) -> int:
    """`resolve()`, reporting any disagreement with the local checkout.

    Warns by default, because packaging another run's artifacts is a supported
    workflow. `fail_on_drift` turns it into an error for release builds, where
    it usually means the wrong artifacts are being packaged.
    """
    if manifest_dir is not None:
        reasons = describe_source_drift(manifest_dir, repo_dir)
        if reasons:
            if fail_on_drift:
                raise RuntimeError(
                    "Source state drift between the artifacts and this "
                    "checkout:\n  - " + "\n  - ".join(reasons)
                )
            for reason in reasons:
                report(f"  Warning: {reason}")
    return resolve(manifest_dir=manifest_dir, repo_dir=repo_dir)


def compute_source_date_epoch(repo_dir: Path = THEROCK_DIR) -> int:
    """Resolves the timestamp to stamp into archives built from `repo_dir`.

    Starts from the superproject's HEAD, which pins every submodule and so
    identifies the whole source state. A submodule commit must exist before the
    superproject can pin it, so on a clean tree the superproject's commit is
    already the newest and the rest of this is a no-op.

    Two cases make that understate reality, both normal in TheRock's development
    flow where subproject sources are edited in place:

    * A submodule checked out past its pin (state '+') carries commits the
      superproject's date does not reflect, so its HEAD time is folded in.
    * Uncommitted changes have no commit time at all, so the mtime of the most
      recently touched one is folded in -- not the current time, so that asking
      twice without touching anything gives the same answer.

    Everything is combined with max(), so a file touched to an old date cannot
    drag the result back before the commit it sits on.
    """
    timestamps = []

    head = _head_timestamp(repo_dir)
    if head is not None:
        timestamps.append(head)

    entries = submodule_entries(repo_dir)
    for entry in entries:
        if not entry.is_populated or not entry.differs_from_pin:
            continue
        submodule_head = _head_timestamp(entry.path)
        if submodule_head is not None:
            timestamps.append(submodule_head)

    dirty = newest_dirty_mtime(repo_dir, entries)
    if dirty is not None:
        timestamps.append(dirty)

    if not timestamps:
        # No git at all. Not reproducible, but a working build beats a
        # deterministically broken one.
        return int(time.time())
    return max(timestamps)


def child_env(
    *,
    export_standard_var: bool = False,
    manifest_dir: Path | None = None,
    fail_on_drift: bool = False,
    repo_dir: Path = THEROCK_DIR,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for workers that write archives.

    Sets `THEROCK_SOURCE_DATE_EPOCH`. Also sets `SOURCE_DATE_EPOCH` when
    `export_standard_var` is true -- see EXPORT_WARNING for what else that
    changes.
    """
    env = dict(os.environ if base_env is None else base_env)
    timestamp = str(
        resolve_checked(
            manifest_dir=manifest_dir,
            repo_dir=repo_dir,
            fail_on_drift=fail_on_drift,
        )
    )
    env[ENV_VAR] = timestamp
    if export_standard_var:
        env[STANDARD_ENV_VAR] = timestamp
    return env


def apply_to_environ(
    *,
    export_standard_var: bool = False,
    manifest_dir: Path | None = None,
    fail_on_drift: bool = False,
    repo_dir: Path = THEROCK_DIR,
) -> int:
    """Publishes the resolved timestamp into this process's environment.

    For orchestrators that write archives in-process rather than by spawning
    workers. Returns the resolved value. Does not overwrite an existing
    `SOURCE_DATE_EPOCH`, which outranks everything (see `archive_util`).
    """
    timestamp = resolve_checked(
        manifest_dir=manifest_dir,
        repo_dir=repo_dir,
        fail_on_drift=fail_on_drift,
    )
    os.environ[ENV_VAR] = str(timestamp)
    if export_standard_var and STANDARD_ENV_VAR not in os.environ:
        os.environ[STANDARD_ENV_VAR] = str(timestamp)
    return timestamp


def add_source_date_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the shared opt-in flag to an orchestrator's argument parser."""
    parser.add_argument(
        "--export-source-date-epoch",
        action="store_true",
        help=(
            f"Also export {STANDARD_ENV_VAR} to child processes, not just "
            f"{ENV_VAR}. This makes compilers, CPython and setuptools behave "
            "reproducibly too, at the cost of a much wider blast radius. See "
            "docs/development/reproducible_archives.md."
        ),
    )
    parser.add_argument(
        "--fail-on-source-drift",
        action="store_true",
        help=(
            "Error instead of warning when the artifacts being packaged were "
            "built from a different commit, or from a dirty tree, than this "
            "checkout. Recommended for release builds."
        ),
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Print the resolved source timestamp for reproducible archives."
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Also report how the value was derived.",
    )
    args = p.parse_args(argv)

    timestamp = compute_source_date_epoch()
    print(timestamp)
    if args.explain:
        head = _head_timestamp(THEROCK_DIR)
        print(f"  superproject HEAD: {head}", file=sys.stderr)
        for entry in submodule_entries():
            if entry.is_populated and entry.differs_from_pin:
                print(
                    f"  submodule past pin: {entry.path.name} "
                    f"({_head_timestamp(entry.path)})",
                    file=sys.stderr,
                )
        print(f"  worktree dirty: {is_worktree_dirty()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
