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


def is_worktree_dirty(
    repo_dir: Path = THEROCK_DIR,
    entries: list[SubmoduleEntry] | None = None,
) -> bool:
    """Whether anything is uncommitted, in the superproject or any submodule.

    Only genuinely uncommitted content counts. A plain `git status --porcelain`
    is not usable here: it reports ` M <path>` for a submodule that merely sits
    at a different commit than its pin, which is committed work already handled
    by folding that submodule's commit time in. Treating it as dirty would jump
    straight to the current time and mask that logic entirely.

    So the superproject is asked with `--ignore-submodules=all`, and each
    populated submodule is asked about its own working tree separately.
    """
    superproject = _git(
        "status", "--porcelain", "--ignore-submodules=all", cwd=repo_dir
    )
    if superproject and superproject.strip():
        return True

    if entries is None:
        entries = submodule_entries(repo_dir)
    for entry in entries:
        if not entry.is_populated:
            continue
        output = _git("status", "--porcelain", cwd=entry.path)
        if output and output.strip():
            return True
    return False


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
    * Uncommitted changes anywhere have no commit time at all, so the current
      time is folded in. Archives from a dirty tree are not reproducible, which
      is inherent rather than a limitation of this approach.
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

    if is_worktree_dirty(repo_dir, entries):
        timestamps.append(int(time.time()))

    if not timestamps:
        # No git at all. Not reproducible, but a working build beats a
        # deterministically broken one.
        return int(time.time())
    return max(timestamps)


def child_env(
    *,
    export_standard_var: bool = False,
    repo_dir: Path = THEROCK_DIR,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for workers that write archives.

    Sets `THEROCK_SOURCE_DATE_EPOCH`. Also sets `SOURCE_DATE_EPOCH` when
    `export_standard_var` is true -- see EXPORT_WARNING for what else that
    changes.
    """
    env = dict(os.environ if base_env is None else base_env)
    timestamp = str(compute_source_date_epoch(repo_dir))
    env[ENV_VAR] = timestamp
    if export_standard_var:
        env[STANDARD_ENV_VAR] = timestamp
    return env


def apply_to_environ(
    *,
    export_standard_var: bool = False,
    repo_dir: Path = THEROCK_DIR,
) -> int:
    """Publishes the resolved timestamp into this process's environment.

    For orchestrators that write archives in-process rather than by spawning
    workers. Returns the resolved value. Does not overwrite an existing
    `SOURCE_DATE_EPOCH`, which outranks everything (see `archive_util`).
    """
    timestamp = compute_source_date_epoch(repo_dir)
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
