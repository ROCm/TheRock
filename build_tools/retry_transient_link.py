#!/usr/bin/env python3
"""Retry a sub-project build once when it hits the transient Windows link flake.

Background
----------
On the Windows CI runners (which are Windows *containers*, not VMs), `lld-link`
intermittently reports every symbol of exactly one translation unit as
undefined, even though that object is present, non-empty, valid COFF, listed in
the link response file, and byte-identical to the object used by a passing build
of the same commit.

This was proven transient: an `if: failure()` audit re-ran the failed
sub-project build with no source change, `ninja -n` reported zero pending
compile edges (so every object was already up to date), it re-ran only the link,
and the link succeeded -- 5 times out of 5, across rocSPARSE and rocSOLVER and
four different GPU architectures.

Why retrying is safe
--------------------
A retry cannot mask a genuine undefined-symbol error, because a genuine error
reproduces on the second attempt and the build still fails. The only cost of a
false retry is the time of one extra link. To keep the flake rate visible rather
than silently absorbed, every retry prints a THEROCK_TRANSIENT_LINK_RETRY marker
line that CI can count.

Usage
-----
    retry_transient_link.py --label rocSPARSE [--max-retries 1] -- <command...>
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# Marker emitted on every retry so the flake rate stays greppable in CI logs.
RETRY_MARKER = "THEROCK_TRANSIENT_LINK_RETRY"

# Signature of the flake. lld-link reports the whole contribution of one
# translation unit as undefined. A real undefined-symbol bug looks identical
# here, which is fine -- it simply reproduces and the build still fails.
LINK_FAILURE_SIGNATURES = (
    "lld-link: error: undefined symbol:",
    "ld.lld: error: undefined symbol:",
)

# Only the tail of the output is inspected, so a multi-GB build log cannot
# exhaust memory. Link errors are always at the end of a failing build.
TAIL_LINES = 400


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a build command, retrying once on the transient link flake."
    )
    parser.add_argument(
        "--label",
        default="",
        help="Sub-project name, used to label retry marker lines.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts after the first failure (default: 1).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The build command, preceded by '--'.",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given; pass it after '--'")
    return args


def run_once(command: Sequence[str], tail: collections.deque[str]) -> int:
    """Run command, streaming merged output through and retaining the tail."""
    tail.clear()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        # Stream immediately so the caller's log keeps accurate timestamps and
        # the build still shows progress.
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
    return process.wait()


def looks_transient(tail: Sequence[str]) -> bool:
    return any(
        signature in line for line in tail for signature in LINK_FAILURE_SIGNATURES
    )


def _audit_module():
    """Imports the audit helpers, or returns None if they are unavailable.

    The wrapper fronts every sub-project build, so a missing or broken helper
    must degrade to the old behaviour rather than fail a build.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "github_actions"))
        import audit_link_inputs

        return audit_link_inputs
    except Exception:
        return None


def find_deficient_libraries(tail: Sequence[str]) -> list[Path]:
    """Returns in-tree libraries that fail to export the symbols the link needs.

    A retry cannot fix this class of failure: the library is already on disk and
    ninja considers it current, so the second attempt re-reads the same bad
    input. Naming the library is what makes it fixable, and it is only reported
    when llvm-nm confirms the symbols really are absent, so a library is never
    deleted on a guess.
    """
    ali = _audit_module()
    if ali is None:
        return []
    text = "".join(tail)
    try:
        symbols = ali.parse_undefined_symbols(text)
        dirs, tokens = ali.parse_link_libraries(text)
        libraries = ali.resolve_libraries(dirs, tokens, Path.cwd())
        if not symbols or not libraries:
            return []
        nm = ali.find_llvm_tool("llvm-nm", Path.cwd(), text)
        if nm is None:
            return []
        exports: dict[Path, set[str]] = {}
        for lib in libraries:
            names, _status = ali.library_symbols(lib, nm)
            exports[lib] = names
        missing = [s for s in symbols if not any(s in n for n in exports.values())]
        if not missing:
            return []
        # Only libraries this build produces can be rebuilt by re-running it.
        # Deleting anything else would turn a flake into a broken tree.
        return [
            lib
            for lib, names in exports.items()
            if names and not any(s in names for s in missing) and _is_rebuildable(lib)
        ]
    except Exception:
        return []


def _is_rebuildable(lib: Path) -> bool:
    """True when the library sits in a build tree rather than a toolchain dir."""
    parts = {p.lower() for p in lib.parts}
    if "dist" in parts or "stage" in parts:
        # Staged copies belong to an already-completed sub-project; removing one
        # breaks a consumer that will not rebuild it.
        return False
    return "build" in parts


def purge_libraries(libraries: Sequence[Path], label: str) -> int:
    """Deletes deficient libraries and their matching import/shared files."""
    removed = 0
    for lib in libraries:
        for candidate in (lib, lib.with_suffix(".dll"), lib.with_suffix(".exp")):
            try:
                if candidate.is_file():
                    candidate.unlink()
                    removed += 1
                    print(f"  removed {candidate}", flush=True)
            except OSError as exc:
                print(f"  could not remove {candidate}: {exc}", flush=True)
    return removed


def main(argv: Sequence[str]) -> int:
    args = parse_arguments(argv)
    label = args.label or "<unlabeled>"
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)

    # This wrapper sits in front of every sub-project build, so it must never be
    # the thing that breaks one. Build output routinely contains characters the
    # active Windows console codepage cannot encode, which would otherwise raise
    # UnicodeEncodeError mid-stream and fail a build that was going fine.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    purged = False
    attempt = 0
    while True:
        returncode = run_once(args.command, tail)
        if returncode == 0:
            if attempt:
                result = "recovered-by-rebuild" if purged else "recovered"
                print(
                    f"{RETRY_MARKER}: label={label} attempt={attempt} result={result}",
                    flush=True,
                )
            return 0

        transient = looks_transient(tail)
        if transient and attempt < args.max_retries:
            attempt += 1
            print(
                f"{RETRY_MARKER}: label={label} attempt={attempt} result=retrying\n"
                f"  A link step reported undefined symbols. This matches a known "
                f"transient read failure on the Windows CI runners, so the build is "
                f"being retried once. A genuine undefined symbol will fail again.",
                flush=True,
            )
            continue

        # Retries are exhausted. If the symbols are missing from a library this
        # build produced, re-running the same link will keep reading the same
        # bad file; removing it makes the next run rebuild it instead.
        if transient and not purged:
            deficient = find_deficient_libraries(tail)
            if deficient:
                purged = True
                attempt += 1
                print(
                    f"{RETRY_MARKER}: label={label} attempt={attempt} "
                    f"result=rebuilding-library\n"
                    f"  {len(deficient)} librar(y/ies) on the link line do not export "
                    f"the missing symbols, so a plain retry cannot help. Removing "
                    f"them so this build regenerates them:",
                    flush=True,
                )
                if purge_libraries(deficient, label):
                    continue
                purged = True

        if attempt:
            print(
                f"{RETRY_MARKER}: label={label} attempt={attempt} result=exhausted",
                flush=True,
            )
        return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
