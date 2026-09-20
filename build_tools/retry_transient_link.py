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


def main(argv: Sequence[str]) -> int:
    args = parse_arguments(argv)
    label = args.label or "<unlabeled>"
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)

    for attempt in range(args.max_retries + 1):
        returncode = run_once(args.command, tail)
        if returncode == 0:
            if attempt:
                print(
                    f"{RETRY_MARKER}: label={label} attempt={attempt} result=recovered",
                    flush=True,
                )
            return 0
        if attempt >= args.max_retries or not looks_transient(tail):
            if attempt:
                print(
                    f"{RETRY_MARKER}: label={label} attempt={attempt} result=exhausted",
                    flush=True,
                )
            return returncode
        print(
            f"{RETRY_MARKER}: label={label} attempt={attempt + 1} result=retrying\n"
            f"  A link step reported undefined symbols. This matches a known "
            f"transient read failure on the Windows CI runners, so the build is "
            f"being retried once. A genuine undefined symbol will fail again.",
            flush=True,
        )
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
