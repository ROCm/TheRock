#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Drift check for the committed consumer graph.

The consumer graph is emitted at CMake configure time by
therock_emit_consumer_graph() and committed to
test_tools/therock_consumer_graph.json so the per-PR change-detection job can
read it without a fetch or configure. This script keeps that committed copy
honest: `--check` regenerates nothing itself but compares the freshly emitted
graph (written to build/ by a configure) against the committed copy after
normalizing BOTH the same way, and fails on any real edge change. `--write`
copies the emitted graph over the committed copy in normalized form, so the
fix for a drift failure is a single command rather than a re-implemented
normalize one-liner.

Normalization (sorted keys, sorted consumer lists, 2-space indent, trailing
newline) makes the check insensitive to the emitter's registration-order output
so it only fails on a genuine dependency-edge change.
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_PATH = _REPO_ROOT / "test_tools" / "therock_consumer_graph.json"
_EMITTED_PATH = _REPO_ROOT / "build" / "therock_consumer_graph.json"


def normalize(path: Path) -> str:
    """Return the graph at `path` as canonical JSON text.

    Sorted keys, sorted consumer lists, 2-space indent, trailing newline. This
    is the single source of truth for the committed graph's on-disk form: both
    the drift comparison and `--write` use it, so the committed copy and the
    check can never disagree on formatting.
    """
    graph = json.loads(path.read_text())
    norm = {
        key: {"consumers": sorted(graph[key].get("consumers", []))}
        for key in sorted(graph)
    }
    return json.dumps(norm, indent=2, sort_keys=True) + "\n"


def check(committed_path: Path, emitted_path: Path) -> int:
    """Fail (return 1) if the committed graph differs from the emitted one."""
    committed = normalize(committed_path)
    emitted = normalize(emitted_path)

    if committed == emitted:
        print("Consumer graph is up to date.")
        return 0

    print("::error::test_tools/therock_consumer_graph.json is out of date.")
    print(
        "A dependency edge changed but the committed consumer graph was not "
        "regenerated."
    )
    print("Fix: run a full-tree configure, then regenerate and re-commit:")
    print("    python3 ./build_tools/fetch_sources.py")
    print(
        "    cmake -B build -GNinja -DTHEROCK_ENABLE_ALL=ON "
        "-DTHEROCK_AMDGPU_FAMILIES=gfx110X-all"
    )
    print(
        "    python3 ./build_tools/github_actions/check_consumer_graph_drift.py --write"
    )
    print("    git add test_tools/therock_consumer_graph.json")
    sys.stdout.writelines(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            emitted.splitlines(keepends=True),
            fromfile="committed",
            tofile="regenerated",
        )
    )
    return 1


def write(committed_path: Path, emitted_path: Path) -> int:
    """Overwrite the committed graph with the normalized emitted graph."""
    committed_path.write_text(normalize(emitted_path))
    print(f"Wrote normalized consumer graph to {committed_path}.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed graph differs from build/'s emitted graph.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Normalize build/'s emitted graph over the committed copy.",
    )
    parser.add_argument(
        "--committed",
        type=Path,
        default=_COMMITTED_PATH,
        help="Path to the committed consumer graph (default: test_tools/).",
    )
    parser.add_argument(
        "--emitted",
        type=Path,
        default=_EMITTED_PATH,
        help="Path to the freshly emitted consumer graph (default: build/).",
    )
    args = parser.parse_args(argv)

    if args.write:
        return write(args.committed, args.emitted)
    return check(args.committed, args.emitted)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
