#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Drift check for the committed consumer graph.

Keeps test_tools/therock_consumer_graph.json in sync with a freshly emitted graph
(written to build/ by a configure). `--check` compares the two, failing on any
edge change; `--write` overwrites the committed copy. Both normalize (sorted keys
and consumer lists, 2-space indent, trailing newline) so only genuine
dependency-edge changes register, not the emitter's registration order.
"""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_FILENAME = "therock_consumer_graph.json"
_COMMITTED_PATH = _REPO_ROOT / "test_tools" / _GRAPH_FILENAME
# Which build tree holds the freshly emitted graph; overridable via --build-dir.
_DEFAULT_BUILD_DIR = Path(os.environ.get("THEROCK_BUILD_DIR", _REPO_ROOT / "build"))


def normalize(path: Path) -> str:
    """Return the graph as canonical JSON: sorted keys and consumer lists,
    2-space indent, trailing newline. Used by both --check and --write."""
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
        "-DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu"
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
        "--build-dir",
        type=Path,
        default=_DEFAULT_BUILD_DIR,
        help=(
            "Build directory holding the freshly emitted graph "
            "(default: $THEROCK_BUILD_DIR or build/). Ignored if --emitted is set."
        ),
    )
    parser.add_argument(
        "--emitted",
        type=Path,
        default=None,
        help="Explicit path to the emitted graph (overrides --build-dir).",
    )
    args = parser.parse_args(argv)

    emitted = args.emitted or (args.build_dir / _GRAPH_FILENAME)

    if args.write:
        return write(args.committed, emitted)
    return check(args.committed, emitted)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
