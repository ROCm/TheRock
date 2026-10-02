#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate test_tools/therock_consumer_graph.json from the super-project CMake.

Statically parses the tracked CMake files (no configure, no submodules) and writes
the reverse-dependency consumer graph. See cmake_consumer_graph.py for the full
description and limitations.

``--check`` analyzes in memory and diffs the result against the committed graph
without writing, exiting non-zero on any difference. While the committed graph is
still emitted by CMake, this parser is a conservative *superset* of it, so
``--check`` reports that expected superset delta; it is intentionally not wired
into any CI gate.
"""

import argparse
import json
import sys
from pathlib import Path

from _therock_utils.cmake_consumer_graph import (
    RepositoryAnalyzer,
    compare_graphs,
    load_consumer_graph,
)

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
CONSUMER_GRAPH_PATH = THEROCK_DIR / "test_tools" / "therock_consumer_graph.json"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_delta(comparison) -> None:
    """Print a concise summary of how the generated graph differs from committed."""
    for label, items in (
        ("generated-only nodes", comparison.generated_only_nodes),
        ("committed-only nodes", comparison.reference_only_nodes),
        ("generated-only edges", comparison.generated_only_edges),
        ("committed-only edges", comparison.reference_only_edges),
    ):
        print(f"  {label} ({len(items)}):")
        for item in items:
            print(f"    {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate test_tools/therock_consumer_graph.json from the super-project CMake."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Diff the generated graph against the committed one without writing; "
            "exit non-zero on any difference. The parser is currently a superset "
            "of the committed graph, so this reports that expected delta and is not "
            "wired into a CI gate."
        ),
    )
    args = parser.parse_args(argv)
    result = RepositoryAnalyzer(THEROCK_DIR).analyze()
    graph = result.build_consumer_graph()

    if args.check:
        committed = load_consumer_graph(CONSUMER_GRAPH_PATH)
        comparison = compare_graphs(graph, committed)
        if (
            comparison.generated_only_nodes
            or comparison.reference_only_nodes
            or comparison.generated_only_edges
            or comparison.reference_only_edges
        ):
            print(
                "Generated consumer graph differs from "
                f"{CONSUMER_GRAPH_PATH.relative_to(THEROCK_DIR).as_posix()}:"
            )
            _print_delta(comparison)
            return 1
        print("Generated consumer graph matches the committed graph.")
        return 0

    _write_json(CONSUMER_GRAPH_PATH, graph)
    print(
        f"Wrote consumer graph for {len(graph)} subprojects to "
        f"{CONSUMER_GRAPH_PATH.relative_to(THEROCK_DIR).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
