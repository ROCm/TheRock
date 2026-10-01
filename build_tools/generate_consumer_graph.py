#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate the committed consumer-graph metadata from the super-project CMake.

Statically parses the tracked CMake files (no configure, no submodules) and writes
three committed files:

* ``test_tools/therock_consumer_graph.json`` — the reverse-dependency consumer graph.
* ``test_tools/therock_subtree_map.json`` — each external source subtree mapped to the
  graph key(s) built from it.
* ``test_tools/therock_source_dir_map.json`` — each artifact mapped to the source
  subtree(s) its therock_provide_artifact() SUBPROJECT_DEPS build from. Report-only
  metadata; it is NOT authoritative for BUILD_TOPOLOGY.toml source_paths, which are
  hand-curated and only partly coincide.

This parser is the authoritative generator of all three files. See cmake_consumer_graph.py
for the full description and limitations.

``--check`` analyzes in memory and diffs the result against ALL committed files
without writing, exiting non-zero on any difference. It is the drift gate: the
``Test build_tools`` pytest step enforces the same invariant, so a stale committed
file fails CI.
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
SUBTREE_MAP_PATH = THEROCK_DIR / "test_tools" / "therock_subtree_map.json"
SOURCE_DIR_MAP_PATH = THEROCK_DIR / "test_tools" / "therock_source_dir_map.json"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_graph_delta(comparison) -> None:
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


def _map_delta(generated: dict, committed: dict, noun: str) -> list[str]:
    """Return human-readable difference lines between two ``str -> list`` maps."""
    lines: list[str] = []
    generated_keys, committed_keys = set(generated), set(committed)
    for key in sorted(generated_keys - committed_keys):
        lines.append(f"    generated-only {noun}: {key} -> {generated[key]}")
    for key in sorted(committed_keys - generated_keys):
        lines.append(f"    committed-only {noun}: {key} -> {committed[key]}")
    for key in sorted(generated_keys & committed_keys):
        if generated[key] != committed[key]:
            lines.append(
                f"    {noun} {key}: generated {generated[key]} != committed {committed[key]}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the committed consumer-graph metadata from the super-project CMake."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Diff the generated graph, subtree map, and source-dir map against the "
            "committed files without writing; exit non-zero on any difference. This "
            "is the drift gate."
        ),
    )
    args = parser.parse_args(argv)
    result = RepositoryAnalyzer(THEROCK_DIR).analyze()
    graph = result.build_consumer_graph()
    subtree_map = result.build_subtree_map()
    source_dir_map = result.build_source_dir_map()

    if args.check:
        differs = False

        committed_graph = load_consumer_graph(CONSUMER_GRAPH_PATH)
        comparison = compare_graphs(graph, committed_graph)
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
            _print_graph_delta(comparison)
            differs = True

        committed_subtree = json.loads(SUBTREE_MAP_PATH.read_text(encoding="utf-8"))
        subtree_delta = _map_delta(subtree_map, committed_subtree, "subtree")
        if subtree_delta:
            print(
                "Generated subtree map differs from "
                f"{SUBTREE_MAP_PATH.relative_to(THEROCK_DIR).as_posix()}:"
            )
            print("\n".join(subtree_delta))
            differs = True

        committed_source_dir = json.loads(
            SOURCE_DIR_MAP_PATH.read_text(encoding="utf-8")
        )
        source_dir_delta = _map_delta(source_dir_map, committed_source_dir, "artifact")
        if source_dir_delta:
            print(
                "Generated source-dir map differs from "
                f"{SOURCE_DIR_MAP_PATH.relative_to(THEROCK_DIR).as_posix()}:"
            )
            print("\n".join(source_dir_delta))
            differs = True

        if differs:
            return 1
        print(
            "Generated consumer graph, subtree map, and source-dir map match the "
            "committed files."
        )
        return 0

    _write_json(CONSUMER_GRAPH_PATH, graph)
    _write_json(SUBTREE_MAP_PATH, subtree_map)
    _write_json(SOURCE_DIR_MAP_PATH, source_dir_map)
    print(
        f"Wrote consumer graph ({len(graph)} subprojects), subtree map "
        f"({len(subtree_map)} subtrees), and source-dir map ({len(source_dir_map)} "
        f"artifacts) to {CONSUMER_GRAPH_PATH.parent.relative_to(THEROCK_DIR).as_posix()}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
