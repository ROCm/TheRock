# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command line interface for the static consumer graph prototype."""

import argparse
import json
import sys
from pathlib import Path

from cmake_consumer_graph.analyzer import (
    RepositoryAnalyzer,
    compare_graphs,
    load_consumer_graph,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str]) -> int:
    """Run the repository analysis and write graph comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--therock-dir",
        type=Path,
        required=True,
        help="Path to the TheRock super-project checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated graph and comparison JSON files",
    )
    parser.add_argument(
        "--reference-graph",
        type=Path,
        help=(
            "Existing graph to compare against; defaults to "
            "<therock-dir>/test_tools/therock_consumer_graph.json"
        ),
    )
    args = parser.parse_args(argv)

    therock_dir = args.therock_dir.resolve()
    reference_path = args.reference_graph or (
        therock_dir / "test_tools" / "therock_consumer_graph.json"
    )
    result = RepositoryAnalyzer(therock_dir).analyze()
    graph = result.build_consumer_graph()
    comparison = compare_graphs(graph, load_consumer_graph(reference_path))

    _write_json(args.output_dir / "prototype_consumer_graph.json", graph)
    _write_json(
        args.output_dir / "comparison.json",
        comparison.to_json_data(),
    )
    _write_json(
        args.output_dir / "analysis_inventory.json",
        {
            "tracked_cmake_files": sorted(
                path.as_posix() for path in result.tracked_cmake_files
            ),
            "parsed_cmake_files": sorted(
                path.as_posix() for path in result.parsed_cmake_files
            ),
            "reachable_cmake_files": sorted(
                path.as_posix() for path in result.reachable_cmake_files
            ),
            "tracked_declaration_files": sorted(
                path.as_posix() for path in result.tracked_declaration_files
            ),
            "declaration_files": sorted(
                path.as_posix() for path in result.declaration_files
            ),
            "unreachable_declaration_files": sorted(
                path.as_posix() for path in result.unreachable_declaration_files
            ),
            "declaration_calls": result.declaration_count,
            "subprojects": len(result.subprojects),
            "dangling_dependencies": result.dangling_dependencies(),
            "skipped_paths": [
                {
                    "location": skipped.location.format(),
                    "expression": skipped.expression,
                    "reason": skipped.reason,
                }
                for skipped in result.skipped_paths
            ],
        },
    )

    summary = comparison.to_json_data()["summary"]
    print(
        f"Parsed {len(result.parsed_cmake_files)}/"
        f"{len(result.tracked_cmake_files)} tracked CMake files."
    )
    print(
        f"Reached {len(result.reachable_cmake_files)} files and discovered "
        f"{result.declaration_count} declarations for "
        f"{len(result.subprojects)} unique subprojects."
    )
    print(json.dumps(summary, indent=2))
    if result.unreachable_declaration_files:
        print("Tracked declaration files were missed by traversal:", file=sys.stderr)
        for path in sorted(result.unreachable_declaration_files):
            print(f"  {path.as_posix()}", file=sys.stderr)
        return 1
    dangling_dependencies = result.dangling_dependencies()
    if dangling_dependencies:
        print("Some dependency names do not resolve to subprojects:", file=sys.stderr)
        print(json.dumps(dangling_dependencies, indent=2), file=sys.stderr)
        return 1
    if comparison.reference_only_nodes or comparison.reference_only_edges:
        print("The static graph missed reference graph data.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
