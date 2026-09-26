# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test for the consumer-graph CLI entry point."""

import json
import subprocess
from pathlib import Path

from cmake_consumer_graph import cli


def test_cli_generates_reports_and_succeeds(tmp_path: Path) -> None:
    # End-to-end smoke test of cli.main() against a throwaway git repo: exercises
    # argument parsing, git-tracked file discovery, analysis, comparison, and JSON
    # output. Self-contained (no dependency on the real tree), so it runs in the
    # default blocking suite rather than the report-only enforcement step.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text(
        "therock_cmake_subproject_declare(amd-llvm)\n"
        "therock_cmake_subproject_declare(hip-clr BUILD_DEPS amd-llvm)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "CMakeLists.txt"], cwd=repo, check=True)

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "amd-llvm": {"consumers": ["hip-clr"]},
                "hip-clr": {"consumers": []},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        [
            "--therock-dir",
            str(repo),
            "--output-dir",
            str(output_dir),
            "--reference-graph",
            str(reference),
        ]
    )

    assert exit_code == 0
    written = {path.name for path in output_dir.iterdir()}
    assert written == {
        "prototype_consumer_graph.json",
        "comparison.json",
        "analysis_inventory.json",
        "subtree_map.json",
    }
    graph = json.loads(
        (output_dir / "prototype_consumer_graph.json").read_text(encoding="utf-8")
    )
    assert graph["amd-llvm"]["consumers"] == ["hip-clr"]
