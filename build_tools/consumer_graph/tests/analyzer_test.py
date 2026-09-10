# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for conservative CMake repository analysis."""

import re
from pathlib import Path

import pytest

from cmake_consumer_graph.analyzer import (
    AnalysisError,
    RepositoryAnalyzer,
    compare_graphs,
    list_tracked_cmake_files,
    load_consumer_graph,
)


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_traversal_stays_within_tracked_files_and_unions_branches(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(optional_deps)
if(WIN32)
  list(APPEND optional_deps windows-runtime)
  add_subdirectory(windows)
else()
  list(APPEND optional_deps linux-runtime)
  add_subdirectory(linux)
endif()
add_subdirectory(external)
therock_cmake_subproject_declare(root
  RUNTIME_DEPS ${optional_deps})
""",
    )
    _write(
        tmp_path / "windows" / "CMakeLists.txt",
        "therock_cmake_subproject_declare(windows-runtime)\n",
    )
    _write(
        tmp_path / "linux" / "CMakeLists.txt",
        "therock_cmake_subproject_declare(linux-runtime)\n",
    )
    _write(
        tmp_path / "external" / "CMakeLists.txt",
        "therock_cmake_subproject_declare(must-not-be-seen)\n",
    )
    tracked = {
        Path("CMakeLists.txt"),
        Path("windows/CMakeLists.txt"),
        Path("linux/CMakeLists.txt"),
    }

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert set(result.subprojects) == {"root", "windows-runtime", "linux-runtime"}
    assert result.subprojects["root"].runtime_deps == {
        "windows-runtime",
        "linux-runtime",
    }
    assert result.declaration_count == 3
    assert result.unreachable_declaration_files == set()
    assert Path("external/CMakeLists.txt") not in result.reachable_cmake_files


def test_graph_includes_explicit_and_toolchain_dependencies(tmp_path: Path) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(amd-llvm)
therock_cmake_subproject_declare(hip-clr BUILD_DEPS amd-llvm)
therock_cmake_subproject_declare(client
  BUILD_DEPS amd-llvm
  RUNTIME_DEPS hip-clr
  COMPILER_TOOLCHAIN
    # Comments inside argument lists must not become values.
    amd-hip)
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    graph = result.build_consumer_graph()

    assert graph["amd-llvm"]["consumers"] == ["client", "hip-clr"]
    assert graph["hip-clr"]["consumers"] == ["client"]


def test_unresolved_dependency_variable_fails_loudly(tmp_path: Path) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(client
  BUILD_DEPS ${unknown_dependency_list})
""",
    )
    tracked = {Path("CMakeLists.txt")}

    with pytest.raises(AnalysisError, match="unknown_dependency_list"):
        RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()


def test_graph_comparison_reports_edge_direction() -> None:
    generated = {
        "a": {"consumers": ["b", "c"]},
        "b": {"consumers": []},
        "c": {"consumers": []},
    }
    reference = {
        "a": {"consumers": ["b"]},
        "b": {"consumers": []},
    }

    comparison = compare_graphs(generated, reference)

    assert comparison.generated_only_nodes == ["c"]
    assert comparison.generated_only_edges == ["a -> c"]
    assert comparison.reference_only_edges == []


def test_defined_but_empty_dependency_variable_is_silently_dropped(
    tmp_path: Path,
) -> None:
    # Unlike the unresolved case above (which fails loud), a variable that is
    # *defined* but expands to nothing is not flagged, so its edge is dropped
    # silently. Pin this behavior so any future fix is a deliberate change.
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(empty_deps)
therock_cmake_subproject_declare(dep)
therock_cmake_subproject_declare(client
  BUILD_DEPS ${empty_deps})
""",
    )
    tracked = {Path("CMakeLists.txt")}

    # No AnalysisError is raised, and the edge is silently absent.
    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    assert result.subprojects["client"].build_deps == set()
    assert result.build_consumer_graph()["dep"]["consumers"] == []


def test_static_graph_is_superset_of_committed_graph() -> None:
    # Run against the real tree: the generated graph must not miss any node or edge
    # in the committed graph (the superset invariant). Skips outside a git checkout.
    repo_root = Path(__file__).resolve().parents[3]
    committed = repo_root / "test_tools" / "therock_consumer_graph.json"
    if not (
        (repo_root / ".git").exists()
        and committed.exists()
        and (repo_root / "CMakeLists.txt").exists()
    ):
        pytest.skip("not a TheRock git checkout with a committed consumer graph")

    result = RepositoryAnalyzer(repo_root).analyze()
    comparison = compare_graphs(
        result.build_consumer_graph(), load_consumer_graph(committed)
    )

    assert comparison.reference_only_nodes == []
    assert comparison.reference_only_edges == []
    assert result.unreachable_declaration_files == set()
    assert result.dangling_dependencies() == {}


def test_subtree_map_relativizes_and_fans_out(tmp_path: Path) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "${THEROCK_SOURCE_DIR}/rocm-libraries")
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/rocm-systems")
therock_cmake_subproject_declare(hip-clr
  EXTERNAL_SOURCE_DIR ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr)
therock_cmake_subproject_declare(ocl-clr
  EXTERNAL_SOURCE_DIR ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr)
therock_cmake_subproject_declare(rocroller
  EXTERNAL_SOURCE_DIR ${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/shared/rocroller)
therock_cmake_subproject_declare(variable-sourced
  EXTERNAL_SOURCE_DIR ${THEROCK_UNMODELED_SOURCE_DIR}/projects/x)
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    subtree_map = result.build_subtree_map()

    # One subtree fans out to several keys; a non-projects/ prefix resolves.
    assert subtree_map == {
        "projects/clr": ["hip-clr", "ocl-clr"],
        "shared/rocroller": ["rocroller"],
    }
    # The declaration whose EXTERNAL_SOURCE_DIR did not resolve is still a
    # subproject, but is silently absent from the subtree map.
    assert "variable-sourced" in result.subprojects


def test_subtree_map_covers_known_subtrees_on_real_tree() -> None:
    # Real-tree anchors: fan-out, name skew, and a non-projects/ prefix all resolve;
    # a variable-sourced (therock_enable_external_source) subtree is skipped. Skips
    # outside a git checkout.
    repo_root = Path(__file__).resolve().parents[3]
    if not ((repo_root / ".git").exists() and (repo_root / "CMakeLists.txt").exists()):
        pytest.skip("not a TheRock git checkout")

    subtree_map = RepositoryAnalyzer(repo_root).analyze().build_subtree_map()

    assert subtree_map.get("projects/clr") == ["hip-clr", "ocl-clr"]
    assert subtree_map.get("projects/composablekernel") == ["composable_kernel"]
    assert "shared/rocroller" in subtree_map
    assert "projects/rocdbgapi" not in subtree_map


def test_subtree_map_covers_all_direct_external_source_dirs() -> None:
    # Coverage cross-check: regex the tracked CMake for every EXTERNAL_SOURCE_DIR
    # written directly under a source root and assert subtree_map covers each,
    # catching a silently-shrinking map. Variable-sourced dirs are not matched.
    repo_root = Path(__file__).resolve().parents[3]
    if not ((repo_root / ".git").exists() and (repo_root / "CMakeLists.txt").exists()):
        pytest.skip("not a TheRock git checkout")

    pattern = re.compile(
        r"EXTERNAL_SOURCE_DIR\s+\"?\$\{THEROCK_ROCM_(?:LIBRARIES|SYSTEMS)_SOURCE_DIR\}"
        r"/([^\"\s)]+)"
    )
    expected: set[str] = set()
    for relative_path in list_tracked_cmake_files(repo_root):
        text = (repo_root / relative_path).read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            # Drop CMake line comments so a commented-out EXTERNAL_SOURCE_DIR (e.g.
            # math-libs/CMakeLists.txt has one) is not treated as a live declaration.
            code = line.split("#", 1)[0]
            expected.update(m.rstrip("/") for m in pattern.findall(code))

    subtree_map = RepositoryAnalyzer(repo_root).analyze().build_subtree_map()

    assert expected, "regex found no direct EXTERNAL_SOURCE_DIR declarations"
    missing = sorted(s for s in expected if s not in subtree_map)
    assert missing == [], f"subtree_map missing direct subtrees: {missing}"
    assert len(subtree_map) >= len(expected)
