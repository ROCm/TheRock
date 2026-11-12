# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for conservative CMake repository analysis."""

import re
from pathlib import Path

import pytest

from _therock_utils.cmake_consumer_graph import (
    THEROCK_REPO,
    AnalysisError,
    RepositoryAnalyzer,
    compare_graphs,
    list_submodule_paths,
    list_tracked_cmake_files,
    load_consumer_graph,
)


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_traversal_within_tracked_files_unions_branches(tmp_path: Path) -> None:
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


def test_graph_explicit_and_toolchain_deps(tmp_path: Path) -> None:
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


def test_unresolved_dep_var_raises(tmp_path: Path) -> None:
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


def test_graph_comparison_edge_direction() -> None:
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


def test_empty_dep_var_silently_dropped(tmp_path: Path) -> None:
    # A *defined* variable that expands to nothing is not flagged, so its edge is
    # dropped silently (an *unresolved* variable, by contrast, raises).
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

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    assert result.subprojects["client"].build_deps == set()
    assert result.build_consumer_graph()["dep"]["consumers"] == []


def test_include_in_caller_scope(tmp_path: Path) -> None:
    # include() shares the caller's scope: a variable set() in the included file
    # is visible to the includer.
    _write(
        tmp_path / "CMakeLists.txt",
        """
include(shared)
therock_cmake_subproject_declare(dep)
therock_cmake_subproject_declare(client BUILD_DEPS ${SHARED_DEPS})
""",
    )
    _write(tmp_path / "shared.cmake", "set(SHARED_DEPS dep)\n")
    tracked = {Path("CMakeLists.txt"), Path("shared.cmake")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert result.subprojects["client"].build_deps == {"dep"}
    assert result.build_consumer_graph()["dep"]["consumers"] == ["client"]


def test_add_subdirectory_scope_isolated(tmp_path: Path) -> None:
    # add_subdirectory() opens a fresh child scope, so a set() inside the child
    # does not overwrite the parent's variable.
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(DEPS parent-dep)
therock_cmake_subproject_declare(parent-dep)
add_subdirectory(child)
therock_cmake_subproject_declare(client BUILD_DEPS ${DEPS})
""",
    )
    _write(
        tmp_path / "child" / "CMakeLists.txt",
        """
set(DEPS child-dep)
therock_cmake_subproject_declare(child-dep)
""",
    )
    tracked = {Path("CMakeLists.txt"), Path("child/CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert result.subprojects["client"].build_deps == {"parent-dep"}
    assert result.build_consumer_graph()["child-dep"]["consumers"] == []


def test_set_parent_scope_writes_parent(tmp_path: Path) -> None:
    # set(... PARENT_SCOPE) in an add_subdirectory child writes the enclosing
    # directory scope, so a dependency list defined there resolves in the parent
    # declaration that consumes it.
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(child-provided-dep)
add_subdirectory(child)
therock_cmake_subproject_declare(client BUILD_DEPS ${CHILD_DEPS})
""",
    )
    _write(
        tmp_path / "child" / "CMakeLists.txt",
        "set(CHILD_DEPS child-provided-dep PARENT_SCOPE)\n",
    )
    tracked = {Path("CMakeLists.txt"), Path("child/CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert result.subprojects["client"].build_deps == {"child-provided-dep"}
    assert result.build_consumer_graph()["child-provided-dep"]["consumers"] == [
        "client"
    ]


def test_foreach_in_lists(tmp_path: Path) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(DEPS alpha beta)
therock_cmake_subproject_declare(alpha)
therock_cmake_subproject_declare(beta)
foreach(item IN LISTS DEPS)
  therock_cmake_subproject_declare(client BUILD_DEPS ${item})
endforeach()
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    graph = result.build_consumer_graph()

    assert graph["alpha"]["consumers"] == ["client"]
    assert graph["beta"]["consumers"] == ["client"]


def test_foreach_in_items(tmp_path: Path) -> None:
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(alpha)
therock_cmake_subproject_declare(beta)
foreach(item IN ITEMS alpha beta)
  therock_cmake_subproject_declare(client BUILD_DEPS ${item})
endforeach()
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    graph = result.build_consumer_graph()

    assert graph["alpha"]["consumers"] == ["client"]
    assert graph["beta"]["consumers"] == ["client"]


def test_foreach_range_binds_nothing(tmp_path: Path) -> None:
    # RANGE yields integers, never names, so the loop variable expands to nothing
    # (the defined-but-empty path) and contributes no bogus dependency.
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(base)
therock_cmake_subproject_declare(client BUILD_DEPS base)
foreach(i RANGE 2)
  therock_cmake_subproject_declare(client BUILD_DEPS ${i})
endforeach()
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert result.subprojects["client"].build_deps == {"base"}
    assert result.dangling_dependencies() == {}


def test_foreach_in_lists_undefined_raises(tmp_path: Path) -> None:
    # foreach(x IN LISTS <undefined>) leaves the loop variable unset, so a
    # dependency built from it raises via the unresolved guard rather than being
    # dropped.
    _write(
        tmp_path / "CMakeLists.txt",
        """
foreach(item IN LISTS undefined_dependency_list)
  therock_cmake_subproject_declare(client BUILD_DEPS ${item})
endforeach()
""",
    )
    tracked = {Path("CMakeLists.txt")}

    with pytest.raises(AnalysisError, match="item"):
        RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()


def test_load_graph_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text('{"a": {"consumers": ["b"]}, "bad": []}', encoding="utf-8")

    with pytest.raises(AnalysisError, match="bad"):
        load_consumer_graph(path)


def test_subtree_map_fans_out(tmp_path: Path) -> None:
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
""",
    )
    tracked = {Path("CMakeLists.txt")}

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()
    subtree_map = result.build_subtree_map()

    assert subtree_map == {
        "projects/clr": ["hip-clr", "ocl-clr"],
        "shared/rocroller": ["rocroller"],
    }


def test_unresolved_source_dir_raises(tmp_path: Path) -> None:
    # An EXTERNAL_SOURCE_DIR that does not resolve is fatal rather than dropped:
    # silently losing it would make the subproject look in-tree to build_repo_map()
    # and mis-attribute its tests to TheRock instead of its owning repository.
    _write(
        tmp_path / "CMakeLists.txt",
        """
therock_cmake_subproject_declare(variable-sourced
  EXTERNAL_SOURCE_DIR ${THEROCK_UNMODELED_SOURCE_DIR}/projects/x)
""",
    )
    tracked = {Path("CMakeLists.txt")}

    with pytest.raises(AnalysisError, match="THEROCK_UNMODELED_SOURCE_DIR"):
        RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()


def test_repo_map_attributes_sources(tmp_path: Path) -> None:
    # The four attribution paths: inside a submodule, inside a nested submodule
    # (the longer path wins), a wrapper directory holding exactly one submodule,
    # and an in-tree source.
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "${THEROCK_SOURCE_DIR}/rocm-libraries")
therock_cmake_subproject_declare(rocblas
  EXTERNAL_SOURCE_DIR ${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocblas)
therock_cmake_subproject_declare(vendored
  EXTERNAL_SOURCE_DIR ${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocblas/vendor)
therock_cmake_subproject_declare(in-tree
  EXTERNAL_SOURCE_DIR ${THEROCK_SOURCE_DIR}/base/aux-overlay)
add_subdirectory(debug-tools)
""",
    )
    # Declared relative to its own directory, and wrapping a submodule rather than
    # being one: debug-tools/rocgdb holds the debug-tools/rocgdb/source submodule.
    _write(
        tmp_path / "debug-tools" / "CMakeLists.txt",
        'therock_cmake_subproject_declare(rocgdb EXTERNAL_SOURCE_DIR "rocgdb")\n',
    )
    tracked = {Path("CMakeLists.txt"), Path("debug-tools/CMakeLists.txt")}
    submodules = {
        "rocm-libraries",
        "rocm-libraries/projects/rocblas/vendor",
        "debug-tools/rocgdb/source",
    }

    result = RepositoryAnalyzer(tmp_path, tracked_cmake_files=tracked).analyze()

    assert result.build_repo_map(submodules) == {
        "rocblas": "rocm-libraries",
        "vendored": "rocm-libraries/projects/rocblas/vendor",
        "rocgdb": "debug-tools/rocgdb/source",
        "in-tree": THEROCK_REPO,
    }


def test_repo_map_declaring_no_source_is_therock(tmp_path: Path) -> None:
    # A subproject with no EXTERNAL_SOURCE_DIR is built from TheRock's own tree.
    _write(
        tmp_path / "CMakeLists.txt",
        "therock_cmake_subproject_declare(therock-googletest)\n",
    )

    result = RepositoryAnalyzer(
        tmp_path, tracked_cmake_files={Path("CMakeLists.txt")}
    ).analyze()

    assert result.build_repo_map({"rocm-libraries"}) == {
        "therock-googletest": THEROCK_REPO
    }


def test_repo_map_root_is_not_a_wrapper(tmp_path: Path) -> None:
    # The repository root contains every submodule, so the "wrapper directory"
    # rule must not fire for it even when only one submodule exists.
    _write(
        tmp_path / "CMakeLists.txt",
        'therock_cmake_subproject_declare(root-sourced EXTERNAL_SOURCE_DIR ".")\n',
    )

    result = RepositoryAnalyzer(
        tmp_path, tracked_cmake_files={Path("CMakeLists.txt")}
    ).analyze()

    assert result.build_repo_map({"rocm-libraries"}) == {"root-sourced": THEROCK_REPO}


def test_repo_map_ambiguous_ownership_raises(tmp_path: Path) -> None:
    # Conditional declarations are unioned, so a subproject sourced from two
    # repositories has no single owner; that must fail rather than pick one.
    _write(
        tmp_path / "CMakeLists.txt",
        """
if(WIN32)
  therock_cmake_subproject_declare(split
    EXTERNAL_SOURCE_DIR ${THEROCK_SOURCE_DIR}/rocm-libraries/projects/x)
else()
  therock_cmake_subproject_declare(split
    EXTERNAL_SOURCE_DIR ${THEROCK_SOURCE_DIR}/rocm-systems/projects/x)
endif()
""",
    )

    result = RepositoryAnalyzer(
        tmp_path, tracked_cmake_files={Path("CMakeLists.txt")}
    ).analyze()

    with pytest.raises(AnalysisError, match="multiple repositories"):
        result.build_repo_map({"rocm-libraries", "rocm-systems"})


def test_repo_map_on_real_tree() -> None:
    # Every committed graph node must resolve to a real repository. A node that
    # silently fell back to TheRock would have its tests attributed to the wrong
    # repository, so the anchors below cover each submodule-shaped case.
    repo_root = Path(__file__).resolve().parents[2]
    committed = repo_root / "test_tools" / "therock_consumer_graph.json"
    if not (
        (repo_root / ".git").exists()
        and committed.exists()
        and (repo_root / "CMakeLists.txt").exists()
    ):
        pytest.skip("not a TheRock git checkout with a committed consumer graph")

    submodules = list_submodule_paths(repo_root)
    repo_map = RepositoryAnalyzer(repo_root).analyze().build_repo_map(submodules)

    missing = sorted(k for k in load_consumer_graph(committed) if k not in repo_map)
    assert missing == [], f"graph nodes with no owning repository: {missing}"
    assert set(repo_map.values()) <= submodules | {THEROCK_REPO}

    assert repo_map["rocblas"] == "rocm-libraries"
    assert repo_map["hipblaslt"] == "rocm-libraries"
    # RCCL and hipFile live in rocm-systems despite being library-shaped.
    assert repo_map["rccl"] == "rocm-systems"
    assert repo_map["hipfile"] == "rocm-systems"
    assert repo_map["hip-clr"] == "rocm-systems"
    # Sourced through therock_enable_external_source() rather than a literal path.
    assert repo_map["amd-dbgapi"] == "rocm-systems"
    assert repo_map["rocr-debug-agent"] == "rocm-systems"
    # Declared against TheRock's autotools wrapper, which holds the submodule.
    assert repo_map["rocgdb"] == "debug-tools/rocgdb/source"
    assert repo_map["libhipcxx"] == "math-libs/libhipcxx"
    assert repo_map["amd-llvm"] == "compiler/amd-llvm"
    # Fetched into the build tree, so owned by no submodule.
    assert repo_map["therock-googletest"] == THEROCK_REPO


def test_static_graph_superset_of_committed() -> None:
    # Run against the real tree: the generated graph must not miss any node or edge
    # in the committed graph (the superset invariant). Skips outside a git checkout.
    repo_root = Path(__file__).resolve().parents[2]
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

    # A reference-only node/edge means the committed graph holds something the
    # parser does not (a stale committed edge, or a parser gap). While the committed
    # graph is still emitted by CMake, regenerate it via the cmake emit path
    # (therock_emit_consumer_graph) — NOT generate_consumer_graph.py, which writes
    # the conservative superset and would itself introduce drift.
    regen_hint = (
        "committed graph has nodes/edges the parser does not; regenerate it via the "
        "cmake emit path (not generate_consumer_graph.py, which writes the superset)"
    )
    assert comparison.reference_only_nodes == [], regen_hint
    assert comparison.reference_only_edges == [], regen_hint
    assert result.unreachable_declaration_files == set()
    assert result.dangling_dependencies() == {}


def test_subtree_map_on_real_tree() -> None:
    # Real-tree anchors: fan-out, name skew, and a non-projects/ prefix all resolve.
    # Skips outside a git checkout.
    repo_root = Path(__file__).resolve().parents[2]
    if not ((repo_root / ".git").exists() and (repo_root / "CMakeLists.txt").exists()):
        pytest.skip("not a TheRock git checkout")

    subtree_map = RepositoryAnalyzer(repo_root).analyze().build_subtree_map()

    assert subtree_map.get("projects/clr") == ["hip-clr", "ocl-clr"]
    assert subtree_map.get("projects/composablekernel") == ["composable_kernel"]
    assert "shared/rocroller" in subtree_map
    # Sourced via therock_enable_external_source() rather than a direct
    # ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/... literal; modeling that command is
    # what brings it into the map.
    assert subtree_map.get("projects/rocdbgapi") == ["amd-dbgapi"]


def test_subtree_map_covers_direct_source_dirs() -> None:
    # Coverage cross-check: regex the tracked CMake for every EXTERNAL_SOURCE_DIR
    # written directly under a source root and assert subtree_map covers each,
    # catching a silently-shrinking map. Variable-sourced dirs are not matched.
    repo_root = Path(__file__).resolve().parents[2]
    if not ((repo_root / ".git").exists() and (repo_root / "CMakeLists.txt").exists()):
        pytest.skip("not a TheRock git checkout")

    # Capture the root-relative subtree of each directly-rooted EXTERNAL_SOURCE_DIR.
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
