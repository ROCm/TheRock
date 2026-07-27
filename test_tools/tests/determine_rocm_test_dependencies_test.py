# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the dynamic consumer-graph test-selection tool.

These tests are hermetic: each builds a self-contained ``--therock-dir`` on
disk carrying

  * ``build/therock_consumer_graph.json`` — the dynamic (never-committed)
    consumer graph the tool loads,
  * a minimal ``BUILD_TOPOLOGY.toml`` — build_stages / artifact_groups /
    artifacts plus the per-artifact ``test_include`` / ``test_exclude`` /
    ``test_fanout_all`` and ``[artifacts.<a>.test_overrides.<sub>]`` keys, and
  * ``artifact-<name>.toml`` (and ``base/artifact.toml``) descriptors whose
    ``components."<path>/stage"`` keys map subprojects to artifacts (and thus to
    build stages).

No test here depends on the committed repo graph. One integration test class
(``TestRealTopologyStageMap``) reads the committed BUILD_TOPOLOGY.toml +
artifact tomls to assert the ``base/artifact.toml`` stage-resolution fix, since
that is about derivation from committed files, not test selection.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THEROCK_DIR = Path(__file__).parent.parent.parent
SCRIPT = Path(__file__).parent.parent / "determine_rocm_test_dependencies.py"

sys.path.insert(0, str(THEROCK_DIR / "test_tools"))

from determine_rocm_test_dependencies import (  # noqa: E402
    _build_subproject_maps,
    _load_consumer_graph,
    get_subprojects_to_test,
    list_subprojects,
)

# ---------------------------------------------------------------------------
# Fixture graph.
#
# Two build stages (via two artifacts), plus a foundational "base" artifact:
#
#   stage math-libs (artifact `prim`, `blas`):
#     hipcub    -> [rocprim]                 (flat prim include)
#     rocthrust -> [rocprim]                 (flat prim include)
#     rocprim   -> []
#     rocsolver -> [hipblas]                 (blas sub-table: rocsolver)
#     rocroller -> [hipblaslt]               (blas sub-table: rocroller)
#     rocblas   -> [hipblas, rocsolver]      (same-stage consumers)
#     hipblas, hipblaslt -> []
#
#   stage debug-tools (artifact `dbg`):
#     amd-dbgapi        -> [rocgdb, rocr-debug-agent, rocr-debug-agent-tests]
#     rocr-debug-agent  -> [rocr-debug-agent-tests]
#     rocgdb, rocr-debug-agent-tests -> []
#     ROCR-Runtime      -> [rocgdb]          (hyphenated + mixed-case)
#
#   artifact `base` (stage runtime): test_fanout_all = true
#     rocm-core -> [rocprim, rocgdb]         (consumers span BOTH stages)
# ---------------------------------------------------------------------------
_GRAPH = {
    "hipcub": {"consumers": []},
    "rocthrust": {"consumers": []},
    "rocprim": {"consumers": ["hipcub", "rocthrust"]},
    "rocsolver": {"consumers": ["hipblas"]},
    "rocroller": {"consumers": ["hipblaslt"]},
    "rocblas": {"consumers": ["hipblas", "rocsolver"]},
    "hipblas": {"consumers": []},
    "hipblaslt": {"consumers": []},
    "amd-dbgapi": {
        "consumers": ["rocgdb", "rocr-debug-agent", "rocr-debug-agent-tests"],
    },
    "rocr-debug-agent": {"consumers": ["rocr-debug-agent-tests"]},
    "rocr-debug-agent-tests": {"consumers": []},
    "rocgdb": {"consumers": []},
    "rocr-runtime": {"consumers": ["rocgdb"]},
    "rocm-core": {"consumers": ["rocprim", "rocgdb"]},
}

# subproject -> owning artifact (artifact name == artifact-<name>.toml stem).
_ARTIFACT_OF = {
    "prim": ["hipcub", "rocthrust", "rocprim"],
    "blas": ["rocsolver", "rocroller", "rocblas", "hipblas", "hipblaslt"],
    "dbg": [
        "amd-dbgapi",
        "rocr-debug-agent",
        "rocr-debug-agent-tests",
        "rocgdb",
        "rocr-runtime",
    ],
}

# artifact -> build stage. `prim` and `blas` share the same stage so the
# same-stage cut is exercised across artifact boundaries within one stage.
_STAGE_OF_ARTIFACT = {
    "prim": "math-libs",
    "blas": "math-libs",
    "dbg": "debug-tools",
    "base": "runtime",
}

# Case-preserving source dir segment for each subproject (mirrors the real
# tomls, e.g. ROCR-Runtime). The tool lowercases when parsing, so casing here is
# just cosmetic; ROCR-Runtime documents the hyphen + mixed-case path shape.
_SUBPROJECT_DIR = {
    "hipcub": "hipCUB",
    "rocthrust": "rocThrust",
    "rocprim": "rocPRIM",
    "rocsolver": "rocSOLVER",
    "rocroller": "rocRoller",
    "rocblas": "rocBLAS",
    "hipblas": "hipBLAS",
    "hipblaslt": "hipBLASLt",
    "amd-dbgapi": "amd-dbgapi",
    "rocr-debug-agent": "rocr-debug-agent",
    "rocr-debug-agent-tests": "rocr-debug-agent-tests",
    "rocgdb": "rocgdb",
    "rocr-runtime": "ROCR-Runtime",
}


def _write_build_topology(root: Path) -> None:
    """Write a minimal BUILD_TOPOLOGY.toml carrying the test_* override keys.

    Each artifact gets its own artifact_group and each stage lists exactly the
    groups that belong to it, so get_stage_for_artifact() resolves correctly.
    """
    lines: list[str] = []

    # Build stages: one group per artifact, artifact name == group name here.
    stage_to_groups: dict[str, list[str]] = {}
    for artifact, stage in _STAGE_OF_ARTIFACT.items():
        stage_to_groups.setdefault(stage, []).append(artifact)
    for stage, groups in stage_to_groups.items():
        lines.append(f"[build_stages.{stage}]")
        lines.append("artifact_groups = [" + ", ".join(f'"{g}"' for g in groups) + "]")
        lines.append("")

    for artifact in _STAGE_OF_ARTIFACT:
        lines.append(f"[artifact_groups.{artifact}]")
        lines.append("")

    # Artifacts + their overrides.
    # `prim`: flat include (hipcub->rocprim, rocthrust->rocprim identical).
    lines += [
        "[artifacts.prim]",
        'artifact_group = "prim"',
        'test_include = ["rocprim"]',
        "",
    ]
    # `blas`: per-subproject sub-tables so rocsolver/rocroller includes do NOT
    # over-apply to siblings (open item A).
    lines += [
        "[artifacts.blas]",
        'artifact_group = "blas"',
        "[artifacts.blas.test_overrides.rocsolver]",
        'test_include = ["hipblas"]',
        "[artifacts.blas.test_overrides.rocroller]",
        'test_include = ["hipblaslt"]',
        "",
    ]
    lines += [
        "[artifacts.dbg]",
        'artifact_group = "dbg"',
        "",
    ]
    # `base`: foundational fan-out.
    lines += [
        "[artifacts.base]",
        'artifact_group = "base"',
        "test_fanout_all = true",
        "",
    ]

    (root / "BUILD_TOPOLOGY.toml").write_text("\n".join(lines))


def _write_artifact_tomls(root: Path) -> None:
    """Write artifact-<name>.toml descriptors + base/artifact.toml."""
    for artifact, subs in _ARTIFACT_OF.items():
        toml_lines = []
        for sub in subs:
            seg = _SUBPROJECT_DIR.get(sub, sub)
            toml_lines.append(f'[components.lib."libs/{seg}/stage"]')
        (root / f"artifact-{artifact}.toml").write_text("\n".join(toml_lines) + "\n")

    # base/artifact.toml is UNSUFFIXED — the plain artifact-*.toml glob misses
    # it; the tool must special-case it (base stage-resolution fix).
    base_dir = root / "base"
    base_dir.mkdir(exist_ok=True)
    (base_dir / "artifact.toml").write_text(
        '[components.lib."base/rocm-core/stage"]\n'
    )


def _write_graph(root: Path, graph: dict | None = None) -> None:
    """Write the dynamic consumer graph to build/therock_consumer_graph.json."""
    build_dir = root / "build"
    build_dir.mkdir(exist_ok=True)
    (build_dir / "therock_consumer_graph.json").write_text(
        json.dumps(_GRAPH if graph is None else graph, indent=2)
    )


def _make_fixture(graph: dict | None = None) -> Path:
    """Create a full hermetic --therock-dir fixture; return its path."""
    root = Path(tempfile.mkdtemp())
    _write_graph(root, graph)
    _write_build_topology(root)
    _write_artifact_tomls(root)
    return root


class _FixtureTestCase(unittest.TestCase):
    """Base class managing a hermetic fixture dir per test."""

    def setUp(self) -> None:
        self.root = _make_fixture()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TestGraphLoading(_FixtureTestCase):
    def test_load_dynamic_graph_from_build_dir(self) -> None:
        graph = _load_consumer_graph(self.root)
        self.assertIn("rocblas", graph)
        self.assertIn("hipblas", graph["rocblas"]["consumers"])

    def test_missing_graph_raises(self) -> None:
        empty = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(FileNotFoundError):
                _load_consumer_graph(empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_stage_and_artifact_maps(self) -> None:
        stage_of, artifact_of = _build_subproject_maps(self.root)
        # prim + blas share stage math-libs; dbg is a separate stage.
        self.assertEqual(stage_of["rocprim"], "math-libs")
        self.assertEqual(stage_of["rocblas"], "math-libs")
        self.assertEqual(stage_of["rocgdb"], "debug-tools")
        self.assertEqual(artifact_of["hipcub"], "prim")
        self.assertEqual(artifact_of["rocblas"], "blas")
        # base/artifact.toml (unsuffixed) resolves rocm-core.
        self.assertEqual(artifact_of["rocm-core"], "base")


class TestSameStageCut(_FixtureTestCase):
    def test_selects_same_stage_consumer(self) -> None:
        # rocblas -> hipblas, rocsolver; all in stage math-libs.
        result = get_subprojects_to_test(["rocblas"], self.root)
        self.assertIn("rocblas", result)
        self.assertIn("hipblas", result)
        self.assertIn("rocsolver", result)

    def test_cross_stage_consumer_is_cut(self) -> None:
        # rocr-runtime (stage debug-tools) -> rocgdb (also debug-tools) is kept,
        # but a cross-stage consumer must NOT be pulled in. Here rocm-core (base
        # fanout) is the fanout case; for the plain same-stage cut we assert that
        # amd-dbgapi's consumers stay within its own stage and no math-libs
        # subproject leaks in.
        result = get_subprojects_to_test(["amd-dbgapi"], self.root)
        self.assertIn("rocgdb", result)
        self.assertIn("rocr-debug-agent", result)
        self.assertNotIn("rocprim", result)
        self.assertNotIn("hipblas", result)

    def test_leaf_selects_only_itself(self) -> None:
        result = get_subprojects_to_test(["hipblas"], self.root)
        self.assertEqual(result, {"hipblas"})


class TestFanoutAll(_FixtureTestCase):
    def test_base_fanout_selects_all_consumers_cross_stage(self) -> None:
        # rocm-core is on artifact `base` (test_fanout_all=true). Its graph
        # consumers span BOTH stages: rocprim (math-libs) + rocgdb (debug-tools).
        # Fanout bypasses the same-stage cut, so both must be selected.
        result = get_subprojects_to_test(["rocm-core"], self.root)
        self.assertEqual(result, {"rocm-core", "rocprim", "rocgdb"})


class TestFlatArtifactInclude(_FixtureTestCase):
    def test_prim_flat_include_applies_to_hipcub(self) -> None:
        # Flat [artifacts.prim] test_include = [rocprim] applies to hipcub.
        result = get_subprojects_to_test(["hipcub"], self.root)
        self.assertIn("rocprim", result)

    def test_prim_flat_include_applies_to_rocthrust(self) -> None:
        # ...and to rocthrust (same artifact, same flat include — no drift).
        result = get_subprojects_to_test(["rocthrust"], self.root)
        self.assertIn("rocprim", result)


class TestOpenItemA(_FixtureTestCase):
    """Per-subproject sub-tables must not over-apply to siblings."""

    def test_rocsolver_include_does_not_pull_rocroller_include(self) -> None:
        # rocsolver sub-table -> hipblas; rocroller sub-table -> hipblaslt.
        # A rocsolver change must include hipblas but NOT hipblaslt.
        result = get_subprojects_to_test(["rocsolver"], self.root)
        self.assertIn("hipblas", result)
        self.assertNotIn("hipblaslt", result)

    def test_rocroller_include_does_not_pull_rocsolver_include(self) -> None:
        # The reverse: a rocroller change must include hipblaslt but NOT hipblas.
        result = get_subprojects_to_test(["rocroller"], self.root)
        self.assertIn("hipblaslt", result)
        self.assertNotIn("hipblas", result)

    def test_sibling_without_override_gets_no_include(self) -> None:
        # rocblas is in artifact `blas` but has no sub-table; it must not inherit
        # rocsolver's or rocroller's includes. Its selections come only from the
        # same-stage cut (hipblas, rocsolver).
        result = get_subprojects_to_test(["rocblas"], self.root)
        self.assertNotIn("hipblaslt", result)


class TestExclude(unittest.TestCase):
    """test_exclude is applied LAST and is order-independent."""

    def _fixture_with_exclude(self) -> Path:
        root = _make_fixture()
        # Add a rocblas sub-table that excludes rocsolver even though it is a
        # same-stage consumer, and hipcub's flat prim include is unaffected.
        topo = (root / "BUILD_TOPOLOGY.toml").read_text()
        topo += (
            "\n[artifacts.blas.test_overrides.rocblas]\n"
            'test_exclude = ["rocsolver"]\n'
        )
        (root / "BUILD_TOPOLOGY.toml").write_text(topo)
        return root

    def test_exclude_applied_last(self) -> None:
        root = self._fixture_with_exclude()
        try:
            result = get_subprojects_to_test(["rocblas"], root)
            self.assertIn("rocblas", result)
            self.assertIn("hipblas", result)
            # rocsolver is a same-stage consumer but excluded LAST.
            self.assertNotIn("rocsolver", result)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_exclude_order_independent_across_changed_projects(self) -> None:
        # rocblas's sub-table excludes rocsolver. When rocblas AND rocsolver
        # change together, rocsolver is added by TWO independent paths (it is a
        # same-stage consumer of rocblas AND a changed project selecting itself),
        # yet the exclude — applied LAST in a single pass over all changed
        # projects — must still win regardless of input order. This guards
        # against exclude being silently undone by another changed project.
        root = self._fixture_with_exclude()
        try:
            ab = get_subprojects_to_test(["rocblas", "rocsolver"], root)
            ba = get_subprojects_to_test(["rocsolver", "rocblas"], root)
            self.assertEqual(ab, ba)
            self.assertNotIn("rocsolver", ab)
            # hipblas is still selected (rocblas same-stage consumer + rocsolver
            # sub-table include), proving only rocsolver was pruned.
            self.assertIn("hipblas", ab)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestNameNormalization(_FixtureTestCase):
    def test_hyphenated_names(self) -> None:
        result = get_subprojects_to_test(["amd-dbgapi"], self.root)
        self.assertIn("amd-dbgapi", result)
        self.assertIn("rocgdb", result)
        self.assertIn("rocr-debug-agent", result)
        self.assertIn("rocr-debug-agent-tests", result)

    def test_mixed_case_hyphenated_normalized(self) -> None:
        # ROCR-Runtime -> lowercased rocr-runtime, whose same-stage consumer is
        # rocgdb.
        result = get_subprojects_to_test(["ROCR-Runtime"], self.root)
        self.assertIn("rocr-runtime", result)
        self.assertIn("rocgdb", result)

    def test_case_insensitive_input(self) -> None:
        result = get_subprojects_to_test(["rocBLAS"], self.root)
        self.assertIn("rocblas", result)
        self.assertIn("hipblas", result)


class TestUnknownProject(_FixtureTestCase):
    def test_unknown_project_selects_only_itself(self) -> None:
        result = get_subprojects_to_test(["nonexistent-lib"], self.root)
        self.assertEqual(result, {"nonexistent-lib"})

    def test_unknown_project_warns_via_cli(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--therock-dir",
                str(self.root),
                "--changed-projects",
                "totallybogus",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Warning: unrecognized project", proc.stderr)
        self.assertIn("totallybogus", proc.stderr)


class TestCliBehaviors(_FixtureTestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--therock-dir", str(self.root), *args],
            capture_output=True,
            text=True,
        )

    def test_comma_separated_input(self) -> None:
        proc = self._run("--changed-projects", "rocblas,hipcub")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        projects = json.loads(proc.stdout.strip())
        self.assertIn("rocblas", projects)
        self.assertIn("hipblas", projects)  # rocblas same-stage consumer
        self.assertIn("rocprim", projects)  # hipcub flat include

    def test_projects_prefix_stripped(self) -> None:
        proc = self._run("--changed-projects", "projects/rocblas")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        projects = json.loads(proc.stdout.strip())
        self.assertIn("rocblas", projects)
        self.assertIn("hipblas", projects)

    def test_empty_changed_projects_outputs_wildcard(self) -> None:
        proc = self._run()
        self.assertEqual(proc.stdout.strip(), "*")

    def test_empty_flag_outputs_wildcard(self) -> None:
        proc = self._run("--changed-projects")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "*")

    def test_format_list(self) -> None:
        proc = self._run("--changed-projects", "hipcub", "--format", "list")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.strip().splitlines()
        self.assertIn("hipcub", lines)
        self.assertIn("rocprim", lines)

    def test_gha_output_format(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as handle:
            output_file = handle.name
        try:
            env = os.environ.copy()
            env["GITHUB_OUTPUT"] = output_file
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--therock-dir",
                    str(self.root),
                    "--changed-projects",
                    "rocblas",
                    "--gha-output",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            content = Path(output_file).read_text()
            self.assertIn("projects_to_test=", content)
            self.assertIn("rocblas", content)
            self.assertIn(",", content)  # comma-separated, not space
        finally:
            os.unlink(output_file)


class TestListSubprojects(_FixtureTestCase):
    def test_list_names(self) -> None:
        names = list_subprojects(self.root, show_deps=False)
        self.assertIn("rocblas", names)
        self.assertIn("rocprim", names)

    def test_list_with_deps(self) -> None:
        deps = list_subprojects(self.root, show_deps=True)
        self.assertIn("hipblas", deps["rocblas"])
        self.assertIn("rocsolver", deps["rocblas"])
        self.assertEqual(deps["hipblas"], "empty")


class TestRealTopologyStageMap(unittest.TestCase):
    """Integration: the committed base/artifact.toml stage-resolution fix.

    This reads the REAL committed BUILD_TOPOLOGY.toml + artifact tomls (no cmake
    configure, no consumer graph). It asserts that the four foundational deps
    packaged by the unsuffixed base/artifact.toml resolve a build stage — the
    gap the tool fixes by special-casing base/artifact.toml.
    """

    def test_foundational_deps_resolve_a_stage(self) -> None:
        stage_of, artifact_of = _build_subproject_maps(THEROCK_DIR)
        for sub in ("rocm-core", "rocm-cmake", "half", "rocprofiler-register"):
            self.assertEqual(
                artifact_of.get(sub),
                "base",
                f"{sub} should map to artifact 'base'",
            )
            self.assertIsNotNone(
                stage_of.get(sub),
                f"{sub} should resolve a build stage via artifact 'base'",
            )


if __name__ == "__main__":
    unittest.main()
