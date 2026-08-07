# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the committed-graph test-selection tool.

These tests target the CURRENT design (committed consumer graph + hand-maintained
``test_policies.toml`` + hop-distance selection), NOT the superseded dynamic
build-stage-cut / BUILD_TOPOLOGY-override design.

Every test is hermetic: it builds a self-contained ``--therock-dir`` on disk
carrying

  * ``test_tools/therock_consumer_graph.json`` — the committed consumer graph the
    tool loads script-relative (or under ``--therock-dir/test_tools/``), and
  * ``test_tools/test_policies.toml`` — the hand-maintained policy file
    (``[component.<name>]`` tables with ``level`` / ``test_include`` /
    ``test_exclude``).

No test depends on the repo's real committed graph/policy. The fixture graph
below is small and purpose-built to exercise each selection rule, including the
amdsmi -> rdc cross-stage regression case and the finding-#8 (no build/ dir) case.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

THEROCK_DIR = Path(__file__).parent.parent.parent
SCRIPT = Path(__file__).parent.parent / "determine_rocm_test_dependencies.py"

sys.path.insert(0, str(THEROCK_DIR / "test_tools"))

from determine_rocm_test_dependencies import (  # noqa: E402
    explain_component,
    get_subprojects_to_test,
    list_subprojects,
    validate_policies,
)

# ---------------------------------------------------------------------------
# Fixture graph (reverse / consumer edges only, all lowercase). The
# rocroller -> hipblaslt -> miopen chain exercises direct (level 4) vs transitive
# (level 3) walks; amdsmi -> rdc is a direct edge; rocm-core reaches a broad
# closure.
#
#   amdsmi     -> [rdc, hipblaslt]
#   rocroller  -> [hipblaslt]
#   hipblaslt  -> [miopen, rocblas]    (miopen is indirect from rocroller)
#   miopen     -> []
#   rocblas    -> []
#   rdc        -> []
#   rocm-core  -> [amdsmi, rocroller]
#   rocgdb     -> []
#   amd-dbgapi -> [rocgdb]
# ---------------------------------------------------------------------------
_GRAPH = {
    "amdsmi": {"consumers": ["rdc", "hipblaslt"]},
    "rocroller": {"consumers": ["hipblaslt"]},
    "hipblaslt": {"consumers": ["miopen", "rocblas"]},
    "miopen": {"consumers": []},
    "rocblas": {"consumers": []},
    "rdc": {"consumers": []},
    "rocm-core": {"consumers": ["amdsmi", "rocroller"]},
    "rocgdb": {"consumers": []},
    "amd-dbgapi": {"consumers": ["rocgdb"]},
}

# Hand-maintained policy fixture.
#   * rocm-core at level 3 -> transitive closure (the level-honored case).
#   * amdsmi test_include adds hip-tests (a test-only value NOT in the graph,
#     proving include values need not be graph keys) and test_exclude prunes
#     rocblas (non-vacuous: the level-3 walk from amdsmi would otherwise reach it
#     via hipblaslt).
#   * rocgdb test_include adds rocgdb-cpu (another test-only, non-graph value).
#   * amd-dbgapi keyed with no body -> defaults (level 4, no include/exclude).
_POLICIES = """\
[component.rocm-core]
level = 3

[component.amdsmi]
level = 3
test_include = ["hip-tests"]
test_exclude = ["rocblas"]

[component.rocgdb]
test_include = ["rocgdb-cpu"]

[component.amd-dbgapi]
"""


def _write_fixture(root: Path, graph: dict | None = None, policies: str | None = None):
    """Populate ``root/test_tools/`` with a graph + policy file."""
    test_tools = root / "test_tools"
    test_tools.mkdir(parents=True, exist_ok=True)
    (test_tools / "therock_consumer_graph.json").write_text(
        json.dumps(_GRAPH if graph is None else graph, indent=2, sort_keys=True) + "\n"
    )
    (test_tools / "test_policies.toml").write_text(
        _POLICIES if policies is None else policies
    )


def _make_fixture(graph: dict | None = None, policies: str | None = None) -> Path:
    """Create a hermetic --therock-dir fixture; return its path."""
    root = Path(tempfile.mkdtemp())
    _write_fixture(root, graph, policies)
    return root


class _FixtureTestCase(unittest.TestCase):
    """Base class managing a hermetic fixture dir per test."""

    def setUp(self) -> None:
        self.root = _make_fixture()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Hop-distance selection: level 5 = self, level 4 = self+direct, level 3 = all.
# ---------------------------------------------------------------------------
class TestHopDistanceSelection(_FixtureTestCase):
    def test_level_5_selects_self_only(self) -> None:
        result = get_subprojects_to_test(["hipblaslt"], self.root, level=5)
        self.assertEqual(result, {"hipblaslt"})

    def test_level_4_selects_self_plus_direct_consumers(self) -> None:
        # hipblaslt -> {miopen, rocblas} directly; NOT their consumers (none here,
        # but rocroller must not appear — it is upstream, not a consumer).
        result = get_subprojects_to_test(["hipblaslt"], self.root, level=4)
        self.assertEqual(result, {"hipblaslt", "miopen", "rocblas"})

    def test_level_4_stops_at_one_hop(self) -> None:
        # rocroller -> hipblaslt (direct). miopen/rocblas are INDIRECT (via
        # hipblaslt) so a level-4 walk must NOT reach them.
        result = get_subprojects_to_test(["rocroller"], self.root, level=4)
        self.assertEqual(result, {"rocroller", "hipblaslt"})
        self.assertNotIn("miopen", result)

    def test_level_3_selects_transitive_closure(self) -> None:
        # rocroller -> hipblaslt -> {miopen, rocblas}. Level 3 reaches all of them.
        result = get_subprojects_to_test(["rocroller"], self.root, level=3)
        self.assertEqual(result, {"rocroller", "hipblaslt", "miopen", "rocblas"})

    def test_default_level_equals_level_4(self) -> None:
        # No policy table for hipblaslt -> default level 4.
        default = get_subprojects_to_test(["hipblaslt"], self.root)
        explicit = get_subprojects_to_test(["hipblaslt"], self.root, level=4)
        self.assertEqual(default, explicit)

    def test_levels_1_and_2_walk_transitively_like_level_3(self) -> None:
        # Levels 1 and 2 are stricter tiers with no separate test-tier output
        # yet, so they walk transitively (same depth as level 3). This
        # keeps the accepted range [1, 5] and the engine in agreement — a level
        # the validator blesses is never silently coerced to the default.
        transitive = get_subprojects_to_test(["rocroller"], self.root, level=3)
        self.assertEqual(
            get_subprojects_to_test(["rocroller"], self.root, level=2), transitive
        )
        self.assertEqual(
            get_subprojects_to_test(["rocroller"], self.root, level=1), transitive
        )


# ---------------------------------------------------------------------------
# A direct consumer is selected regardless of build stage (amdsmi -> rdc).
# ---------------------------------------------------------------------------
class TestAmdsmiRdcRegression(_FixtureTestCase):
    def test_amdsmi_selects_rdc_at_level_4(self) -> None:
        result = get_subprojects_to_test(["amdsmi"], self.root, level=4)
        self.assertIn("rdc", result)

    def test_amdsmi_selects_rdc_at_default_level(self) -> None:
        result = get_subprojects_to_test(["amdsmi"], self.root)
        self.assertIn("rdc", result)

    def test_rdc_is_a_direct_consumer_not_transitive(self) -> None:
        # rdc is reached at one hop, not via a transitive walk.
        one_hop = get_subprojects_to_test(["amdsmi"], self.root, level=4)
        self.assertIn("rdc", one_hop)


# ---------------------------------------------------------------------------
# Policy `level` honored: the changed component's own table sets the walk depth.
# ---------------------------------------------------------------------------
class TestPolicyLevelHonored(_FixtureTestCase):
    def test_level_3_component_walks_transitively(self) -> None:
        # rocm-core is level=3 in the policy -> transitive closure over the graph.
        result = get_subprojects_to_test(["rocm-core"], self.root)
        self.assertEqual(
            result,
            {
                "rocm-core",
                "amdsmi",
                "rocroller",
                "rdc",
                "hipblaslt",
                "miopen",
                "rocblas",
            },
        )

    def test_default_component_walks_one_hop(self) -> None:
        # rocroller has no policy table -> default level 4 -> one hop only.
        result = get_subprojects_to_test(["rocroller"], self.root)
        self.assertEqual(result, {"rocroller", "hipblaslt"})

    def test_cli_level_overrides_policy_level(self) -> None:
        # rocm-core policy is level 3, but --level 5 must win (self only).
        result = get_subprojects_to_test(["rocm-core"], self.root, level=5)
        self.assertEqual(result, {"rocm-core"})


# ---------------------------------------------------------------------------
# test_include additive (incl. a test-only value not in the graph).
# ---------------------------------------------------------------------------
class TestIncludeAdditive(_FixtureTestCase):
    def test_include_adds_test_only_value_absent_from_graph(self) -> None:
        # amdsmi's test_include = ["hip-tests"]; hip-tests is NOT a graph key,
        # proving include values need not be graph keys.
        self.assertNotIn("hip-tests", _GRAPH)
        result = get_subprojects_to_test(["amdsmi"], self.root)
        self.assertIn("hip-tests", result)

    def test_include_is_unioned_onto_walk(self) -> None:
        # The walk result is preserved AND the include is added.
        result = get_subprojects_to_test(["amdsmi"], self.root)
        self.assertIn("amdsmi", result)  # self (from walk)
        self.assertIn("rdc", result)  # direct consumer (from walk)
        self.assertIn("hip-tests", result)  # from include


# ---------------------------------------------------------------------------
# test_exclude subtractive-LAST + order-independent + non-vacuous.
# ---------------------------------------------------------------------------
class TestExcludeSubtractiveLast(_FixtureTestCase):
    def test_exclude_prunes_a_project_the_walk_would_select(self) -> None:
        # amdsmi is level=3; the transitive walk reaches rocblas (via hipblaslt),
        # but test_exclude=["rocblas"] prunes it LAST. Non-vacuous: hipblaslt (the
        # path to rocblas) IS still selected, so only rocblas was removed.
        result = get_subprojects_to_test(["amdsmi"], self.root)
        self.assertIn("hipblaslt", result)
        self.assertNotIn("rocblas", result)

    def test_exclude_order_independent(self) -> None:
        # amdsmi excludes rocblas. When amdsmi AND rocblas change together,
        # rocblas is added by TWO paths (self-selection of the changed project
        # `rocblas`, and amdsmi's transitive walk), yet exclude — applied LAST in
        # one pass — must still win regardless of input order.
        ab = get_subprojects_to_test(["amdsmi", "rocblas"], self.root)
        ba = get_subprojects_to_test(["rocblas", "amdsmi"], self.root)
        self.assertEqual(ab, ba)
        self.assertNotIn("rocblas", ab)
        # hipblaslt still present, proving only rocblas was pruned.
        self.assertIn("hipblaslt", ab)


# ---------------------------------------------------------------------------
# Name normalization + projects/ prefix strip + comma-split input (CLI).
# ---------------------------------------------------------------------------
class TestNameNormalization(_FixtureTestCase):
    def test_case_insensitive_input(self) -> None:
        result = get_subprojects_to_test(["AMDSMI"], self.root, level=4)
        self.assertIn("amdsmi", result)
        self.assertIn("rdc", result)

    def test_hyphenated_name(self) -> None:
        result = get_subprojects_to_test(["amd-dbgapi"], self.root, level=4)
        self.assertIn("amd-dbgapi", result)
        self.assertIn("rocgdb", result)

    def test_mixed_case_hyphenated_normalized(self) -> None:
        result = get_subprojects_to_test(["Amd-DbgApi"], self.root, level=4)
        self.assertIn("amd-dbgapi", result)
        self.assertIn("rocgdb", result)


class TestCliInputParsing(_FixtureTestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--therock-dir", str(self.root), *args],
            capture_output=True,
            text=True,
        )

    def test_projects_prefix_stripped(self) -> None:
        proc = self._run("--changed-projects", "projects/amdsmi", "--level", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        projects = json.loads(proc.stdout.strip())
        self.assertIn("amdsmi", projects)
        self.assertIn("rdc", projects)

    def test_comma_separated_input(self) -> None:
        proc = self._run("--changed-projects", "amdsmi,rocroller", "--level", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        projects = json.loads(proc.stdout.strip())
        self.assertIn("amdsmi", projects)
        self.assertIn("rdc", projects)  # amdsmi direct consumer
        self.assertIn("rocroller", projects)
        self.assertIn("hipblaslt", projects)  # rocroller direct consumer

    def test_empty_changed_projects_outputs_wildcard(self) -> None:
        proc = self._run()
        self.assertEqual(proc.stdout.strip(), "*")

    def test_empty_flag_outputs_wildcard(self) -> None:
        proc = self._run("--changed-projects")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "*")

    def test_format_list(self) -> None:
        proc = self._run(
            "--changed-projects", "amdsmi", "--level", "4", "--format", "list"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.strip().splitlines()
        self.assertIn("amdsmi", lines)
        self.assertIn("rdc", lines)


# ---------------------------------------------------------------------------
# Unknown changed project warns and selects only itself.
# ---------------------------------------------------------------------------
class TestUnknownProject(_FixtureTestCase):
    def test_unknown_project_selects_only_itself(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = get_subprojects_to_test(["nonexistent-lib"], self.root)
        self.assertEqual(result, {"nonexistent-lib"})
        self.assertIn("unrecognized project", buf.getvalue())

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


# ---------------------------------------------------------------------------
# --gha-output writes projects_to_test=... to $GITHUB_OUTPUT.
# ---------------------------------------------------------------------------
class TestGhaOutput(_FixtureTestCase):
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
                    "amdsmi",
                    "--level",
                    "4",
                    "--gha-output",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            content = Path(output_file).read_text()
            self.assertIn("projects_to_test=", content)
            self.assertIn("amdsmi", content)
            self.assertIn("rdc", content)
            self.assertIn(",", content)  # comma-separated, not space
        finally:
            os.unlink(output_file)


# ---------------------------------------------------------------------------
# --validate-policies: passes on valid fixture; fails on bogus KEY; does NOT
# fail on include VALUE absent from graph (the corrected rule).
# ---------------------------------------------------------------------------
class TestValidatePolicies(_FixtureTestCase):
    def test_valid_fixture_passes(self) -> None:
        ok, messages = validate_policies(self.root)
        self.assertTrue(ok, "\n".join(messages))
        self.assertTrue(any(m.startswith("OK:") for m in messages))

    def test_test_only_include_value_does_not_fail(self) -> None:
        # hip-tests / rocgdb-cpu are test-only include VALUES absent from the
        # graph. Per the corrected rule they must NOT fail validation; they are
        # noted informationally.
        ok, messages = validate_policies(self.root)
        self.assertTrue(ok)
        joined = "\n".join(messages)
        self.assertIn("hip-tests", joined)
        self.assertIn("rocgdb-cpu", joined)

    def test_bogus_component_key_fails(self) -> None:
        bogus = _POLICIES + "\n[component.does-not-exist]\n"
        root = _make_fixture(policies=bogus)
        try:
            ok, messages = validate_policies(root)
            self.assertFalse(ok)
            joined = "\n".join(messages)
            self.assertIn("does-not-exist", joined)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_out_of_range_level_fails(self) -> None:
        bad = _POLICIES + "\n[component.miopen]\nlevel = 9\n"
        root = _make_fixture(policies=bad)
        try:
            ok, messages = validate_policies(root)
            self.assertFalse(ok)
            self.assertIn("outside [1, 5]", "\n".join(messages))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_level_2_is_accepted(self) -> None:
        # Level 2 is within the engine's implemented range (_LEVEL_TO_DEPTH), so
        # validation must accept it — the accepted range is derived from the
        # engine, never a hardcoded literal that could drift from it.
        good = _POLICIES + "\n[component.miopen]\nlevel = 2\n"
        root = _make_fixture(policies=good)
        try:
            ok, messages = validate_policies(root)
            self.assertTrue(ok, "\n".join(messages))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_non_integer_level_rejected_cleanly(self) -> None:
        # A float/string level is a typo. _load_policies raises a clear ValueError
        # instead of truncating (4.9 -> 4) or emitting a raw int() traceback.
        bad = _POLICIES + "\n[component.miopen]\nlevel = 4.9\n"
        root = _make_fixture(policies=bad)
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_policies(root)
            self.assertIn("non-integer level", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_validate_cli_passes(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--therock-dir",
                str(self.root),
                "--validate-policies",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK:", proc.stdout)

    def test_validate_cli_fails_on_bogus_key(self) -> None:
        bogus = _POLICIES + "\n[component.does-not-exist]\n"
        root = _make_fixture(policies=bogus)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--therock-dir",
                    str(root),
                    "--validate-policies",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("does-not-exist", proc.stderr)
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# The real committed test_policies.toml validates against the real committed
# graph, so a stale policy key is caught.
# ---------------------------------------------------------------------------
class TestRealCommittedPolicies(unittest.TestCase):
    def test_committed_policies_validate_against_committed_graph(self) -> None:
        ok, messages = validate_policies(THEROCK_DIR)
        self.assertTrue(ok, "\n".join(messages))


# ---------------------------------------------------------------------------
# Identifier-space contract: inputs must be consumer-graph keys. The function only
# lowercases; the CLI also strips a leading `projects/`. Subtree paths and
# hyphenated matrix keys are NOT graph keys and must be mapped by the caller.
# ---------------------------------------------------------------------------
class TestIdentifierSpaceContract(_FixtureTestCase):
    def test_graph_key_input_resolves(self) -> None:
        selected = get_subprojects_to_test(["amdsmi"], self.root, level=4)
        self.assertIn("amdsmi", selected)
        self.assertIn("rdc", selected)

    def test_subtree_path_is_not_a_graph_key(self) -> None:
        # The CLI strips `projects/` -> `clr`, still not the graph key `hip-clr`,
        # so it selects only itself with a warning.
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--therock-dir",
                str(self.root),
                "--changed-projects",
                "projects/clr",
                "--level",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        selected = set(json.loads(proc.stdout.strip()))
        self.assertEqual(selected, {"clr"})
        self.assertIn("unrecognized", proc.stderr.lower())

    def test_shared_prefix_not_stripped(self) -> None:
        # Only `projects/` is stripped, not `shared/`. `shared/rocroller` is not a
        # graph key, so it does not resolve to `rocroller`.
        buf = io.StringIO()
        with redirect_stderr(buf):
            selected = get_subprojects_to_test(["shared/rocroller"], self.root, level=4)
        self.assertNotIn("hipblaslt", selected)  # rocroller's real consumer
        self.assertIn("unrecognized", buf.getvalue().lower())

    def test_graph_keys_use_underscores_not_matrix_hyphens(self) -> None:
        # Selection returns graph keys (underscore-form), not the hyphenated CI
        # matrix keys — asserting the skew a future mapping step must bridge.
        graph = {
            "miopen": {"consumers": ["hipdnn_integration_tests"]},
            "hipdnn_integration_tests": {"consumers": []},
        }
        root = _make_fixture(graph=graph, policies="[component.miopen]\n")
        try:
            selected = get_subprojects_to_test(["miopen"], root, level=4)
            self.assertIn("hipdnn_integration_tests", selected)
            self.assertNotIn("hipdnn-integration-tests", selected)
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# --explain: names level, walk set, includes (test-only marked), excludes, final
# set — and the final set matches the actual selection (explain can't drift).
# ---------------------------------------------------------------------------
class TestExplain(_FixtureTestCase):
    def test_explain_reports_level_walk_include_exclude_final(self) -> None:
        text = explain_component("amdsmi", self.root)
        self.assertIn("Explain: amdsmi", text)
        self.assertIn("level:", text)
        self.assertIn("graph walk:", text)
        self.assertIn("test_include:", text)
        self.assertIn("test_exclude:", text)
        self.assertIn("final:", text)

    def test_explain_marks_test_only_include(self) -> None:
        text = explain_component("amdsmi", self.root)
        # hip-tests is not a graph key -> marked (test-only).
        self.assertIn("hip-tests (test-only)", text)

    def test_explain_names_exclude(self) -> None:
        text = explain_component("amdsmi", self.root)
        # test_exclude=["rocblas"] appears in the excludes line.
        exclude_line = [line for line in text.splitlines() if "test_exclude:" in line][
            0
        ]
        self.assertIn("rocblas", exclude_line)

    def test_explain_final_matches_selection(self) -> None:
        # The `final:` line must equal get_subprojects_to_test — explain cannot
        # drift from real selection.
        text = explain_component("amdsmi", self.root)
        final_line = [line for line in text.splitlines() if "final:" in line][0]
        final_set = {
            s.strip() for s in final_line.split("final:")[1].split(",") if s.strip()
        }
        selection = get_subprojects_to_test(["amdsmi"], self.root)
        self.assertEqual(final_set, selection)

    def test_explain_reports_level_3_walk_depth(self) -> None:
        text = explain_component("rocm-core", self.root)
        self.assertIn("level:\t3", text)
        self.assertIn("unbounded", text)


# ---------------------------------------------------------------------------
# --list-subprojects works with NO build/ dir (finding #8): committed graph is
# read from --therock-dir/test_tools/, not build/.
# ---------------------------------------------------------------------------
class TestListSubprojectsNoBuildDir(_FixtureTestCase):
    def test_no_build_dir_present(self) -> None:
        # The fixture root has NO build/ directory — assert it stays that way.
        self.assertFalse((self.root / "build").exists())

    def test_list_names_from_clean_therock_dir(self) -> None:
        names = list_subprojects(self.root, show_deps=False)
        self.assertEqual(set(names), set(_GRAPH.keys()))

    def test_list_with_deps(self) -> None:
        deps = list_subprojects(self.root, show_deps=True)
        self.assertIn("rdc", deps["amdsmi"])
        self.assertEqual(deps["rdc"], "empty")

    def test_list_subprojects_cli_no_build_dir(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--therock-dir",
                str(self.root),
                "--list-subprojects",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        names = json.loads(proc.stdout)
        self.assertEqual(set(names), set(_GRAPH.keys()))


if __name__ == "__main__":
    unittest.main()
