# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import configure_coverage_ci
from _therock_utils.build_topology import get_topology


class ParseProjectsTest(unittest.TestCase):
    def test_empty_selects_every_buildable_project(self):
        # Registered-but-unmeasurable projects are deliberately left out: an
        # empty selection should not schedule a job that cannot report. Nor
        # should it pick projects whose stage this workflow cannot build.
        self.assertEqual(
            configure_coverage_ci.parse_projects(""),
            sorted(configure_coverage_ci.DEFAULT_PROJECTS),
        )

    def test_default_selection_is_one_the_workflow_can_actually_build(self):
        configure_coverage_ci.resolve_build_stages(
            configure_coverage_ci.parse_projects("")
        )

    def test_default_leaves_out_what_cannot_be_built_or_is_known_broken(self):
        self.assertEqual(
            sorted(
                configure_coverage_ci.SUPPORTED_PROJECTS
                - configure_coverage_ci.DEFAULT_PROJECTS
            ),
            # rccl and rocshmem for want of a comm-libs build job, hipblaslt
            # and hiptensor because their instrumented builds do not link.
            ["hipblaslt", "hiptensor", "rccl", "rocshmem"],
        )

    def test_blocked_project_is_left_out_of_every_alias(self):
        # An alias is the caller delegating the choice, so it must not hand
        # back a project whose build is known to fail.
        for alias in ("all", "rocm_libraries_all", "rocm_systems_all"):
            with self.subTest(alias=alias):
                self.assertFalse(
                    configure_coverage_ci.BLOCKED_PROJECTS
                    & configure_coverage_ci._GROUP_ALIASES[alias]
                )

    def test_blocked_project_is_still_honoured_when_named(self):
        # Retesting is the only way to notice the block has been lifted.
        for name in sorted(configure_coverage_ci.BLOCKED_PROJECTS):
            with self.subTest(project=name):
                self.assertEqual(configure_coverage_ci.parse_projects(name), [name])

    def test_naming_a_blocked_project_warns_with_the_reason(self):
        for name in sorted(configure_coverage_ci.BLOCKED_PROJECTS):
            with self.subTest(project=name):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    configure_coverage_ci.parse_projects(name)
                self.assertIn(
                    configure_coverage_ci.COVERAGE_PROJECTS[name].blocked_reason,
                    stderr.getvalue(),
                )

    def test_whitespace_and_case_are_normalized(self):
        self.assertEqual(configure_coverage_ci.parse_projects(" HipRand "), ["hiprand"])

    def test_duplicates_are_dropped_but_order_is_kept(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("hiprand,hiprand"), ["hiprand"]
        )

    def test_unknown_project_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            configure_coverage_ci.parse_projects("hiprand,not-a-project")
        self.assertIn("not-a-project", str(context.exception))

    def test_all_alias_expands_to_every_measurable_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("all"),
            sorted(
                configure_coverage_ci.SUPPORTED_PROJECTS
                - configure_coverage_ci.BLOCKED_PROJECTS
            ),
        )

    def test_rocm_libraries_all_expands_to_that_group(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
            sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS),
        )

    def test_group_aliases_are_case_insensitive(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("  ALL  "),
            configure_coverage_ci.parse_projects("all"),
        )
        self.assertEqual(
            configure_coverage_ci.parse_projects("Rocm_Libraries_All"),
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
        )

    def test_alias_and_explicit_name_overlap_is_deduped(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("rocm_libraries_all,hiprand"),
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
        )

    def test_empty_group_alias_is_rejected(self):
        # Both groups are populated, so the empty case needs an empty group to
        # exist at all. Selecting nothing is a caller mistake worth reporting
        # rather than silently producing no jobs.
        with mock.patch.dict(
            configure_coverage_ci._GROUP_ALIASES,
            {"rocm_systems_all": frozenset()},
        ):
            with self.assertRaises(ValueError):
                configure_coverage_ci.parse_projects("rocm_systems_all")

    def test_unknown_alias_like_token_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_projects("rocm_everything_all")


class SourceRepoPartitionTest(unittest.TestCase):
    def test_groups_partition_the_measurable_projects(self):
        libraries = set(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
        systems = set(configure_coverage_ci.ROCM_SYSTEMS_PROJECTS)
        self.assertTrue(libraries.isdisjoint(systems))
        self.assertEqual(
            libraries | systems,
            set(
                configure_coverage_ci.SUPPORTED_PROJECTS
                - configure_coverage_ci.BLOCKED_PROJECTS
            ),
        )


class ParseAmdgpuFamiliesTest(unittest.TestCase):
    def test_empty_falls_back_to_the_default_family(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families(""),
            [configure_coverage_ci.DEFAULT_AMDGPU_FAMILIES],
        )

    def test_comma_separated_families_are_split(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families("gfx94X-dcgpu, gfx110X-all"),
            ["gfx94X-dcgpu", "gfx110X-all"],
        )


class ParseConfigSourceTest(unittest.TestCase):
    def test_repository_and_ref_are_split(self):
        self.assertEqual(
            configure_coverage_ci.parse_config_source("ROCm/rocm-libraries@develop"),
            ("ROCm/rocm-libraries", "develop"),
        )

    def test_empty_falls_back_to_the_default_source(self):
        self.assertEqual(
            configure_coverage_ci.parse_config_source(""),
            ("ROCm/rocm-libraries", "main"),
        )

    def test_missing_ref_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_config_source("ROCm/rocm-libraries")


class BuildCoverageMatrixTest(unittest.TestCase):
    def test_one_entry_per_project_and_family(self):
        matrix = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"],
            ["gfx94X-dcgpu", "gfx110X-all"],
            "ROCm/rocm-libraries",
            "main",
        )
        self.assertEqual(len(matrix), 2)
        self.assertEqual(
            [entry["amdgpu_families"] for entry in matrix],
            ["gfx94X-dcgpu", "gfx110X-all"],
        )

    def test_entries_are_json_serializable(self):
        matrix = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(json.loads(json.dumps(matrix)), matrix)

    def test_every_registered_project_declares_report_inputs(self):
        for name, project in configure_coverage_ci.COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertTrue(project.object_globs, "needs objects for llvm-cov")
                self.assertTrue(project.fetch_artifact_args, "needs artifacts to fetch")
                self.assertTrue(project.stage)
                self.assertTrue(project.test_component)

    def test_every_measurable_project_names_an_upstream_option(self):
        # TheRock's <PROJECT>_ENABLE_COVERAGE is only a knob; instrumentation
        # happens when the subproject's own option is set. An entry without one
        # builds and tests cleanly and then reports nothing.
        for name in sorted(configure_coverage_ci.SUPPORTED_PROJECTS):
            with self.subTest(project=name):
                self.assertTrue(
                    configure_coverage_ci.COVERAGE_PROJECTS[name].coverage_option
                )

    def test_unsupported_project_is_rejected_with_the_reason(self):
        with self.assertRaises(ValueError) as context:
            configure_coverage_ci.parse_projects("miopen")
        self.assertIn("not supported", str(context.exception))
        self.assertIn("no coverage option", str(context.exception))

    def test_unsupported_projects_stay_out_of_the_group_aliases(self):
        # They remain registered so the gap is tracked, but must not be
        # scheduled by a group selection.
        for name in ("miopen", "hipsparse", "rocalution", "rocprofiler-sdk"):
            with self.subTest(project=name):
                self.assertIn(name, configure_coverage_ci.COVERAGE_PROJECTS)
                self.assertNotIn(name, configure_coverage_ci.parse_projects("all"))

    def test_object_globs_reach_the_matrix(self):
        # object_globs is what scopes a report to one project, so it has to
        # survive into the matrix in the spelling llvm-cov is handed.
        (entry,) = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(entry["object_globs"], "lib/libhiprand.so*")
        self.assertEqual(entry["fetch_artifact_args"], "--rand")


class BuildCoverageCmakeOptionsTest(unittest.TestCase):
    def test_full_selection_collapses_to_the_group_option(self):
        # What the nightly does: instrument every onboarded project at once.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(
                sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
            ),
            ["-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"],
        )

    def test_both_groups_collapse_to_the_combined_option(self):
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(
                sorted(
                    configure_coverage_ci.SUPPORTED_PROJECTS
                    - configure_coverage_ci.BLOCKED_PROJECTS
                )
            ),
            ["-DTHEROCK_COVERAGE_ALL=ON"],
        )

    def test_blocked_project_is_named_rather_than_folded_into_a_group(self):
        # It is outside the group lists, so the group flag would not reach it
        # and the explicit ask would silently build uninstrumented.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(["hipblaslt"]),
            ["-DHIPBLASLT_ENABLE_COVERAGE=ON"],
        )

    def test_partial_selection_names_each_project_in_upper_case(self):
        # therock_subproject.cmake only forwards the upper case spelling.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(["rocblas"]),
            ["-DROCBLAS_ENABLE_COVERAGE=ON"],
        )

    def test_partial_selection_also_names_extra_targets(self):
        # rocPRIM reports against binaries built by rocPRIM_tests, so naming
        # rocPRIM alone would instrument nothing the report reads.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(["rocprim"]),
            ["-DROCPRIM_ENABLE_COVERAGE=ON", "-DROCPRIM_TESTS_ENABLE_COVERAGE=ON"],
        )

    def test_one_group_does_not_claim_to_cover_the_other(self):
        options = configure_coverage_ci.build_coverage_cmake_options(
            sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
        )
        self.assertEqual(options, ["-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"])

    def test_no_selection_produces_no_options(self):
        self.assertEqual(configure_coverage_ci.build_coverage_cmake_options([]), [])


class ResolveBuildStagesTest(unittest.TestCase):
    def test_math_libs_project_needs_the_math_libs_stage(self):
        self.assertEqual(
            configure_coverage_ci.resolve_build_stages(["hiprand"]),
            {"math-libs"},
        )

    def test_compiler_runtime_project_needs_no_math_libs_build(self):
        # compiler-runtime is built unconditionally, so a project living there
        # must not drag in the per-architecture math-libs job.
        self.assertEqual(
            configure_coverage_ci.resolve_build_stages(["amdsmi"]),
            {"compiler-runtime"},
        )

    def test_whole_rocm_libraries_group_stays_within_math_libs(self):
        keys = configure_coverage_ci.parse_projects("rocm_libraries_all")
        self.assertEqual(
            configure_coverage_ci.resolve_build_stages(keys), {"math-libs"}
        )

    def test_comm_libs_project_is_rejected_with_a_actionable_message(self):
        # comm-libs has no build job, so selecting rccl would leave the test
        # job with nothing to overlay. It has to fail here instead.
        with self.assertRaises(ValueError) as caught:
            configure_coverage_ci.resolve_build_stages(["rccl"])
        message = str(caught.exception)
        self.assertIn("rccl", message)
        self.assertIn("comm-libs", message)
        self.assertIn("multi_arch_ci_coverage_linux.yml", message)

    def test_rejection_names_every_unbuildable_project_not_just_the_first(self):
        with self.assertRaises(ValueError) as caught:
            configure_coverage_ci.resolve_build_stages(["hiprand", "rccl", "rocshmem"])
        message = str(caught.exception)
        self.assertIn("rccl", message)
        self.assertIn("rocshmem", message)

    def test_every_buildable_stage_is_one_a_project_can_register_against(self):
        self.assertTrue(
            configure_coverage_ci.BUILDABLE_STAGES <= configure_coverage_ci.KNOWN_STAGES
        )


class RegistryMatchesBuildTopologyTest(unittest.TestCase):
    """CoverageProject.stage reads the build topology instead of repeating it.

    The derivation looks at the first artifact only, so it is well defined
    only while every artifact of a project comes from one stage. A project
    whose artifacts straddled two stages would otherwise get a stage picked by
    list order, and the overlay would find nothing hours into a run.
    """

    def test_all_artifacts_of_a_project_come_from_one_stage(self):
        topology = get_topology()
        for key, project in sorted(configure_coverage_ci.COVERAGE_PROJECTS.items()):
            for artifact in project.artifact_names:
                with self.subTest(project=key, artifact=artifact):
                    self.assertEqual(
                        topology.get_stage_for_artifact(artifact),
                        project.stage,
                    )

    def test_every_project_resolves_to_a_known_stage(self):
        for key, project in sorted(configure_coverage_ci.COVERAGE_PROJECTS.items()):
            with self.subTest(project=key):
                self.assertIn(project.stage, configure_coverage_ci.KNOWN_STAGES)

    def test_unknown_artifact_is_rejected_rather_than_guessed(self):
        project = configure_coverage_ci.CoverageProject(
            cmake_target="notAProject",
            artifact_names=["no-such-artifact"],
        )
        with self.assertRaises(ValueError) as caught:
            project.stage
        self.assertIn("no-such-artifact", str(caught.exception))

    def test_stage_override_wins_over_the_topology(self):
        # The escape hatch for a project the topology cannot answer for. No
        # entry needs it today, so this is the only thing exercising it.
        project = configure_coverage_ci.CoverageProject(
            cmake_target="hipRAND",
            artifact_names=["no-such-artifact"],
            stage_override=configure_coverage_ci.STAGE_COMM_LIBS,
        )
        self.assertEqual(project.stage, configure_coverage_ci.STAGE_COMM_LIBS)

    def test_stage_needs_an_artifact_or_an_override(self):
        project = configure_coverage_ci.CoverageProject(cmake_target="hipRAND")
        with self.assertRaises(ValueError) as caught:
            project.stage
        self.assertIn("artifact_names", str(caught.exception))


class ValidateRegistryTest(unittest.TestCase):
    """The registry's hand-written parts are checked, not assumed.

    These ran as bare asserts once, which PYTHONOPTIMIZE strips; CMake runs
    this script on every configure, so a stripped check meant a quietly wrong
    therock_coverage_projects.cmake rather than a failed configure.
    """

    def _project(self, **kwargs):
        defaults = dict(
            cmake_target="hipRAND",
            coverage_option="BUILD_CODE_COVERAGE",
            artifact_names=["rand"],
            artifact_relpaths=["math-libs/hipRAND/stage"],
        )
        return configure_coverage_ci.CoverageProject(**{**defaults, **kwargs})

    def test_the_shipped_registry_is_valid(self):
        configure_coverage_ci.validate_registry(configure_coverage_ci.COVERAGE_PROJECTS)

    def test_raises_rather_than_asserts(self):
        # A ValueError subclass, so callers can keep catching ValueError.
        self.assertTrue(
            issubclass(configure_coverage_ci.CoverageRegistryError, ValueError)
        )

    def test_unknown_source_repo_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(source_repo="rocm-something")}
            )
        self.assertIn("source_repo", str(caught.exception))

    def test_key_disagreeing_with_cmake_target_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry({"hip_rand": self._project()})
        self.assertIn("cmake_target lower cased", str(caught.exception))

    def test_entry_with_neither_option_nor_reason_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(coverage_option="")}
            )
        self.assertIn("set either coverage_option", str(caught.exception))

    def test_entry_with_both_option_and_reason_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(unsupported_reason="no")}
            )
        self.assertIn("set only one of", str(caught.exception))

    def test_measurable_project_without_overlay_inputs_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(artifact_relpaths=[])}
            )
        self.assertIn("artifact_relpaths", str(caught.exception))

    def test_artifact_the_topology_does_not_know_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(artifact_names=["no-such-artifact"])}
            )
        self.assertIn("no-such-artifact", str(caught.exception))

    def test_stage_outside_the_known_set_is_rejected(self):
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {"hiprand": self._project(stage_override="not-a-stage")}
            )
        self.assertIn("not-a-stage", str(caught.exception))

    def test_every_problem_is_reported_not_just_the_first(self):
        # Adding a project should surface all of its mistakes in one run.
        with self.assertRaises(configure_coverage_ci.CoverageRegistryError) as caught:
            configure_coverage_ci.validate_registry(
                {
                    "hip_rand": self._project(
                        source_repo="nope",
                        artifact_relpaths=[],
                    )
                }
            )
        message = str(caught.exception)
        self.assertIn("source_repo", message)
        self.assertIn("cmake_target lower cased", message)
        self.assertIn("artifact_relpaths", message)


class DerivedAttributesTest(unittest.TestCase):
    """The fields that used to be written out per entry.

    Each was identical to something already known -- the registry key, the
    cmake_target, or the build topology -- for every project, so they are
    derived now and these tests pin the shape the coverage workflows read.
    """

    def test_key_and_test_component_follow_the_cmake_target(self):
        for key, project in sorted(configure_coverage_ci.COVERAGE_PROJECTS.items()):
            with self.subTest(project=key):
                self.assertEqual(project.key, key)
                self.assertEqual(project.test_component, key)

    def test_coverage_config_path_follows_the_key(self):
        project = configure_coverage_ci.COVERAGE_PROJECTS["hiprand"]
        self.assertEqual(
            project.coverage_config,
            "projects/hiprand/test_categories_coverage.yaml",
        )

    def test_codecov_flag_defaults_to_the_cmake_target(self):
        # The cased spelling, not the key: Codecov reports "hipRAND".
        self.assertEqual(
            configure_coverage_ci.COVERAGE_PROJECTS["hiprand"].codecov_flag, "hipRAND"
        )

    def test_codecov_flag_override_is_honoured(self):
        # rocshmem is the only project whose flag is not its cmake_target.
        self.assertEqual(
            configure_coverage_ci.COVERAGE_PROJECTS["rocshmem"].codecov_flag, "rocSHMEM"
        )

    def test_every_project_declares_the_inputs_the_derivations_need(self):
        for key, project in sorted(configure_coverage_ci.COVERAGE_PROJECTS.items()):
            with self.subTest(project=key):
                self.assertTrue(project.artifact_names, "stage is derived from these")
                self.assertTrue(project.codecov_flag)
                self.assertTrue(project.coverage_config)


class MainTest(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "github_output"
            output_path.touch()
            os.environ["GITHUB_OUTPUT"] = os.fspath(output_path)
            os.environ["PROJECTS_TO_TEST"] = "hiprand"
            os.environ["AMDGPU_FAMILIES"] = "gfx94X-dcgpu"
            os.environ["COVERAGE_CONFIG_SOURCE"] = "ROCm/rocm-libraries@main"

            self.assertEqual(configure_coverage_ci.main([]), 0)

            written = output_path.read_text()
            self.assertIn("coverage_matrix", written)
            self.assertIn("dist_amdgpu_families", written)
            self.assertIn("families_matrix_json", written)
            self.assertIn("coverage_cmake_options", written)
            # One project out of a populated group names that project alone.
            self.assertIn("-DHIPRAND_ENABLE_COVERAGE=ON", written)
            # The per-project test and overlay inputs the nightly reads back.
            self.assertIn('"test_component": "hiprand"', written)
            self.assertIn('"object_globs": "lib/libhiprand.so*"', written)
            # hipRAND is a math-libs project, so that stage has to be built.
            self.assertIn("needs_math_libs=true", written)


class EmitCmakeTest(unittest.TestCase):
    def test_emits_group_lists_from_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            text = out.read_text()
            self.assertIn("set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS", text)
            self.assertIn("hipRAND", text)
            self.assertIn("set(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS", text)
            self.assertIn("rccl", text)
            # A group flag has to reach the sibling subproject that builds a
            # header-only project's tests, or the group instruments rocPRIM and
            # leaves its only reportable binaries without a coverage mapping.
            self.assertIn("rocPRIM_tests", text)
            # Group membership follows source_repo, not the stage.
            libraries_line = next(
                line
                for line in text.splitlines()
                if line.startswith("set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS")
            )
            self.assertNotIn("rccl", libraries_line)
            # Unmeasurable projects are excluded, or a group build would set a
            # flag their CMake does not implement.
            self.assertNotIn("MIOpen", text)
            self.assertNotIn("hipSPARSE ", text)
            # The CMake group lists and the Python group sets decide the same
            # thing in two places, so a blocked project has to leave both or a
            # group build sets the flag the Python side just declined to.
            self.assertNotIn("hipBLASLt", libraries_line)

    def test_blocked_project_keeps_its_option_mapping(self):
        # Excluded from the group lists but still nameable, which needs the
        # translation to its upstream option name to survive.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_HIPBLASLT HIPBLASLT_ENABLE_COVERAGE)",
                out.read_text(),
            )

    def test_emits_the_upstream_option_name_per_subproject(self):
        # The generic names are why this is a per-subproject map: setting
        # BUILD_CODE_COVERAGE globally would instrument everything that happens
        # to understand it.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            text = out.read_text()
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_HIPRAND BUILD_CODE_COVERAGE)", text
            )
            self.assertIn("set(THEROCK_COVERAGE_OPTION_ROCRAND CODE_COVERAGE)", text)
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_RCCL ENABLE_CODE_COVERAGE)", text
            )
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_HIPBLASLT HIPBLASLT_ENABLE_COVERAGE)", text
            )
            # A sibling subproject builds from the same source, so it takes the
            # same option as its parent.
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_ROCPRIM_TESTS BUILD_CODE_COVERAGE)", text
            )
            self.assertNotIn("THEROCK_COVERAGE_OPTION_MIOPEN", text)

    def test_emit_cmake_needs_no_env(self):
        # Must work without PROJECTS_TO_TEST / AMDGPU_FAMILIES / GITHUB_OUTPUT set.
        import os

        saved = {
            k: os.environ.pop(k, None)
            for k in (
                "PROJECTS_TO_TEST",
                "AMDGPU_FAMILIES",
                "COVERAGE_CONFIG_SOURCE",
                "GITHUB_OUTPUT",
            )
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "c.cmake"
                self.assertEqual(
                    configure_coverage_ci.main(["--emit-cmake", str(out)]), 0
                )
                self.assertTrue(out.exists())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
