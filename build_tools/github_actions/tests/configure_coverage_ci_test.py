# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_coverage_ci


class ParseProjectsTest(unittest.TestCase):
    def test_empty_selects_every_measurable_project(self):
        # Registered-but-unmeasurable projects are deliberately left out: an
        # empty selection should not schedule a job that cannot report.
        self.assertEqual(
            configure_coverage_ci.parse_projects(""),
            sorted(configure_coverage_ci.SUPPORTED_PROJECTS),
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
            sorted(configure_coverage_ci.SUPPORTED_PROJECTS),
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
            set(configure_coverage_ci.SUPPORTED_PROJECTS),
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
                sorted(configure_coverage_ci.SUPPORTED_PROJECTS)
            ),
            ["-DTHEROCK_COVERAGE_ALL=ON"],
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


class BuildStagePlanTest(unittest.TestCase):
    def test_math_libs_only_selection_needs_no_generic_stage(self):
        generic, needs_math_libs = configure_coverage_ci.build_stage_plan(["hiprand"])
        self.assertEqual(generic, [])
        self.assertTrue(needs_math_libs)

    def test_compiler_runtime_project_adds_no_stage_of_its_own(self):
        # compiler-runtime is built unconditionally, so a project living there
        # must not also appear as a generic stage to build.
        generic, needs_math_libs = configure_coverage_ci.build_stage_plan(["amdsmi"])
        self.assertEqual(generic, [])
        self.assertFalse(needs_math_libs)

    def test_generic_stage_project_is_named_for_the_build_job(self):
        generic, needs_math_libs = configure_coverage_ci.build_stage_plan(["rccl"])
        self.assertEqual(
            generic,
            [
                {
                    "stage_name": "comm-libs",
                    "stage_display_name": "Stage - Coverage Comm Libs",
                }
            ],
        )
        self.assertFalse(needs_math_libs)

    def test_stages_are_deduplicated_across_projects(self):
        generic, _ = configure_coverage_ci.build_stage_plan(["rccl", "rocshmem"])
        self.assertEqual([entry["stage_name"] for entry in generic], ["comm-libs"])

    def test_selecting_everything_names_every_non_default_stage(self):
        generic, needs_math_libs = configure_coverage_ci.build_stage_plan(
            sorted(configure_coverage_ci.COVERAGE_PROJECTS)
        )
        self.assertEqual(
            [entry["stage_name"] for entry in generic],
            ["comm-libs", "profiler-apps"],
        )
        self.assertTrue(needs_math_libs)


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
            # hipRAND is a math-libs project, so no other stage is built.
            self.assertIn("needs_math_libs=true", written)
            self.assertIn("generic_stages_json=[]", written)


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
