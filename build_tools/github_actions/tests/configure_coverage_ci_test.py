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
    def test_empty_selects_every_coverage_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects(""),
            sorted(configure_coverage_ci.COVERAGE_PROJECTS),
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

    def test_all_alias_expands_to_every_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("all"),
            sorted(configure_coverage_ci.COVERAGE_PROJECTS),
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
        # Update this test once a rocm-systems project is onboarded to coverage
        # (the group becomes non-empty).
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_projects("rocm_systems_all")

    def test_unknown_alias_like_token_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_projects("rocm_everything_all")


class SourceRepoPartitionTest(unittest.TestCase):
    def test_groups_partition_all_projects(self):
        libraries = set(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
        systems = set(configure_coverage_ci.ROCM_SYSTEMS_PROJECTS)
        self.assertTrue(libraries.isdisjoint(systems))
        self.assertEqual(
            libraries | systems,
            set(configure_coverage_ci.COVERAGE_PROJECTS),
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

    def test_every_registered_project_declares_hybrid_fetch_inputs(self):
        # Without these the nightly test job cannot tell which files belong to
        # the project, and would test an entirely non-instrumented install.
        for name, project in configure_coverage_ci.COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertTrue(project.artifact_names, "needs artifacts to overlay")
                self.assertTrue(project.artifact_relpaths, "needs relpaths to overlay")

    def test_hybrid_fetch_inputs_reach_the_matrix(self):
        (entry,) = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(entry["artifact_names"], "rand")
        self.assertEqual(entry["artifact_relpaths"], "math-libs/hipRAND/stage")


class BuildCoverageCmakeOptionsTest(unittest.TestCase):
    def test_full_selection_collapses_to_the_group_option(self):
        # What the nightly does: instrument every onboarded project at once.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(
                sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
            ),
            ["-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"],
        )

    def test_partial_selection_names_each_project_in_upper_case(self):
        # therock_subproject.cmake only forwards the upper case spelling. Today
        # hipRAND is the whole rocm-libraries group, so a second project has to
        # be stubbed in to reach the per-project fallback at all.
        registry = dict(configure_coverage_ci.COVERAGE_PROJECTS)
        registry["rocblas"] = configure_coverage_ci.CoverageProject(
            cmake_target="rocBLAS",
            stage="math-libs",
            test_component="rocblas",
            coverage_config="projects/rocblas/test_categories_coverage.yaml",
        )
        with (
            mock.patch.object(configure_coverage_ci, "COVERAGE_PROJECTS", registry),
            mock.patch.object(
                configure_coverage_ci,
                "ROCM_LIBRARIES_PROJECTS",
                frozenset(registry),
            ),
        ):
            self.assertEqual(
                configure_coverage_ci.build_coverage_cmake_options(["rocblas"]),
                ["-DROCBLAS_ENABLE_COVERAGE=ON"],
            )

    def test_empty_group_does_not_produce_a_group_option(self):
        # rocm-systems has no onboarded projects, so selecting only
        # rocm-libraries must not claim to cover everything.
        options = configure_coverage_ci.build_coverage_cmake_options(
            sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
        )
        self.assertNotIn("-DTHEROCK_COVERAGE_ALL=ON", options)
        self.assertNotIn("-DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON", options)

    def test_no_selection_produces_no_options(self):
        self.assertEqual(configure_coverage_ci.build_coverage_cmake_options([]), [])


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
            # hipRAND is currently the whole rocm-libraries group, so selecting
            # it selects the group option.
            self.assertIn("-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON", written)
            # The per-project test and overlay inputs the nightly reads back.
            self.assertIn('"test_component": "hiprand"', written)
            self.assertIn('"artifact_relpaths": "math-libs/hipRAND/stage"', written)


class EmitCmakeTest(unittest.TestCase):
    def test_emits_group_lists_from_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            text = out.read_text()
            self.assertIn("set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS", text)
            self.assertIn("hipRAND", text)
            # rocm-systems group is empty today -> an empty set().
            self.assertRegex(text, r"set\(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS\s*\)")

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
