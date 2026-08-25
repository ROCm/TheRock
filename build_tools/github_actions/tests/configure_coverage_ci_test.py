# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_coverage_flag_uses_the_upper_case_project_name(self):
        # therock_subproject.cmake only forwards the upper case spelling.
        (entry,) = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(entry["coverage_flag"], "HIPRAND_ENABLE_COVERAGE")

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
                self.assertTrue(project.stage_project)
                self.assertTrue(project.test_component)


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
            self.assertIn("-DHIPRAND_ENABLE_COVERAGE=ON", written)
            # compiler-runtime is always built so the profiling runtime exists.
            self.assertIn("compiler-runtime,math-libs", written)


if __name__ == "__main__":
    unittest.main()
