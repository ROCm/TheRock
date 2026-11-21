# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_coverage_ci
import fetch_test_configurations
from configure_coverage_ci import (
    COVERAGE_PROJECTS,
    NIGHTLY_CMAKE_OPTIONS,
    CoverageProject,
)

THEROCK_ROOT = Path(__file__).resolve().parents[3]


class CoverageProjectsTest(unittest.TestCase):
    def test_test_components_exist(self):
        for name, project in COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertIn(
                    project.test_component, fetch_test_configurations.test_matrix
                )

    def test_projects_are_in_the_built_stage(self):
        for name, project in COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertEqual(project.build_stage, "math-libs")

    def test_cmake_options_survive_whitespace_splitting(self):
        for name, project in COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertTrue(project.cmake_options)
                for option in project.cmake_options:
                    self.assertRegex(option, r"^-D\S+=\S+$")


class NightlyCmakeOptionsTest(unittest.TestCase):
    def test_options_are_declared_in_cmake_lists(self):
        cmake_lists = (THEROCK_ROOT / "CMakeLists.txt").read_text()
        for option in NIGHTLY_CMAKE_OPTIONS:
            with self.subTest(option=option):
                match = re.fullmatch(r"-D(\w+)=ON", option)
                self.assertIsNotNone(match)
                self.assertRegex(cmake_lists, rf"\boption\(\s*{match[1]}\s")


class ParseProjectsTest(unittest.TestCase):
    def test_empty_selects_every_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects(" , "), list(COVERAGE_PROJECTS)
        )

    def test_names_are_case_insensitive_and_deduplicated(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("hipDNN, HIPDNN,hipdnn"), ["hipdnn"]
        )

    def test_unknown_project_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rocprim"):
            configure_coverage_ci.parse_projects("hipdnn,rocprim")


class ParseAmdgpuFamiliesTest(unittest.TestCase):
    def test_empty_uses_default(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families(""), ["gfx94X-dcgpu"]
        )

    def test_splits_and_strips(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families(" gfx94X-dcgpu ,gfx950-dcgpu,"),
            ["gfx94X-dcgpu", "gfx950-dcgpu"],
        )


class BuildOutputsTest(unittest.TestCase):
    def test_one_report_job_per_project_and_family(self):
        outputs = configure_coverage_ci.build_outputs(
            ["hipdnn"], ["gfx94X-dcgpu", "gfx950-dcgpu"]
        )

        self.assertEqual(
            json.loads(outputs["coverage_matrix"]),
            [
                {
                    "project_name": "hipdnn",
                    "test_component": "hipdnn",
                    "object_globs": "lib/libhipdnn_backend.so*",
                    "fetch_artifact_args": "--hipdnn",
                    "build_stage": "math-libs",
                    "codecov_flag": "hipDNN",
                    "amdgpu_families": family,
                }
                for family in ["gfx94X-dcgpu", "gfx950-dcgpu"]
            ],
        )
        self.assertEqual(outputs["dist_amdgpu_families"], "gfx94X-dcgpu;gfx950-dcgpu")
        self.assertEqual(
            json.loads(outputs["families_matrix_json"]),
            [{"amdgpu_family": "gfx94X-dcgpu"}, {"amdgpu_family": "gfx950-dcgpu"}],
        )
        self.assertEqual(
            outputs["coverage_cmake_options"], "-DHIPDNN_ENABLE_COVERAGE=ON"
        )

    def test_shared_cmake_options_are_passed_once(self):
        hipdnn = COVERAGE_PROJECTS["hipdnn"]
        sibling = CoverageProject(
            cmake_options=hipdnn.cmake_options + ("-DOTHER_ENABLE_COVERAGE=ON",),
            test_component="sibling",
            object_globs=("lib/libsibling.so*", "bin/sibling_test"),
            fetch_artifact_args="--sibling",
            build_stage="math-libs",
            codecov_flag="sibling",
        )
        with mock.patch.dict(COVERAGE_PROJECTS, {"sibling": sibling}):
            outputs = configure_coverage_ci.build_outputs(
                ["hipdnn", "sibling"], ["gfx94X-dcgpu"]
            )

        self.assertEqual(
            outputs["coverage_cmake_options"],
            "-DHIPDNN_ENABLE_COVERAGE=ON -DOTHER_ENABLE_COVERAGE=ON",
        )
        self.assertEqual(
            json.loads(outputs["coverage_matrix"])[1]["object_globs"],
            "lib/libsibling.so*,bin/sibling_test",
        )

    def test_nightly_adds_group_options_but_not_report_jobs(self):
        regular = configure_coverage_ci.build_outputs(["hipdnn"], ["gfx94X-dcgpu"])
        nightly = configure_coverage_ci.build_outputs(
            ["hipdnn"], ["gfx94X-dcgpu"], nightly=True
        )

        self.assertEqual(
            nightly["coverage_cmake_options"],
            "-DHIPDNN_ENABLE_COVERAGE=ON"
            " -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"
            " -DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON"
            " -DTHEROCK_COVERAGE_ALL=ON",
        )
        self.assertEqual(nightly["coverage_matrix"], regular["coverage_matrix"])


class MainTest(unittest.TestCase):
    def test_writes_github_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "github_output"
            output_path.touch()
            env = {
                "GITHUB_OUTPUT": os.fspath(output_path),
                "PROJECTS_TO_TEST": "hipdnn",
                "AMDGPU_FAMILIES": "gfx94X-dcgpu",
            }
            with mock.patch.dict(os.environ, env):
                self.assertEqual(configure_coverage_ci.main([]), 0)
            written = output_path.read_text()

        for name in [
            "coverage_matrix",
            "dist_amdgpu_families",
            "families_matrix_json",
            "coverage_cmake_options",
        ]:
            self.assertIn(f"{name}=", written)
        self.assertIn("coverage_cmake_options=-DHIPDNN_ENABLE_COVERAGE=ON", written)

    def test_nightly_env_adds_group_options(self):
        env = {"PROJECTS_TO_TEST": "hipdnn", "NIGHTLY": "true"}
        with mock.patch.dict(os.environ, env):
            with mock.patch.object(configure_coverage_ci, "gha_set_output") as output:
                self.assertEqual(configure_coverage_ci.main([]), 0)

        cmake_options = output.call_args.args[0]["coverage_cmake_options"].split()
        self.assertEqual(
            cmake_options,
            ["-DHIPDNN_ENABLE_COVERAGE=ON", *NIGHTLY_CMAKE_OPTIONS],
        )

    def test_unknown_project_fails(self):
        with mock.patch.dict(os.environ, {"PROJECTS_TO_TEST": "nope"}):
            with mock.patch("sys.stderr"):
                self.assertEqual(configure_coverage_ci.main(["--print"]), 1)


if __name__ == "__main__":
    unittest.main()
