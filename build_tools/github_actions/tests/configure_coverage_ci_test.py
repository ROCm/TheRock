# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

import configure_coverage_ci
import fetch_test_configurations
from _therock_utils.build_topology import get_topology
from configure_coverage_ci import COVERAGE_PROJECTS, GROUP_CMAKE_OPTIONS

THEROCK_ROOT = Path(__file__).resolve().parents[3]


class CoverageProjectsTest(unittest.TestCase):
    def test_names_are_subprojects_in_a_coverage_group(self):
        cmake_lists = (THEROCK_ROOT / "CMakeLists.txt").read_text()
        subprojects = set()
        for group in ["LIBRARIES", "SYSTEMS"]:
            match = re.search(
                rf"set\(THEROCK_COVERAGE_ROCM_{group}_PROJECTS\s([^)]*)\)", cmake_lists
            )
            subprojects.update(match.group(1).split())
        for name in COVERAGE_PROJECTS:
            with self.subTest(project=name):
                self.assertIn(name, subprojects)

    def test_projects_have_tests(self):
        for name in COVERAGE_PROJECTS:
            with self.subTest(project=name):
                self.assertIn(name.lower(), fetch_test_configurations.test_matrix)

    def test_report_artifacts_are_built_by_the_build_stage(self):
        producers = get_topology().get_artifact_to_producer_stages()
        outputs = configure_coverage_ci.build_outputs(
            list(COVERAGE_PROJECTS), ["gfx94X-dcgpu"]
        )
        for job in json.loads(outputs["coverage_matrix"]):
            for arg in job["fetch_artifact_args"].split():
                artifact = arg.removeprefix("--")
                with self.subTest(project=job["project_name"], artifact=artifact):
                    self.assertIn(job["build_stage"], producers.get(artifact, []))


class GroupCmakeOptionsTest(unittest.TestCase):
    def test_options_are_declared_in_cmake_lists(self):
        cmake_lists = (THEROCK_ROOT / "CMakeLists.txt").read_text()
        for option in GROUP_CMAKE_OPTIONS.values():
            name = re.fullmatch(r"-D(\w+)=ON", option).group(1)
            with self.subTest(option=name):
                self.assertRegex(cmake_lists, rf"\boption\(\s*{name}\s")


class ParseProjectsTest(unittest.TestCase):
    def test_empty_selects_every_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects(" , "), list(COVERAGE_PROJECTS)
        )

    def test_names_are_case_insensitive_and_deduplicated(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("hipDNN, HIPDNN,hipdnn"), ["hipDNN"]
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


class ParseGroupCmakeOptionsTest(unittest.TestCase):
    def test_unset_and_false_select_nothing(self):
        self.assertEqual(
            configure_coverage_ci.parse_group_cmake_options(
                {"COVERAGE_ROCM_SYSTEMS_ALL": "false"}
            ),
            [],
        )

    def test_true_selects_matching_options_in_declared_order(self):
        self.assertEqual(
            configure_coverage_ci.parse_group_cmake_options(
                {"COVERAGE_ALL": "true", "COVERAGE_ROCM_LIBRARIES_ALL": "true"}
            ),
            [
                "-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON",
                "-DTHEROCK_COVERAGE_ALL=ON",
                "-DHIPCCL_ENABLE_COVERAGE=OFF",
            ],
        )

    def test_hipccl_is_turned_off_with_the_groups_that_include_it(self):
        for env_var, turned_off in [
            ("COVERAGE_ROCM_LIBRARIES_ALL", True),
            ("COVERAGE_ROCM_SYSTEMS_ALL", False),
            ("COVERAGE_ALL", True),
        ]:
            with self.subTest(env_var=env_var):
                options = configure_coverage_ci.parse_group_cmake_options(
                    {env_var: "true"}
                )
                self.assertEqual("-DHIPCCL_ENABLE_COVERAGE=OFF" in options, turned_off)


class BuildOutputsTest(unittest.TestCase):
    def test_one_report_job_per_project_and_family(self):
        outputs = configure_coverage_ci.build_outputs(
            ["hipDNN"], ["gfx94X-dcgpu", "gfx950-dcgpu"]
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

    def test_settings_follow_from_the_subproject_name(self):
        with mock.patch.dict(
            COVERAGE_PROJECTS,
            {"rocSOLVER": ("lib/librocsolver.so*", "bin/rocsolver-bench")},
        ), mock.patch.dict(
            fetch_test_configurations.test_matrix,
            {"rocsolver": {"fetch_artifact_args": "--solver --tests --blas"}},
        ):
            outputs = configure_coverage_ci.build_outputs(
                ["hipDNN", "rocSOLVER"], ["gfx94X-dcgpu"]
            )

        self.assertEqual(
            outputs["coverage_cmake_options"],
            "-DHIPDNN_ENABLE_COVERAGE=ON -DROCSOLVER_ENABLE_COVERAGE=ON",
        )
        self.assertEqual(
            json.loads(outputs["coverage_matrix"])[1],
            {
                "project_name": "rocsolver",
                "test_component": "rocsolver",
                "object_globs": "lib/librocsolver.so*,bin/rocsolver-bench",
                "fetch_artifact_args": "--solver --blas",
                "build_stage": "math-libs",
                "codecov_flag": "rocSOLVER",
                "amdgpu_families": "gfx94X-dcgpu",
            },
        )

    def test_group_options_add_cmake_options_but_not_report_jobs(self):
        outputs = configure_coverage_ci.build_outputs(
            ["hipDNN"], ["gfx94X-dcgpu"], ["-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"]
        )

        self.assertEqual(
            outputs["coverage_cmake_options"],
            "-DHIPDNN_ENABLE_COVERAGE=ON -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON",
        )
        self.assertEqual(len(json.loads(outputs["coverage_matrix"])), 1)


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

    def test_group_env_vars_add_group_options(self):
        env = {
            "PROJECTS_TO_TEST": "hipdnn",
            "COVERAGE_ROCM_LIBRARIES_ALL": "true",
            "COVERAGE_ROCM_SYSTEMS_ALL": "true",
            "COVERAGE_ALL": "false",
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(configure_coverage_ci.main(["--print"]), 0)

        self.assertEqual(
            json.loads(stdout.getvalue())["coverage_cmake_options"],
            "-DHIPDNN_ENABLE_COVERAGE=ON"
            " -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"
            " -DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON"
            " -DHIPCCL_ENABLE_COVERAGE=OFF",
        )

    def test_unknown_project_fails(self):
        with mock.patch.dict(os.environ, {"PROJECTS_TO_TEST": "nope"}):
            with mock.patch("sys.stderr"):
                self.assertEqual(configure_coverage_ci.main(["--print"]), 1)


if __name__ == "__main__":
    unittest.main()
