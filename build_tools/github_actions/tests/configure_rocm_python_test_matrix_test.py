# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_rocm_python_test_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_matrix_references,
    get_workflow_job,
    load_workflow,
)


class ConfigureRocmPythonTestMatrixTest(unittest.TestCase):
    def test_linux_matrix_expands_runnable_family_across_versions_and_images(self):
        matrix = m.build_rocm_python_test_matrix(
            per_family_info=[
                {
                    "amdgpu_family": "gfxMOCKLINUX",
                    "test-runs-on": "mock-linux-runner",
                }
            ],
            platform="linux",
        )

        self.assertEqual(len(matrix), 6)
        python_versions = {row["python_version"] for row in matrix}
        container_image_names = {row["container_image_name"] for row in matrix}
        self.assertEqual(python_versions, {"3.10", "3.11", "3.12"})
        self.assertEqual(container_image_names, {"ubuntu24.04", "ubi10"})
        self.assertEqual({row["amdgpu_family"] for row in matrix}, {"gfxMOCKLINUX"})
        self.assertEqual({row["test_runs_on"] for row in matrix}, {"mock-linux-runner"})

    def test_windows_matrix_uses_native_python_312(self):
        matrix = m.build_rocm_python_test_matrix(
            per_family_info=[
                {
                    "amdgpu_family": "gfxMOCKWINDOWS",
                    "test-runs-on": "mock-windows-runner",
                }
            ],
            platform="windows",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "amdgpu_family": "gfxMOCKWINDOWS",
                    "test_runs_on": "mock-windows-runner",
                    "python_version": "3.12",
                    "container_image_name": "native",
                    "container_image_url": "",
                }
            ],
        )

    def test_families_without_runners_are_skipped(self):
        # test-runs-on is required, test-runs-on-labels is not used on its own
        matrix = m.build_rocm_python_test_matrix(
            per_family_info=[
                {
                    "amdgpu_family": "gfxMOCKTARGET",
                    "test-runs-on": "",
                    "test-runs-on-labels": [
                        {"label": "mock-weighted-runner", "weight": 1.0}
                    ],
                }
            ],
            platform="linux",
        )

        self.assertEqual(matrix, [])

    def test_weighted_runner_labels_override_fallback_runner(self):
        with mock.patch.object(
            m, "select_weighted_label", return_value="mock-weighted-runner"
        ) as select_weighted_label:
            matrix = m.build_rocm_python_test_matrix(
                per_family_info=[
                    {
                        "amdgpu_family": "gfxMOCKWEIGHTED",
                        "test-runs-on": "mock-fallback-runner",
                        "test-runs-on-labels": [
                            {"label": "mock-weighted-runner", "weight": 1.0}
                        ],
                    }
                ],
                platform="linux",
            )

        self.assertEqual(len(matrix), 6)
        self.assertEqual(select_weighted_label.call_count, len(matrix))
        self.assertEqual(
            {row["test_runs_on"] for row in matrix}, {"mock-weighted-runner"}
        )

    def test_unknown_platform_errors(self):
        with self.assertRaisesRegex(ValueError, "not-a-platform"):
            m.build_rocm_python_test_matrix(
                per_family_info=[],
                platform="not-a-platform",
            )

    def test_generated_rows_cover_workflow_matrix_inputs(self):
        # The build_rocm_python_test_matrix script produces matrix JSON for use
        # in workflow files like:
        #
        #   matrix:
        #     include: ${{ fromJSON(inputs.build_config).test_python_packages_matrix }}
        #
        # Then it passes values to the test workflow using expressions like:
        #   with:
        #     amdgpu_family: ${{ matrix.amdgpu_family }}
        #     test_runs_on: ${{ matrix.test_runs_on }}
        #     python_version: ${{ matrix.python_version }}
        #
        # This test checks that all `${{ matrix. }}` values are produced for
        # every row in the generated matrix. It intentionally does not check
        # that every generated key is consumed by each workflow; if we want to
        # enforce exact schemas, do that with generator-local tests.
        test_cases = [
            ("multi_arch_ci_linux.yml", "linux"),
            ("multi_arch_ci_windows.yml", "windows"),
        ]

        for workflow_filename, platform in test_cases:
            with self.subTest(workflow_filename=workflow_filename):
                workflow = load_workflow(WORKFLOWS_DIR / workflow_filename)
                job = get_workflow_job(workflow, "test_python_packages_per_family")
                matrix_references = get_matrix_references(job["with"])

                matrix = m.build_rocm_python_test_matrix(
                    per_family_info=[
                        {
                            "amdgpu_family": "gfxMOCKTEST",
                            "test-runs-on": "mock-runner",
                        }
                    ],
                    platform=platform,
                )

                self.assertGreater(len(matrix), 0)
                for row in matrix:
                    # This checks the row schema, not whether values are
                    # truthy. Empty values are allowed, such as
                    # container_image_url="" for native Windows tests.
                    # Undefined values are not: if the workflow reads
                    # `matrix.unknown`, this test fails until the generator
                    # emits that key for every row.
                    self.assertEqual(matrix_references - set(row), set())


if __name__ == "__main__":
    unittest.main()
