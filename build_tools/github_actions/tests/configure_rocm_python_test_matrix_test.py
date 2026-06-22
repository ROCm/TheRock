# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_rocm_python_test_matrix as m


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


if __name__ == "__main__":
    unittest.main()
