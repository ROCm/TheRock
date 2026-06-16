# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_pytorch_release_matrix as m


class ConfigurePyTorchReleaseMatrixTest(unittest.TestCase):
    def test_ci_linux_matrix_uses_reduced_defaults(self):
        matrix = m.generate_pytorch_matrix(
            release_type="ci",
            platform="linux",
            amdgpu_families="gfx94X-dcgpu",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.10",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.11",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.12",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
            ],
        )

    def test_linux_filters_amdgpu_families_per_ref(self):
        matrix = m.generate_pytorch_matrix(
            release_type="ci",
            platform="linux",
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.10",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.11",
                    "amdgpu_families": "gfx94X-dcgpu;gfx125X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.12",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
            ],
        )

    def test_rows_with_no_remaining_amdgpu_families_are_skipped(self):
        matrix = m.generate_pytorch_matrix(
            release_type="ci",
            platform="linux",
            amdgpu_families="gfx125X-dcgpu",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.11",
                    "amdgpu_families": "gfx125X-dcgpu",
                }
            ],
        )

    @patch("configure_pytorch_release_matrix.gha_set_output")
    def test_main_writes_matrix_output_from_arguments(self, mock_set_output):
        result = m.main(
            [
                "--release-type=dev",
                "--platform=linux",
                "--python-versions=3.12;3.13",
                "--amdgpu-families=gfx94X-dcgpu",
            ]
        )

        self.assertEqual(result, 0)
        outputs = mock_set_output.call_args.args[0]
        self.assertEqual(outputs["build_pytorch"], "true")
        matrix = json.loads(outputs["pytorch_matrix"])
        python_versions = {row["python_version"] for row in matrix}
        families = {row["amdgpu_families"] for row in matrix}
        self.assertEqual(python_versions, {"3.12", "3.13"})
        self.assertEqual(families, {"gfx94X-dcgpu"})


if __name__ == "__main__":
    unittest.main()
