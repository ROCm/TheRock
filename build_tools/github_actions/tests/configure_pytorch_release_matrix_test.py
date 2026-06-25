# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_pytorch_release_matrix as m


class ConfigurePytorchReleaseMatrixTest(unittest.TestCase):
    def test_ci_linux_uses_reduced_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="ci",
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        # Compared to releases:
        #   * limited to python 3.12
        #   * not including "nightly" pytorch_git_ref
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

    def test_ci_windows_uses_reduced_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="ci",
            amdgpu_families="gfx110X-all",
            platform="windows",
        )

        # Compared to releases:
        #   * limited to python 3.12
        # Compared to Linux:
        #   * limited to only a single pytorch_git_ref
        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.10",
                    "amdgpu_families": "gfx110X-all",
                },
            ],
        )

    def test_explicit_versions_and_refs_narrow_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            python_versions=["3.13"],
            pytorch_git_refs=["nightly"],
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.13",
                    "pytorch_git_ref": "nightly",
                    "amdgpu_families": "gfx94X-dcgpu",
                }
            ],
        )

    def test_filters_exact_unsupported_family(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["release/2.10"],
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
            platform="linux",
        )

        # gfx125X-dcgpu not supported on the release/2.10 ref, should filter
        self.assertEqual(matrix[0]["amdgpu_families"], "gfx94X-dcgpu")
        matrix_families = ";".join(row["amdgpu_families"] for row in matrix)
        self.assertNotIn("gfx125X", matrix_families)

    def test_unknown_explicit_ref_keeps_families(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["users/alice/gfx125x-bringup"],
            amdgpu_families="gfx125X-dcgpu",
            platform="linux",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "users/alice/gfx125x-bringup",
                    "amdgpu_families": "gfx125X-dcgpu",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
