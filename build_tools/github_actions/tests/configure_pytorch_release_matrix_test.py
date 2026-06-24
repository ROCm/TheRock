# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_pytorch_release_matrix as m


class ConfigurePytorchReleaseMatrixTest(unittest.TestCase):
    def test_linux_matrix_uses_all_release_python_versions_and_refs(self):
        matrix = m.generate_pytorch_matrix(
            python_versions=None,
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        self.assertEqual(len(matrix), 25)
        self.assertEqual(
            matrix[0],
            {
                "python_version": "3.10",
                "pytorch_git_ref": "release/2.9",
                "amdgpu_families": "gfx94X-dcgpu",
            },
        )
        self.assertEqual(
            matrix[-1],
            {
                "python_version": "3.14",
                "pytorch_git_ref": "nightly",
                "amdgpu_families": "gfx94X-dcgpu",
            },
        )

    def test_windows_matrix_does_not_filter_linux_only_family(self):
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx125X-dcgpu",
            platform="windows",
        )

        self.assertEqual(
            matrix[0],
            {
                "python_version": "3.12",
                "pytorch_git_ref": "release/2.9",
                "amdgpu_families": "gfx125X-dcgpu",
            },
        )
        self.assertEqual(len(matrix), 5)

    def test_explicit_python_versions_narrow_matrix(self):
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        self.assertEqual({row["python_version"] for row in matrix}, {"3.12"})
        self.assertEqual(len(matrix), 5)

    def test_filters_unsupported_family_by_substring(self):
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
            platform="linux",
        )

        self.assertEqual(
            {row["amdgpu_families"] for row in matrix},
            {"gfx94X-dcgpu"},
        )


if __name__ == "__main__":
    unittest.main()
