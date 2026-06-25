# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent))

import configure_pytorch_release_matrix as m


class ConfigurePyTorchReleaseMatrixTest(unittest.TestCase):
    def test_release_2_10_excludes_gfx125x(self) -> None:
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
            platform="linux",
            pytorch_refs=["release/2.10"],
        )
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["pytorch_git_ref"], "release/2.10")
        self.assertEqual(matrix[0]["amdgpu_families"], "gfx94X-dcgpu")

    def test_release_2_11_includes_gfx125x(self) -> None:
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
            platform="linux",
            pytorch_refs=["release/2.11"],
        )
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["amdgpu_families"], "gfx94X-dcgpu;gfx125X-dcgpu")

    def test_pytorch_refs_filter_limits_matrix_rows(self) -> None:
        matrix = m.generate_pytorch_matrix(
            python_versions=["3.12"],
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
            pytorch_refs=["release/2.10", "release/2.11"],
        )
        refs = {row["pytorch_git_ref"] for row in matrix}
        self.assertEqual(refs, {"release/2.10", "release/2.11"})


if __name__ == "__main__":
    unittest.main()
