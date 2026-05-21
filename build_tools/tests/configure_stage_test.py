#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_stage project resolution."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from configure_stage import generate_cmake_args


class ProjectResolutionTest(unittest.TestCase):
    """Test project name to CMake flag resolution."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def test_single_project_generates_correct_flag(self):
        """Test that rocblas generates -DTHEROCK_ENABLE_BLAS=ON."""
        args = generate_cmake_args(
            stage_name=None,
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=["rocblas"],
        )
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_multiple_projects_generate_correct_flags(self):
        """Test that rocblas + miopen generates both flags."""
        args = generate_cmake_args(
            stage_name=None,
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=["rocblas", "miopen"],
        )
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)


if __name__ == "__main__":
    unittest.main()
