#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from configure_stage import generate_cmake_args


class ProjectResolutionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def _get_flags(self, projects):
        return generate_cmake_args(
            stage_name=None,
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=projects,
        )

    def test_artifact_name_resolution(self):
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["blas"]))
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["fft"]))

    def test_subproject_resolution(self):
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["rocfft"]))
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["rocblas"]))

    def test_multiple_projects(self):
        args = self._get_flags(["blas", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)


if __name__ == "__main__":
    unittest.main()
