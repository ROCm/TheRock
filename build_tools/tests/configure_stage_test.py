#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_stage project resolution."""

import os
import platform
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from configure_stage import generate_cmake_args

IS_WINDOWS = platform.system().lower() == "windows"


class ProjectResolutionTest(unittest.TestCase):
    """Test project name to CMake flag resolution."""

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

    # rocm-systems: runtime projects
    def test_clr(self):
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", self._get_flags(["clr"]))

    @unittest.skipIf(IS_WINDOWS, "core-runtime disabled on Windows")
    def test_rocr_runtime(self):
        self.assertIn(
            "-DTHEROCK_ENABLE_CORE_RUNTIME=ON", self._get_flags(["rocr-runtime"])
        )

    # rocm-systems: profiler projects
    @unittest.skipIf(IS_WINDOWS, "rocprofiler-sdk disabled on Windows")
    def test_rocprofiler_sdk(self):
        self.assertIn(
            "-DTHEROCK_ENABLE_ROCPROFV3=ON", self._get_flags(["rocprofiler-sdk"])
        )

    @unittest.skipIf(IS_WINDOWS, "rocprofiler-compute disabled on Windows")
    def test_rocprofiler_compute(self):
        self.assertIn(
            "-DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=ON",
            self._get_flags(["rocprofiler-compute"]),
        )

    # rocm-systems: debug/dc tools
    def test_rocdbgapi(self):
        self.assertIn("-DTHEROCK_ENABLE_AMD_DBGAPI=ON", self._get_flags(["rocdbgapi"]))

    @unittest.skipIf(IS_WINDOWS, "rdc disabled on Windows")
    def test_rdc(self):
        self.assertIn("-DTHEROCK_ENABLE_RDC=ON", self._get_flags(["rdc"]))

    # rocm-libraries: math libs
    def test_rocblas(self):
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["rocblas"]))

    def test_tensile(self):
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["tensile"]))

    def test_rocprim(self):
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", self._get_flags(["rocprim"]))

    def test_rocfft(self):
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["rocfft"]))

    # rocm-libraries: ml libs
    def test_miopen(self):
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", self._get_flags(["miopen"]))

    def test_composablekernel(self):
        self.assertIn(
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            self._get_flags(["composablekernel"]),
        )

    # rocm-libraries: dnn-providers
    def test_miopen_provider(self):
        self.assertIn(
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON", self._get_flags(["miopen-provider"])
        )

    # multiple projects
    def test_multiple_projects(self):
        args = self._get_flags(["rocblas", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)


if __name__ == "__main__":
    unittest.main()
