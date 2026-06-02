#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_stage project resolution.

Note: Subproject name resolution (e.g., 'rocfft' -> 'fft') requires the
CMake-generated artifact_subprojects.json manifest. Tests here use artifact
names directly or split_databases entries which are available from TOML.

For full subproject resolution testing, run cmake configure first and use
--build-dir to point to the build directory.
"""

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
    """Test project name to CMake flag resolution.

    These tests verify resolution using:
    - Artifact names (always work)
    - split_databases entries (always work from TOML)

    Subproject name resolution (e.g., 'rocfft', 'tensile') requires the
    CMake manifest and is tested separately with a configured build.
    """

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

    # Tests using artifact names (always available)
    def test_core_hip_artifact(self):
        """Test core-hip artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", self._get_flags(["core-hip"]))

    @unittest.skipIf(IS_WINDOWS, "core-runtime disabled on Windows")
    def test_core_runtime_artifact(self):
        """Test core-runtime artifact name resolves correctly."""
        self.assertIn(
            "-DTHEROCK_ENABLE_CORE_RUNTIME=ON", self._get_flags(["core-runtime"])
        )

    @unittest.skipIf(IS_WINDOWS, "rocprofiler-sdk disabled on Windows")
    def test_rocprofiler_sdk_artifact(self):
        """Test rocprofiler-sdk artifact name resolves correctly."""
        self.assertIn(
            "-DTHEROCK_ENABLE_ROCPROFV3=ON", self._get_flags(["rocprofiler-sdk"])
        )

    @unittest.skipIf(IS_WINDOWS, "rocprofiler-compute disabled on Windows")
    def test_rocprofiler_compute_artifact(self):
        """Test rocprofiler-compute artifact name resolves correctly."""
        self.assertIn(
            "-DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=ON",
            self._get_flags(["rocprofiler-compute"]),
        )

    def test_amd_dbgapi_artifact(self):
        """Test amd-dbgapi artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_AMD_DBGAPI=ON", self._get_flags(["amd-dbgapi"]))

    @unittest.skipIf(IS_WINDOWS, "rdc disabled on Windows")
    def test_rdc_artifact(self):
        """Test rdc artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_RDC=ON", self._get_flags(["rdc"]))

    def test_blas_artifact(self):
        """Test blas artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["blas"]))

    def test_prim_artifact(self):
        """Test prim artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", self._get_flags(["prim"]))

    def test_fft_artifact(self):
        """Test fft artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["fft"]))

    def test_miopen_artifact(self):
        """Test miopen artifact name resolves correctly."""
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", self._get_flags(["miopen"]))

    def test_composable_kernel_artifact(self):
        """Test composable-kernel artifact name resolves correctly."""
        self.assertIn(
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            self._get_flags(["composable-kernel"]),
        )

    def test_miopenprovider_artifact(self):
        """Test miopenprovider artifact name resolves correctly."""
        self.assertIn(
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON", self._get_flags(["miopenprovider"])
        )

    # Tests using split_databases entries (available from TOML)
    def test_rocblas_split_database(self):
        """Test rocblas (split_database entry) resolves to blas artifact."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["rocblas"]))

    def test_hipblaslt_split_database(self):
        """Test hipblaslt (split_database entry) resolves to blas artifact."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["hipblaslt"]))

    # Multiple projects test
    def test_multiple_artifacts(self):
        """Test multiple artifact names resolve correctly."""
        args = self._get_flags(["blas", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)


if __name__ == "__main__":
    unittest.main()
