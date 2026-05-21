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

    def _get_flags(self, projects):
        return generate_cmake_args(
            stage_name=None,
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=projects,
        )

    # --- rocm-systems mappings ---

    def test_amdsmi(self):
        args = self._get_flags(["amdsmi"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_AMDSMI=ON", args)

    def test_rocm_smi_lib(self):
        args = self._get_flags(["rocm-smi-lib"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_AMDSMI=ON", args)

    def test_aqlprofile(self):
        args = self._get_flags(["aqlprofile"])
        self.assertIn("-DTHEROCK_ENABLE_AQLPROFILE=ON", args)

    def test_clr(self):
        args = self._get_flags(["clr"])
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", args)

    def test_hip(self):
        args = self._get_flags(["hip"])
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", args)

    def test_hip_tests(self):
        args = self._get_flags(["hip-tests"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_HIPTESTS=ON", args)

    def test_hipother(self):
        args = self._get_flags(["hipother"])
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", args)

    def test_hotswap(self):
        args = self._get_flags(["hotswap"])
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", args)

    def test_rdc(self):
        args = self._get_flags(["rdc"])
        self.assertIn("-DTHEROCK_ENABLE_RDC=ON", args)

    def test_cuid(self):
        args = self._get_flags(["cuid"])
        self.assertIn("-DTHEROCK_ENABLE_RDC=ON", args)

    def test_rocdbgapi(self):
        args = self._get_flags(["rocdbgapi"])
        self.assertIn("-DTHEROCK_ENABLE_AMD_DBGAPI=ON", args)

    def test_amd_dbgapi(self):
        args = self._get_flags(["amd-dbgapi"])
        self.assertIn("-DTHEROCK_ENABLE_AMD_DBGAPI=ON", args)

    def test_rocprofiler(self):
        args = self._get_flags(["rocprofiler"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", args)

    def test_rocprofiler_sdk(self):
        args = self._get_flags(["rocprofiler-sdk"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", args)

    def test_rocprofiler_register(self):
        args = self._get_flags(["rocprofiler-register"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", args)

    def test_roctracer(self):
        args = self._get_flags(["roctracer"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", args)

    def test_rocprofiler_compute(self):
        args = self._get_flags(["rocprofiler-compute"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFILER_COMPUTE=ON", args)

    def test_rocprofiler_systems(self):
        args = self._get_flags(["rocprofiler-systems"])
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFSYS=ON", args)

    def test_rocr_debug_agent(self):
        args = self._get_flags(["rocr-debug-agent"])
        self.assertIn("-DTHEROCK_ENABLE_ROCR_DEBUG_AGENT=ON", args)

    def test_rocr_runtime(self):
        args = self._get_flags(["rocr-runtime"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_RUNTIME=ON", args)

    def test_rocshmem(self):
        args = self._get_flags(["rocshmem"])
        self.assertIn("-DTHEROCK_ENABLE_ROCSHMEM=ON", args)

    def test_rocjitsu(self):
        args = self._get_flags(["rocjitsu"])
        self.assertIn("-DTHEROCK_ENABLE_ROCJITSU=ON", args)

    def test_mirage(self):
        args = self._get_flags(["mirage"])
        self.assertIn("-DTHEROCK_ENABLE_ROCJITSU=ON", args)

    # --- rocm-libraries mappings ---

    def test_composablekernel(self):
        args = self._get_flags(["composablekernel"])
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", args)

    def test_hipblas(self):
        args = self._get_flags(["hipblas"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_hipblas_common(self):
        args = self._get_flags(["hipblas-common"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_hipblaslt(self):
        args = self._get_flags(["hipblaslt"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_rocblas(self):
        args = self._get_flags(["rocblas"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_tensile(self):
        args = self._get_flags(["tensile"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_rocroller(self):
        args = self._get_flags(["rocroller"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_hipcub(self):
        args = self._get_flags(["hipcub"])
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", args)

    def test_rocprim(self):
        args = self._get_flags(["rocprim"])
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", args)

    def test_rocthrust(self):
        args = self._get_flags(["rocthrust"])
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", args)

    def test_hipdnn(self):
        args = self._get_flags(["hipdnn"])
        self.assertIn("-DTHEROCK_ENABLE_HIPDNN=ON", args)

    def test_hipfft(self):
        args = self._get_flags(["hipfft"])
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", args)

    def test_rocfft(self):
        args = self._get_flags(["rocfft"])
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", args)

    def test_hiprand(self):
        args = self._get_flags(["hiprand"])
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", args)

    def test_rocrand(self):
        args = self._get_flags(["rocrand"])
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", args)

    def test_hipsolver(self):
        args = self._get_flags(["hipsolver"])
        self.assertIn("-DTHEROCK_ENABLE_SOLVER=ON", args)

    def test_rocsolver(self):
        args = self._get_flags(["rocsolver"])
        self.assertIn("-DTHEROCK_ENABLE_SOLVER=ON", args)

    def test_hipsparse(self):
        args = self._get_flags(["hipsparse"])
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", args)

    def test_rocsparse(self):
        args = self._get_flags(["rocsparse"])
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", args)

    def test_hipsparselt(self):
        args = self._get_flags(["hipsparselt"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_miopen(self):
        args = self._get_flags(["miopen"])
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)

    def test_rocwmma(self):
        args = self._get_flags(["rocwmma"])
        self.assertIn("-DTHEROCK_ENABLE_ROCWMMA=ON", args)

    # --- dnn-providers mappings ---

    def test_fusilli_provider(self):
        args = self._get_flags(["fusilli-provider"])
        self.assertIn("-DTHEROCK_ENABLE_FUSILLIPROVIDER=ON", args)

    def test_hipblaslt_provider(self):
        args = self._get_flags(["hipblaslt-provider"])
        self.assertIn("-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON", args)

    def test_hip_kernel_provider(self):
        args = self._get_flags(["hip-kernel-provider"])
        self.assertIn("-DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON", args)

    def test_miopen_provider(self):
        args = self._get_flags(["miopen-provider"])
        self.assertIn("-DTHEROCK_ENABLE_MIOPENPROVIDER=ON", args)

    def test_integration_tests(self):
        args = self._get_flags(["integration-tests"])
        self.assertIn("-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON", args)

    # --- Multiple projects ---

    def test_multiple_projects(self):
        args = self._get_flags(["rocblas", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)


if __name__ == "__main__":
    unittest.main()
