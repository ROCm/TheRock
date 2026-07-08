#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from configure_stage import (
    generate_cmake_args,
    get_project_features,
    get_stage_features,
)


class ProjectResolutionTest(unittest.TestCase):
    """Tests for --projects flag project name resolution."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def _get_flags(self, projects, **kwargs):
        return generate_cmake_args(
            stage_name=kwargs.get("stage_name"),
            amdgpu_families=kwargs.get("amdgpu_families", ""),
            dist_amdgpu_families=kwargs.get("dist_amdgpu_families", ""),
            topology=self.topology,
            project_names=projects,
            platform_name=kwargs.get("platform_name", "linux"),
            build_dir=kwargs.get("build_dir"),
        )

    # --- Basic artifact name resolution ---

    def test_artifact_name_resolution(self):
        """Test that artifact names resolve to correct flags."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["blas"]))
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["fft"]))
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", self._get_flags(["rand"]))
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", self._get_flags(["miopen"]))

    def test_subproject_resolution(self):
        """Test that subproject names resolve to artifact flags."""
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["rocfft"]))
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["hipfft"]))
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["rocblas"]))
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", self._get_flags(["rocrand"]))
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", self._get_flags(["hiprand"]))

    # --- Case insensitivity ---

    def test_case_insensitive_resolution(self):
        """Test that project names are case-insensitive."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["ROCBLAS"]))
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["RocBLAS"]))
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["ROCFFT"]))
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", self._get_flags(["MIOpen"]))

    # --- Multiple projects ---

    def test_multiple_projects(self):
        """Test that multiple projects enable multiple flags."""
        args = self._get_flags(["blas", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)

    def test_multiple_subprojects_same_artifact(self):
        """Test that multiple subprojects from same artifact don't duplicate flags."""
        args = self._get_flags(["rocfft", "hipfft"])
        # Should only have one FFT flag
        fft_count = sum(1 for a in args if "THEROCK_ENABLE_FFT=ON" in a)
        self.assertEqual(fft_count, 1)

    def test_mixed_artifacts_and_subprojects(self):
        """Test mixing artifact names and subproject names."""
        args = self._get_flags(["blas", "rocfft", "miopen"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)

    # --- split_databases resolution ---

    def test_split_database_rocblas(self):
        """Test that rocblas (split_database) maps to blas artifact."""
        args = self._get_flags(["rocblas"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_split_database_hipblaslt(self):
        """Test that hipblaslt (split_database) maps to blas artifact."""
        args = self._get_flags(["hipblaslt"])
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_hipsparselt_enables_sparse(self):
        """Test that hipsparselt maps to sparse artifact (not blas)."""
        args = self._get_flags(["hipsparselt"])
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", args)
        # Should NOT enable BLAS for hipsparselt
        self.assertNotIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_split_database_miopen(self):
        """Test that miopen (split_database) maps to miopen artifact."""
        args = self._get_flags(["miopen"])
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)

    # --- Empty and edge cases ---

    def test_empty_projects_list(self):
        """Test that empty projects list returns only ENABLE_ALL=OFF."""
        args = self._get_flags([])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        # Should have no ENABLE_*=ON flags except ALL=OFF
        enable_on_flags = [a for a in args if "=ON" in a]
        self.assertEqual(len(enable_on_flags), 0)

    def test_unknown_project_returns_no_features(self):
        """Test that unknown project names don't produce features."""
        features = get_project_features(
            self.topology, ["nonexistent_project"], platform_name="linux"
        )
        self.assertEqual(len(features), 0)

    # --- Core artifacts ---

    def test_core_runtime_resolution(self):
        """Test core-runtime artifact and subprojects."""
        args = self._get_flags(["core-runtime"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_RUNTIME=ON", args)

    def test_rocr_runtime_subproject(self):
        """Test ROCR-Runtime subproject resolution."""
        args = self._get_flags(["rocr-runtime"])
        self.assertIn("-DTHEROCK_ENABLE_CORE_RUNTIME=ON", args)

    def test_hip_clr_resolution(self):
        """Test hip-clr subproject resolution."""
        args = self._get_flags(["hip-clr"])
        self.assertIn("-DTHEROCK_ENABLE_HIP_RUNTIME=ON", args)

    # --- Compiler artifacts ---

    def test_amd_llvm_resolution(self):
        """Test amd-llvm artifact resolution."""
        args = self._get_flags(["amd-llvm"])
        self.assertIn("-DTHEROCK_ENABLE_COMPILER=ON", args)

    def test_hipcc_subproject(self):
        """Test hipcc subproject maps to compiler."""
        args = self._get_flags(["hipcc"])
        self.assertIn("-DTHEROCK_ENABLE_COMPILER=ON", args)

    # --- Profiler artifacts ---

    def test_rocprofiler_sdk_resolution(self):
        """Test rocprofiler-sdk artifact resolution."""
        args = self._get_flags(["rocprofiler-sdk"])
        # rocprofiler-sdk artifact has feature_name ROCPROFV3
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", args)

    # --- Communication libraries ---

    def test_rccl_resolution(self):
        """Test rccl artifact resolution."""
        args = self._get_flags(["rccl"])
        self.assertIn("-DTHEROCK_ENABLE_RCCL=ON", args)


class StageResolutionTest(unittest.TestCase):
    """Tests for --stage flag stage resolution."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def test_stage_features(self):
        """Test that stage resolution returns expected features."""
        features = get_stage_features(self.topology, "math-libs", platform_name="linux")
        self.assertIn("BLAS", features)
        self.assertIn("FFT", features)
        self.assertIn("RAND", features)

    def test_compiler_runtime_stage(self):
        """Test compiler-runtime stage includes compiler features."""
        features = get_stage_features(
            self.topology, "compiler-runtime", platform_name="linux"
        )
        self.assertIn("COMPILER", features)


class AliasMapTest(unittest.TestCase):
    """Tests for alias map generation and resolution."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def test_alias_map_includes_artifact_names(self):
        """Test that alias map includes artifact names."""
        alias_map = self.topology.get_alias_to_artifact_map()
        self.assertIn("blas", alias_map)
        self.assertIn("fft", alias_map)
        self.assertIn("miopen", alias_map)

    def test_alias_map_includes_split_databases(self):
        """Test that alias map includes split_database names."""
        alias_map = self.topology.get_alias_to_artifact_map()
        self.assertIn("rocblas", alias_map)
        self.assertIn("hipblaslt", alias_map)
        self.assertIn("hipsparselt", alias_map)

    def test_alias_map_lowercases_keys(self):
        """Test that alias map keys are lowercase."""
        alias_map = self.topology.get_alias_to_artifact_map()
        for key in alias_map.keys():
            self.assertEqual(key, key.lower())

    def test_resolve_project_to_artifact(self):
        """Test direct project to artifact resolution."""
        self.assertEqual(self.topology.resolve_project_to_artifact("rocblas"), "blas")
        self.assertEqual(
            self.topology.resolve_project_to_artifact("hipsparselt"), "sparse"
        )
        self.assertEqual(self.topology.resolve_project_to_artifact("miopen"), "miopen")

    def test_resolve_unknown_returns_none(self):
        """Test that unknown project returns None."""
        self.assertIsNone(
            self.topology.resolve_project_to_artifact("nonexistent_project")
        )


class PlatformFilteringTest(unittest.TestCase):
    """Tests for platform-specific filtering."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def test_windows_disabled_artifacts_excluded(self):
        """Test that artifacts disabled on windows are excluded."""
        features = get_project_features(
            self.topology, ["rccl"], platform_name="windows"
        )
        # rccl is disabled on windows
        self.assertNotIn("RCCL", features)

    def test_linux_includes_all_artifacts(self):
        """Test that linux includes all artifacts."""
        features = get_project_features(self.topology, ["rccl"], platform_name="linux")
        self.assertIn("RCCL", features)


class StageSkipTest(unittest.TestCase):
    """Tests for --skip-stages functionality."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def test_hip_clr_includes_runtime_tests(self):
        """Test that hip-clr needs compiler-runtime and runtime-tests (via test_artifacts)."""
        required = self.topology.get_stages_for_projects(["hip-clr"])
        self.assertIn("compiler-runtime", required)
        self.assertIn("runtime-tests", required)  # Via test_artifacts
        self.assertNotIn("math-libs", required)
        self.assertNotIn("comm-libs", required)

    def test_rocblas_needs_math_libs(self):
        """Test that rocblas needs math-libs stage."""
        required = self.topology.get_stages_for_projects(["rocblas"])
        self.assertIn("compiler-runtime", required)
        self.assertIn("math-libs", required)
        self.assertNotIn("comm-libs", required)

    def test_rccl_needs_comm_libs(self):
        """Test that rccl needs comm-libs stage."""
        required = self.topology.get_stages_for_projects(["rccl"])
        self.assertIn("compiler-runtime", required)
        self.assertIn("comm-libs", required)
        self.assertNotIn("math-libs", required)

    def test_hip_tests_needs_runtime_tests(self):
        """Test that hip-tests needs runtime-tests stage."""
        required = self.topology.get_stages_for_projects(["hip-tests"])
        self.assertIn("compiler-runtime", required)
        self.assertIn("runtime-tests", required)
        self.assertNotIn("math-libs", required)

    def test_rocr_runtime_includes_test_artifacts(self):
        """Test that ROCR-Runtime includes hip-tests and rocrtst via test_artifacts."""
        required = self.topology.get_stages_for_projects(["rocr-runtime"])
        self.assertIn("compiler-runtime", required)
        self.assertIn(
            "runtime-tests", required
        )  # Via test_artifacts (core-hiptests, rocrtst)
        self.assertNotIn("math-libs", required)

    def test_multiple_projects_combines_stages(self):
        """Test that multiple projects combine their stage requirements."""
        required = self.topology.get_stages_for_projects(["rocblas", "rccl"])
        self.assertIn("compiler-runtime", required)
        self.assertIn("math-libs", required)
        self.assertIn("comm-libs", required)


if __name__ == "__main__":
    unittest.main()
