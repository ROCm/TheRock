#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from configure_stage import generate_cmake_args, get_project_features


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

    def test_artifact_and_subproject_resolution(self):
        """Test artifact and subproject names resolve to correct flags."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["blas"]))
        self.assertIn("-DTHEROCK_ENABLE_FFT=ON", self._get_flags(["rocfft"]))
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["RocBLAS"]))

    def test_split_database_resolution(self):
        """Test split_database names resolve correctly."""
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", self._get_flags(["hipblaslt"]))
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", self._get_flags(["hipsparselt"]))

    def test_multiple_projects(self):
        """Test multiple projects enable multiple flags."""
        args = self._get_flags(["blas", "miopen", "rccl"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_RCCL=ON", args)


class FeatureOrientedResolutionTest(unittest.TestCase):
    """Tests for feature-oriented project resolution."""

    def setUp(self):
        self.topology = get_topology()

    def test_hipsparse_resolves_to_sparse(self):
        """hipSPARSE is in blas artifact but gated by SPARSE."""
        features = self.topology.resolve_projects_to_features(["hipSPARSE"])
        self.assertIn("SPARSE", features)
        self.assertNotIn("BLAS", features)

    def test_hipsolver_resolves_to_solver(self):
        """hipSOLVER is gated by SOLVER."""
        features = self.topology.resolve_projects_to_features(["hipSOLVER"])
        self.assertIn("SOLVER", features)

    def test_rocblas_resolves_to_blas(self):
        """rocBLAS resolves to BLAS (no override)."""
        features = self.topology.resolve_projects_to_features(["rocBLAS"])
        self.assertIn("BLAS", features)


class RocmSystemsMappingTest(unittest.TestCase):
    """Tests for rocm-systems project mappings."""

    def setUp(self):
        self.topology = get_topology()

    def test_hip_maps_to_core_hip(self):
        """rocm-systems 'hip' directory maps to core-hip artifact."""
        alias_map = self.topology.get_alias_to_artifact_map()
        self.assertEqual(alias_map.get("hip"), "core-hip")

    def test_canonical_artifact_not_overridden(self):
        """Canonical artifact names should not be overridden."""
        alias_map = self.topology.get_alias_to_artifact_map()
        self.assertEqual(alias_map.get("rocprofiler-compute"), "rocprofiler-compute")


class ManifestValidationTest(unittest.TestCase):
    """Tests for manifest validation."""

    def test_project_mappings_has_valid_features(self):
        """Verify subproject_features in project_mappings.json has valid feature names."""
        manifest_path = Path(__file__).parent.parent / "project_mappings.json"
        if not manifest_path.exists():
            self.skipTest("project_mappings.json not found")

        with manifest_path.open() as f:
            mappings = json.load(f)

        topology = get_topology()
        valid_features = {
            topology.get_artifact_feature_name(a) for a in topology.artifacts.values()
        }

        for subproject, feature in mappings.get("subproject_features", {}).items():
            self.assertIn(
                feature,
                valid_features,
                f"Invalid feature '{feature}' for subproject '{subproject}'",
            )

    def test_artifact_subprojects_matches_cmake(self):
        """Verify artifact_subprojects.json matches what CMake generates."""
        repo_root = Path(__file__).parent.parent.parent
        # Skip if submodules aren't fetched (required for CMake configure)
        hip_version = repo_root / "rocm-systems" / "projects" / "hip" / "VERSION"
        if not hip_version.exists():
            self.skipTest("Submodules not fetched")

        script = repo_root / "build_tools" / "generate_subproject_manifest.py"
        result = subprocess.run(
            [sys.executable, str(script), "--verify"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"artifact_subprojects.json is out of sync:\n{result.stdout}{result.stderr}",
        )


class StageProjectFilterTest(unittest.TestCase):
    """Tests for stage-specific project filtering."""

    def setUp(self):
        self.topology = get_topology()

    def test_filter_hip_for_compiler_runtime(self):
        """Test that hip is included when filtering for compiler-runtime."""
        filtered = self.topology.filter_projects_for_stage(
            ["hip", "rccl", "rocblas"], "compiler-runtime"
        )
        self.assertIn("hip", filtered)
        self.assertNotIn("rccl", filtered)
        self.assertNotIn("rocblas", filtered)

    def test_filter_rccl_for_comm_libs(self):
        """Test that rccl is included when filtering for comm-libs."""
        filtered = self.topology.filter_projects_for_stage(
            ["hip", "rccl", "rocblas"], "comm-libs"
        )
        self.assertNotIn("hip", filtered)
        self.assertIn("rccl", filtered)
        self.assertNotIn("rocblas", filtered)

    def test_filter_rocblas_for_math_libs(self):
        """Test that rocblas is included when filtering for math-libs."""
        filtered = self.topology.filter_projects_for_stage(
            ["hip", "rccl", "rocblas"], "math-libs"
        )
        self.assertNotIn("hip", filtered)
        self.assertNotIn("rccl", filtered)
        self.assertIn("rocblas", filtered)

    def test_filter_empty_for_unrelated_stage(self):
        """Test that filtering returns empty for unrelated stage."""
        filtered = self.topology.filter_projects_for_stage(
            ["hip", "rccl"], "media-libs"
        )
        self.assertEqual(filtered, [])

    def test_filter_unknown_stage_returns_empty(self):
        """Test that filtering for unknown stage returns empty."""
        filtered = self.topology.filter_projects_for_stage(["hip"], "nonexistent-stage")
        self.assertEqual(filtered, [])


class StageAndProjectsCombinedTest(unittest.TestCase):
    """Tests for combined --stage and --projects behavior."""

    def setUp(self):
        self.topology = get_topology()

    def test_stage_with_matching_project(self):
        """Test that --stage compiler-runtime --projects hip enables only HIP."""
        from build_tools.configure_stage import generate_cmake_args

        args = generate_cmake_args(
            stage_name="compiler-runtime",
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=["hip"],
        )
        args_str = " ".join(args)
        self.assertIn("THEROCK_ENABLE_HIP_RUNTIME=ON", args_str)
        # Should NOT enable unrelated features
        self.assertNotIn("THEROCK_ENABLE_RCCL=ON", args_str)
        self.assertNotIn("THEROCK_ENABLE_BLAS=ON", args_str)

    def test_stage_with_non_matching_project(self):
        """Test that --stage math-libs --projects hip falls back to stage defaults."""
        from build_tools.configure_stage import generate_cmake_args

        args = generate_cmake_args(
            stage_name="math-libs",
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=["hip"],  # hip is NOT in math-libs
        )
        args_str = " ".join(args)
        # Should fall back to math-libs stage features (BLAS, FFT, etc.)
        self.assertIn("THEROCK_ENABLE_BLAS=ON", args_str)
        self.assertIn("THEROCK_ENABLE_FFT=ON", args_str)
        # HIP_RUNTIME is included as an inbound dependency for math-libs
        # (this is expected - stages need their dependencies enabled)

    def test_stage_with_mixed_projects(self):
        """Test --stage comm-libs --projects hip rccl enables only rccl."""
        from build_tools.configure_stage import generate_cmake_args

        args = generate_cmake_args(
            stage_name="comm-libs",
            amdgpu_families="",
            dist_amdgpu_families="",
            topology=self.topology,
            project_names=["hip", "rccl"],  # Only rccl is in comm-libs
        )
        args_str = " ".join(args)
        self.assertIn("THEROCK_ENABLE_RCCL=ON", args_str)
        # hip should be filtered out since it's not in comm-libs
        self.assertNotIn("THEROCK_ENABLE_HIP_RUNTIME=ON", args_str)


if __name__ == "__main__":
    unittest.main()
