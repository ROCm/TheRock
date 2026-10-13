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
from configure_stage import (
    filter_artifacts_for_stage,
    generate_cmake_args,
    get_artifact_features,
)


class ArtifactResolutionTest(unittest.TestCase):
    """Tests for --artifacts flag artifact name resolution."""

    @classmethod
    def setUpClass(cls):
        cls.topology = get_topology()

    def _get_flags(self, artifacts, **kwargs):
        return generate_cmake_args(
            stage_name=kwargs.get("stage_name"),
            amdgpu_families=kwargs.get("amdgpu_families", ""),
            dist_amdgpu_families=kwargs.get("dist_amdgpu_families", ""),
            topology=self.topology,
            artifact_names=artifacts,
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

    def test_multiple_artifacts(self):
        """Test multiple artifacts enable multiple flags."""
        args = self._get_flags(["blas", "miopen", "rccl"])
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", args)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", args)
        self.assertIn("-DTHEROCK_ENABLE_RCCL=ON", args)


class FeatureOrientedResolutionTest(unittest.TestCase):
    """Tests for feature-oriented artifact resolution."""

    def setUp(self):
        self.topology = get_topology()

    def test_hipsparse_resolves_to_sparse(self):
        """hipSPARSE is in blas artifact but gated by SPARSE."""
        features = self.topology.resolve_artifacts_to_features(["hipSPARSE"])
        self.assertIn("SPARSE", features)
        self.assertNotIn("BLAS", features)

    def test_hipsolver_resolves_to_solver(self):
        """hipSOLVER is gated by SOLVER."""
        features = self.topology.resolve_artifacts_to_features(["hipSOLVER"])
        self.assertIn("SOLVER", features)

    def test_rocblas_resolves_to_blas(self):
        """rocBLAS resolves to BLAS (no override)."""
        features = self.topology.resolve_artifacts_to_features(["rocBLAS"])
        self.assertIn("BLAS", features)


class RocmSystemsMappingTest(unittest.TestCase):
    """Tests for rocm-systems artifact mappings."""

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


class StageArtifactFilteringTest(unittest.TestCase):
    """Tests for filtering artifacts by stage."""

    def setUp(self):
        self.topology = get_topology()

    def test_artifacts_filtered_to_stage(self):
        """Artifacts not belonging to a stage are filtered out in cmake args."""
        # RPP is in cv-libs, blas is in math-libs - passing both to math-libs
        # should only enable blas
        args = generate_cmake_args(
            stage_name="math-libs",
            amdgpu_families="gfx1100",
            dist_amdgpu_families="",
            topology=self.topology,
            artifact_names=["rpp", "blas"],
            platform_name="linux",
        )
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", args)
        self.assertNotIn("-DTHEROCK_ENABLE_RPP=ON", args)

    def test_filter_returns_other_stage_artifacts(self):
        """filter_artifacts_for_stage returns artifacts from other stages."""
        filtered, other_stage, unknown = filter_artifacts_for_stage(
            self.topology, "math-libs", ["rpp", "blas", "miopen"]
        )
        # blas is in math-libs, miopen is in ml-libs (built in same stage), rpp is cv-libs
        self.assertIn("blas", filtered)
        self.assertIn("miopen", filtered)
        self.assertIn("rpp", other_stage)
        self.assertNotIn("blas", other_stage)
        self.assertEqual(unknown, [])

    def test_filter_returns_unknown_artifacts(self):
        """filter_artifacts_for_stage returns unknown artifact names."""
        filtered, other_stage, unknown = filter_artifacts_for_stage(
            self.topology, "math-libs", ["blas", "not-a-real-artifact", "also-fake"]
        )
        self.assertIn("blas", filtered)
        self.assertEqual(other_stage, [])
        self.assertIn("not-a-real-artifact", unknown)
        self.assertIn("also-fake", unknown)

    def test_sparse_subprojects_resolve_to_sparse_feature(self):
        """hipSPARSE/rocSPARSE resolve to SPARSE, not BLAS."""
        # These subprojects are in the blas artifact's source paths but should
        # enable SPARSE feature via project_mappings.json
        args = generate_cmake_args(
            stage_name="math-libs",
            amdgpu_families="gfx1100",
            dist_amdgpu_families="",
            topology=self.topology,
            artifact_names=["hipSPARSE"],
            platform_name="linux",
        )
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", args)
        self.assertNotIn("-DTHEROCK_ENABLE_BLAS=ON", args)

    def test_hipsparse_and_rocsparse_both_sparse(self):
        """Both hipSPARSE and rocSPARSE map to SPARSE feature."""
        for subproject in ["hipSPARSE", "rocSPARSE", "hipSPARSELt"]:
            features = self.topology.resolve_artifacts_to_features([subproject])
            self.assertIn("SPARSE", features, f"{subproject} should map to SPARSE")
            self.assertNotIn("BLAS", features, f"{subproject} should not map to BLAS")

    def test_hipsparselt_resolves_to_sparse(self):
        """hipSPARSELt is a split_database in sparse and resolves to SPARSE."""
        # hipsparselt is listed in sparse artifact's split_databases
        args = generate_cmake_args(
            stage_name="math-libs",
            amdgpu_families="gfx1100",
            dist_amdgpu_families="",
            topology=self.topology,
            artifact_names=["hipsparselt"],
            platform_name="linux",
        )
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", args)
        self.assertNotIn("-DTHEROCK_ENABLE_BLAS=ON", args)


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

    @unittest.skipIf(
        sys.platform == "win32",
        "manifest verification requires a Linux CMake configuration",
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


if __name__ == "__main__":
    unittest.main()
