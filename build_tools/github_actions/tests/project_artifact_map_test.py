#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for project_artifact_map module."""

import os
import sys
import unittest
from pathlib import Path

# build_tools/github_actions/tests -> build_tools
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from _therock_utils.project_artifact_map import (
    get_artifact_for_path,
    get_all_rocm_libraries_artifacts,
    parse_changed_path,
    resolve_artifacts_for_paths,
    ROCM_LIBRARIES_PROJECT_MAP,
    ROCM_LIBRARIES_GLOBAL_PATHS,
)


class GetArtifactForPathTest(unittest.TestCase):
    """Test cases for get_artifact_for_path function."""

    def test_rocblas_maps_to_blas(self):
        """rocblas project should map to blas artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocblas/src/foo.cpp"),
            "blas",
        )

    def test_hipblas_maps_to_blas(self):
        """hipblas project should map to blas artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipblas/include/foo.hpp"),
            "blas",
        )

    def test_hipblaslt_maps_to_blas(self):
        """hipblaslt project should map to blas artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipblaslt/CMakeLists.txt"),
            "blas",
        )

    def test_rocfft_maps_to_fft(self):
        """rocfft project should map to fft artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocfft/src/kernel.cpp"),
            "fft",
        )

    def test_hipfft_maps_to_fft(self):
        """hipfft project should map to fft artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipfft/src/hipfft.cpp"),
            "fft",
        )

    def test_rocprim_maps_to_prim(self):
        """rocprim project should map to prim artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocprim/include/prim.hpp"),
            "prim",
        )

    def test_hipcub_maps_to_prim(self):
        """hipcub project should map to prim artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipcub/src/cub.cpp"),
            "prim",
        )

    def test_rocthrust_maps_to_prim(self):
        """rocthrust project should map to prim artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocthrust/include/foo.hpp"),
            "prim",
        )

    def test_rocrand_maps_to_rand(self):
        """rocrand project should map to rand artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocrand/src/random.cpp"),
            "rand",
        )

    def test_hiprand_maps_to_rand(self):
        """hiprand project should map to rand artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hiprand/include/hiprand.h"),
            "rand",
        )

    def test_composablekernel_maps_to_composable_kernel(self):
        """composablekernel project should map to composable-kernel artifact."""
        self.assertEqual(
            get_artifact_for_path(
                "rocm-libraries", "projects/composablekernel/src/ck.cpp"
            ),
            "composable-kernel",
        )

    def test_hiptensor_maps_to_hiptensor(self):
        """hiptensor project should map to hiptensor artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hiptensor/src/tensor.cpp"),
            "hiptensor",
        )

    def test_rocwmma_maps_to_rocwmma(self):
        """rocwmma project should map to rocwmma artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocwmma/src/wmma.cpp"),
            "rocwmma",
        )

    def test_rocalution_maps_to_rocalution(self):
        """rocalution project should map to rocalution artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocalution/src/solver.cpp"),
            "rocalution",
        )

    def test_miopen_maps_to_miopen(self):
        """miopen project should map to miopen artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/miopen/src/conv.cpp"),
            "miopen",
        )

    def test_hipdnn_maps_to_hipdnn(self):
        """hipdnn project should map to hipdnn artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipdnn/src/dnn.cpp"),
            "hipdnn",
        )

    def test_solver_projects(self):
        """Solver projects should map to solver artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipsolver/src/foo.cpp"),
            "solver",
        )
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocsolver/src/foo.cpp"),
            "solver",
        )

    def test_sparse_projects(self):
        """Sparse projects should map to sparse artifact."""
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/hipsparse/src/foo.cpp"),
            "sparse",
        )
        self.assertEqual(
            get_artifact_for_path("rocm-libraries", "projects/rocsparse/src/foo.cpp"),
            "sparse",
        )


class GlobalPathsTest(unittest.TestCase):
    """Test cases for global paths that affect all artifacts."""

    def test_shared_dir_returns_none(self):
        """shared/ directory should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", "shared/common.hpp")
        )

    def test_cmake_dir_returns_none(self):
        """cmake/ directory should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", "cmake/FindHIP.cmake")
        )

    def test_root_cmakelists_returns_none(self):
        """Root CMakeLists.txt should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", "CMakeLists.txt")
        )

    def test_github_dir_returns_none(self):
        """.github/ directory should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", ".github/workflows/ci.yml")
        )

    def test_dnn_providers_returns_none(self):
        """dnn-providers/ directory should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", "dnn-providers/foo/bar.cpp")
        )

    def test_tools_dir_returns_none(self):
        """tools/ directory should return None (affects all)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-libraries", "tools/script.py")
        )


class NonRocmLibrariesTest(unittest.TestCase):
    """Test cases for non-rocm-libraries submodules."""

    def test_llvm_project_returns_none(self):
        """llvm-project submodule should return None (no granular mapping)."""
        self.assertIsNone(
            get_artifact_for_path("llvm-project", "llvm/lib/Target/AMDGPU/foo.cpp")
        )

    def test_rocm_systems_returns_none(self):
        """rocm-systems submodule should return None (no granular mapping)."""
        self.assertIsNone(
            get_artifact_for_path("rocm-systems", "projects/hip/src/hip.cpp")
        )


class ParseChangedPathTest(unittest.TestCase):
    """Test cases for parse_changed_path function."""

    def test_simple_path(self):
        """Simple two-component path should split correctly."""
        submodule, subpath = parse_changed_path("rocm-libraries/projects/rocblas/foo.cpp")
        self.assertEqual(submodule, "rocm-libraries")
        self.assertEqual(subpath, "projects/rocblas/foo.cpp")

    def test_single_component(self):
        """Single component path should return empty subpath."""
        submodule, subpath = parse_changed_path("rocm-libraries")
        self.assertEqual(submodule, "rocm-libraries")
        self.assertEqual(subpath, "")

    def test_nested_path(self):
        """Deeply nested path should split at first slash."""
        submodule, subpath = parse_changed_path("build_tools/github_actions/tests/foo.py")
        self.assertEqual(submodule, "build_tools")
        self.assertEqual(subpath, "github_actions/tests/foo.py")


class GetAllRocmLibrariesArtifactsTest(unittest.TestCase):
    """Test cases for get_all_rocm_libraries_artifacts function."""

    def test_returns_frozenset(self):
        """Should return a frozenset."""
        artifacts = get_all_rocm_libraries_artifacts()
        self.assertIsInstance(artifacts, frozenset)

    def test_contains_expected_artifacts(self):
        """Should contain all expected artifact names."""
        artifacts = get_all_rocm_libraries_artifacts()
        expected = {"blas", "fft", "prim", "rand", "miopen", "composable-kernel"}
        self.assertTrue(expected.issubset(artifacts))


class ResolveArtifactsForPathsTest(unittest.TestCase):
    """Test cases for resolve_artifacts_for_paths function."""

    def test_single_project_change(self):
        """Single project change should return single artifact."""
        impacted, conservative = resolve_artifacts_for_paths(
            ["rocm-libraries/projects/rocblas/src/foo.cpp"],
            {"rocm-libraries": ["rocm-libraries"]},
        )
        self.assertEqual(impacted, {"blas"})
        self.assertFalse(conservative)

    def test_multiple_project_changes(self):
        """Multiple project changes should return multiple artifacts."""
        impacted, conservative = resolve_artifacts_for_paths(
            [
                "rocm-libraries/projects/rocblas/src/foo.cpp",
                "rocm-libraries/projects/rocfft/src/bar.cpp",
            ],
            {"rocm-libraries": ["rocm-libraries"]},
        )
        self.assertEqual(impacted, {"blas", "fft"})
        self.assertFalse(conservative)

    def test_global_path_triggers_conservative(self):
        """Global path change should trigger conservative fallback."""
        impacted, conservative = resolve_artifacts_for_paths(
            ["rocm-libraries/shared/common.hpp"],
            {"rocm-libraries": ["rocm-libraries"]},
        )
        self.assertTrue(conservative)
        # Should include all artifacts from rocm-libraries
        self.assertTrue(len(impacted) > 5)

    def test_mixed_changes(self):
        """Mixed specific and global changes should be conservative."""
        impacted, conservative = resolve_artifacts_for_paths(
            [
                "rocm-libraries/projects/rocblas/src/foo.cpp",
                "rocm-libraries/cmake/FindHIP.cmake",
            ],
            {"rocm-libraries": ["rocm-libraries"]},
        )
        self.assertTrue(conservative)
        self.assertIn("blas", impacted)

    def test_non_rocm_libraries_ignored(self):
        """Non-rocm-libraries paths should be ignored."""
        impacted, conservative = resolve_artifacts_for_paths(
            ["llvm-project/llvm/lib/Target/foo.cpp"],
            {"compilers": ["llvm-project"]},
        )
        self.assertEqual(impacted, set())
        self.assertFalse(conservative)

    def test_empty_input(self):
        """Empty input should return empty set."""
        impacted, conservative = resolve_artifacts_for_paths(
            [],
            {"rocm-libraries": ["rocm-libraries"]},
        )
        self.assertEqual(impacted, set())
        self.assertFalse(conservative)


class ProjectMappingCompletenessTest(unittest.TestCase):
    """Test that the project mapping covers actual rocm-libraries projects."""

    def test_all_projects_have_mapping(self):
        """Known rocm-libraries projects should have artifact mappings."""
        # These are the actual project directories in rocm-libraries/projects/
        expected_projects = [
            "rocblas",
            "hipblas",
            "hipblaslt",
            "rocfft",
            "hipfft",
            "rocprim",
            "hipcub",
            "rocthrust",
            "rocrand",
            "hiprand",
            "composablekernel",
            "hiptensor",
            "rocwmma",
            "rocalution",
            "miopen",
            "hipdnn",
            "hipsolver",
            "rocsolver",
            "hipsparse",
            "rocsparse",
        ]
        for project in expected_projects:
            artifact = get_artifact_for_path(
                "rocm-libraries", f"projects/{project}/src/foo.cpp"
            )
            self.assertIsNotNone(
                artifact,
                f"Project 'projects/{project}' should have an artifact mapping",
            )


if __name__ == "__main__":
    unittest.main()
