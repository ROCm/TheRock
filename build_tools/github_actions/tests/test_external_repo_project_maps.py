#!/usr/bin/env python3

"""Unit tests for external_repo_project_maps.py

These tests verify:
1. Project paths referenced in the maps actually exist in the external repos
2. The collect_projects_to_run logic works correctly
3. Dependency resolution and merging works as expected
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_repo_project_maps import (
    ROCM_LIBRARIES_SUBTREE_TO_PROJECT_MAP,
    ROCM_LIBRARIES_PROJECT_MAP,
    ROCM_SYSTEMS_SUBTREE_TO_PROJECT_MAP,
    ROCM_SYSTEMS_PROJECT_MAP,
    collect_projects_to_run,
    get_repo_config,
)


# Path to the workspace root (TheRock/)
THEROCK_ROOT = Path(__file__).parent.parent.parent.parent
ROCM_LIBRARIES_ROOT = THEROCK_ROOT / "rocm-libraries"
ROCM_SYSTEMS_ROOT = THEROCK_ROOT / "rocm-systems"


class TestRocmLibrariesPaths(unittest.TestCase):
    """Verify that paths in rocm-libraries project maps exist."""

    @unittest.skipIf(
        not ROCM_LIBRARIES_ROOT.exists(), "rocm-libraries submodule not initialized"
    )
    def test_subtree_paths_exist(self):
        """Verify all subtree paths in ROCM_LIBRARIES_SUBTREE_TO_PROJECT_MAP exist."""
        missing_paths = []

        for subtree_path in ROCM_LIBRARIES_SUBTREE_TO_PROJECT_MAP.keys():
            full_path = ROCM_LIBRARIES_ROOT / subtree_path
            if not full_path.exists():
                missing_paths.append(subtree_path)

        self.assertEqual(
            missing_paths,
            [],
            f"The following subtree paths do not exist in rocm-libraries: {missing_paths}",
        )


class TestRocmSystemsPaths(unittest.TestCase):
    """Verify that paths in rocm-systems project maps exist."""

    @unittest.skipIf(
        not ROCM_SYSTEMS_ROOT.exists(), "rocm-systems submodule not initialized"
    )
    def test_subtree_paths_exist(self):
        """Verify all subtree paths in ROCM_SYSTEMS_SUBTREE_TO_PROJECT_MAP exist."""
        missing_paths = []

        for subtree_path in ROCM_SYSTEMS_SUBTREE_TO_PROJECT_MAP.keys():
            full_path = ROCM_SYSTEMS_ROOT / subtree_path
            if not full_path.exists():
                missing_paths.append(subtree_path)

        self.assertEqual(
            missing_paths,
            [],
            f"The following subtree paths do not exist in rocm-systems: {missing_paths}",
        )


class TestCollectProjectsToRun(unittest.TestCase):
    """Test the collect_projects_to_run logic."""

    def test_basic_project_collection_rocm_libraries(self):
        """Test basic project collection for rocm-libraries."""
        # Test single project
        projects = collect_projects_to_run(
            subtrees=["projects/rocprim"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        self.assertEqual(len(projects), 1)
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", projects[0]["cmake_options"])
        self.assertEqual(projects[0]["project_to_test"], "rocprim,rocthrust,hipcub")

    def test_basic_project_collection_rocm_systems(self):
        """Test basic project collection for rocm-systems."""
        # Test single project
        projects = collect_projects_to_run(
            subtrees=["projects/hip"],
            platform="linux",
            repo_name="rocm-systems",
        )

        self.assertEqual(len(projects), 1)
        self.assertIn("-DTHEROCK_ENABLE_CORE=ON", projects[0]["cmake_options"])

    def test_multiple_subtrees_same_project(self):
        """Test that multiple subtrees mapping to the same project are deduplicated."""
        # Both rocprim and hipcub map to "prim"
        projects = collect_projects_to_run(
            subtrees=["projects/rocprim", "projects/hipcub"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        # Should result in only one "prim" project
        self.assertEqual(len(projects), 1)
        self.assertIn("-DTHEROCK_ENABLE_PRIM=ON", projects[0]["cmake_options"])

    def test_dependency_merging(self):
        """Test that dependencies are properly merged."""
        # miopen depends on blas and rand
        projects = collect_projects_to_run(
            subtrees=["projects/miopen", "projects/rocblas"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        # Should have miopen (with blas merged) - blas should be removed
        self.assertEqual(len(projects), 1)

        # Check that miopen has both its own and blas cmake options
        cmake_opts = projects[0]["cmake_options"]
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", cmake_opts)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", cmake_opts)

    def test_optional_component_merging(self):
        """Test that optional components (sparse, solver) are merged correctly."""
        # sparse is optional and should be added to blas
        projects = collect_projects_to_run(
            subtrees=["projects/rocsparse", "projects/rocblas"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        # Should result in one blas project with sparse options added
        self.assertEqual(len(projects), 1)
        cmake_opts = projects[0]["cmake_options"]
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", cmake_opts)
        self.assertIn("-DTHEROCK_ENABLE_SPARSE=ON", cmake_opts)

        # Check tests include both blas and sparse projects
        project_tests = projects[0]["project_to_test"]
        self.assertIn("rocblas", project_tests)
        self.assertIn("rocsparse", project_tests)

    def test_platform_specific_flags(self):
        """Test that platform-specific flags are added correctly."""
        # miopen uses CK from rocm-libraries/projects/composablekernel
        projects_linux = collect_projects_to_run(
            subtrees=["projects/miopen"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        projects_windows = collect_projects_to_run(
            subtrees=["projects/miopen"],
            platform="windows",
            repo_name="rocm-libraries",
        )

        # Should have CK enabled with source dir pointing to rocm-libraries CK
        linux_opts = projects_linux[0]["cmake_options"]
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", linux_opts)
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", linux_opts)
        self.assertIn("-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON", linux_opts)
        self.assertIn(
            "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../source-repo/projects/composablekernel",
            linux_opts,
        )

        windows_opts = projects_windows[0]["cmake_options"]
        self.assertIn("-DTHEROCK_ENABLE_MIOPEN=ON", windows_opts)
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", windows_opts)
        self.assertIn("-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON", windows_opts)
        self.assertIn(
            "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../source-repo/projects/composablekernel",
            windows_opts,
        )

    def test_enable_all_off_is_added(self):
        """Test that -DTHEROCK_ENABLE_ALL=OFF is always added."""
        projects = collect_projects_to_run(
            subtrees=["projects/rocprim"],
            platform="linux",
            repo_name="rocm-libraries",
        )

        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", projects[0]["cmake_options"])


class TestGetRepoConfig(unittest.TestCase):
    """Test get_repo_config function."""

    def test_rocm_libraries_config(self):
        """Test getting rocm-libraries configuration."""
        config = get_repo_config("rocm-libraries")

        self.assertIn("subtree_to_project_map", config)
        self.assertIn("project_map", config)
        self.assertIn("additional_options", config)
        self.assertIn("dependency_graph", config)

        # Verify it has the expected content
        self.assertIn("projects/rocprim", config["subtree_to_project_map"])
        self.assertIn("prim", config["project_map"])

    def test_rocm_systems_config(self):
        """Test getting rocm-systems configuration."""
        config = get_repo_config("rocm-systems")

        self.assertIn("subtree_to_project_map", config)
        self.assertIn("project_map", config)

        # Verify it has the expected content
        self.assertIn("projects/hip", config["subtree_to_project_map"])
        self.assertIn("core", config["project_map"])

    def test_case_insensitive_repo_name(self):
        """Test that repo name matching is case-insensitive."""
        config1 = get_repo_config("ROCm-Libraries")
        config2 = get_repo_config("rocm-LIBRARIES")

        self.assertEqual(config1, config2)

    def test_unknown_repo_raises_error(self):
        """Test that unknown repo name raises ValueError."""
        with self.assertRaises(ValueError):
            get_repo_config("unknown-repo")


if __name__ == "__main__":
    unittest.main()
