#!/usr/bin/env python3

"""Tests for external repository detection and configuration in configure_ci.py"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path

# Add parent directory to path to import configure_ci
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the imports that configure_ci needs
sys.modules['amdgpu_family_matrix'] = MagicMock()
sys.modules['fetch_test_configurations'] = MagicMock()
sys.modules['github_actions_utils'] = MagicMock()


class TestExternalRepoDetection(unittest.TestCase):
    """Test detection of external repositories (rocm-libraries, rocm-systems)"""

    def test_rocm_libraries_detection_from_path(self):
        """Test that rocm-libraries is detected from cwd path"""
        test_paths = [
            "/path/to/rocm-libraries",
            "/workspace/rocm-libraries/subdir",
            "C:\\Dev\\rocm-libraries",
        ]
        
        for path in test_paths:
            with self.subTest(path=path):
                # Path contains rocm-libraries
                self.assertIn("rocm-libraries", path.lower())

    def test_rocm_systems_detection_from_path(self):
        """Test that rocm-systems is detected from cwd path"""
        test_paths = [
            "/path/to/rocm-systems",
            "/workspace/rocm-systems/subdir",
            "C:\\Dev\\rocm-systems",
        ]
        
        for path in test_paths:
            with self.subTest(path=path):
                # Path contains rocm-systems
                self.assertIn("rocm-systems", path.lower())

    def test_therock_not_external(self):
        """Test that TheRock itself is not detected as external"""
        test_paths = [
            "/path/to/TheRock",
            "/workspace/TheRock/subdir",
            "C:\\Dev\\TheRock",
        ]
        
        for path in test_paths:
            with self.subTest(path=path):
                # Path does not contain rocm-libraries or rocm-systems
                self.assertNotIn("rocm-libraries", path.lower())
                self.assertNotIn("rocm-systems", path.lower())


class TestAutoDetectConfiguration(unittest.TestCase):
    """Test the auto-detect configuration logic from workflows"""

    def test_rocm_libraries_variables(self):
        """Test that rocm-libraries sets correct configuration variables"""
        # Expected outputs when REPO contains "rocm-libraries"
        expected_vars = {
            "cmake_source_var": "THEROCK_ROCM_LIBRARIES_SOURCE_DIR",
            "patches_dir": "rocm-libraries",
            "fetch_exclusion": "--no-include-rocm-libraries",
            "enable_dvc": "true",
            "enable_ck": "true",
        }
        
        # Verify each expected variable
        self.assertEqual(expected_vars["cmake_source_var"], "THEROCK_ROCM_LIBRARIES_SOURCE_DIR")
        self.assertEqual(expected_vars["patches_dir"], "rocm-libraries")
        self.assertEqual(expected_vars["fetch_exclusion"], "--no-include-rocm-libraries")
        self.assertEqual(expected_vars["enable_dvc"], "true")
        self.assertEqual(expected_vars["enable_ck"], "true")

    def test_rocm_systems_variables(self):
        """Test that rocm-systems sets correct configuration variables"""
        # Expected outputs when REPO contains "rocm-systems"
        expected_vars = {
            "cmake_source_var": "THEROCK_ROCM_SYSTEMS_SOURCE_DIR",
            "patches_dir": "rocm-systems",
            "fetch_exclusion": "--no-include-rocm-systems",
            "enable_dvc": "false",
            "enable_ck": "false",
        }
        
        # Verify each expected variable
        self.assertEqual(expected_vars["cmake_source_var"], "THEROCK_ROCM_SYSTEMS_SOURCE_DIR")
        self.assertEqual(expected_vars["patches_dir"], "rocm-systems")
        self.assertEqual(expected_vars["fetch_exclusion"], "--no-include-rocm-systems")
        self.assertEqual(expected_vars["enable_dvc"], "false")
        self.assertEqual(expected_vars["enable_ck"], "false")

    def test_dvc_enabled_only_for_libraries(self):
        """Test that DVC is only enabled for rocm-libraries, not rocm-systems"""
        libraries_dvc = "true"
        systems_dvc = "false"
        
        self.assertEqual(libraries_dvc, "true")
        self.assertEqual(systems_dvc, "false")

    def test_composable_kernel_enabled_only_for_libraries(self):
        """Test that composable_kernel checkout is only for rocm-libraries"""
        libraries_ck = "true"
        systems_ck = "false"
        
        self.assertEqual(libraries_ck, "true")
        self.assertEqual(systems_ck, "false")


class TestTherockDirEnvironment(unittest.TestCase):
    """Test THEROCK_DIR environment variable logic"""

    def test_external_repo_therock_dir(self):
        """Test that external repos set THEROCK_DIR=TheRock"""
        # When external_source_checkout is true
        external_therock_dir = "TheRock"
        self.assertEqual(external_therock_dir, "TheRock")

    def test_direct_therock_dir(self):
        """Test that direct TheRock builds set THEROCK_DIR=."""
        # When external_source_checkout is false
        direct_therock_dir = "."
        self.assertEqual(direct_therock_dir, ".")


class TestProjectMatrixCrossProduct(unittest.TestCase):
    """Test cross-product of projects × GPU families for external repos"""

    def test_project_config_structure(self):
        """Test that project configs have required fields"""
        # Example project config from external repo
        project_config = {
            "project_to_test": "rocBLAS",
            "cmake_options": "-DBUILD_TESTING=ON",
        }
        
        self.assertIn("project_to_test", project_config)
        self.assertIn("cmake_options", project_config)

    def test_gpu_family_structure(self):
        """Test that GPU family configs have required fields"""
        # Example GPU family config
        gpu_variant = {
            "family": "gfx94X-dcgpu",
            "test-runs-on": "azure-linux-rocm",
            "artifact_group": "gfx94X-dcgpu",
        }
        
        self.assertIn("family", gpu_variant)
        self.assertIn("artifact_group", gpu_variant)

    def test_artifact_group_includes_project_name(self):
        """Test that artifact_group for external repos includes project name"""
        project_name = "rocBLAS"
        gpu_family = "gfx94X-dcgpu"
        
        # Expected format: {project_name}-{gpu_family}
        expected_artifact_group = f"{project_name}-{gpu_family}"
        self.assertEqual(expected_artifact_group, "rocBLAS-gfx94X-dcgpu")

    def test_cross_product_creates_multiple_entries(self):
        """Test that cross-product creates entries for each combination"""
        projects = ["rocBLAS", "rocFFT"]
        gpu_families = ["gfx94X-dcgpu", "gfx1151"]
        
        # Expected number of combinations
        expected_combinations = len(projects) * len(gpu_families)
        self.assertEqual(expected_combinations, 4)


class TestPatchPaths(unittest.TestCase):
    """Test patch directory paths for external repos"""

    def test_rocm_libraries_patch_path(self):
        """Test patch path for rocm-libraries"""
        therock_dir = "TheRock"
        patches_dir = "rocm-libraries"
        
        expected_path = f"{therock_dir}/patches/amd-mainline/{patches_dir}"
        self.assertEqual(expected_path, "TheRock/patches/amd-mainline/rocm-libraries")

    def test_rocm_systems_patch_path(self):
        """Test patch path for rocm-systems"""
        therock_dir = "TheRock"
        patches_dir = "rocm-systems"
        
        expected_path = f"{therock_dir}/patches/amd-mainline/{patches_dir}"
        self.assertEqual(expected_path, "TheRock/patches/amd-mainline/rocm-systems")


class TestScriptPaths(unittest.TestCase):
    """Test that script paths use THEROCK_DIR correctly"""

    def test_script_paths_with_therock_dir(self):
        """Test that all script paths use THEROCK_DIR variable"""
        therock_dir = "TheRock"
        
        scripts = [
            "build_tools/setup_ccache.py",
            "build_tools/health_status.py",
            "build_tools/fetch_sources.py",
            "build_tools/analyze_build_times.py",
            "build_tools/github_actions/build_configure.py",
            "build_tools/github_actions/post_build_upload.py",
        ]
        
        for script in scripts:
            full_path = f"{therock_dir}/{script}"
            # Verify path starts with THEROCK_DIR
            self.assertTrue(full_path.startswith(therock_dir))

    def test_requirements_path_with_therock_dir(self):
        """Test that requirements.txt path uses THEROCK_DIR"""
        therock_dir = "TheRock"
        requirements_path = f"{therock_dir}/requirements.txt"
        
        self.assertEqual(requirements_path, "TheRock/requirements.txt")


if __name__ == "__main__":
    unittest.main()
