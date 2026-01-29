#!/usr/bin/env python3

"""Unit tests for external repo CI detection logic

Tests for both detect_external_repo_config.py and configure_ci.py functions.
These tests verify:
1. Dynamic import from external repos works correctly
2. Skip pattern detection uses external repo patterns
3. Test list extraction from external repo project maps
4. Full build configuration is returned (ignoring cmake_options)
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_external_repo_config import (
    import_external_repo_module,
    get_skip_patterns,
    get_test_list,
)
from configure_ci import _detect_external_repo_projects


class TestDynamicImport(unittest.TestCase):
    """Test dynamic import from external repos.

    These tests will skip when external repos are not available, which happens when:
    - Submodules are not populated (local development without fetch_sources)
    - External repo is not checked out in alternate CI paths

    The import_external_repo_module function checks these locations in order:
    1. EXTERNAL_SOURCE_PATH environment variable (test integration workflows)
    2. Current working directory (external repo calling TheRock CI)
    3. GITHUB_WORKSPACE (GitHub Actions workspace)
    4. TheRock submodule path (TheRock's own CI after fetch_sources)
    """

    def test_import_rocm_libraries_matrix(self):
        """Test importing therock_matrix from rocm-libraries."""
        module = import_external_repo_module("rocm-libraries", "therock_matrix")
        if module is None:
            self.skipTest(
                "rocm-libraries not available (submodule not populated or not in CI paths)"
            )
        self.assertTrue(hasattr(module, "project_map"))
        self.assertTrue(hasattr(module, "subtree_to_project_map"))

    def test_import_rocm_systems_matrix(self):
        """Test importing therock_matrix from rocm-systems."""
        module = import_external_repo_module("rocm-systems", "therock_matrix")
        if module is None:
            self.skipTest(
                "rocm-systems not available (submodule not populated or not in CI paths)"
            )
        self.assertTrue(hasattr(module, "project_map"))

    def test_import_rocm_systems_configure_ci(self):
        """Test importing therock_configure_ci from rocm-systems (may have deps)."""
        module = import_external_repo_module("rocm-systems", "therock_configure_ci")
        # May fail due to internal import dependencies, which is OK
        # We mainly care that the import mechanism works for simpler modules
        if module:
            self.assertTrue(hasattr(module, "SKIPPABLE_PATH_PATTERNS"))

    def test_import_nonexistent_module(self):
        """Test that importing non-existent module returns None gracefully."""
        module = import_external_repo_module("nonexistent-repo", "nonexistent_module")
        self.assertIsNone(module)


class TestSkipPatternExtraction(unittest.TestCase):
    """Test extracting skip patterns from external repos."""

    def test_get_skip_patterns_rocm_systems(self):
        """Test getting skip patterns from rocm-systems."""
        patterns = get_skip_patterns("rocm-systems")
        self.assertIsInstance(patterns, list)
        # May be empty if import fails, that's OK
        if patterns:
            self.assertIn("docs/*", patterns)

    def test_get_skip_patterns_rocm_libraries(self):
        """Test getting skip patterns for rocm-libraries."""
        patterns = get_skip_patterns("rocm-libraries")
        self.assertIsInstance(patterns, list)
        # rocm-libraries may not have therock_configure_ci, returns empty list

    def test_skip_patterns_fallback(self):
        """Test that empty list is returned when external repo doesn't have patterns."""
        patterns = get_skip_patterns("nonexistent-repo")
        self.assertIsInstance(patterns, list)
        # Should return empty list, configure_ci.py provides defaults


class TestTestListExtraction(unittest.TestCase):
    """Test extracting test lists from external repo project maps."""

    def test_get_test_list_rocm_libraries(self):
        """Test getting test list from rocm-libraries."""
        tests = get_test_list("rocm-libraries")
        self.assertIsInstance(tests, list)
        if tests:  # May be empty if import fails
            # rocm-libraries should have various project tests
            possible_tests = {"rocprim", "rocblas", "hipblas", "rocfft", "miopen"}
            self.assertTrue(any(test in possible_tests for test in tests))

    def test_get_test_list_rocm_systems(self):
        """Test getting test list from rocm-systems."""
        tests = get_test_list("rocm-systems")
        self.assertIsInstance(tests, list)
        if tests:  # May be empty if import fails
            # rocm-systems should have hip-tests, rocprofiler-tests
            self.assertTrue("hip-tests" in tests or "rocprofiler-tests" in tests)

    def test_test_list_fallback(self):
        """Test that empty list is returned when external repo doesn't have project maps."""
        tests = get_test_list("nonexistent-repo")
        self.assertIsInstance(tests, list)
        # Should return empty list, configure_ci.py provides ["all"] default


class TestFullBuildDetection(unittest.TestCase):
    """Test the _detect_external_repo_projects function for full builds."""

    def test_schedule_event_always_builds(self):
        """Test that schedule events always trigger builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = []

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="schedule",
                projects_input="",
            )

            self.assertEqual(len(result["linux_projects"]), 1)
            self.assertEqual(len(result["windows_projects"]), 1)
            # Should have test list, not just ["all"]
            self.assertIsInstance(result["linux_projects"][0]["project_to_test"], list)

    def test_manual_all_override_builds(self):
        """Test that explicit 'all' projects input triggers builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = []

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="all",
            )

            self.assertEqual(len(result["linux_projects"]), 1)
            self.assertEqual(len(result["windows_projects"]), 1)

    def test_only_skippable_paths_skips_build(self):
        """Test that only docs/metadata changes skip builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = [
                "README.md",
                "docs/api.rst",
                ".gitignore",
            ]

            result = _detect_external_repo_projects(
                repo_name="rocm-systems",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="",
            )

            self.assertEqual(result["linux_projects"], [])
            self.assertEqual(result["windows_projects"], [])

    def test_code_changes_trigger_build(self):
        """Test that code changes trigger builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = [
                "README.md",
                "projects/rocprim/src/main.cpp",
            ]

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="",
            )

            self.assertEqual(len(result["linux_projects"]), 1)
            self.assertEqual(len(result["windows_projects"]), 1)
            # Should have actual test list from rocm-libraries
            test_list = result["linux_projects"][0]["project_to_test"]
            self.assertIsInstance(test_list, list)
            self.assertGreater(len(test_list), 0)

    def test_no_modified_files_skips_build(self):
        """Test that no file changes skip builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = []

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="",
            )

            self.assertEqual(result["linux_projects"], [])
            self.assertEqual(result["windows_projects"], [])

    def test_git_diff_error_skips_build(self):
        """Test that git diff errors skip builds."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = None

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="",
            )

            self.assertEqual(result["linux_projects"], [])
            self.assertEqual(result["windows_projects"], [])

    def test_returns_test_list_not_cmake_options(self):
        """Test that builds return test lists but NOT cmake options (full builds)."""
        with patch("configure_ci.get_modified_paths") as mock_get_paths:
            mock_get_paths.return_value = ["projects/rocprim/src/main.cpp"]

            result = _detect_external_repo_projects(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="",
            )

            # Should have test list
            self.assertIn("project_to_test", result["linux_projects"][0])
            # Should NOT have cmake_options (we do full builds)
            self.assertNotIn("cmake_options", result["linux_projects"][0])

            # Test list should be from external repo, not just ["all"]
            test_list = result["linux_projects"][0]["project_to_test"]
            self.assertIsInstance(test_list, list)


if __name__ == "__main__":
    unittest.main()
