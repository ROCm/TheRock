#!/usr/bin/env python3

"""Unit tests for external repository CI configuration (configure_ci.py).

Tests for external repository CI configuration logic, including:
- setup_external_repo_configs() - Uses EXTERNAL_REPO_NAME when set
- parse_projects_input() - Project input string parsing
- get_test_list_for_build() - Test list determination
- detect_external_repo_projects_to_build() - Main orchestration function
- cross_product_projects_with_gpu_variants() - Matrix generation

Testing Strategy:
    Some tests use mocking to isolate orchestration logic from I/O operations.
    This follows the repo's testing patterns (see fetch_test_configurations_test.py,
    github_actions_utils_test.py) where orchestration functions that coordinate
    multiple I/O operations are tested by mocking those operations.

    Mocked operations include:
    - get_git_modified_paths: Calls git diff (subprocess)
    - get_external_repo_path: Resolves external repo directory
    - get_test_list: Reads external repo configuration files
    - get_skip_patterns: Reads external repo configuration files

    Pure logic and parsing functions are tested without mocking.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_ci import (
    parse_projects_input,
    get_test_list_for_build,
    cross_product_projects_with_gpu_variants,
    detect_external_repo_projects_to_build,
    setup_external_repo_configs,
)


class TestSetupExternalRepoConfigs(unittest.TestCase):
    """Test setup_external_repo_configs() with EXTERNAL_REPO_NAME set or unset."""

    def test_returns_none_when_external_repo_name_unset(self):
        """When EXTERNAL_REPO_NAME is unset or empty, we are not an external run."""
        with patch.dict(os.environ, {"EXTERNAL_REPO_NAME": ""}, clear=False):
            result = setup_external_repo_configs(
                base_args={"base_ref": "HEAD^1", "github_event_name": "pull_request"},
                output_empty_matrix_and_exit_func=lambda: None,
            )
        self.assertIsNone(result)

    def test_returns_none_when_external_repo_name_unknown(self):
        """When EXTERNAL_REPO_NAME is set to unknown repo, return None."""
        with patch.dict(os.environ, {"EXTERNAL_REPO_NAME": "unknown-repo"}):
            result = setup_external_repo_configs(
                base_args={"base_ref": "HEAD^1", "github_event_name": "pull_request"},
                output_empty_matrix_and_exit_func=lambda: None,
            )
        self.assertIsNone(result)

    @patch("configure_ci.detect_external_repo_projects_to_build")
    def test_returns_config_when_external_repo_name_known(self, mock_projects_to_build):
        """When EXTERNAL_REPO_NAME is set to a known repo, return config from detect_external_repo_projects_to_build."""
        mock_projects_to_build.return_value = {
            "linux_projects": [{"projects_to_test": "rocprim,rocblas"}],
            "windows_projects": [{"projects_to_test": "rocprim,rocblas"}],
        }
        with patch.dict(os.environ, {"EXTERNAL_REPO_NAME": "rocm-libraries"}):
            result = setup_external_repo_configs(
                base_args={"base_ref": "HEAD^1", "github_event_name": "pull_request"},
                output_empty_matrix_and_exit_func=MagicMock(),
            )
        self.assertIsNotNone(result)
        self.assertEqual(
            result["linux_external_project_configs"],
            [{"projects_to_test": "rocprim,rocblas"}],
        )
        self.assertEqual(
            result["windows_external_project_configs"],
            [{"projects_to_test": "rocprim,rocblas"}],
        )
        mock_projects_to_build.assert_called_once()
        call_kwargs = mock_projects_to_build.call_args[1]
        self.assertEqual(call_kwargs["repo_name"], "rocm-libraries")


class TestDetectExternalRepoProjects(unittest.TestCase):
    """Test detect_external_repo_projects_to_build() function.

    This function orchestrates multiple I/O operations (git diff, config file reading)
    to decide whether to build and what to test. We mock the I/O operations to test
    the orchestration logic in isolation, following patterns in:
    - fetch_test_configurations_test.py (mocks gha_set_output)
    - github_actions_utils_test.py (mocks subprocess, HTTP requests)
    """

    def test_schedule_event_always_builds(self):
        """Test that schedule events always trigger builds."""
        with patch("configure_ci.get_test_list") as mock_get_test_list:
            mock_get_test_list.return_value = ["rocprim", "rocblas"]

            result = detect_external_repo_projects_to_build(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="schedule",
                projects_input="",
            )

        self.assertEqual(len(result["linux_projects"]), 1)
        self.assertEqual(len(result["windows_projects"]), 1)
        self.assertIn("projects_to_test", result["linux_projects"][0])
        self.assertEqual(
            result["linux_projects"][0]["projects_to_test"], "rocprim,rocblas"
        )
        self.assertEqual(
            result["windows_projects"][0]["projects_to_test"], "rocprim,rocblas"
        )

    def test_manual_all_override_builds(self):
        """Test that explicit 'all' projects input triggers builds."""
        with patch("configure_ci.get_test_list") as mock_get_test_list:
            mock_get_test_list.return_value = ["rocprim", "rocblas"]

            result = detect_external_repo_projects_to_build(
                repo_name="rocm-libraries",
                base_ref="origin/develop",
                github_event_name="pull_request",
                projects_input="all",
            )

        self.assertEqual(len(result["linux_projects"]), 1)
        self.assertEqual(len(result["windows_projects"]), 1)

    def test_specific_projects_override(self):
        """Test that specific projects input triggers builds."""
        result = detect_external_repo_projects_to_build(
            repo_name="rocm_libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="projects/rocprim,projects/rocblas",
        )

        self.assertEqual(len(result["linux_projects"]), 1)
        self.assertEqual(len(result["windows_projects"]), 1)
        self.assertEqual(
            result["linux_projects"][0]["projects_to_test"], "rocprim,rocblas"
        )

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_only_skippable_paths_skips_build(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_skip_patterns,
    ):
        """Test that only docs/metadata changes skip builds."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = ["*.md", "docs/*", ".gitignore"]
        mock_get_git_modified_paths.return_value = [
            "README.md",
            "docs/api.rst",
            ".gitignore",
        ]

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-systems",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(result["linux_projects"], [])
        self.assertEqual(result["windows_projects"], [])

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_test_list")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_code_changes_trigger_build(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_test_list,
        mock_get_skip_patterns,
    ):
        """Test that code changes trigger builds."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = []
        mock_get_git_modified_paths.return_value = [
            "README.md",
            "projects/rocprim/src/main.cpp",
        ]
        mock_get_test_list.return_value = ["rocprim", "rocblas"]

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(len(result["linux_projects"]), 1)
        self.assertEqual(len(result["windows_projects"]), 1)
        test_list = result["linux_projects"][0]["projects_to_test"]
        self.assertIsInstance(test_list, str)
        self.assertGreater(len(test_list), 0)

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_no_modified_files_skips_build(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_skip_patterns,
    ):
        """Test that no file changes skip builds."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = []
        mock_get_git_modified_paths.return_value = []

        result = detect_external_repo_projects_to_build(
            repo_name="rocm_libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(result["linux_projects"], [])
        self.assertEqual(result["windows_projects"], [])

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_git_diff_error_skips_build(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_skip_patterns,
    ):
        """Test that git diff failure (None) skips build, same as internal is_ci_run_required(None)."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = []
        mock_get_git_modified_paths.return_value = None

        result = detect_external_repo_projects_to_build(
            repo_name="rocm_libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(result["linux_projects"], [])
        self.assertEqual(result["windows_projects"], [])

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_test_list")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_returns_test_list_not_cmake_options(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_test_list,
        mock_get_skip_patterns,
    ):
        """Test that builds return test lists but NOT cmake options (full builds)."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = []
        mock_get_git_modified_paths.return_value = ["projects/rocprim/src/main.cpp"]
        mock_get_test_list.return_value = ["rocprim", "rocblas"]

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        # Should have test list
        self.assertIn("projects_to_test", result["linux_projects"][0])
        # Should NOT have cmake_options (we do full builds)
        self.assertNotIn("cmake_options", result["linux_projects"][0])

        # Should have test list, not just ["all"]
        test_list = result["linux_projects"][0]["projects_to_test"]
        self.assertIsInstance(test_list, str)
        self.assertEqual(test_list, "rocprim,rocblas")

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_test_list")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_fallback_to_default_test_list(
        self,
        mock_get_external_repo_path,
        mock_get_git_modified_paths,
        mock_get_test_list,
        mock_get_skip_patterns,
    ):
        """Test that fallback to ['all'] works when external repo doesn't provide test list."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_skip_patterns.return_value = []
        mock_get_git_modified_paths.return_value = ["projects/rocprim/src/main.cpp"]
        mock_get_test_list.return_value = []  # Empty list

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(len(result["linux_projects"]), 1)
        self.assertEqual(result["linux_projects"][0]["projects_to_test"], "all")

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_uses_external_repo_skip_patterns(
        self, mock_get_external_repo_path, mock_get_git_modified_paths, mock_get_skip
    ):
        """Test that external repo skip patterns are used when available."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_git_modified_paths.return_value = [
            "docs/README.md"
        ]  # Should be skipped
        mock_get_skip.return_value = ["docs/*"]

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        self.assertEqual(result["linux_projects"], [])
        self.assertEqual(result["windows_projects"], [])

    @patch("configure_ci.get_skip_patterns")
    @patch("configure_ci.get_git_modified_paths")
    @patch("configure_ci.get_external_repo_path")
    def test_fallback_to_default_skip_patterns(
        self, mock_get_external_repo_path, mock_get_git_modified_paths, mock_get_skip
    ):
        """Test that default skip patterns are used when external repo doesn't provide them."""
        mock_get_external_repo_path.return_value = MagicMock()
        mock_get_git_modified_paths.return_value = [
            "README.md"
        ]  # Should be skipped by default patterns
        mock_get_skip.return_value = []  # Empty list

        result = detect_external_repo_projects_to_build(
            repo_name="rocm-libraries",
            base_ref="origin/develop",
            github_event_name="pull_request",
            projects_input="",
        )

        # Default patterns include "*.md" so README.md should be skipped
        self.assertEqual(result["linux_projects"], [])
        self.assertEqual(result["windows_projects"], [])


class TestCrossProductProjectsWithGpuVariants(unittest.TestCase):
    """Test cross_product_projects_with_gpu_variants() function.

    Tests pure matrix generation logic. No mocking needed - pure function
    that takes lists and returns combined list.
    """

    def test_single_project_single_variant(self):
        """Test cross-product with one project and one GPU variant."""
        project_configs = [{"projects_to_test": "rocprim,rocblas"}]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[0]["projects_to_test"], "rocprim,rocblas")
        self.assertNotIn("cmake_options", result[0])

    def test_single_project_multiple_variants(self):
        """Test cross-product with one project and multiple GPU variants."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = [
            {"family": "gfx94x", "platform": "linux"},
            {"family": "gfx110x", "platform": "linux"},
            {"family": "gfx94x", "platform": "windows"},
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["projects_to_test"], "rocprim")
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[1]["projects_to_test"], "rocprim")
        self.assertEqual(result[1]["family"], "gfx110x")
        self.assertEqual(result[1]["platform"], "linux")
        self.assertEqual(result[2]["projects_to_test"], "rocprim")
        self.assertEqual(result[2]["family"], "gfx94x")
        self.assertEqual(result[2]["platform"], "windows")

    def test_multiple_projects_single_variant(self):
        """Test cross-product with multiple projects and one GPU variant."""
        project_configs = [
            {"projects_to_test": "rocprim"},
            {"projects_to_test": "rocblas"},
            {"projects_to_test": "rocfft"},
        ]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["projects_to_test"], "rocprim")
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[1]["projects_to_test"], "rocblas")
        self.assertEqual(result[1]["family"], "gfx94x")
        self.assertEqual(result[2]["projects_to_test"], "rocfft")
        self.assertEqual(result[2]["family"], "gfx94x")

    def test_multiple_projects_multiple_variants(self):
        """Test cross-product with multiple projects and multiple GPU variants."""
        project_configs = [
            {"projects_to_test": "rocprim"},
            {"projects_to_test": "rocblas"},
        ]
        gpu_variants = [
            {"family": "gfx94x", "platform": "linux"},
            {"family": "gfx110x", "platform": "linux"},
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 4)  # 2 projects * 2 variants = 4
        # Verify all combinations exist
        combinations = [
            (r["projects_to_test"], r["family"], r["platform"]) for r in result
        ]
        self.assertIn(("rocprim", "gfx94x", "linux"), combinations)
        self.assertIn(("rocprim", "gfx110x", "linux"), combinations)
        self.assertIn(("rocblas", "gfx94x", "linux"), combinations)
        self.assertIn(("rocblas", "gfx110x", "linux"), combinations)

    def test_empty_project_configs(self):
        """Test cross-product with empty project configs."""
        project_configs = []
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 0)

    def test_empty_gpu_variants(self):
        """Test cross-product with empty GPU variants."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = []

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 0)

    def test_preserves_gpu_variant_fields(self):
        """Test that all GPU variant fields are preserved in result."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = [
            {
                "family": "gfx94x",
                "platform": "linux",
                "build_variant": "release",
                "extra_field": "value",
            }
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[0]["build_variant"], "release")
        self.assertEqual(result[0]["extra_field"], "value")
        self.assertEqual(result[0]["projects_to_test"], "rocprim")

    def test_no_cmake_options_in_result(self):
        """Test that cmake_options are NOT included in result (full builds only)."""
        project_configs = [
            {"projects_to_test": "rocprim", "cmake_options": "-DROCBLAS=ON"}
        ]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertNotIn("cmake_options", result[0])
        self.assertEqual(result[0]["projects_to_test"], "rocprim")

    def test_complex_gpu_variant_structure(self):
        """Test with complex GPU variant structure."""
        project_configs = [
            {"projects_to_test": "rocprim,rocblas"},
            {"projects_to_test": "rocfft"},
        ]
        gpu_variants = [
            {
                "family": "gfx94x",
                "platform": "linux",
                "build_variant": "release",
                "test_labels": ["smoke"],
            },
            {
                "family": "gfx110x",
                "platform": "windows",
                "build_variant": "debug",
                "test_labels": ["full"],
            },
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 4)  # 2 projects * 2 variants
        # Verify structure is preserved
        for r in result:
            self.assertIn("family", r)
            self.assertIn("platform", r)
            self.assertIn("build_variant", r)
            self.assertIn("test_labels", r)
            self.assertIn("projects_to_test", r)
            self.assertNotIn("cmake_options", r)


if __name__ == "__main__":
    unittest.main()
