#!/usr/bin/env python3
"""Unit tests for detect_external_repo_config.py"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_external_repo_config import (
    detect_repo_name,
    get_repo_config,
    main as detect_external_repo_config_main,
    output_github_actions_vars,
    REPO_CONFIGS,
)


class TestDetectRepoName(unittest.TestCase):
    """Tests for detect_repo_name function"""

    def test_full_repo_name(self):
        """Test with full repository name (org/repo format)"""
        self.assertEqual(detect_repo_name("ROCm/rocm-libraries"), "rocm-libraries")
        self.assertEqual(detect_repo_name("ROCm/rocm-systems"), "rocm-systems")

    def test_short_repo_name(self):
        """Test with short repository name"""
        self.assertEqual(detect_repo_name("rocm-libraries"), "rocm-libraries")
        self.assertEqual(detect_repo_name("rocm-systems"), "rocm-systems")


class TestGetRepoConfig(unittest.TestCase):
    """Tests for get_repo_config function"""

    def test_rocm_libraries_config(self):
        """Test rocm-libraries configuration"""
        config = get_repo_config("rocm-libraries")
        self.assertEqual(
            config["cmake_source_var"], "THEROCK_ROCM_LIBRARIES_SOURCE_DIR"
        )
        self.assertEqual(config["patches_dir"], "rocm-libraries")
        self.assertEqual(
            config["fetch_exclusion"],
            "--no-include-rocm-libraries --no-include-ml-frameworks",
        )
        # enable_dvc is platform-specific for rocm-libraries
        self.assertIsInstance(config["enable_dvc"], dict)
        self.assertTrue(config["enable_dvc"]["linux"])
        self.assertTrue(config["enable_dvc"]["windows"])

    def test_rocm_systems_config(self):
        """Test rocm-systems configuration"""
        config = get_repo_config("rocm-systems")
        self.assertEqual(config["cmake_source_var"], "THEROCK_ROCM_SYSTEMS_SOURCE_DIR")
        self.assertEqual(config["patches_dir"], "rocm-systems")
        self.assertEqual(
            config["fetch_exclusion"],
            "--no-include-rocm-systems --no-include-rocm-libraries --no-include-ml-frameworks",
        )
        # enable_dvc is platform-specific for rocm-systems
        self.assertIsInstance(config["enable_dvc"], dict)
        self.assertFalse(config["enable_dvc"]["linux"])
        self.assertTrue(config["enable_dvc"]["windows"])

    def test_unknown_repo_raises_error(self):
        """Test that unknown repository raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_repo_config("unknown-repo")
        self.assertIn("Unknown external repository", str(context.exception))
        self.assertIn("unknown-repo", str(context.exception))

    def test_all_repos_have_required_keys(self):
        """Test that all repo configs have required keys"""
        required_keys = {
            "cmake_source_var",
            "patches_dir",
            "fetch_exclusion",
            "enable_dvc",
        }
        for repo_name, config in REPO_CONFIGS.items():
            with self.subTest(repo=repo_name):
                self.assertTrue(
                    required_keys.issubset(config.keys()),
                    f"Repo {repo_name} missing required keys: {required_keys - config.keys()}",
                )


class TestOutputGithubActionsVars(unittest.TestCase):
    """Tests for output_github_actions_vars function"""

    def test_output_to_file(self):
        """Test output to GITHUB_OUTPUT file"""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_file = f.name

        try:
            # Set GITHUB_OUTPUT environment variable
            os.environ["GITHUB_OUTPUT"] = temp_file

            config = {
                "cmake_source_var": "TEST_VAR",
                "patches_dir": "test-dir",
                "enable_dvc": True,
                "enable_ck": False,
            }

            output_github_actions_vars(config)

            # Read the output file
            with open(temp_file, "r") as f:
                output = f.read()

            # Verify output format
            self.assertIn("cmake_source_var=TEST_VAR", output)
            self.assertIn("patches_dir=test-dir", output)
            self.assertIn("enable_dvc=true", output)  # Boolean converted to lowercase
            self.assertIn("enable_ck=false", output)  # Boolean converted to lowercase

        finally:
            # Cleanup
            if "GITHUB_OUTPUT" in os.environ:
                del os.environ["GITHUB_OUTPUT"]
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_boolean_conversion(self):
        """Test that booleans are converted to lowercase strings"""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_file = f.name

        try:
            os.environ["GITHUB_OUTPUT"] = temp_file

            config = {
                "bool_true": True,
                "bool_false": False,
            }

            output_github_actions_vars(config)

            with open(temp_file, "r") as f:
                output = f.read()

            # Verify lowercase (important for bash conditionals)
            self.assertIn("bool_true=true", output)
            self.assertIn("bool_false=false", output)
            self.assertNotIn("True", output)
            self.assertNotIn("False", output)

        finally:
            if "GITHUB_OUTPUT" in os.environ:
                del os.environ["GITHUB_OUTPUT"]
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_extra_cmake_options_generated(self):
        """Test that extra_cmake_options is generated by main() when --workspace is provided."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_file = f.name

        try:
            os.environ["GITHUB_OUTPUT"] = temp_file
            # main() requires --platform for repos with platform-specific config values.
            old_argv = sys.argv[:]
            sys.argv = [
                "detect_external_repo_config.py",
                "--repository",
                "ROCm/rocm-libraries",
                "--platform",
                "linux",
                "--workspace",
                "/workspace",
            ]
            rc = detect_external_repo_config_main()
            self.assertEqual(rc, 0)

            with open(temp_file, "r") as f:
                output = f.read()

            # Verify extra_cmake_options is included
            self.assertIn(
                "extra_cmake_options=-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=/workspace",
                output,
            )

        finally:
            sys.argv = old_argv
            if "GITHUB_OUTPUT" in os.environ:
                del os.environ["GITHUB_OUTPUT"]
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
