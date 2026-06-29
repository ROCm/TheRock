#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for DVC support in external repository configuration.

Tests the fix for https://github.com/ROCm/rocm-systems/actions/runs/28359613355/job/84014838210
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_external_repo_config import (
    get_repo_config,
    main as detect_external_repo_config_main,
)


class TestDVCProjectsConfiguration(unittest.TestCase):
    """Tests for DVC projects configuration in external repos."""

    def test_rocm_systems_has_dvc_projects(self):
        """Test that rocm-systems config includes dvc_projects."""
        config = get_repo_config("rocm-systems")

        # Verify dvc_projects key exists
        self.assertIn(
            "dvc_projects",
            config,
            "rocm-systems config must have dvc_projects for wsl-rocdxg builds",
        )

        # Verify it contains external-rocm-systems
        self.assertEqual(
            config["dvc_projects"],
            ["external-rocm-systems"],
            "dvc_projects should contain external-rocm-systems path",
        )

    def test_rocm_systems_has_all_required_keys(self):
        """Test that rocm-systems config has all required keys including dvc_projects."""
        config = get_repo_config("rocm-systems")

        required_keys = {
            "cmake_source_var",
            "submodule_path",
            "skip_submodules",
            "dvc_projects",  # New requirement
        }

        self.assertTrue(
            required_keys.issubset(config.keys()),
            f"rocm-systems config missing required keys: {required_keys - config.keys()}",
        )

    def test_rocm_libraries_no_dvc_projects(self):
        """Test that rocm-libraries doesn't have dvc_projects (not needed)."""
        config = get_repo_config("rocm-libraries")

        # rocm-libraries doesn't use DVC for external builds
        self.assertNotIn(
            "dvc_projects", config, "rocm-libraries doesn't need dvc_projects"
        )


class TestFetchSourcesArgsGeneration(unittest.TestCase):
    """Tests for fetch_sources_args generation with DVC support."""

    def setUp(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            self.temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = self.temp_file

    def tearDown(self):
        """Clean up test fixtures."""
        if "GITHUB_OUTPUT" in os.environ:
            del os.environ["GITHUB_OUTPUT"]
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def test_rocm_systems_generates_dvc_args(self):
        """Test that rocm-systems generates --dvc-projects in fetch_sources_args."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-systems",
                "--external-repo-json",
                '{"repository": "ROCm/rocm-systems", "ref": "develop"}',
            ]
        )

        self.assertEqual(rc, 0, "detect_external_repo_config should succeed")

        # Read the output
        with open(self.temp_file, "r") as f:
            output = f.read()

        # Verify fetch_sources_args contains both skip-submodules and dvc-projects
        self.assertIn("fetch_sources_args=", output)
        self.assertIn(
            "--skip-submodules rocm-systems",
            output,
            "Should include --skip-submodules rocm-systems",
        )
        self.assertIn(
            "--dvc-projects external-rocm-systems",
            output,
            "Should include --dvc-projects external-rocm-systems",
        )

    def test_rocm_systems_config_json_includes_dvc(self):
        """Test that config_json includes dvc_projects in fetch_sources_args."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-systems",
            ]
        )

        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # Verify config_json structure
        self.assertIn("config_json=", output)
        self.assertIn('"checkout_path": "external-rocm-systems"', output)
        self.assertIn('"fetch_sources_args":', output)

        # Extract and verify the fetch_sources_args within config_json
        # It should contain both arguments
        config_json_line = [
            line for line in output.split("\n") if "config_json=" in line
        ][0]
        self.assertIn("--skip-submodules rocm-systems", config_json_line)
        self.assertIn("--dvc-projects external-rocm-systems", config_json_line)

    def test_rocm_libraries_no_dvc_args(self):
        """Test that rocm-libraries doesn't generate --dvc-projects (not needed)."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-libraries",
            ]
        )

        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # rocm-libraries should only have skip-submodules
        self.assertIn("--skip-submodules rocm-libraries", output)
        self.assertNotIn(
            "--dvc-projects",
            output,
            "rocm-libraries shouldn't generate dvc-projects args",
        )


class TestDVCArgsFormat(unittest.TestCase):
    """Tests for proper formatting of DVC arguments."""

    def setUp(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            self.temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = self.temp_file

    def tearDown(self):
        if "GITHUB_OUTPUT" in os.environ:
            del os.environ["GITHUB_OUTPUT"]
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def test_args_separated_by_space(self):
        """Test that skip-submodules and dvc-projects are separated by space."""
        rc = detect_external_repo_config_main(["--repository", "rocm-systems"])

        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # Find the fetch_sources_args line
        for line in output.split("\n"):
            if "fetch_sources_args=" in line and "config_json" not in line:
                # Should be: --skip-submodules rocm-systems --dvc-projects external-rocm-systems
                self.assertRegex(
                    line,
                    r"--skip-submodules rocm-systems --dvc-projects external-rocm-systems",
                    "Arguments should be space-separated with correct order",
                )

    def test_external_prefix_in_dvc_path(self):
        """Test that DVC projects use 'external-' prefix for checkout path."""
        rc = detect_external_repo_config_main(["--repository", "rocm-systems"])

        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # DVC projects should reference external-rocm-systems, not just rocm-systems
        self.assertIn(
            "external-rocm-systems", output, "DVC path should use external- prefix"
        )

        # Skip-submodules should reference the base name without prefix
        self.assertIn(
            "--skip-submodules rocm-systems",
            output,
            "Skip-submodules should use base repo name",
        )


if __name__ == "__main__":
    unittest.main()
