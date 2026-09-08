#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for DVC support in fetch_sources.py.

Tests the fix for https://github.com/ROCm/rocm-systems/actions/runs/28359613355/job/84014838210
"""

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch


# Mock PullResult before importing fetch_sources
@dataclass
class MockPullResult:
    fetched: int = 0
    cached: int = 0
    skipped: int = 0


# Mock fetch_dvc_artifacts before import to avoid boto3 dependency
fetch_dvc_mock = Mock()
fetch_dvc_mock.DEFAULT_JOBS = 4
fetch_dvc_mock.PullResult = MockPullResult
sys.modules["fetch_dvc_artifacts"] = fetch_dvc_mock

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import fetch_sources


class TestPullLargeFilesExternalRepo(unittest.TestCase):
    """Tests for pull_large_files() with external repositories."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.therock_dir = Path(self.temp_dir)

        # Create mock directory structure
        self.external_rocm_systems = self.therock_dir / "external-rocm-systems"
        self.external_rocm_systems.mkdir()

        # Create .dvc/config for external-rocm-systems
        dvc_dir = self.external_rocm_systems / ".dvc"
        dvc_dir.mkdir()
        dvc_config = dvc_dir / "config"
        dvc_config.write_text("[core]\n    remote = origin\n")

        # Create submodule directory for comparison
        self.rocm_systems_submodule = self.therock_dir / "base" / "rocm-systems"
        self.rocm_systems_submodule.mkdir(parents=True)
        submodule_dvc_dir = self.rocm_systems_submodule / ".dvc"
        submodule_dvc_dir.mkdir()
        submodule_dvc_config = submodule_dvc_dir / "config"
        submodule_dvc_config.write_text("[core]\n    remote = origin\n")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_external_repo_uses_direct_path(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test that external repos use direct path instead of get_submodule_path."""
        mock_therock_dir.__truediv__ = lambda self, other: self.therock_dir / other
        mock_therock_dir.return_value = self.therock_dir
        fetch_sources.THEROCK_DIR = self.therock_dir

        # get_submodule_path should only be called for submodules, not external repos
        mock_get_submodule_path.return_value = "base/rocm-systems"
        mock_dvc_pull.return_value = MockPullResult()

        # List of DVC projects (external repo + submodule)
        dvc_projects = ["external-rocm-systems", "rocm-systems"]
        # List of known submodules (only rocm-systems)
        projects = ["rocm-systems"]

        fetch_sources.pull_large_files(dvc_projects, projects, jobs=4)

        # Verify get_submodule_path was only called for submodule, not external repo
        mock_get_submodule_path.assert_called_once_with("rocm-systems")

        # Verify DVC pull was called for both paths
        self.assertEqual(mock_dvc_pull.call_count, 2)

        # Extract the paths passed to dvc pull
        call_args = [call_obj[0][0] for call_obj in mock_dvc_pull.call_args_list]

        # External repo should use direct path
        self.assertIn(self.external_rocm_systems, call_args)
        # Submodule should use get_submodule_path result
        self.assertIn(self.rocm_systems_submodule, call_args)

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_external_repo_only(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test DVC pull with only external repos (no submodules)."""
        fetch_sources.THEROCK_DIR = self.therock_dir
        mock_dvc_pull.return_value = MockPullResult()

        # Only external repos, no submodules
        dvc_projects = ["external-rocm-systems"]
        projects = []  # Empty list of submodules

        fetch_sources.pull_large_files(dvc_projects, projects, jobs=4)

        # get_submodule_path should not be called
        mock_get_submodule_path.assert_not_called()

        # DVC pull should be called once for external repo
        mock_dvc_pull.assert_called_once()
        call_args = mock_dvc_pull.call_args[0]
        self.assertEqual(call_args[0], self.external_rocm_systems)

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_submodule_only(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test DVC pull with only submodules (no external repos)."""
        fetch_sources.THEROCK_DIR = self.therock_dir
        mock_get_submodule_path.return_value = "base/rocm-systems"
        mock_dvc_pull.return_value = MockPullResult()

        # Only submodules, no external repos
        dvc_projects = ["rocm-systems"]
        projects = ["rocm-systems"]

        fetch_sources.pull_large_files(dvc_projects, projects, jobs=4)

        # get_submodule_path should be called once
        mock_get_submodule_path.assert_called_once_with("rocm-systems")

        # DVC pull should be called once
        mock_dvc_pull.assert_called_once()
        call_args = mock_dvc_pull.call_args[0]
        self.assertEqual(call_args[0], self.rocm_systems_submodule)

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_dvc_artifacts.pull")
    def test_missing_dvc_config_warning(self, mock_dvc_pull, mock_therock_dir):
        """Test warning when .dvc/config is missing."""
        fetch_sources.THEROCK_DIR = self.therock_dir

        # Create directory without .dvc/config
        no_dvc_dir = self.therock_dir / "external-no-dvc"
        no_dvc_dir.mkdir()

        dvc_projects = ["external-no-dvc"]
        projects = []

        with patch("builtins.print") as mock_print:
            fetch_sources.pull_large_files(dvc_projects, projects)

            # Check that warning was printed
            warning_calls = [
                call_obj
                for call_obj in mock_print.call_args_list
                if len(call_obj[0]) > 0 and "WARNING" in str(call_obj[0][0])
            ]
            self.assertGreater(len(warning_calls), 0, "Should print warning")

        # DVC pull should not be called for directory without config
        mock_dvc_pull.assert_not_called()

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_jobs_parameter_passed(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test that jobs parameter is passed to dvc pull."""
        fetch_sources.THEROCK_DIR = self.therock_dir
        mock_dvc_pull.return_value = MockPullResult()

        dvc_projects = ["external-rocm-systems"]
        projects = []

        # Call with custom jobs parameter
        fetch_sources.pull_large_files(dvc_projects, projects, jobs=8)

        # Verify jobs parameter was passed
        mock_dvc_pull.assert_called_once()
        self.assertEqual(mock_dvc_pull.call_args[1]["jobs"], 8)

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_dvc_artifacts.pull")
    def test_empty_dvc_projects(self, mock_dvc_pull, mock_therock_dir):
        """Test behavior with empty dvc_projects list."""
        fetch_sources.THEROCK_DIR = self.therock_dir

        with patch("builtins.print") as mock_print:
            fetch_sources.pull_large_files([], [], jobs=4)

            # Should print skip message
            skip_calls = [
                call_obj
                for call_obj in mock_print.call_args_list
                if len(call_obj[0]) > 0 and "No DVC projects" in str(call_obj[0][0])
            ]
            self.assertGreater(len(skip_calls), 0, "Should print skip message")

        # DVC pull should not be called
        mock_dvc_pull.assert_not_called()

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_none_jobs_uses_default(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test that None jobs parameter uses DEFAULT_JOBS."""
        fetch_sources.THEROCK_DIR = self.therock_dir
        mock_dvc_pull.return_value = MockPullResult()

        dvc_projects = ["external-rocm-systems"]
        projects = []

        # Call with jobs=None (default)
        fetch_sources.pull_large_files(dvc_projects, projects, jobs=None)

        # Verify DEFAULT_JOBS was used
        mock_dvc_pull.assert_called_once()
        self.assertEqual(mock_dvc_pull.call_args[1]["jobs"], 4)  # DEFAULT_JOBS


class TestPullLargeFilesIntegration(unittest.TestCase):
    """Integration tests simulating real-world usage."""

    @patch("fetch_sources.THEROCK_DIR")
    @patch("fetch_sources.get_submodule_path")
    @patch("fetch_dvc_artifacts.pull")
    def test_wsl_rocdxg_scenario(
        self, mock_dvc_pull, mock_get_submodule_path, mock_therock_dir
    ):
        """Test the exact scenario from wsl-rocdxg CI failure.

        This simulates the fix for:
        https://github.com/ROCm/rocm-systems/actions/runs/28359613355/job/84014838210
        """
        temp_dir = tempfile.mkdtemp()
        try:
            therock_dir = Path(temp_dir)
            fetch_sources.THEROCK_DIR = therock_dir

            # Set up external-rocm-systems with DVC config
            external_repo = therock_dir / "external-rocm-systems"
            external_repo.mkdir()
            dvc_dir = external_repo / ".dvc"
            dvc_dir.mkdir()
            (dvc_dir / "config").write_text("[core]\n    remote = origin\n")

            # Create libwkmi.a.dvc file to simulate real scenario
            wkmi_dir = (
                external_repo
                / "shared"
                / "amdgpu-windows-interop"
                / "wkmi"
                / "lnx"
                / "lib"
            )
            wkmi_dir.mkdir(parents=True)
            (wkmi_dir / "libwkmi.a.dvc").write_text(
                "md5: 5c48f0f50d868cd048400223df6115bc\nsize: 337200\n"
            )

            mock_get_submodule_path.return_value = "base/rocm-systems"
            mock_dvc_pull.return_value = MockPullResult()

            # These are the arguments from detect_external_repo_config.py
            dvc_projects = ["external-rocm-systems"]
            projects = []  # rocm-systems is skipped, so not in projects list

            fetch_sources.pull_large_files(dvc_projects, projects)

            # Verify DVC pull was called for external-rocm-systems
            mock_dvc_pull.assert_called_once()
            call_path = mock_dvc_pull.call_args[0][0]
            self.assertEqual(call_path, external_repo)

            # Verify get_submodule_path was NOT called (external repo only)
            mock_get_submodule_path.assert_not_called()

        finally:
            import shutil

            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
