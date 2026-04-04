#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Add parent directory to path to import the module
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import setup_python_cmd


class ResolvePythonCmdTest(unittest.TestCase):
    """Tests for resolve_python_cmd function."""

    def test_ubuntu_profiles(self):
        """Test Ubuntu profiles return python3.12."""
        self.assertEqual(
            setup_python_cmd.resolve_python_cmd("ubuntu2404"), "python3.12"
        )
        self.assertEqual(
            setup_python_cmd.resolve_python_cmd("ubuntu2204"), "python3.12"
        )
        self.assertEqual(
            setup_python_cmd.resolve_python_cmd("ubuntu2004"), "python3.12"
        )

    def test_debian_profiles(self):
        """Test Debian profiles return python3.12."""
        self.assertEqual(setup_python_cmd.resolve_python_cmd("debian12"), "python3.12")
        self.assertEqual(setup_python_cmd.resolve_python_cmd("debian11"), "python3.12")

    def test_sles_profiles(self):
        """Test SLES profiles return python3.13."""
        self.assertEqual(setup_python_cmd.resolve_python_cmd("sles16"), "python3.13")
        self.assertEqual(setup_python_cmd.resolve_python_cmd("sles15"), "python3.13")

    def test_rhel_profiles(self):
        """Test RHEL profiles return python3.12."""
        self.assertEqual(setup_python_cmd.resolve_python_cmd("rhel10"), "python3.12")
        self.assertEqual(setup_python_cmd.resolve_python_cmd("rhel9"), "python3.12")

    def test_centos_profiles(self):
        """Test CentOS profiles return python3.12 (default case)."""
        self.assertEqual(setup_python_cmd.resolve_python_cmd("centos9"), "python3.12")


class InstallPythonRuntimeTest(unittest.TestCase):
    """Tests for install_python_runtime function."""

    @patch("setup_python_cmd.subprocess.run")
    def test_ubuntu_install(self, mock_run):
        """Test Ubuntu installation uses apt-get."""
        setup_python_cmd.install_python_runtime("ubuntu2404")

        # Should call apt-get update and install
        self.assertEqual(mock_run.call_count, 2)

        # First call: apt-get update
        update_call = mock_run.call_args_list[0]
        self.assertEqual(update_call[0][0], ["apt-get", "update", "-qq"])
        self.assertTrue(update_call[1]["check"])
        self.assertEqual(update_call[1]["env"]["DEBIAN_FRONTEND"], "noninteractive")

        # Second call: apt-get install
        install_call = mock_run.call_args_list[1]
        self.assertEqual(
            install_call[0][0],
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "python3.12",
                "python3.12-venv",
                "python3-pip",
            ],
        )
        self.assertTrue(install_call[1]["check"])
        self.assertEqual(install_call[1]["env"]["DEBIAN_FRONTEND"], "noninteractive")

    @patch("setup_python_cmd.subprocess.run")
    def test_debian_install(self, mock_run):
        """Test Debian installation uses apt-get."""
        setup_python_cmd.install_python_runtime("debian12")

        # Should call apt-get update and install
        self.assertEqual(mock_run.call_count, 2)

        # Verify it's using apt-get for debian
        self.assertEqual(mock_run.call_args_list[0][0][0], ["apt-get", "update", "-qq"])
        self.assertIn("python3.12", mock_run.call_args_list[1][0][0])

    @patch("setup_python_cmd.subprocess.run")
    def test_sles_install(self, mock_run):
        """Test SLES installation uses zypper."""
        setup_python_cmd.install_python_runtime("sles16")

        # Should call zypper refresh and install
        self.assertEqual(mock_run.call_count, 2)

        # First call: zypper refresh
        refresh_call = mock_run.call_args_list[0]
        self.assertEqual(refresh_call[0][0], ["zypper", "--non-interactive", "refresh"])
        self.assertTrue(refresh_call[1]["check"])

        # Second call: zypper install
        install_call = mock_run.call_args_list[1]
        self.assertEqual(
            install_call[0][0],
            [
                "zypper",
                "--non-interactive",
                "install",
                "-y",
                "python313",
                "python313-pip",
            ],
        )
        self.assertTrue(install_call[1]["check"])

    @patch("setup_python_cmd.subprocess.run")
    def test_rhel_install(self, mock_run):
        """Test RHEL installation uses dnf."""
        setup_python_cmd.install_python_runtime("rhel10")

        # Should call dnf install
        self.assertEqual(mock_run.call_count, 1)

        # dnf install call
        install_call = mock_run.call_args_list[0]
        self.assertEqual(
            install_call[0][0],
            [
                "dnf",
                "install",
                "-y",
                "--allowerasing",
                "python3.12",
                "python3.12-pip",
            ],
        )
        self.assertTrue(install_call[1]["check"])

    @patch("setup_python_cmd.subprocess.run")
    def test_centos_install(self, mock_run):
        """Test CentOS installation uses dnf (default case)."""
        setup_python_cmd.install_python_runtime("centos9")

        # Should use dnf (default case)
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args_list[0][0][0][0], "dnf")


class EmitOutputTest(unittest.TestCase):
    """Tests for emit_output function."""

    def test_json_format(self):
        """Test JSON output format."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            setup_python_cmd.emit_output("python3.12", "json")
            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), '{"python_cmd": "python3.12"}')

    def test_github_format(self):
        """Test GitHub Actions output format."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            setup_python_cmd.emit_output("python3.12", "github")
            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), "PYTHON_CMD=python3.12")

    def test_env_format(self):
        """Test shell environment variable format."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            setup_python_cmd.emit_output("python3.12", "env")
            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), "export PYTHON_CMD=python3.12")

    def test_different_python_versions(self):
        """Test output with different Python versions."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            setup_python_cmd.emit_output("python3.13", "github")
            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), "PYTHON_CMD=python3.13")


class MainIntegrationTest(unittest.TestCase):
    """Integration tests for main function."""

    @patch("setup_python_cmd.install_python_runtime")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_without_install(self, mock_stdout, mock_install):
        """Test main without --install-runtime."""
        exit_code = setup_python_cmd.main(["--os-profile", "ubuntu2404"])

        self.assertEqual(exit_code, 0)
        mock_install.assert_not_called()
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), "PYTHON_CMD=python3.12")

    @patch("setup_python_cmd.install_python_runtime")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_with_install(self, mock_stdout, mock_install):
        """Test main with --install-runtime."""
        exit_code = setup_python_cmd.main(
            ["--os-profile", "ubuntu2404", "--install-runtime"]
        )

        self.assertEqual(exit_code, 0)
        mock_install.assert_called_once_with("ubuntu2404")
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), "PYTHON_CMD=python3.12")

    @patch("setup_python_cmd.install_python_runtime")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_json_output(self, mock_stdout, mock_install):
        """Test main with JSON output format."""
        exit_code = setup_python_cmd.main(
            ["--os-profile", "rhel10", "--output-format", "json"]
        )

        self.assertEqual(exit_code, 0)
        mock_install.assert_not_called()
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), '{"python_cmd": "python3.12"}')

    @patch("setup_python_cmd.install_python_runtime")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_env_output(self, mock_stdout, mock_install):
        """Test main with env output format."""
        exit_code = setup_python_cmd.main(
            ["--os-profile", "sles16", "--output-format", "env"]
        )

        self.assertEqual(exit_code, 0)
        mock_install.assert_not_called()
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), "export PYTHON_CMD=python3.13")

    @patch("setup_python_cmd.install_python_runtime")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_install_then_emit(self, mock_stdout, mock_install):
        """Test that installation happens before output emission."""
        setup_python_cmd.main(
            [
                "--os-profile",
                "ubuntu2404",
                "--install-runtime",
                "--output-format",
                "github",
            ]
        )

        # Verify install was called with correct profile
        mock_install.assert_called_once_with("ubuntu2404")
        # Verify output was emitted
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), "PYTHON_CMD=python3.12")


class OsProfileMappingConsistencyTest(unittest.TestCase):
    """Tests to ensure resolve_python_cmd and install_python_runtime are consistent."""

    @patch("setup_python_cmd.subprocess.run")
    def test_ubuntu_consistency(self, mock_run):
        """Test Ubuntu mapping is consistent between resolve and install."""
        os_profile = "ubuntu2404"
        python_cmd = setup_python_cmd.resolve_python_cmd(os_profile)
        setup_python_cmd.install_python_runtime(os_profile)

        # Verify install includes the expected python version
        install_call = mock_run.call_args_list[1][0][0]
        self.assertIn(python_cmd, install_call)

    @patch("setup_python_cmd.subprocess.run")
    def test_sles_consistency(self, mock_run):
        """Test SLES mapping is consistent between resolve and install."""
        os_profile = "sles16"
        python_cmd = setup_python_cmd.resolve_python_cmd(os_profile)
        setup_python_cmd.install_python_runtime(os_profile)

        # For SLES, python3.13 maps to package python313
        install_call = mock_run.call_args_list[1][0][0]
        self.assertIn("python313", install_call)
        self.assertEqual(python_cmd, "python3.13")

    @patch("setup_python_cmd.subprocess.run")
    def test_rhel_consistency(self, mock_run):
        """Test RHEL mapping is consistent between resolve and install."""
        os_profile = "rhel10"
        python_cmd = setup_python_cmd.resolve_python_cmd(os_profile)
        setup_python_cmd.install_python_runtime(os_profile)

        # Verify install includes the expected python version
        install_call = mock_run.call_args_list[0][0][0]
        self.assertIn(python_cmd, install_call)


if __name__ == "__main__":
    unittest.main()
