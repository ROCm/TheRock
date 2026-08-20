# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import os

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from setup_venv import install_packages_into_venv


class InstallPackagesTest(unittest.TestCase):
    """Tests for install_packages_into_venv() command generation."""

    def setUp(self):
        self.venv_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.venv_dir, ignore_errors=True)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_basic_pip_usage(self, mock_run, mock_find_python):
        """The most basic usage should run `python -m pip install [packages]`"""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
        )

        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "python")
        self.assertEqual(cmd[1], "-m")
        self.assertEqual(cmd[2], "pip")
        self.assertEqual(cmd[3], "install")
        self.assertIn("rocm", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_basic_uv_usage(self, mock_run, mock_find_python):
        """Using uv generates a different command structure."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            use_uv=True,
        )

        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "uv")
        self.assertEqual(cmd[1], "pip")
        self.assertEqual(cmd[2], "install")
        self.assertEqual(cmd[3], "--python")
        self.assertIn("rocm", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_multiple_packages(self, mock_run, mock_find_python):
        """Multiple packages can be installed at once."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["torch", "torchaudio"],
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("torch", cmd)
        self.assertIn("torchaudio", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_pre_flag_pip(self, mock_run, mock_find_python):
        """--pre flag uses pip syntax."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            pre=True,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--pre", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_pre_flag_uv(self, mock_run, mock_find_python):
        """--pre flag uses uv syntax when use_uv=True."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            use_uv=True,
            pre=True,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--prerelease=allow", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_index_url(self, mock_run, mock_find_python):
        """Passing index_url uses the URL as-is."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://example.com/full/path/",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://example.com/full/path/", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_index_name(self, mock_run, mock_find_python):
        """Passing index_name without index_url uses a known url."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_name="stable",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://repo.amd.com/rocm/whl-multi-arch/", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_find_links_only(self, mock_run, mock_find_python):
        """Passing just find_links uses it."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            find_links="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)
        self.assertFalse(any("--index-url" in str(a) for a in cmd))

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_empty_index_url_disables_index(self, mock_run, mock_find_python):
        """An explicit empty index_url means install only from find-links."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="",
            find_links="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--no-index", cmd)
        self.assertIn("--no-build-isolation", cmd)
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)
        self.assertFalse(any("--index-url" in str(a) for a in cmd))

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_index_url_and_find_links(self, mock_run, mock_find_python):
        """Both index_url and find_links can be used together."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://deps/simple/",
            find_links="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://deps/simple/", cmd)
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)

    @patch("setup_venv.time.sleep")
    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_package_install_retries_then_succeeds(
        self, mock_run, mock_find_python, mock_sleep
    ):
        """Transient package install failures are retried."""
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["pip"]),
            None,
        ]

        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            install_retry_timeout_seconds=60,
            install_retry_wait_between_seconds=30,
        )

        self.assertEqual(mock_run.call_count, 2)
        mock_sleep.assert_called_once_with(30)

    @patch("setup_venv.time.sleep")
    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_package_install_retries_can_be_disabled(
        self, mock_run, mock_find_python, mock_sleep
    ):
        """Setting the retry window to zero preserves fail-fast behavior."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip"])

        with self.assertRaises(subprocess.CalledProcessError):
            install_packages_into_venv(
                venv_dir=self.venv_dir,
                packages=["rocm"],
                install_retry_timeout_seconds=0,
                install_retry_wait_between_seconds=30,
            )

        self.assertEqual(mock_run.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
