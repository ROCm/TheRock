from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import os
import re

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from setup_venv import (
    GFX_TARGET_REGEX,
    install_packages_into_venv,
)


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
    def test_index_url_complete(self, mock_run, mock_find_python):
        """Passing index_url without index_subdir uses the URL as-is."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://example.com/full/path/",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://example.com/full/path/", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_index_name_with_subdir(self, mock_run, mock_find_python):
        """Passing index_name with index_subdir constructs full URL."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_name="stable",
            index_subdir="gfx110X-all",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://repo.amd.com/rocm/whl/gfx110X-all", cmd)

    @patch("setup_venv.find_venv_python_exe", return_value="python")
    @patch("setup_venv.run_command")
    def test_index_url_with_subdir(self, mock_run, mock_find_python):
        """Passing index_url with index_subdir constructs full URL."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://example.com/base",
            index_subdir="gfx94X-dcgpu",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://example.com/base/gfx94X-dcgpu", cmd)

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


class GfxRegexPatternTest(unittest.TestCase):
    def test_valid_match(self):
        html_snippet = '<a href="relpath/to/wherever/gfx103X-dgpu">gfx103X-dgpu</a><br><a href="/relpath/gfx120X-all">gfx120X-all</a>'
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(["gfx103X-dgpu", "gfx120X-all"], matches)

    def test_match_without_suffix(self):
        html_snippet = "<a>gfx940</a><br><a>gfx1030</a>"
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(["gfx940", "gfx1030"], matches)

    def test_invalid_match(self):
        html_snippet = "<a>gfx94000</a><br><a>gfx1030X-dgpu</a>"
        matches = re.findall(GFX_TARGET_REGEX, html_snippet)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
