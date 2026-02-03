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
    main,
)


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


class InstallPackagesTest(unittest.TestCase):
    """Tests for package installation command generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_pip_install_call(self, mock_run_command):
        """Extract the pip install command from mock calls."""
        for call in mock_run_command.call_args_list:
            args = call[0][0]  # First positional arg is the command list
            # Look for pip install with a package (not just --upgrade pip)
            if "install" in args and any("rocm" in str(a) for a in args):
                return args
        return None

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_name_with_subdir(self, mock_scrape, mock_run):
        """--index-name with --index-subdir constructs full URL."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--index-name",
                "nightly",
                "--index-subdir",
                "gfx110X-all",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn("--index-url=https://rocm.nightlies.amd.com/v2/gfx110X-all", cmd)
        self.assertIn("rocm", cmd)

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_url_complete(self, mock_scrape, mock_run):
        """--index-url without --index-subdir uses URL as-is."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--index-url",
                "https://example.com/full/path/",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn("--index-url=https://example.com/full/path/", cmd)

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_url_with_subdir(self, mock_scrape, mock_run):
        """--index-url with --index-subdir constructs full URL."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--index-url",
                "https://example.com/base",
                "--index-subdir",
                "gfx94X-dcgpu",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn("--index-url=https://example.com/base/gfx94X-dcgpu", cmd)

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_find_links_only(self, mock_scrape, mock_run):
        """--find-links-url alone works without --index-url."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--find-links-url",
                "https://bucket/run-123/index.html",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)
        # Should not have --index-url when only --find-links-url is specified
        self.assertFalse(any("--index-url" in str(a) for a in cmd))

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_url_and_find_links(self, mock_scrape, mock_run):
        """Both --index-url and --find-links-url can be used together."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--index-url",
                "https://deps/simple/",
                "--find-links-url",
                "https://bucket/run-123/index.html",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn("--index-url=https://deps/simple/", cmd)
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)

    @patch("setup_venv.run_command")
    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_name_and_find_links(self, mock_scrape, mock_run):
        """--index-name with --index-subdir and --find-links-url together."""
        main(
            [
                self.temp_dir,
                "--packages",
                "rocm",
                "--index-name",
                "dev",
                "--index-subdir",
                "gfx110X-all",
                "--find-links-url",
                "https://bucket/run-123/index.html",
            ]
        )

        cmd = self._get_pip_install_call(mock_run)
        self.assertIsNotNone(cmd)
        self.assertIn(
            "--index-url=https://rocm.devreleases.amd.com/v2/gfx110X-all", cmd
        )
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)


class ValidationTest(unittest.TestCase):
    """Tests for argument validation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_packages_requires_index_source(self, mock_scrape):
        """--packages without any index option should error."""
        with self.assertRaises(SystemExit):
            main(
                [
                    self.temp_dir,
                    "--packages",
                    "rocm",
                ]
            )

    @patch("setup_venv.scrape_subdirs", return_value=None)
    def test_index_name_requires_subdir(self, mock_scrape):
        """--index-name without --index-subdir should error."""
        with self.assertRaises(SystemExit):
            main(
                [
                    self.temp_dir,
                    "--packages",
                    "rocm",
                    "--index-name",
                    "nightly",
                ]
            )


if __name__ == "__main__":
    unittest.main()
