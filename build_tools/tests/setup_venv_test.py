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
    main,
)


class InstallPackagesTest(unittest.TestCase):
    """Tests for install_packages_into_venv() command generation."""

    def setUp(self):
        self.venv_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.venv_dir, ignore_errors=True)

    @patch("setup_venv.run_command")
    def test_index_name_with_subdir(self, mock_run):
        """index_name with index_subdir constructs full URL."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_name="nightly",
            index_subdir="gfx110X-all",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://rocm.nightlies.amd.com/v2/gfx110X-all", cmd)
        self.assertIn("rocm", cmd)

    @patch("setup_venv.run_command")
    def test_index_url_complete(self, mock_run):
        """index_url without index_subdir uses URL as-is."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://example.com/full/path/",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://example.com/full/path/", cmd)

    @patch("setup_venv.run_command")
    def test_index_url_with_subdir(self, mock_run):
        """index_url with index_subdir constructs full URL."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://example.com/base",
            index_subdir="gfx94X-dcgpu",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://example.com/base/gfx94X-dcgpu", cmd)

    @patch("setup_venv.run_command")
    def test_find_links_only(self, mock_run):
        """find_links_url alone works without index_url."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            find_links_url="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)
        self.assertFalse(any("--index-url" in str(a) for a in cmd))

    @patch("setup_venv.run_command")
    def test_index_url_and_find_links(self, mock_run):
        """Both index_url and find_links_url can be used together."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_url="https://deps/simple/",
            find_links_url="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--index-url=https://deps/simple/", cmd)
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)

    @patch("setup_venv.run_command")
    def test_index_name_and_find_links(self, mock_run):
        """index_name with index_subdir and find_links_url together."""
        install_packages_into_venv(
            venv_dir=self.venv_dir,
            packages=["rocm"],
            index_name="dev",
            index_subdir="gfx110X-all",
            find_links_url="https://bucket/run-123/index.html",
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn(
            "--index-url=https://rocm.devreleases.amd.com/v2/gfx110X-all", cmd
        )
        self.assertIn("--find-links=https://bucket/run-123/index.html", cmd)

    def test_index_url_and_index_name_raises(self):
        """Setting both index_url and index_name raises ValueError."""
        with self.assertRaises(ValueError):
            install_packages_into_venv(
                venv_dir=self.venv_dir,
                packages=["rocm"],
                index_url="https://example.com/",
                index_name="nightly",
            )


class ValidationTest(unittest.TestCase):
    """Tests for argument validation in main()."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("setup_venv._scrape_rocm_index_subdirs", return_value=None)
    def test_index_name_requires_subdir(self, mock_scrape):
        """--index-name without --index-subdir should error."""
        with self.assertRaises(SystemExit):
            main([self.temp_dir, "--packages", "rocm", "--index-name", "nightly"])


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
