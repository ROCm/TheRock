#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for presence-only ``native_linux_package_pre_upload_test.py``.

Covers Created Packages filename parsing (Failed/Skipped ignored), path-safe
presence checks, report formatting, and CLI exit codes.

Run::

    python3.12 -m unittest build_tools.packaging.linux.tests.native_linux_package_pre_upload_ut_test -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

for _path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = os.fspath(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import native_linux_package_pre_upload_test as verify  # noqa: E402

SAMPLE_MANIFEST = """# Built Packages Manifest
# Package Type: DEB
# Successfully built: 3
# Failed to build: 1

# Created Packages:
amdrocm-core-sdk7.15_7.15.0-1_amd64.deb
amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb
amdrocm-fft7.15_7.15.0-1_amd64.deb

# Failed Packages:
amdrocm-ck

# Skipped Packages:
# Note: Package names shown are base names from package.json
amdrocm-optional
"""


def _cli_args(packages_dir: Path, *extra: str) -> list[str]:
    """Build minimal CLI argv for ``verify.main`` against ``packages_dir``."""
    return [
        "--packages-dir",
        str(packages_dir),
        "--pkg-type",
        "deb",
        *extra,
    ]


class ParseBuiltPackageNamesTest(unittest.TestCase):
    """Manifest parsing keeps created filenames and ignores failed/skipped."""

    def test_parses_created_package_filenames(self) -> None:
        """Created ``.deb`` lines are returned; failed/skipped base names are not."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(SAMPLE_MANIFEST, encoding="utf-8")
            names = verify.parse_built_package_names(manifest)

        self.assertEqual(
            names,
            [
                "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
                "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
                "amdrocm-fft7.15_7.15.0-1_amd64.deb",
            ],
        )

    def test_pkg_type_filters_extension(self) -> None:
        """``pkg_type=deb`` drops ``.rpm`` filenames from the result."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(
                "# Created Packages:\na.deb\nb.rpm\n",
                encoding="utf-8",
            )
            names = verify.parse_built_package_names(manifest, pkg_type="deb")
        self.assertEqual(names, ["a.deb"])

    def test_name_prefix_filter(self) -> None:
        """``name_prefix`` keeps only matching Created filenames."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(SAMPLE_MANIFEST, encoding="utf-8")
            names = verify.parse_built_package_names(
                manifest, name_prefix="amdrocm-fft"
            )
        self.assertEqual(names, ["amdrocm-fft7.15_7.15.0-1_amd64.deb"])

    def test_missing_manifest(self) -> None:
        """Missing manifest path raises ``FileNotFoundError``."""
        with self.assertRaises(FileNotFoundError):
            verify.parse_built_package_names(Path("/no/such/built_packages.txt"))


class VerifyManifestPackagesTest(unittest.TestCase):
    """Presence check and path-safety behavior."""

    def test_found_and_missing(self) -> None:
        """Present files go to ``found``; absent names go to ``missing``."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.deb").write_bytes(b"x")
            result = verify.verify_manifest_packages(root, ["a.deb", "b.deb"])
        self.assertEqual(result.found, ["a.deb"])
        self.assertEqual(result.missing, ["b.deb"])
        self.assertFalse(result.passed)

    def test_path_traversal_rejected(self) -> None:
        """Filenames that escape ``packages_dir`` raise ``ValueError``."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify.verify_manifest_packages(Path(tmp), ["../escape.deb"])


class FormatReportTextTest(unittest.TestCase):
    """Human-readable report content."""

    def test_includes_overall(self) -> None:
        """PASS report includes overall status and the manifest path."""
        result = verify.ManifestVerifyResult(expected=["a.deb"], found=["a.deb"])
        text = verify.format_report_text(result, Path("/tmp/built_packages.txt"))
        self.assertIn("Overall result: PASS", text)
        self.assertIn("built_packages.txt", text)


class RunCliTest(unittest.TestCase):
    """CLI exit codes for presence verification."""

    def test_pass_when_files_exist(self) -> None:
        """Exit ``0`` when every Created filename exists under packages-dir."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "built_packages.txt").write_text(SAMPLE_MANIFEST, encoding="utf-8")
            for name in (
                "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
                "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
                "amdrocm-fft7.15_7.15.0-1_amd64.deb",
            ):
                (root / name).write_bytes(b"x")
            self.assertEqual(verify.main(_cli_args(root)), 0)

    def test_missing_package_fails(self) -> None:
        """Exit ``1`` when Created filenames are listed but files are absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "built_packages.txt").write_text(SAMPLE_MANIFEST, encoding="utf-8")
            self.assertEqual(verify.main(_cli_args(root)), 1)

    def test_missing_manifest_is_config_error(self) -> None:
        """Exit ``2`` when ``built_packages.txt`` is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(verify.main(_cli_args(Path(tmp))), 2)


if __name__ == "__main__":
    unittest.main()
