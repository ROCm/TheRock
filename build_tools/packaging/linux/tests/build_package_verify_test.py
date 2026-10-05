#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for manifest-based ``build_package_verify.py``.

Validates Created Packages filename presence under packages-dir; Failed /
Skipped base-name sections are excluded from the presence check.

Run::

    python3.12 -m unittest build_tools.packaging.linux.tests.build_package_verify_test -v
"""

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

import build_package_verify as verify  # noqa: E402

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


class ParseBuiltPackageNamesTest(unittest.TestCase):
    """Manifest parsing keeps created filenames and ignores failed/skipped."""

    def test_parses_created_package_filenames(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(
                "\n".join(
                    (
                        "# Created Packages:",
                        "a.deb",
                        "b.rpm",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            names = verify.parse_built_package_names(manifest, pkg_type="rpm")

        self.assertEqual(names, ["b.rpm"])

    def test_name_prefix_filters_smoke_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(SAMPLE_MANIFEST, encoding="utf-8")
            names = verify.parse_built_package_names(
                manifest,
                name_prefix="amdrocm-core-sdk",
            )

        self.assertEqual(
            names,
            [
                "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
                "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
            ],
        )

    def test_missing_manifest_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            verify.parse_built_package_names(Path("/tmp/does-not-exist-xyz.txt"))

    def test_invalid_pkg_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text("a.deb\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify.parse_built_package_names(manifest, pkg_type="msi")


class VerifyManifestPackagesTest(unittest.TestCase):
    """Presence checks against packages-dir."""

    def test_all_present_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / "a.deb").write_bytes(b"deb")
            (packages_dir / "b.deb").write_bytes(b"deb")
            result = verify.verify_manifest_packages(
                packages_dir,
                ["a.deb", "b.deb"],
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing, [])

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / "a.deb").write_bytes(b"deb")
            result = verify.verify_manifest_packages(
                packages_dir,
                ["a.deb", "missing.deb"],
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.missing, ["missing.deb"])
        self.assertEqual(result.found, ["a.deb"])

    def test_path_traversal_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            with self.assertRaises(ValueError):
                verify.verify_manifest_packages(
                    packages_dir,
                    ["../escape.deb"],
                )


class FormatReportTextTest(unittest.TestCase):
    """Stdout report includes overall status and missing package names."""

    def test_report_lists_missing(self) -> None:
        result = verify.ManifestVerifyResult(
            expected=["a.deb", "b.deb"],
            found=["a.deb"],
            missing=["b.deb"],
        )
        text = verify.format_report_text(result, Path("/tmp/built_packages.txt"))
        self.assertIn("Overall result: FAIL", text)
        self.assertIn("b.deb", text)


class RunCliTest(unittest.TestCase):
    """End-to-end CLI against a temp packages dir + manifest."""

    def test_run_passes_when_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            deb_name = "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb"
            (packages_dir / deb_name).write_bytes(b"deb")
            (packages_dir / verify.MANIFEST_NAME).write_text(
                f"# Created Packages:\n{deb_name}\n",
                encoding="utf-8",
            )
            args = verify.parse_args(
                [
                    "--packages-dir",
                    str(packages_dir),
                    "--pkg-type",
                    "deb",
                ],
            )
            self.assertEqual(verify.run(args), 0)

    def test_run_fails_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / verify.MANIFEST_NAME).write_text(
                "missing.deb\n",
                encoding="utf-8",
            )
            args = verify.parse_args(["--packages-dir", str(packages_dir)])
            self.assertEqual(verify.run(args), 1)

    def test_run_errors_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            args = verify.parse_args(["--packages-dir", str(packages_dir)])
            self.assertEqual(verify.run(args), 2)

    def test_run_errors_when_filter_matches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / verify.MANIFEST_NAME).write_text(
                "amdrocm-fft7.15_7.15.0-1_amd64.deb\n",
                encoding="utf-8",
            )
            args = verify.parse_args(
                [
                    "--packages-dir",
                    str(packages_dir),
                    "--name-prefix",
                    "amdrocm-core-sdk",
                ],
            )
            self.assertEqual(verify.run(args), 2)

    def test_run_errors_on_path_traversal_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / verify.MANIFEST_NAME).write_text(
                "../escape.deb\n",
                encoding="utf-8",
            )
            args = verify.parse_args(["--packages-dir", str(packages_dir)])
            self.assertEqual(verify.run(args), 2)

    def test_run_passes_with_failed_and_skipped_sections(self) -> None:
        """Presence check uses Created filenames only; Failed/Skipped are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            for name in (
                "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
                "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
                "amdrocm-fft7.15_7.15.0-1_amd64.deb",
            ):
                (packages_dir / name).write_bytes(b"deb")
            (packages_dir / verify.MANIFEST_NAME).write_text(
                SAMPLE_MANIFEST,
                encoding="utf-8",
            )
            args = verify.parse_args(
                ["--packages-dir", str(packages_dir), "--pkg-type", "deb"],
            )
            self.assertEqual(verify.run(args), 0)

    def test_run_errors_when_only_failed_base_names(self) -> None:
        """Manifest with no .deb/.rpm filenames is a configuration error."""
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / verify.MANIFEST_NAME).write_text(
                "# Failed Packages:\namdrocm-ck\n",
                encoding="utf-8",
            )
            args = verify.parse_args(["--packages-dir", str(packages_dir)])
            self.assertEqual(verify.run(args), 2)

    def test_run_errors_when_packages_dir_missing(self) -> None:
        args = verify.parse_args(
            ["--packages-dir", "/tmp/does-not-exist-pkg-verify-xyz"],
        )
        self.assertEqual(verify.run(args), 2)

    def test_directory_entry_counts_as_missing(self) -> None:
        """A directory with the package name is not a present package file."""
        with tempfile.TemporaryDirectory() as tmp:
            packages_dir = Path(tmp)
            (packages_dir / "a.deb").mkdir()
            result = verify.verify_manifest_packages(packages_dir, ["a.deb"])
        self.assertFalse(result.passed)
        self.assertEqual(result.missing, ["a.deb"])


if __name__ == "__main__":
    unittest.main()
