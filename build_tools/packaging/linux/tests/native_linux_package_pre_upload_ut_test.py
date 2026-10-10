#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pre-upload verify and its ``built_packages`` claim inputs.

Covers:
- Presence-only checks against ``built_packages.txt``
  (``native_linux_package_pre_upload_test.py``).
- Enriched ``built_packages.json`` written by ``packaging_summary`` (control
  fields pre-upload will compare in a follow-up).

Run::

    python3.12 -m unittest build_tools.packaging.linux.tests.native_linux_package_pre_upload_ut_test -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

for _path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = os.fspath(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import native_linux_package_pre_upload_test as verify  # noqa: E402
import packaging_summary as summary  # noqa: E402
from packaging_utils import PackageConfig  # noqa: E402

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


def _package_config(dest: Path, *, pkg_type: str = "deb") -> PackageConfig:
    return PackageConfig(
        artifacts_dir=dest / "artifacts",
        dest_dir=dest,
        pkg_type=pkg_type,
        rocm_version="7.15.0",
        version_suffix="1",
        install_prefix="/opt/rocm",
        gfx_arch="gfx1100",
        versioned_pkg=True,
    )


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


class EnrichedPackageSummaryClaimTest(unittest.TestCase):
    """``packaging_summary`` JSON claim used by future pre-upload field checks."""

    @patch("packaging_summary.read_package_control_fields")
    def test_write_build_manifest_emits_json_claim_for_pre_upload(
        self, mock_read: MagicMock
    ) -> None:
        """TXT inventory stays filename-only; JSON carries control fields."""
        mock_read.return_value = {
            "Package": "amdrocm-fft7.15",
            "Version": "7.15.0-1",
            "Architecture": "amd64",
            "Depends": "amdrocm-runtime7.15, libc6",
            "Pre-Depends": "",
            "Recommends": "amdrocm-optional7.15",
            "Suggests": "",
            "Provides": "",
            "Replaces": "",
            "Conflicts": "",
            "Breaks": "",
            "Priority": "optional",
            "Section": "devel",
            "Maintainer": "AMD",
            "Homepage": "https://rocm.docs.amd.com",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "amdrocm-fft7.15_7.15.0-1_amd64.deb"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest, pkg_type="deb"),
                summary.PackageList(
                    total=["amdrocm-fft"],
                    built=[fname],
                    skipped=["amdrocm-optional"],
                    failed=["amdrocm-ck"],
                ),
            )

            txt = (dest / summary.MANIFEST_TXT_NAME).read_text(encoding="utf-8")
            self.assertIn("# Created Packages:", txt)
            self.assertIn(fname, txt)

            doc = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(doc["schema_version"], summary.MANIFEST_SCHEMA_VERSION)
            self.assertEqual(doc["pkg_type"], "deb")
            self.assertEqual(doc["rocm_version"], "7.15.0")
            self.assertEqual(doc["version_suffix"], "1")
            self.assertEqual(doc["failed"], ["amdrocm-ck"])
            self.assertEqual(doc["skipped"], ["amdrocm-optional"])
            self.assertEqual(len(doc["packages"]), 1)
            pkg = doc["packages"][0]
            self.assertEqual(pkg["filename"], fname)
            control = pkg["control"]
            self.assertEqual(control["Package"], "amdrocm-fft7.15")
            self.assertEqual(control["Version"], "7.15.0-1")
            self.assertEqual(control["Depends"], "amdrocm-runtime7.15, libc6")
            self.assertEqual(control["Recommends"], "amdrocm-optional7.15")
            for key in summary.DEB_CONTROL_FIELDS:
                self.assertIn(key, control)

    @patch("packaging_summary.subprocess.run")
    def test_rpm_claim_includes_requires_and_related_fields(
        self, mock_run: MagicMock
    ) -> None:
        """RPM JSON claim includes Requires/Provides/Recommends for pre-upload."""
        scalar = "amdrocm-fft7.15\n7.15.0\n1\nx86_64\nDevelopment/Libraries\nMIT\nAMD\n"

        def _run(cmd, **_kwargs):
            m = MagicMock(returncode=0, stdout="", stderr="")
            if len(cmd) >= 3 and cmd[2] == "--queryformat":
                m.stdout = scalar
            elif len(cmd) >= 2 and cmd[1] == "-qpR":
                m.stdout = (
                    "amdrocm-runtime7.15 = 7.15.0-1\nrpmlib(CompressedFileNames)\n"
                )
            elif "--provides" in cmd:
                m.stdout = "amdrocm-fft7.15 = 7.15.0-1\n"
            elif "--recommends" in cmd:
                m.stdout = "amdrocm-extra7.15\n"
            return m

        mock_run.side_effect = _run
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "amdrocm-fft7.15-7.15.0-1.x86_64.rpm"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest, pkg_type="rpm"),
                summary.PackageList(
                    total=["amdrocm-fft"],
                    built=[fname],
                    skipped=[],
                    failed=[],
                ),
            )
            doc = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )
            control = doc["packages"][0]["control"]
            self.assertEqual(doc["pkg_type"], "rpm")
            self.assertEqual(control["Name"], "amdrocm-fft7.15")
            self.assertEqual(control["Version"], "7.15.0")
            self.assertEqual(control["Requires"], "amdrocm-runtime7.15 = 7.15.0-1")
            self.assertNotIn("rpmlib(", control["Requires"])
            self.assertEqual(control["Provides"], "amdrocm-fft7.15 = 7.15.0-1")
            self.assertEqual(control["Recommends"], "amdrocm-extra7.15")
            for name, _ in summary.RPM_LIST_QUERIES:
                self.assertIn(name, control)

    @patch("packaging_summary.read_package_control_fields")
    def test_control_read_failure_recorded_as_error(self, mock_read: MagicMock) -> None:
        """Unreadable package still appears in the claim with ``error`` (no crash)."""
        mock_read.side_effect = RuntimeError("dpkg-deb failed")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "broken.deb"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest),
                summary.PackageList(
                    total=["broken"], built=[fname], skipped=[], failed=[]
                ),
            )
            pkg = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )["packages"][0]
            self.assertEqual(pkg["filename"], fname)
            self.assertIn("error", pkg)
            self.assertNotIn("control", pkg)


if __name__ == "__main__":
    unittest.main()
