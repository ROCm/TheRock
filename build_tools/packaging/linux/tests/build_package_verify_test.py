#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``build_package_verify.py`` (pre-upload Step 1).

Run (Python 3.10+)::

    python3 -m unittest build_tools.packaging.linux.tests.build_package_verify_test -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_LINUX_DIR = Path(__file__).resolve().parent.parent
if str(_LINUX_DIR) not in sys.path:
    sys.path.insert(0, str(_LINUX_DIR))

import build_package_verify as verify  # noqa: E402
from packaging_utils import PackageConfig  # noqa: E402


class ExpectedControlVersionTest(unittest.TestCase):
    """``expected_control_version`` matches deb/rpm packaging rules."""

    def test_deb_version_with_suffix(self):
        cfg = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp/out"),
            pkg_type="deb",
            rocm_version="7.14.0",
            version_suffix="12345",
            install_prefix="/opt/rocm/core",
            gfx_arch="",
        )
        self.assertEqual(verify.expected_control_version(cfg), "7.14.0-12345")

    def test_deb_version_without_suffix(self):
        cfg = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp/out"),
            pkg_type="deb",
            rocm_version="7.14.0",
            version_suffix="",
            install_prefix="/opt/rocm/core",
            gfx_arch="",
        )
        self.assertEqual(verify.expected_control_version(cfg), "7.14.0")

    def test_rpm_version_release(self):
        cfg = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp/out"),
            pkg_type="rpm",
            rocm_version="7.14.0",
            version_suffix="12345",
            install_prefix="/opt/rocm/core",
            gfx_arch="",
        )
        self.assertEqual(verify.expected_control_version(cfg), "7.14.0-12345")

    def test_rpm_default_release(self):
        cfg = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp/out"),
            pkg_type="rpm",
            rocm_version="7.14.0",
            version_suffix="",
            install_prefix="/opt/rocm/core",
            gfx_arch="",
        )
        self.assertEqual(verify.expected_control_version(cfg), "7.14.0-1")


class VersionsMatchTest(unittest.TestCase):
    """``versions_match`` tolerates DEB ``~`` vs ``-`` normalization."""

    def test_exact_match(self):
        self.assertTrue(verify.versions_match("7.14.0-1", "7.14.0-1", "deb"))

    def test_deb_tilde_normalization(self):
        self.assertTrue(verify.versions_match("7.14.0-12345", "7.14.0~12345", "deb"))

    def test_rpm_no_tilde_normalization(self):
        self.assertFalse(verify.versions_match("7.14.0-12345", "7.14.0~12345", "rpm"))


class VerifyVariantTest(unittest.TestCase):
    """``verify_variant`` presence and version checks."""

    def setUp(self):
        self.config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp/out"),
            pkg_type="deb",
            rocm_version="7.14.0",
            version_suffix="daily",
            install_prefix="/opt/rocm/core",
            gfx_arch="",
        )

    def test_missing_package_fails(self):
        result = verify.verify_variant(
            "amdrocm-core-sdk",
            "main",
            "amdrocm-core-sdk7.14",
            {},
            self.config,
            check_version=True,
        )
        self.assertFalse(result.found)
        self.assertFalse(result.passed)
        self.assertIn("not found", result.errors[0])

    @patch.object(verify, "read_package_file_version", return_value="7.14.0~daily")
    def test_found_with_matching_version_passes(self, _mock_read):
        path = Path("/tmp/pkg.deb")
        files = {"amdrocm-core-sdk7.14": path}
        result = verify.verify_variant(
            "amdrocm-core-sdk",
            "main",
            "amdrocm-core-sdk7.14",
            files,
            self.config,
            check_version=True,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.version_ok)
        self.assertTrue(result.passed)

    @patch.object(verify, "read_package_file_version", return_value="7.14.0~wrong")
    def test_version_mismatch_fails(self, _mock_read):
        path = Path("/tmp/pkg.deb")
        files = {"amdrocm-core-sdk7.14": path}
        result = verify.verify_variant(
            "amdrocm-core-sdk",
            "main",
            "amdrocm-core-sdk7.14",
            files,
            self.config,
            check_version=True,
        )
        self.assertTrue(result.found)
        self.assertFalse(result.version_ok)
        self.assertFalse(result.passed)


class ExtraFilesTest(unittest.TestCase):
    """Unexpected package file detection."""

    def test_collect_extra_package_files(self):
        files = {
            "amdrocm-core-sdk7.14": Path("/tmp/a.deb"),
            "unexpected-pkg7.14": Path("/tmp/b.deb"),
        }
        expected = {"amdrocm-core-sdk7.14"}
        extra = verify.collect_extra_package_files(files, expected)
        self.assertEqual(extra, ["unexpected-pkg7.14"])


class BuildSummaryTest(unittest.TestCase):
    """``BuildVerifySummary`` aggregation and pass/fail."""

    def _variant(self, name: str, *, found: bool, version_ok: bool) -> verify.VariantBuildCheck:
        return verify.VariantBuildCheck(
            base_package="amdrocm-core-sdk",
            label="main",
            expected_name=name,
            file_path=Path("/tmp/x.deb") if found else None,
            found=found,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily" if found else None,
            version_ok=version_ok,
            errors=[] if found and version_ok else ["err"],
        )

    def test_all_passed(self):
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[self._variant("amdrocm-core-sdk7.14", found=True, version_ok=True)],
        )
        summary = verify.build_summary(
            ["amdrocm-core-sdk"],
            [report],
            {"amdrocm-core-sdk7.14": Path("/tmp/x.deb")},
            fail_on_extra=False,
        )
        self.assertTrue(summary.passed)
        self.assertEqual(summary.variants_passed, 1)

    def test_missing_variant_fails(self):
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[self._variant("amdrocm-core-sdk7.14", found=False, version_ok=False)],
        )
        summary = verify.build_summary(
            ["amdrocm-core-sdk"],
            [report],
            {},
            fail_on_extra=False,
        )
        self.assertFalse(summary.passed)
        self.assertEqual(summary.missing_variants(), ["amdrocm-core-sdk7.14"])

    def test_extra_files_fail_when_enabled(self):
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[self._variant("amdrocm-core-sdk7.14", found=True, version_ok=True)],
        )
        summary = verify.build_summary(
            ["amdrocm-core-sdk"],
            [report],
            {
                "amdrocm-core-sdk7.14": Path("/tmp/x.deb"),
                "extra7.14": Path("/tmp/y.deb"),
            },
            fail_on_extra=True,
        )
        self.assertFalse(summary.passed)
        self.assertEqual(summary.extra_package_files, ["extra7.14"])


class ReportFormatTest(unittest.TestCase):
    """JSON and text report serialization."""

    def test_json_roundtrip_fields(self):
        variant = verify.VariantBuildCheck(
            base_package="amdrocm-core-sdk",
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=["amdrocm-core-sdk"],
            reports=[report],
            package_files_found=["amdrocm-core-sdk7.14"],
        )
        payload = json.loads(verify.format_report_json(summary))
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["variants_expected"], 1)
        text = verify.format_report_text(summary)
        self.assertIn("ROCm build package verification report", text)

    def test_text_contains_pass(self):
        variant = verify.VariantBuildCheck(
            base_package="amdrocm-core-sdk",
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=["amdrocm-core-sdk"],
            reports=[report],
            package_files_found=["amdrocm-core-sdk7.14"],
        )
        text = verify.format_report_text(summary)
        self.assertIn("Overall result: PASS", text)
        self.assertIn("amdrocm-core-sdk7.14", text)


class WriteReportFilesTest(unittest.TestCase):
    """``write_report_files`` creates txt and json under report dir."""

    def test_writes_both_reports(self):
        variant = verify.VariantBuildCheck(
            base_package="amdrocm-core-sdk",
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package="amdrocm-core-sdk",
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=["amdrocm-core-sdk"],
            reports=[report],
            package_files_found=["amdrocm-core-sdk7.14"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            verify.write_report_files(summary, report_dir)
            self.assertTrue((report_dir / "build_status_report.txt").is_file())
            self.assertTrue((report_dir / "build_status_report.json").is_file())
            payload = json.loads(
                (report_dir / "build_status_report.json").read_text(encoding="utf-8"),
            )
            self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
