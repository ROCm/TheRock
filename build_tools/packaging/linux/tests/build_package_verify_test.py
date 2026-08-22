#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``build_package_verify.py``.

Run::

    python3.12 -m unittest build_tools.packaging.linux.tests.build_package_verify_test -v
"""

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

TEST_ROCM_VERSION = "7.14.0"
TEST_VERSION_SUFFIX = "daily"
TEST_INSTALL_PREFIX = "/opt/rocm/core"
TEST_GFX_TARGET = "gfx1100"
TEST_GFX_TARGET_ALT = "gfx942"
TEST_PKG_TYPE_DEB = "deb"

PKG_CORE_SDK = "amdrocm-core-sdk"
PKG_DEVELOPER_TOOLS = "amdrocm-developer-tools"
PKG_FFT = "amdrocm-fft"


def _setup_import_path() -> None:
    for path in (BUILD_TOOLS_DIR, LINUX_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_setup_import_path()

import build_package  # noqa: E402
import build_package_verify as verify  # noqa: E402
from packaging_utils import (  # noqa: E402
    GFX_META,
    PackageConfig,
    get_package_info,
    is_gfxarch_package,
)


def _args(tmp: Path, **overrides: object) -> Namespace:
    artifacts = tmp / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, object] = {
        "artifacts_dir": artifacts,
        "dest_dir": tmp / "output",
        "target": [TEST_GFX_TARGET, TEST_GFX_TARGET_ALT],
        "pkg_type": TEST_PKG_TYPE_DEB,
        "rocm_version": TEST_ROCM_VERSION,
        "version_suffix": TEST_VERSION_SUFFIX,
        "install_prefix": TEST_INSTALL_PREFIX,
        "runpath_pkg": False,
        "enable_kpack": False,
        "build_variant": "",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_kpack_manifest(artifacts_dir: Path) -> None:
    manifest_dir = artifacts_dir / "pkg"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "therock_manifest.json").write_text(
        json.dumps({"flags": {"KPACK_SPLIT_ARTIFACTS": True}}),
        encoding="utf-8",
    )


def _kpack_config(tmp: Path, **overrides: object) -> PackageConfig:
    root = Path(tmp)
    _write_kpack_manifest(root / "artifacts")
    return build_package.create_package_config(
        _args(root, enable_kpack=True, **overrides),
    )


class BuildPackageVerifyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_context = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_context.name)

    def tearDown(self) -> None:
        self._temp_context.cleanup()


class ExpectedControlVersionTest(unittest.TestCase):
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
    def test_exact_match(self):
        self.assertTrue(verify.versions_match("7.14.0-1", "7.14.0-1", "deb"))

    def test_deb_tilde_normalization(self):
        self.assertTrue(verify.versions_match("7.14.0-12345", "7.14.0~12345", "deb"))

    def test_rpm_no_tilde_normalization(self):
        self.assertFalse(verify.versions_match("7.14.0-12345", "7.14.0~12345", "rpm"))


class IterPackageVariantSpecsRoutingTest(BuildPackageVerifyTestCase):
    def test_core_sdk_kpack_lists_meta_and_device_variants(self):
        cfg = _kpack_config(self.temp_dir)
        labels = [
            spec.label
            for spec in verify.iter_package_variant_specs(PKG_CORE_SDK, cfg)
        ]
        self.assertIn("meta", labels)
        self.assertIn("non-versioned", labels)
        self.assertIn(f"device-{TEST_GFX_TARGET}", labels)
        self.assertNotIn("host", labels)

    def test_developer_tools_kpack_lists_simple_variants(self):
        cfg = _kpack_config(self.temp_dir)
        labels = [
            spec.label
            for spec in verify.iter_package_variant_specs(
                PKG_DEVELOPER_TOOLS, cfg
            )
        ]
        self.assertEqual(labels, ["versioned", "non-versioned"])

    def test_fft_without_gfx_artifacts_lists_simple_variants(self):
        cfg = _kpack_config(self.temp_dir)
        pkg_info = get_package_info(PKG_FFT)
        self.assertFalse(
            is_gfxarch_package(
                pkg_info=pkg_info,
                enable_kpack=True,
                artifacts_dir=cfg.artifacts_dir,
            ),
        )
        labels = [
            spec.label
            for spec in verify.iter_package_variant_specs(PKG_FFT, cfg)
        ]
        self.assertEqual(labels, ["versioned", "non-versioned"])

    def test_gfx_meta_variant_uses_meta_arch_suffix(self):
        cfg = _kpack_config(self.temp_dir)
        meta_variants = [
            (spec.versioned_pkg, spec.gfx_arch)
            for spec in verify.iter_package_variant_specs(PKG_CORE_SDK, cfg)
            if spec.label == "meta"
        ]
        self.assertEqual(meta_variants, [(True, GFX_META)])


class VerifyVariantTest(unittest.TestCase):
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
            PKG_CORE_SDK,
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
            PKG_CORE_SDK,
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
            PKG_CORE_SDK,
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
    def test_collect_extra_package_files(self):
        files = {
            "amdrocm-core-sdk7.14": Path("/tmp/a.deb"),
            "unexpected-pkg7.14": Path("/tmp/b.deb"),
        }
        expected = {"amdrocm-core-sdk7.14"}
        extra = verify.collect_extra_package_files(files, expected)
        self.assertEqual(extra, ["unexpected-pkg7.14"])


class BuildSummaryTest(unittest.TestCase):
    def _variant(
        self, name: str, *, found: bool, version_ok: bool
    ) -> verify.VariantBuildCheck:
        return verify.VariantBuildCheck(
            base_package=PKG_CORE_SDK,
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
            base_package=PKG_CORE_SDK,
            variants=[
                self._variant("amdrocm-core-sdk7.14", found=True, version_ok=True)
            ],
        )
        summary = verify.build_summary(
            [PKG_CORE_SDK],
            [report],
            {"amdrocm-core-sdk7.14": Path("/tmp/x.deb")},
            fail_on_extra=False,
        )
        self.assertTrue(summary.passed)
        self.assertEqual(summary.variants_passed, 1)

    def test_missing_variant_fails(self):
        report = verify.BuildVerifyReport(
            base_package=PKG_CORE_SDK,
            variants=[
                self._variant("amdrocm-core-sdk7.14", found=False, version_ok=False)
            ],
        )
        summary = verify.build_summary(
            [PKG_CORE_SDK],
            [report],
            {},
            fail_on_extra=False,
        )
        self.assertFalse(summary.passed)
        self.assertEqual(summary.missing_variants(), ["amdrocm-core-sdk7.14"])

    def test_extra_files_fail_when_enabled(self):
        report = verify.BuildVerifyReport(
            base_package=PKG_CORE_SDK,
            variants=[
                self._variant("amdrocm-core-sdk7.14", found=True, version_ok=True)
            ],
        )
        summary = verify.build_summary(
            [PKG_CORE_SDK],
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
    def test_json_roundtrip_fields(self):
        variant = verify.VariantBuildCheck(
            base_package=PKG_CORE_SDK,
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package=PKG_CORE_SDK,
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=[PKG_CORE_SDK],
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
            base_package=PKG_CORE_SDK,
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package=PKG_CORE_SDK,
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=[PKG_CORE_SDK],
            reports=[report],
            package_files_found=["amdrocm-core-sdk7.14"],
        )
        text = verify.format_report_text(summary)
        self.assertIn("Overall result: PASS", text)
        self.assertIn("amdrocm-core-sdk7.14", text)


class WriteReportFilesTest(unittest.TestCase):
    def test_writes_both_reports(self):
        variant = verify.VariantBuildCheck(
            base_package=PKG_CORE_SDK,
            label="main",
            expected_name="amdrocm-core-sdk7.14",
            file_path=Path("/tmp/x.deb"),
            found=True,
            expected_version="7.14.0-daily",
            actual_version="7.14.0~daily",
            version_ok=True,
        )
        report = verify.BuildVerifyReport(
            base_package=PKG_CORE_SDK,
            variants=[variant],
        )
        summary = verify.BuildVerifySummary(
            packages_requested=[PKG_CORE_SDK],
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
