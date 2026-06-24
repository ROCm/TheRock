#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``pkg_dependency_checker.py``."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json

_LINUX_DIR = Path(__file__).resolve().parent.parent
if str(_LINUX_DIR) not in sys.path:
    sys.path.insert(0, str(_LINUX_DIR))

import pkg_dependency_checker as checker  # noqa: E402
from packaging_utils import GFX_META, PackageConfig  # noqa: E402


class ParseDependencyNamesTest(unittest.TestCase):
    """``parse_dependency_names`` extracts installable names from control fields."""

    def test_deb_simple_depends(self):
        """DEB ``Depends`` with version constraints yields bare package names."""
        deps = checker.parse_dependency_names(
            "amdrocm-core-sdk7.14-gfx1100 (= 7.14.0~daily), libc6",
            "deb",
        )
        self.assertEqual(deps, ["amdrocm-core-sdk7.14-gfx1100", "libc6"])

    def test_deb_alternatives(self):
        """DEB alternatives (``a | b``) return every alternative name."""
        deps = checker.parse_dependency_names(
            "libstdc++6 | libstdc++8, amdrocm-llvm7.14",
            "deb",
        )
        self.assertEqual(deps[0], "libstdc++6")
        self.assertEqual(deps[1], "libstdc++8")
        self.assertEqual(deps[2], "amdrocm-llvm7.14")

    def test_rpm_requires_skips_rpmlib(self):
        """RPM requires skip ``rpmlib(...)`` virtual constraints."""
        deps = checker.parse_dependency_names(
            "rpmlib(CompressedFileNames) <= 3.0.4-1, amdrocm-core7.14 = 7.14.0",
            "rpm",
        )
        self.assertEqual(deps, ["amdrocm-core7.14"])


class IterVariantConfigsTest(unittest.TestCase):
    """``iter_variant_configs`` mirrors ``build_package`` variant enumeration."""

    def test_kpack_gfxarch_metapackage_includes_meta_and_arch_variants(self):
        """``amdrocm-core-sdk`` in kpack lists non-versioned, GFX_META, and per-arch metas."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackageConfig(
                artifacts_dir=Path(tmp),
                dest_dir=Path(tmp) / "out",
                pkg_type="deb",
                rocm_version="7.14.0",
                version_suffix="daily",
                install_prefix="/opt/rocm/core-7.14",
                gfx_arch="",
                enable_kpack=True,
                gfxarch_list=("gfx1100", "gfx942"),
            )
            labels = [v[0] for v in checker.iter_variant_configs("amdrocm-core-sdk", cfg)]
            self.assertIn("non-versioned metapackage", labels)
            self.assertIn("versioned meta (GFX_META)", labels)
            self.assertTrue(any("gfx1100" in label for label in labels))
            self.assertTrue(any("gfx942" in label for label in labels))

    def test_kpack_non_gfxarch_has_versioned_and_nonversioned(self):
        """Non-gfx metapackage ``amdrocm-developer-tools`` has two kpack variants."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackageConfig(
                artifacts_dir=Path(tmp),
                dest_dir=Path(tmp) / "out",
                pkg_type="deb",
                rocm_version="7.14.0",
                version_suffix="",
                install_prefix="/opt/rocm/core-7.14",
                gfx_arch="",
                enable_kpack=True,
                gfxarch_list=("gfx1100",),
            )
            labels = [
                v[0] for v in checker.iter_variant_configs("amdrocm-developer-tools", cfg)
            ]
            self.assertEqual(labels, ["versioned package", "non-versioned metapackage"])


class ExpectedDependenciesTest(unittest.TestCase):
    """Expected dependency lists use the same rules as packaging."""

    @patch("packaging_utils.print_function_name", lambda: None)
    def test_core_sdk_meta_kpack_depends_on_gfx_variants(self):
        """GFX_META ``amdrocm-core-sdk`` must depend on per-arch SDK metas (issue #6093 guard)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackageConfig(
                artifacts_dir=Path(tmp),
                dest_dir=Path(tmp) / "out",
                pkg_type="deb",
                rocm_version="7.14.0",
                version_suffix="daily",
                install_prefix="/opt/rocm/core-7.14",
                gfx_arch="",
                enable_kpack=True,
                gfxarch_list=("gfx1100", "gfx942"),
            )
            deps = checker.expected_dependencies(
                "amdrocm-core-sdk",
                cfg,
                versioned_pkg=True,
                gfx_arch=GFX_META,
            )
            self.assertTrue(
                any("gfx1100" in dep for dep in deps),
                f"expected gfx1100 variant in deps, got: {deps}",
            )
            self.assertTrue(
                any("gfx942" in dep for dep in deps),
                f"expected gfx942 variant in deps, got: {deps}",
            )
            self.assertFalse(
                any(dep.startswith("amdrocm-core-devel") for dep in deps),
                "regression: must not depend on raw DEBDepends entries only",
            )


class CompareDepListsTest(unittest.TestCase):
    """``_compare_dep_lists`` flags missing declared and missing installed deps."""

    def test_missing_from_actual(self):
        """Dependencies absent from the declared field are reported as missing."""
        missing, not_installed, extra = checker._compare_dep_lists(
            ["a", "b"],
            ["a"],
            verify_installed=False,
            pkg_type="deb",
        )
        self.assertEqual(missing, ["b"])
        self.assertEqual(not_installed, [])
        self.assertEqual(extra, [])

    @patch.object(checker, "is_installed", return_value=False)
    def test_not_installed_when_verify_enabled(self, _mock_installed):
        """``--verify-installed`` marks declared deps that are not on the system."""
        missing, not_installed, extra = checker._compare_dep_lists(
            ["amdrocm-foo"],
            ["amdrocm-foo"],
            verify_installed=True,
            pkg_type="deb",
        )
        self.assertEqual(missing, [])
        self.assertEqual(not_installed, ["amdrocm-foo"])
        self.assertEqual(extra, [])

    def test_extra_deps_in_actual(self):
        """Dependencies present but not expected are reported as extra."""
        _missing, _not_installed, extra = checker._compare_dep_lists(
            ["a"],
            ["a", "b"],
            verify_installed=False,
            pkg_type="deb",
        )
        self.assertEqual(extra, ["b"])


class GuessBasePackageNameTest(unittest.TestCase):
    """``_guess_base_package_name`` maps installed names back to package.json keys."""

    def test_maps_versioned_gfx_sdk_name(self):
        """``amdrocm-core-sdk7.14-gfx1100`` resolves to ``amdrocm-core-sdk``."""
        self.assertEqual(
            checker._guess_base_package_name("amdrocm-core-sdk7.14-gfx1100"),
            "amdrocm-core-sdk",
        )

    def test_maps_deb_dev_suffix_to_devel(self):
        """Debian ``-dev`` suffix maps back to ``-devel`` in package.json."""
        self.assertEqual(
            checker._guess_base_package_name("amdrocm-llvm-dev7.14"),
            "amdrocm-llvm-devel",
        )


class RunSummaryTest(unittest.TestCase):
    """``RunSummary`` aggregates variant results."""

    def test_summary_counts_and_unique_deps(self):
        report = checker.CheckReport(
            base_package="amdrocm-core-sdk",
            variants=[
                checker.VariantCheck(
                    label="meta",
                    installed_name="amdrocm-core-sdk7.14",
                    package_found=True,
                    expected_deps=["a", "b"],
                    actual_deps=["a"],
                    missing=["b"],
                    passed=False,
                ),
                checker.VariantCheck(
                    label="nv",
                    installed_name="amdrocm-core-sdk",
                    package_found=True,
                    expected_deps=["c"],
                    actual_deps=["c"],
                    passed=True,
                ),
            ],
        )
        summary = checker.build_run_summary(["amdrocm-core-sdk"], [report])
        self.assertEqual(summary.variants_checked, 2)
        self.assertEqual(summary.variants_passed, 1)
        self.assertEqual(summary.variants_failed, 1)
        self.assertFalse(summary.passed)
        self.assertEqual(summary.unique_expected_deps(), ["a", "b", "c"])
        self.assertEqual(summary.unique_missing_deps(), ["b"])

    def test_format_report_json_contains_inventory(self):
        report = checker.CheckReport(
            base_package="amdrocm-core-sdk",
            variants=[
                checker.VariantCheck(
                    label="meta",
                    installed_name="amdrocm-core-sdk7.14",
                    package_found=True,
                    expected_deps=["dep-a"],
                    actual_deps=["dep-a"],
                    passed=True,
                ),
            ],
        )
        summary = checker.build_run_summary(["amdrocm-core-sdk"], [report])
        payload = json.loads(checker.format_report_json(summary))
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["packages_requested"], ["amdrocm-core-sdk"])
        self.assertEqual(payload["reports"][0]["variants"][0]["actual_deps"], ["dep-a"])


if __name__ == "__main__":
    unittest.main()
