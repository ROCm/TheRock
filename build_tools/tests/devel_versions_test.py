# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Development-tree version checks use real synthetic wheel metadata, without a GPU."""

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROCM_SDK_SRC = (
    Path(__file__).resolve().parent.parent / "packaging/python/templates/rocm/src"
)
sys.path.insert(0, os.fspath(ROCM_SDK_SRC))
from rocm_sdk import _devel


VERSION = "7.13.0a20260425"
OTHER_VERSION = "7.13.0a20260512"


class DevelVersionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name)
        for name in ("rocm-sdk-devel", "rocm-sdk-core", "rocm-sdk-libraries-gfx1151"):
            self.install_metadata(name, VERSION)

    def install_metadata(self, name, version):
        directory = self.site / (name.replace("-", "_") + ".dist-info")
        directory.mkdir(exist_ok=True)
        (directory / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )

    def test_matching_versions(self):
        _devel._validate_devel_versions(self.site, VERSION)

    def test_reported_mixed_nightly_is_rejected(self):
        self.install_metadata("rocm-sdk-devel", OTHER_VERSION)
        with self.assertRaisesRegex(ImportError, f"rocm-sdk-devel=={OTHER_VERSION}"):
            _devel._validate_devel_versions(self.site, VERSION)

    def test_mismatched_runtime_is_rejected(self):
        for name in (
            "rocm-sdk-core",
            "rocm-sdk-libraries-gfx1151",
            "rocm-sdk-libraries",
        ):
            with self.subTest(package=name):
                self.install_metadata(name, OTHER_VERSION)
                with self.assertRaisesRegex(ImportError, f"{name}=={OTHER_VERSION}"):
                    _devel._validate_devel_versions(self.site, VERSION)
                self.install_metadata(name, VERSION)

    def test_post_releases_follow_device_compatibility_policy(self):
        self.install_metadata("rocm-sdk-devel", VERSION + ".post2")
        self.install_metadata("rocm-sdk-core", VERSION + ".post1")
        _devel._validate_devel_versions(self.site, VERSION + ".post3")

    def test_unrelated_packages_are_not_version_locked(self):
        self.install_metadata("rocm-profiler", "1.0")
        self.install_metadata("numpy", "2.0")
        _devel._validate_devel_versions(self.site, VERSION)

    def test_normalized_distribution_names(self):
        self.install_metadata("ROCM_SDK_LIBRARIES_gfx942", OTHER_VERSION)
        with self.assertRaisesRegex(ImportError, "rocm-sdk-libraries-gfx942"):
            _devel._validate_devel_versions(self.site, VERSION)

    def test_check_precedes_both_expansion_and_reuse(self):
        self.install_metadata("rocm-sdk-devel", OTHER_VERSION)
        module = types.ModuleType("rocm_sdk_devel")
        module.__file__ = str(self.site / "rocm_sdk_devel" / "__init__.py")
        expanded = self.site / "_rocm_sdk_devel"
        package_entry = mock.Mock()
        package_entry.get_py_package_name.return_value = expanded.name
        with (
            mock.patch.dict(sys.modules, {"rocm_sdk_devel": module}),
            mock.patch.object(_devel.di, "__version__", VERSION, create=True),
            mock.patch.dict(_devel.di.ALL_PACKAGES, {"devel": package_entry}),
            mock.patch.object(_devel, "_expand_devel_contents") as expand,
            mock.patch.object(_devel, "_reconcile_device_links") as reconcile,
        ):
            for already_expanded in (False, True):
                with self.subTest(already_expanded=already_expanded):
                    if already_expanded:
                        expanded.mkdir()
                        (expanded / "__init__.py").touch()
                    with self.assertRaisesRegex(ImportError, "same SDK build"):
                        _devel.get_devel_root()
                    expand.assert_not_called()
                    reconcile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
