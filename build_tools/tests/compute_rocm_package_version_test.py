# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_rocm_package_version


# Note: the regex matches in here aren't exact, but they should be "good enough"
# to cover the general structure of each version string while allowing for
# future changes like using X.Y versions instead of X.Y.Z versions.


class DetermineVersionTest(unittest.TestCase):
    def test_dev_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type="dev",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 7.9.0.dev0+abcdef
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   .dev0+
        #   [0-9a-z]+   Git SHA (short or long)
        self.assertRegex(version, r"^[0-9]+[0-9\.]*\.dev0\+[0-9a-z]+$")

    def test_nightly_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type="nightly",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 7.9.0rc20251001 (YYYYMMDD)
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   a
        #   [0-9]{8}    Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*a[0-9]{8}$")

    def test_prerelease_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type="prerelease",
            custom_version_suffix=None,
            prerelease_version="5",
            override_base_version=None,
        )
        # For example: 7.9.0rc5
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   rc
        #   .*          Arbitrary suffix (typically a build number)
        self.assertRegex(version, r"^[0-9]+[0-9\.]*rc.*$")

    def test_custom_version_suffix(self):
        version = compute_rocm_package_version.compute_version(
            release_type=None,
            custom_version_suffix="abc",
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 7.9.0.dev0+abcdef
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   abd         Our custom suffix
        self.assertRegex(version, r"^[0-9]+[0-9\.]*abc$")

    def test_override_base_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type=None,
            custom_version_suffix="abc",
            prerelease_version=None,
            override_base_version="1000",
        )
        self.assertEqual(version, "1000abc")


class DebPackageVersionTest(unittest.TestCase):
    """Tests for Debian package version computation."""

    def test_dev_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="dev",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~dev20251203
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~dev
        #   [0-9]{8}    Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~dev[0-9]{8}$")

    def test_nightly_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="nightly",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251203
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~
        #   [0-9]{8}    Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}$")

    def test_prerelease_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="prerelease",
            custom_version_suffix=None,
            prerelease_version="2",
            override_base_version=None,
        )
        # For example: 8.1.0~pre2
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~pre
        #   .*          Prerelease number
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~pre.*$")

    def test_release_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="release",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.0",
        )
        # For example: 8.1.0 (no suffix)
        self.assertEqual(version, "8.1.0")

    def test_custom_version_suffix(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type=None,
            custom_version_suffix="~custom1",
            prerelease_version=None,
            override_base_version="8.0.0",
        )
        self.assertEqual(version, "8.0.0~custom1")


class RpmPackageVersionTest(unittest.TestCase):
    """Tests for RPM package version computation."""

    def test_dev_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="dev",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251203gabcdef1
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~
        #   [0-9]{8}    Date as YYYYMMDD
        #   g
        #   [0-9a-z]{8} Short git SHA (8 characters)
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}g[0-9a-z]{8}$")

    def test_nightly_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="nightly",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251203
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~
        #   [0-9]{8}    Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}$")

    def test_prerelease_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="prerelease",
            custom_version_suffix=None,
            prerelease_version="2",
            override_base_version=None,
        )
        # For example: 8.1.0~rc2
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   ~rc
        #   .*          Prerelease number
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~rc.*$")

    def test_release_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="release",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.0",
        )
        # For example: 8.1.0 (no suffix)
        self.assertEqual(version, "8.1.0")

    def test_custom_version_suffix(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type=None,
            custom_version_suffix="~custom1",
            prerelease_version=None,
            override_base_version="8.0.0",
        )
        self.assertEqual(version, "8.0.0~custom1")


class BackwardsCompatibilityTest(unittest.TestCase):
    """Tests for backwards compatibility with old API."""

    def test_old_api_without_package_type(self):
        """Test that old calls without package_type still work (defaults to wheel)."""
        version = compute_rocm_package_version.compute_version(
            release_type="nightly",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version="7.9.0",
        )
        # Should default to wheel format
        self.assertEqual(version[:5], "7.9.0")
        self.assertIn("a", version)  # alpha version for nightly


class MainFunctionMultiplePackageTypesTest(unittest.TestCase):
    """Tests for main() function with new behavior: compute all package types when --package-type is omitted."""

    def test_compute_all_package_types_without_flag(self):
        """Test that when --package-type is not provided, all types are computed."""
        captured_outputs = {}
        original_gha_set_output = compute_rocm_package_version.gha_set_output

        def mock_gha_set_output(outputs):
            captured_outputs.update(outputs)

        compute_rocm_package_version.gha_set_output = mock_gha_set_output

        try:
            compute_rocm_package_version.main(
                ["--release-type", "dev", "--override-base-version", "8.0.0"]
            )

            # Should have all three outputs
            self.assertIn("rocm_package_version", captured_outputs)
            self.assertIn("rocm_deb_package_version", captured_outputs)
            self.assertIn("rocm_rpm_package_version", captured_outputs)

            # Verify formats
            self.assertRegex(
                captured_outputs["rocm_package_version"], r"^8\.0\.0\.dev0\+[0-9a-z]+$"
            )
            self.assertRegex(
                captured_outputs["rocm_deb_package_version"], r"^8\.0\.0~dev[0-9]{8}$"
            )
            self.assertRegex(
                captured_outputs["rocm_rpm_package_version"],
                r"^8\.0\.0~[0-9]{8}g[0-9a-z]{8}$",
            )
        finally:
            compute_rocm_package_version.gha_set_output = original_gha_set_output

    def test_backward_compatible_with_package_type_specified(self):
        """Test that when --package-type is provided, only that type is computed (backward compatible)."""
        captured_outputs = {}
        original_gha_set_output = compute_rocm_package_version.gha_set_output

        def mock_gha_set_output(outputs):
            captured_outputs.update(outputs)

        compute_rocm_package_version.gha_set_output = mock_gha_set_output

        try:
            compute_rocm_package_version.main(
                [
                    "--package-type",
                    "deb",
                    "--release-type",
                    "nightly",
                    "--override-base-version",
                    "8.1.0",
                ]
            )

            # Should only have deb output
            self.assertIn("rocm_deb_package_version", captured_outputs)
            self.assertNotIn("rocm_package_version", captured_outputs)
            self.assertNotIn("rocm_rpm_package_version", captured_outputs)

            # Verify format
            self.assertRegex(
                captured_outputs["rocm_deb_package_version"], r"^8\.1\.0~[0-9]{8}$"
            )
        finally:
            compute_rocm_package_version.gha_set_output = original_gha_set_output

    def test_existing_workflows_still_work(self):
        """Test that existing workflows reading rocm_package_version still work."""
        captured_outputs = {}
        original_gha_set_output = compute_rocm_package_version.gha_set_output

        def mock_gha_set_output(outputs):
            captured_outputs.update(outputs)

        compute_rocm_package_version.gha_set_output = mock_gha_set_output

        try:
            # This mimics setup_multi_arch.yml line 66: no --package-type specified
            compute_rocm_package_version.main(["--release-type", "dev"])

            # Existing workflow reads rocm_package_version, which should still exist
            self.assertIn("rocm_package_version", captured_outputs)

            # Additional outputs don't break existing workflows (they just ignore them)
            self.assertIn("rocm_deb_package_version", captured_outputs)
            self.assertIn("rocm_rpm_package_version", captured_outputs)
        finally:
            compute_rocm_package_version.gha_set_output = original_gha_set_output


if __name__ == "__main__":
    unittest.main()
