import argparse
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_rocm_native_package_version


# Note: the regex matches in here aren't exact, but they should be "good enough"
# to cover the general structure of each version string while allowing for
# future changes like using X.Y versions instead of X.Y.Z versions.


class NativePackageVersionTest(unittest.TestCase):
    """Test suite for ROCm native package version computation."""

    # RPM Tests
    def test_rpm_dev_version(self):
        """Test RPM development version format: <version>~<YYYYMMDD>g<short-git-sha>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="dev",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251201gf689a8e
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~              Tilde separator
        #   [0-9]{8}       Date as YYYYMMDD
        #   g              Git prefix
        #   [0-9a-f]{8}    Short git SHA (8 characters)
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}g[0-9a-f]{8}$")

    def test_rpm_nightly_version(self):
        """Test RPM nightly version format: <version>~<YYYYMMDD>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="nightly",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251203
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~              Tilde separator
        #   [0-9]{8}       Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}$")

    def test_rpm_prerelease_version(self):
        """Test RPM prerelease version format: <version>~rc<N>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="rpm",
            version_suffix=None,
            prerelease_version="1",
            override_base_version=None,
        )
        # For example: 8.1.0~rc1
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~rc            Tilde separator + rc prefix
        #   [0-9]+         Prerelease number
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~rc[0-9]+$")

    def test_rpm_prerelease_version_2(self):
        """Test RPM prerelease version with different number"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="rpm",
            version_suffix=None,
            prerelease_version="2",
            override_base_version=None,
        )
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~rc2$")

    def test_rpm_release_version(self):
        """Test RPM final release version format: <version>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   $              No suffix
        self.assertRegex(version, r"^[0-9]+[0-9\.]*$")
        # Ensure no tilde is present in release versions
        self.assertNotIn("~", version)

    # DEB Tests
    def test_deb_dev_version(self):
        """Test DEB development version format: <version>~dev<YYYYMMDD>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="dev",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~dev20251201
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~dev           Tilde separator + dev prefix
        #   [0-9]{8}       Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~dev[0-9]{8}$")

    def test_deb_nightly_version(self):
        """Test DEB nightly version format: <version>~<YYYYMMDD>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="nightly",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~20251203
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~              Tilde separator
        #   [0-9]{8}       Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}$")

    def test_deb_prerelease_version(self):
        """Test DEB prerelease version format: <version>~pre<N>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="deb",
            version_suffix=None,
            prerelease_version="1",
            override_base_version=None,
        )
        # For example: 8.1.0~pre1
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   ~pre           Tilde separator + pre prefix
        #   [0-9]+         Prerelease number
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~pre[0-9]+$")

    def test_deb_prerelease_version_2(self):
        """Test DEB prerelease version with different number"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="deb",
            version_suffix=None,
            prerelease_version="2",
            override_base_version=None,
        )
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~pre2$")

    def test_deb_release_version(self):
        """Test DEB final release version format: <version>"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0
        #   [0-9]+         Must start with a number
        #   [0-9\.]*       Some additional numbers and/or periods
        #   $              No suffix
        self.assertRegex(version, r"^[0-9]+[0-9\.]*$")
        # Ensure no tilde is present in release versions
        self.assertNotIn("~", version)

    # Custom Suffix Tests
    def test_custom_version_suffix_rpm(self):
        """Test custom version suffix for RPM"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="dev",
            package_type="rpm",
            version_suffix=".dev0",
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0.dev0
        self.assertRegex(version, r"^[0-9]+[0-9\.]*\.dev0$")

    def test_custom_version_suffix_deb(self):
        """Test custom version suffix for DEB"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="nightly",
            package_type="deb",
            version_suffix="~custom123",
            prerelease_version=None,
            override_base_version=None,
        )
        # For example: 8.1.0~custom123
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~custom123$")

    # Override Base Version Tests
    def test_override_base_version_rpm(self):
        """Test override base version for RPM"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.1",
        )
        self.assertEqual(version, "8.1.1")

    def test_override_base_version_deb(self):
        """Test override base version for DEB"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.1",
        )
        self.assertEqual(version, "8.1.1")

    def test_override_with_custom_suffix(self):
        """Test override base version with custom suffix"""
        version = compute_rocm_native_package_version.compute_version(
            release_type="dev",
            package_type="rpm",
            version_suffix="abc",
            prerelease_version=None,
            override_base_version="1000",
        )
        self.assertEqual(version, "1000abc")

    # Version Ordering Tests (conceptual - these test that versions follow expected patterns)
    def test_rpm_version_sequence(self):
        """Test that RPM versions follow expected patterns for ordering"""
        # Create versions with mocked dates and git SHAs
        with patch.object(
            compute_rocm_native_package_version, "get_current_date", return_value="20251201"
        ):
            with patch.object(
                compute_rocm_native_package_version, "get_git_sha", return_value="f689a8ea"
            ):
                dev1 = compute_rocm_native_package_version.compute_version(
                    release_type="dev",
                    package_type="rpm",
                    version_suffix=None,
                    prerelease_version=None,
                    override_base_version="8.1.0",
                )
                self.assertEqual(dev1, "8.1.0~20251201gf689a8ea")

        with patch.object(
            compute_rocm_native_package_version, "get_current_date", return_value="20251203"
        ):
            nightly = compute_rocm_native_package_version.compute_version(
                release_type="nightly",
                package_type="rpm",
                version_suffix=None,
                prerelease_version=None,
                override_base_version="8.1.0",
            )
            self.assertEqual(nightly, "8.1.0~20251203")

        rc1 = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="rpm",
            version_suffix=None,
            prerelease_version="1",
            override_base_version="8.1.0",
        )
        self.assertEqual(rc1, "8.1.0~rc1")

        rc2 = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="rpm",
            version_suffix=None,
            prerelease_version="2",
            override_base_version="8.1.0",
        )
        self.assertEqual(rc2, "8.1.0~rc2")

        release = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.0",
        )
        self.assertEqual(release, "8.1.0")

        next_release = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="rpm",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.1",
        )
        self.assertEqual(next_release, "8.1.1")

    def test_deb_version_sequence(self):
        """Test that DEB versions follow expected patterns for ordering"""
        # Create versions with mocked dates
        with patch.object(
            compute_rocm_native_package_version, "get_current_date", return_value="20251201"
        ):
            dev1 = compute_rocm_native_package_version.compute_version(
                release_type="dev",
                package_type="deb",
                version_suffix=None,
                prerelease_version=None,
                override_base_version="8.1.0",
            )
            self.assertEqual(dev1, "8.1.0~dev20251201")

        with patch.object(
            compute_rocm_native_package_version, "get_current_date", return_value="20251202"
        ):
            dev2 = compute_rocm_native_package_version.compute_version(
                release_type="dev",
                package_type="deb",
                version_suffix=None,
                prerelease_version=None,
                override_base_version="8.1.0",
            )
            self.assertEqual(dev2, "8.1.0~dev20251202")

        with patch.object(
            compute_rocm_native_package_version, "get_current_date", return_value="20251203"
        ):
            nightly = compute_rocm_native_package_version.compute_version(
                release_type="nightly",
                package_type="deb",
                version_suffix=None,
                prerelease_version=None,
                override_base_version="8.1.0",
            )
            self.assertEqual(nightly, "8.1.0~20251203")

        pre1 = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="deb",
            version_suffix=None,
            prerelease_version="1",
            override_base_version="8.1.0",
        )
        self.assertEqual(pre1, "8.1.0~pre1")

        pre2 = compute_rocm_native_package_version.compute_version(
            release_type="prerelease",
            package_type="deb",
            version_suffix=None,
            prerelease_version="2",
            override_base_version="8.1.0",
        )
        self.assertEqual(pre2, "8.1.0~pre2")

        release = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.0",
        )
        self.assertEqual(release, "8.1.0")

        next_release = compute_rocm_native_package_version.compute_version(
            release_type="release",
            package_type="deb",
            version_suffix=None,
            prerelease_version=None,
            override_base_version="8.1.1",
        )
        self.assertEqual(next_release, "8.1.1")


if __name__ == "__main__":
    unittest.main()
