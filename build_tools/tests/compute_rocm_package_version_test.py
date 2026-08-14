# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_rocm_package_version


# Note: the regex matches in here aren't exact, but they should be "good enough"
# to cover the general structure of each version string while allowing for
# future changes like using X.Y versions instead of X.Y.Z versions.


class PythonPackageVersionTest(unittest.TestCase):
    def test_ci_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            release_type="ci",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        self.assertRegex(version, r"^[0-9]+[0-9\.]*\.dev0\+[0-9a-z]+$")

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

    def test_dev_version_with_git_sha_override(self):
        version = compute_rocm_package_version.compute_version(
            release_type="dev",
            override_base_version="7.9.0",
            override_git_sha="abcdef1234567890abcdef1234567890abcdef12",
        )
        self.assertEqual(version, "7.9.0.dev0+abcdef1234567890abcdef1234567890abcdef12")

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

    def test_nightly_with_override_base_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type="nightly",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version="7.9.0",
        )
        self.assertRegex(version, r"^7\.9\.0a[0-9]{8}$")

    def test_versions_are_valid_and_canonical(self):
        # Version() rejects non-PEP 440 versions such as "7.10.0~rc0".
        # See https://packaging.python.org/en/latest/specifications/version-specifiers/.
        versions = self._compute_versions_by_release_type()

        for release_type, version in versions.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(str(Version(version)), version)

    def test_versions_sort_by_release_type(self):
        # pip install --upgrade selects the greatest available version, so enforce:
        # release > prerelease > nightly > dev.
        versions = self._compute_versions_by_release_type()
        print(f"versions: {versions}")

        self.assertGreater(
            Version(versions["nightly"]),
            Version(versions["dev"]),
        )
        self.assertGreater(
            Version(versions["prerelease"]),
            Version(versions["nightly"]),
        )
        self.assertGreater(
            Version(versions["release"]),
            Version(versions["prerelease"]),
        )

    @staticmethod
    def _compute_versions_by_release_type() -> dict[str, str]:
        common_args = {
            "package_type": "wheel",
            "override_base_version": "7.10.0",
        }
        return {
            "dev": compute_rocm_package_version.compute_version(
                release_type="dev",
                override_git_sha="abcdef1234567890abcdef1234567890abcdef12",
                **common_args,
            ),
            "nightly": compute_rocm_package_version.compute_version(
                release_type="nightly",
                **common_args,
            ),
            "prerelease": compute_rocm_package_version.compute_version(
                release_type="prerelease",
                prerelease_version="0",
                **common_args,
            ),
            "release": compute_rocm_package_version.compute_version(
                release_type="release",
                **common_args,
            ),
        }


class DebPackageVersionTest(unittest.TestCase):
    """Tests for Debian package version computation."""

    def test_ci_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="ci",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~dev[0-9]{8}$")

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

    def test_ci_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="ci",
            custom_version_suffix=None,
            prerelease_version=None,
            override_base_version=None,
        )
        self.assertRegex(version, r"^[0-9]+[0-9\.]*~[0-9]{8}g[0-9a-z]{8}$")

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

    def test_dev_version_with_git_sha_override(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="dev",
            override_base_version="8.1.0",
            override_git_sha="abcdef1234567890",
        )
        self.assertRegex(version, r"^8\.1\.0~[0-9]{8}gabcdef12$")

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


class MainFunctionTest(unittest.TestCase):
    def test_sets_package_version_outputs(self):
        with mock.patch.object(
            compute_rocm_package_version, "gha_set_output"
        ) as gha_set_output:
            compute_rocm_package_version.main(
                [
                    "--release-type",
                    "release",
                    "--override-base-version",
                    "7.99.0",
                ]
            )

        gha_set_output.assert_called_once_with(
            {
                "rocm_package_version": "7.99.0",
                "rocm_deb_package_version": "7.99.0",
                "rocm_rpm_package_version": "7.99.0",
            }
        )


if __name__ == "__main__":
    unittest.main()
