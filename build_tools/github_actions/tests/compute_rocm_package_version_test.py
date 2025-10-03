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
            override_base_version=None,
        )
        # For example: 7.9.0rc20251001 (YYYYMMDD)
        #   [0-9]+      Must start with a number
        #   [0-9\.]*    Some additional numbers and/or periods
        #   rc
        #   [0-9]{8}    Date as YYYYMMDD
        self.assertRegex(version, r"^[0-9]+[0-9\.]*rc[0-9]{8}$")

    def test_custom_version_suffix(self):
        version = compute_rocm_package_version.compute_version(
            release_type=None,
            custom_version_suffix="abc",
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
            override_base_version="1000",
        )
        self.assertEqual(version, "1000abc")


if __name__ == "__main__":
    unittest.main()
