from pathlib import Path
import os
import sys
import unittest
from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import determine_version


class DetermineVersionTest(unittest.TestCase):
    def test_dev_version(self):
        rocm_version = "7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73"
        optional_build_prod_arguments = determine_version.derive_versions(
            rocm_version=rocm_version, verbose_output=True
        )

        self.assertEqual(
            optional_build_prod_arguments,
            "--rocm-sdk-version ==7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73 --version-suffix +devrocm7.0.0.dev0-515115ea2cb85a0b71b5507ce56a627d14c7ae73",
        )

    def test_nightly_version(self):
        rocm_version = "7.0.0rc20250707"
        optional_build_prod_arguments = determine_version.derive_versions(
            rocm_version=rocm_version, verbose_output=True
        )

        self.assertEqual(
            optional_build_prod_arguments,
            "--rocm-sdk-version ==7.0.0rc20250707 --version-suffix +rocm7.0.0rc20250707",
        )

    def test_version_suffix_sorting(self):
        # This tests that version suffixes follow this ordering:
        # final > prerelease > alpha > dev
        rocm_version_final = "7.0.0"
        rocm_version_prerelease = "7.0.0rc1"
        rocm_version_alpha = "7.0.0a20251202"
        rocm_version_dev = "7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73"

        suffix_final = determine_version.derive_version_suffix(rocm_version_final)
        suffix_prerelease = determine_version.derive_version_suffix(
            rocm_version_prerelease
        )
        suffix_alpha = determine_version.derive_version_suffix(rocm_version_alpha)
        suffix_dev = determine_version.derive_version_suffix(rocm_version_dev)

        version_final = Version("1" + suffix_final)
        version_prerelease = Version("1" + suffix_prerelease)
        version_alpha = Version("1" + suffix_alpha)
        version_dev = Version("1" + suffix_dev)

        self.assertGreater(version_final, version_prerelease)
        self.assertGreater(version_prerelease, version_alpha)
        self.assertGreater(version_alpha, version_dev)


if __name__ == "__main__":
    unittest.main()
