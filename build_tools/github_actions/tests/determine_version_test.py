from pathlib import Path
import os
import sys
import unittest

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
            "--rocm-sdk-version ==7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73 --version-suffix +rocm7.0.0.dev0-515115ea2cb85a0b71b5507ce56a627d14c7ae73",
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


if __name__ == "__main__":
    unittest.main()
