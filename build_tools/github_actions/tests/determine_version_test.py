# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest import mock

from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import determine_version


class DetermineVersionTest(unittest.TestCase):
    def test_dev_version_suffix(self):
        rocm_version = "7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73"
        suffix = determine_version.derive_version_suffix(rocm_version)
        self.assertEqual(
            suffix,
            "+devrocm7.0.0.dev0-515115ea2cb85a0b71b5507ce56a627d14c7ae73",
        )

    def test_nightly_version_suffix(self):
        rocm_version = "7.0.0rc20250707"
        suffix = determine_version.derive_version_suffix(rocm_version)
        self.assertEqual(suffix, "+rocm7.0.0rc20250707")

    def test_bkc_version_suffix(self):
        rocm_version = "10.1.0a20260811+bkc.20260813"
        suffix = determine_version.derive_version_suffix(rocm_version)
        self.assertEqual(suffix, "+rocm10.1.0a20260811-bkc.20260813")

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

    def test_rocm_version_round_trip_from_torch_local_version(self):
        cases = [
            "7.0.0",
            "7.0.0rc20250707",
            "10.3.0a20261027",
            "10.1.0a20260811+bkc.20260813",
            "7.0.0.dev0+515115ea2cb85a0b71b5507ce56a627d14c7ae73",
        ]
        for rocm_version in cases:
            suffix = determine_version.derive_version_suffix(rocm_version)
            expected = str(Version(rocm_version))
            with self.subTest(rocm_version=rocm_version, form="plain"):
                self.assertEqual(
                    determine_version.rocm_version_from_torch_version(
                        f"2.15.0{suffix}"
                    ),
                    expected,
                )
            with self.subTest(rocm_version=rocm_version, form="git"):
                self.assertEqual(
                    determine_version.rocm_version_from_torch_version(
                        f"2.15.0+git91321091.{suffix.lstrip('+')}"
                    ),
                    expected,
                )

    def test_rocm_version_from_failing_full_suite_torch_wheel(self):
        self.assertEqual(
            determine_version.rocm_version_from_torch_version(
                "2.15.0+git91321091.rocm10.3.0a20261027"
            ),
            "10.3.0a20261027",
        )

    def test_rocm_version_from_torch_version_requires_suffix(self):
        with self.assertRaises(ValueError):
            determine_version.rocm_version_from_torch_version("2.15.0+cpu")

    def test_torch_version_cli_prints_rocm_version_without_env(self):
        with mock.patch.object(determine_version, "gha_set_env") as gha_set_env:
            with mock.patch("sys.stdout") as stdout:
                determine_version.main(
                    ["--torch-version", "2.15.0+git91321091.rocm10.3.0a20261027"]
                )
        gha_set_env.assert_not_called()
        stdout.write.assert_any_call("10.3.0a20261027")

    def test_write_env_file_sets_build_arguments_and_suffix(self):
        rocm_version = "7.0.0"
        with mock.patch.object(determine_version, "gha_set_env") as gha_set_env:
            determine_version.main(["--rocm-version", rocm_version, "--write-env-file"])

        gha_set_env.assert_called_once_with(
            {
                "optional_build_prod_arguments": (
                    "--rocm-sdk-version ==7.0.0 --version-suffix +rocm7.0.0"
                ),
                "version_suffix": "+rocm7.0.0",
            }
        )


if __name__ == "__main__":
    unittest.main()
