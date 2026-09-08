# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from debian.debian_support import NativeVersion
from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_rocm_package_version

TEST_BKC_VERSION_DATA = {"release-metadata": {"base-date": "20260811"}}


# Note: the regex matches in here aren't exact, but they should be "good enough"
# to cover the general structure of each version string while allowing for
# future changes like using X.Y versions instead of X.Y.Z versions.


class VersionFileTest(unittest.TestCase):
    def test_loads_repository_version_file(self):
        # The version file in this repository should always parse correctly.
        version_data = compute_rocm_package_version.load_version_file()

        self.assertIsInstance(version_data["rocm-version"], str)
        self.assertIsInstance(version_data["release-metadata"], dict)

    def test_loads_version_file_without_metadata(self):
        # Version files prior to introducing the metadata field should still
        # parse and load without errors.
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "version.json"
            version_file.write_text(
                """{
  "rocm-version": "7.9.0"
}
""",
                encoding="utf-8",
            )

            self.assertEqual(
                compute_rocm_package_version.load_version_file(version_file),
                {"rocm-version": "7.9.0"},
            )

    def test_loads_version_file_with_empty_metadata(self):
        # Metadata is expected to be empty on the default branch.
        # It may be populated on release branches.
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "version.json"
            version_file.write_text(
                """{
  "rocm-version": "7.9.0",
  "release-metadata": {
    "base-date": ""
  }
}
""",
                encoding="utf-8",
            )

            self.assertEqual(
                compute_rocm_package_version.load_version_file(version_file),
                {
                    "rocm-version": "7.9.0",
                    "release-metadata": {"base-date": ""},
                },
            )

    def test_loads_version_file_with_populated_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "version.json"
            version_file.write_text(
                """{
  "rocm-version": "7.9.0",
  "release-metadata": {
    "base-date": "20260811"
  }
}
""",
                encoding="utf-8",
            )

            self.assertEqual(
                compute_rocm_package_version.load_version_file(version_file),
                {
                    "rocm-version": "7.9.0",
                    "release-metadata": {"base-date": "20260811"},
                },
            )

    def test_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "version.json"
            # Missing closing }
            version_file.write_text(
                '{ "rocm-version": "7.9.0"',
                encoding="utf-8",
            )

            with self.assertRaises(json.JSONDecodeError):
                compute_rocm_package_version.load_version_file(version_file)

    def test_rejects_invalid_base_date_metadata(self):
        for base_date in ("2026081", "20261301"):
            with (
                self.subTest(base_date=base_date),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                version_file = Path(temp_dir) / "version.json"
                version_file.write_text(
                    f"""{{
  "rocm-version": "7.9.0",
  "release-metadata": {{
    "base-date": "{base_date}"
  }}
}}
""",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    ValueError, "valid date in YYYYMMDD format"
                ):
                    compute_rocm_package_version.load_version_file(version_file)


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

    def test_dev_bkc_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            release_type="dev-bkc",
            override_base_version="7.9.0",
            override_git_sha="abcdef1234567890abcdef1234567890abcdef12",
            version_data=TEST_BKC_VERSION_DATA,
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

    def test_nightly_bkc_version(self):
        version = compute_rocm_package_version.compute_version(
            release_type="nightly-bkc",
            override_base_version="7.9.0",
            version_data=TEST_BKC_VERSION_DATA,
        )
        self.assertRegex(
            version,
            r"^7\.9\.0a20260811\+bkc\.[0-9]{8}$",
        )

    def test_bkc_version_requires_base_date_metadata(self):
        for release_type in ("dev-bkc", "nightly-bkc"):
            with (
                self.subTest(release_type=release_type),
                self.assertRaisesRegex(ValueError, "release-metadata.base-date"),
            ):
                compute_rocm_package_version.compute_version(
                    release_type=release_type,
                    override_base_version="7.9.0",
                    version_data={},
                )

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
        # release > prerelease > nightly > nightly-bkc > dev-bkc == dev.
        versions = self._compute_versions_by_release_type()

        self.assertEqual(
            Version(versions["dev-bkc"]),
            Version(versions["dev"]),
        )
        self.assertGreater(
            Version(versions["nightly-bkc"]),
            Version(versions["dev-bkc"]),
        )
        self.assertGreater(
            Version(versions["nightly"]),
            Version(versions["nightly-bkc"]),
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
            "dev-bkc": compute_rocm_package_version.compute_version(
                release_type="dev-bkc",
                override_git_sha="abcdef1234567890abcdef1234567890abcdef12",
                version_data=TEST_BKC_VERSION_DATA,
                **common_args,
            ),
            "nightly-bkc": compute_rocm_package_version.compute_version(
                release_type="nightly-bkc",
                version_data=TEST_BKC_VERSION_DATA,
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

    def test_dev_bkc_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="dev-bkc",
            override_base_version="8.1.0",
            version_data=TEST_BKC_VERSION_DATA,
        )
        self.assertRegex(version, r"^8\.1\.0~dev[0-9]{8}$")

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

    def test_nightly_bkc_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="deb",
            release_type="nightly-bkc",
            override_base_version="8.1.0",
            version_data=TEST_BKC_VERSION_DATA,
        )
        self.assertRegex(
            version,
            r"^8\.1\.0~20260811\.bkc\.[0-9]{8}$",
        )

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

    def test_versions_are_valid(self):
        # NativeVersion() rejects invalid Debian versions such as "7.10.0/rc0".
        # See https://www.debian.org/doc/debian-policy/ch-controlfields.html#version.
        versions = self._compute_versions_by_release_type()

        for release_type, version in versions.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(str(NativeVersion(version)), version)

    def test_versions_sort_by_release_type(self):
        # apt upgrade selects the greatest available version.
        # To match other package types, we want:
        #     release > prerelease > nightly > dev.
        # Debian currently sorts:
        #     release > prerelease > dev > nightly.
        versions = self._compute_versions_by_release_type()

        # This test is disabled since current sorting is dev > nightly!
        # See https://github.com/ROCm/TheRock/issues/7638.
        # self.assertGreater(
        #     NativeVersion(versions["nightly"]),
        #     NativeVersion(versions["dev"]),
        # )
        self.assertGreater(
            NativeVersion(versions["prerelease"]),
            NativeVersion(versions["nightly"]),
        )
        self.assertGreater(
            NativeVersion(versions["release"]),
            NativeVersion(versions["prerelease"]),
        )

    @staticmethod
    def _compute_versions_by_release_type() -> dict[str, str]:
        common_args = {
            "package_type": "deb",
            "override_base_version": "7.10.0",
        }
        return {
            "dev": compute_rocm_package_version.compute_version(
                release_type="dev",
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

    def test_dev_bkc_version_uses_dev_version_shape(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="dev-bkc",
            override_base_version="8.1.0",
            override_git_sha="abcdef1234567890",
            version_data=TEST_BKC_VERSION_DATA,
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

    def test_nightly_bkc_version(self):
        version = compute_rocm_package_version.compute_version(
            package_type="rpm",
            release_type="nightly-bkc",
            override_base_version="8.1.0",
            version_data=TEST_BKC_VERSION_DATA,
        )
        self.assertRegex(
            version,
            r"^8\.1\.0~20260811\.bkc\.[0-9]{8}$",
        )

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


# Test meaningful combinations of argparse options through the real computation path,
# including main()'s side effect of writing versions to GitHub Actions outputs.
class MainFunctionTest(unittest.TestCase):
    def test_sets_dev_outputs_with_version_overrides(self):
        override_git_sha = "abcdef1234567890abcdef1234567890abcdef12"
        with (
            mock.patch.dict(os.environ, {"GITHUB_SHA": "f" * 40}),
            mock.patch.object(
                compute_rocm_package_version, "gha_set_output"
            ) as gha_set_output,
        ):
            compute_rocm_package_version.main(
                [
                    "--release-type",
                    "dev",
                    "--override-base-version",
                    "7.99.0",
                    "--override-git-sha",
                    override_git_sha,
                ]
            )

        gha_set_output.assert_called_once()
        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(
            set(outputs),
            {
                "rocm_package_version",
                "rocm_deb_package_version",
                "rocm_rpm_package_version",
            },
        )
        self.assertEqual(
            outputs["rocm_package_version"],
            f"7.99.0.dev0+{override_git_sha}",
        )
        self.assertTrue(outputs["rocm_deb_package_version"].startswith("7.99.0~dev"))
        self.assertTrue(outputs["rocm_rpm_package_version"].startswith("7.99.0~"))
        self.assertTrue(outputs["rocm_rpm_package_version"].endswith("gabcdef12"))

    def test_sets_prerelease_outputs(self):
        with mock.patch.object(
            compute_rocm_package_version, "gha_set_output"
        ) as gha_set_output:
            compute_rocm_package_version.main(
                [
                    "--release-type",
                    "prerelease",
                    "--prerelease-version",
                    "2",
                    "--override-base-version",
                    "7.99.0",
                ]
            )

        gha_set_output.assert_called_once_with(
            {
                "rocm_package_version": "7.99.0rc2",
                "rocm_deb_package_version": "7.99.0~pre2",
                "rocm_rpm_package_version": "7.99.0~rc2",
            }
        )

    def test_sets_custom_suffix_outputs(self):
        with mock.patch.object(
            compute_rocm_package_version, "gha_set_output"
        ) as gha_set_output:
            compute_rocm_package_version.main(
                [
                    "--custom-version-suffix",
                    ".custom1",
                    "--override-base-version",
                    "7.99.0",
                ]
            )

        gha_set_output.assert_called_once_with(
            {
                "rocm_package_version": "7.99.0.custom1",
                "rocm_deb_package_version": "7.99.0.custom1",
                "rocm_rpm_package_version": "7.99.0.custom1",
            }
        )


if __name__ == "__main__":
    unittest.main()
