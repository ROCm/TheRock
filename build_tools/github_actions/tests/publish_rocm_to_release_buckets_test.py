#!/usr/bin/env python
"""Unit tests for publish_rocm_to_release_buckets.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_rocm_to_release_buckets import main
from _therock_utils.storage_location import StorageLocation


class TestPublishRocmToReleaseBuckets(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_dev_linux_copies_tarballs_and_python(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        # Calls: tarballs, python -> v3/whl-staging, python -> v3/whl
        self.assertEqual(mock_copy.call_count, 3)
        # First call: tarballs
        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.bucket, "therock-dev-artifacts")
        self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
        self.assertEqual(tarball_dest.bucket, "therock-dev-tarball")
        self.assertEqual(tarball_dest.relative_path, "v4/tarball")
        # Python staging then release
        python_source, python_dest_staging = mock_copy.call_args_list[1].args
        self.assertEqual(python_source.bucket, "therock-dev-artifacts")
        self.assertEqual(python_source.relative_path, "123-linux/python")
        self.assertEqual(python_dest_staging.bucket, "therock-dev-python")
        self.assertEqual(python_dest_staging.relative_path, "v3/whl-staging")
        _, python_dest_release = mock_copy.call_args_list[2].args
        self.assertEqual(python_dest_release.relative_path, "v3/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_nightly_windows_copies_to_correct_buckets(self, mock_copy):
        mock_copy.return_value = 1
        main(
            [
                "--run-id",
                "99",
                "--platform",
                "windows",
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.bucket, "therock-nightly-artifacts")
        self.assertEqual(tarball_source.relative_path, "99-windows/tarballs")
        self.assertEqual(tarball_dest.bucket, "therock-nightly-tarball")

        python_source, python_dest = mock_copy.call_args_list[1].args
        self.assertEqual(python_source.bucket, "therock-nightly-artifacts")
        self.assertEqual(python_source.relative_path, "99-windows/python")
        self.assertEqual(python_dest.bucket, "therock-nightly-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_kpack_split_uses_v4_whl_directly(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--kpack-split",
                "true",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        # Calls: tarballs, python -> v4/whl (no staging for multi-arch)
        self.assertEqual(mock_copy.call_count, 2)
        _, python_dest = mock_copy.call_args_list[1].args
        self.assertEqual(python_dest.relative_path, "v4/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_dev_linux_copies_native_packages(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        # Calls: tarballs, python -> v3/whl-staging, python -> v3/whl, deb, rpm
        self.assertEqual(mock_copy.call_count, 5)
        # deb packages
        deb_source, deb_dest = mock_copy.call_args_list[3].args
        self.assertEqual(deb_source.bucket, "therock-dev-artifacts")
        self.assertEqual(deb_source.relative_path, "123-linux/packages/deb")
        self.assertEqual(deb_dest.bucket, "therock-dev-packages")
        self.assertRegex(deb_dest.relative_path, r"^v4/deb/\d{8}-123$")
        # rpm packages
        rpm_source, rpm_dest = mock_copy.call_args_list[4].args
        self.assertEqual(rpm_source.bucket, "therock-dev-artifacts")
        self.assertEqual(rpm_source.relative_path, "123-linux/packages/rpm")
        self.assertEqual(rpm_dest.bucket, "therock-dev-packages")
        self.assertRegex(rpm_dest.relative_path, r"^v4/rpm/\d{8}-123$")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_windows_skips_native_packages(self, mock_copy):
        mock_copy.return_value = 1
        main(
            [
                "--run-id",
                "99",
                "--platform",
                "windows",
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )
        # Only tarballs + python x2 (3 calls) — native packages skipped for windows
        self.assertEqual(mock_copy.call_count, 3)

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_raises_when_no_tarballs_found(self, mock_copy):
        mock_copy.return_value = 0
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--run-id",
                    "123",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_asan_skips_python_packages(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--build-variant",
                "asan",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        # Only tarballs should be copied (python packages skipped for ASAN)
        self.assertEqual(mock_copy.call_count, 1)
        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
        # ASAN tarballs go to separate folder
        self.assertEqual(tarball_dest.relative_path, "v4/tarball-asan")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_structured_places_rocm_packages_in_package_dirs(
        self, mock_copy_dir, mock_list, mock_copy_file
    ):
        # Structured multi-arch: python packages go into per-package dirs via
        # list_files + copy_file, not the flat copy_directory.
        mock_copy_dir.return_value = 2  # tarballs still use copy_directory
        mock_list.return_value = [
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
            ),
            StorageLocation(
                "therock-dev-artifacts", "123-linux/python/rocm-7.13.0.tar.gz"
            ),
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
            ),
            # Non-accepted artifact in the listing must be ignored.
            StorageLocation("therock-dev-artifacts", "123-linux/python/index.html"),
        ]
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--kpack-split",
                "true",
                "--structured",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        dest_by_src = {
            call.args[0].relative_path: call.args[1].relative_path
            for call in mock_copy_file.call_args_list
        }
        self.assertNotIn("123-linux/python/index.html", dest_by_src)
        self.assertEqual(
            dest_by_src[
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl"
            ],
            "v5/core/whl/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
        )
        self.assertEqual(
            dest_by_src["123-linux/python/rocm-7.13.0.tar.gz"],
            "v5/core/whl/rocm/rocm-7.13.0.tar.gz",
        )
        self.assertEqual(
            dest_by_src[
                "123-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl"
            ],
            "v5/core/whl/rocm-sdk-device-gfx1100/"
            "rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
        )
        # Destination bucket is the release python bucket.
        for call in mock_copy_file.call_args_list:
            self.assertEqual(call.args[1].bucket, "therock-dev-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    def test_structured_whl_next(self, mock_list, mock_copy_file, mock_copy_dir):
        mock_copy_dir.return_value = 2  # tarballs
        mock_list.return_value = [
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
            ),
        ]
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--kpack-split",
                "true",
                "--structured",
                "--python-index",
                "whl-next",
                "--skip-native-packages",
                "--dry-run",
            ]
        )
        _, dest = mock_copy_file.call_args_list[0].args
        self.assertEqual(
            dest.relative_path,
            "v5/core/whl-next/rocm-sdk-core/"
            "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    def test_structured_raises_when_no_python_packages(
        self, mock_list, mock_copy_file, mock_copy_dir
    ):
        mock_copy_dir.return_value = 2  # tarballs succeed
        mock_list.return_value = []
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--run-id",
                    "123",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                    "--kpack-split",
                    "true",
                    "--structured",
                    "--skip-native-packages",
                    "--dry-run",
                ]
            )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_structured_requires_kpack_split(self, mock_copy):
        # Structured only applies to the multi-arch (kpack-split) path; using it
        # with the legacy per-family layout is rejected.
        mock_copy.return_value = 2
        with self.assertRaises(SystemExit):
            main(
                [
                    "--run-id",
                    "123",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                    "--structured",
                    "--skip-native-packages",
                    "--dry-run",
                ]
            )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_asan_native_packages_use_separate_path(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--build-variant",
                "asan",
                "--dry-run",
            ]
        )

        # Calls: tarballs, deb, rpm (no python for ASAN)
        self.assertEqual(mock_copy.call_count, 3)
        # deb packages go to packages-asan path
        deb_source, deb_dest = mock_copy.call_args_list[1].args
        self.assertEqual(deb_source.relative_path, "123-linux/packages/deb")
        self.assertRegex(deb_dest.relative_path, r"^v4/packages-asan/deb/\d{8}-123$")
        # rpm packages go to packages-asan path
        rpm_source, rpm_dest = mock_copy.call_args_list[2].args
        self.assertEqual(rpm_source.relative_path, "123-linux/packages/rpm")
        self.assertRegex(rpm_dest.relative_path, r"^v4/packages-asan/rpm/\d{8}-123$")


if __name__ == "__main__":
    unittest.main()
