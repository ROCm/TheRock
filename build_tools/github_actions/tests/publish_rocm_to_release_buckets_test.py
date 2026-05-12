#!/usr/bin/env python
"""Unit tests for publish_rocm_to_release_buckets.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_rocm_to_release_buckets import main


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
    def test_kpack_split_uses_v4_staging_then_release(self, mock_copy):
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

        # Calls: tarballs, python -> v4/whl-staging, python -> v4/whl
        self.assertEqual(mock_copy.call_count, 3)
        _, python_dest_staging = mock_copy.call_args_list[1].args
        self.assertEqual(python_dest_staging.relative_path, "v4/whl-staging")
        _, python_dest_release = mock_copy.call_args_list[2].args
        self.assertEqual(python_dest_release.relative_path, "v4/whl")

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


if __name__ == "__main__":
    unittest.main()
