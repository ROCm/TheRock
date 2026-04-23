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
    def test_dev_linux_copies_from_artifacts_to_tarball_bucket(self, mock_copy):
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

        mock_copy.assert_called_once()
        source, dest = mock_copy.call_args.args[0], mock_copy.call_args.args[1]
        self.assertEqual(source.bucket, "therock-dev-artifacts")
        self.assertEqual(source.relative_path, "123-linux/tarballs")
        self.assertEqual(dest.bucket, "therock-dev-tarball")
        self.assertEqual(dest.relative_path, "v4/tarball")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_nightly_windows_copies_from_artifacts_to_tarball_bucket(self, mock_copy):
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

        source, dest = mock_copy.call_args.args[0], mock_copy.call_args.args[1]
        self.assertEqual(source.bucket, "therock-nightly-artifacts")
        self.assertEqual(source.relative_path, "99-windows/tarballs")
        self.assertEqual(dest.bucket, "therock-nightly-tarball")
        self.assertEqual(dest.relative_path, "v4/tarball")

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
