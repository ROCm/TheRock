#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for publish_lmcache_to_release_bucket.py."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_lmcache_to_release_bucket import main


class TestPublishLmcacheToReleaseBucket(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_lmcache_to_release_bucket.gha_set_output")
    def test_dev_uploads_to_multi_arch_index(self, mock_set_output, mock_upload):
        mock_upload.return_value = 1

        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        source, destination = mock_upload.call_args.args
        self.assertEqual(source, self.source_dir)
        self.assertEqual(destination.bucket, "therock-dev-python")
        self.assertEqual(destination.relative_path, "v4/whl")
        self.assertEqual(mock_upload.call_args.kwargs["include"], ["lmcache-*.whl"])
        mock_set_output.assert_called_once_with(
            {"package_index_url": "https://rocm.devreleases.amd.com/whl-multi-arch/"}
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_nightly_selects_nightly_bucket(self, mock_upload):
        mock_upload.return_value = 1

        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        _source, destination = mock_upload.call_args.args
        self.assertEqual(destination.bucket, "therock-nightly-python")
        self.assertEqual(destination.relative_path, "v4/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_fails_when_no_lmcache_wheel_is_uploaded(self, mock_upload):
        mock_upload.return_value = 0

        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_fails_when_source_directory_is_missing(self):
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir / "missing"),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
