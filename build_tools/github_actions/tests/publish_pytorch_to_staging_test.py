#!/usr/bin/env python
"""Unit tests for publish_pytorch_to_staging.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_pytorch_to_staging import main


class TestPublishPytorchToStaging(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    def setUp(self):
        # Real directory so the script's existence check passes; the
        # upload itself is mocked so no S3 contact happens.
        self._tmp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_dev_uploads_to_v4_whl_staging_in_dev_python(self, mock_upload):
        mock_upload.return_value = 3
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        self.assertEqual(mock_upload.call_count, 1)
        call_args = mock_upload.call_args
        source, dest = call_args.args
        self.assertEqual(source, self.source_dir)
        self.assertEqual(dest.bucket, "therock-dev-python")
        self.assertEqual(dest.relative_path, "v4/whl-staging")
        self.assertEqual(call_args.kwargs.get("include"), ["*.whl"])

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_nightly_selects_nightly_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-nightly-python")
        self.assertEqual(dest.relative_path, "v4/whl-staging")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_prerelease_selects_prerelease_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "prerelease",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-prerelease-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_raises_when_no_wheels_uploaded(self, mock_upload):
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

    def test_raises_when_source_dir_missing(self):
        missing = self.source_dir / "does-not-exist"
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(missing),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_invalid_release_type_rejected(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "release",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
