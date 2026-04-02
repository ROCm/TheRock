# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for storage_location module."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.storage_location import StorageLocation


class StorageLocationTest(unittest.TestCase):
    """Tests for StorageLocation dataclass."""

    def test_s3_uri(self):
        """Test S3 URI construction."""
        loc = StorageLocation("my-bucket", "path/to/file.tar.xz")
        self.assertEqual(loc.s3_uri, "s3://my-bucket/path/to/file.tar.xz")

    def test_https_url(self):
        """Test HTTPS URL construction."""
        loc = StorageLocation("my-bucket", "path/to/file.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://my-bucket.s3.amazonaws.com/path/to/file.tar.xz",
        )

    def test_local_path(self):
        """Test local path construction."""
        loc = StorageLocation("my-bucket", "path/to/file.tar.xz")
        staging_dir = Path("/tmp/staging")
        result = loc.local_path(staging_dir)
        self.assertEqual(result, Path("/tmp/staging/path/to/file.tar.xz"))

    def test_relative_path_attribute(self):
        """Test that relative_path is stored correctly."""
        loc = StorageLocation("bucket", "some/relative/path")
        self.assertEqual(loc.relative_path, "some/relative/path")

    def test_bucket_attribute(self):
        """Test that bucket is stored correctly."""
        loc = StorageLocation("test-bucket", "file.txt")
        self.assertEqual(loc.bucket, "test-bucket")

    def test_frozen_dataclass(self):
        """Test that StorageLocation is immutable (frozen)."""
        loc = StorageLocation("bucket", "path")
        with self.assertRaises(AttributeError):
            loc.bucket = "new-bucket"  # type: ignore

    def test_equality(self):
        """Test equality comparison between StorageLocations."""
        loc1 = StorageLocation("bucket", "path/file.tar.xz")
        loc2 = StorageLocation("bucket", "path/file.tar.xz")
        loc3 = StorageLocation("other-bucket", "path/file.tar.xz")
        self.assertEqual(loc1, loc2)
        self.assertNotEqual(loc1, loc3)

    def test_s3_uri_with_nested_path(self):
        """Test S3 URI with deeply nested paths."""
        loc = StorageLocation("bucket", "a/b/c/d/e/file.tar.zst")
        self.assertEqual(loc.s3_uri, "s3://bucket/a/b/c/d/e/file.tar.zst")

    def test_https_url_with_simple_path(self):
        """Test HTTPS URL with a simple filename."""
        loc = StorageLocation("bucket", "file.txt")
        self.assertEqual(
            loc.https_url, "https://bucket.s3.amazonaws.com/file.txt"
        )

    def test_local_path_with_different_staging_dirs(self):
        """Test local_path with various staging directories."""
        loc = StorageLocation("bucket", "run-123/artifact.tar.xz")
        self.assertEqual(
            loc.local_path(Path("/tmp/a")),
            Path("/tmp/a/run-123/artifact.tar.xz"),
        )
        self.assertEqual(
            loc.local_path(Path("/opt/staging")),
            Path("/opt/staging/run-123/artifact.tar.xz"),
        )


if __name__ == "__main__":
    unittest.main()
