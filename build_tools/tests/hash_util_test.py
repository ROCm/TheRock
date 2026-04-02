# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for hash_util module."""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.hash_util import calculate_hash, write_hash


class CalculateHashTest(unittest.TestCase):
    """Tests for calculate_hash function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_file.bin"

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()
        os.rmdir(self.temp_dir)

    def test_sha256_known_content(self):
        """Test SHA256 hash of known content matches expected value."""
        content = b"hello world"
        self.test_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        digest = calculate_hash(self.test_file, "sha256")
        self.assertEqual(digest.hexdigest(), expected)

    def test_md5_known_content(self):
        """Test MD5 hash of known content matches expected value."""
        content = b"test data for md5"
        self.test_file.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        digest = calculate_hash(self.test_file, "md5")
        self.assertEqual(digest.hexdigest(), expected)

    def test_sha512_known_content(self):
        """Test SHA512 hash of known content."""
        content = b"sha512 test content"
        self.test_file.write_bytes(content)
        expected = hashlib.sha512(content).hexdigest()
        digest = calculate_hash(self.test_file, "sha512")
        self.assertEqual(digest.hexdigest(), expected)

    def test_empty_file(self):
        """Test hash of an empty file."""
        self.test_file.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        digest = calculate_hash(self.test_file, "sha256")
        self.assertEqual(digest.hexdigest(), expected)

    def test_large_file(self):
        """Test hash of a file larger than the internal buffer (64KB)."""
        # Create a file larger than 2**16 (65536) bytes
        content = b"A" * 100_000
        self.test_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        digest = calculate_hash(self.test_file, "sha256")
        self.assertEqual(digest.hexdigest(), expected)

    def test_binary_content(self):
        """Test hash of binary content with all byte values."""
        content = bytes(range(256)) * 10
        self.test_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        digest = calculate_hash(self.test_file, "sha256")
        self.assertEqual(digest.hexdigest(), expected)

    def test_returns_hash_object(self):
        """Test that calculate_hash returns a hash object with standard methods."""
        self.test_file.write_bytes(b"test")
        digest = calculate_hash(self.test_file, "sha256")
        # Should have hexdigest method
        self.assertIsInstance(digest.hexdigest(), str)
        # Should have digest method
        self.assertIsInstance(digest.digest(), bytes)


class WriteHashTest(unittest.TestCase):
    """Tests for write_hash function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.hash_file = Path(self.temp_dir) / "test.sha256"
        self.source_file = Path(self.temp_dir) / "source.bin"

    def tearDown(self):
        for f in [self.hash_file, self.source_file]:
            if f.exists():
                f.unlink()
        os.rmdir(self.temp_dir)

    def test_writes_hex_digest_with_newline(self):
        """Test that write_hash writes the hex digest followed by a newline."""
        self.source_file.write_bytes(b"test content")
        digest = calculate_hash(self.source_file, "sha256")
        write_hash(self.hash_file, digest)
        content = self.hash_file.read_text()
        self.assertEqual(content, digest.hexdigest() + "\n")

    def test_writes_correct_sha256(self):
        """Test that written hash matches expected SHA256."""
        data = b"known data"
        self.source_file.write_bytes(data)
        expected_hex = hashlib.sha256(data).hexdigest()
        digest = calculate_hash(self.source_file, "sha256")
        write_hash(self.hash_file, digest)
        content = self.hash_file.read_text().strip()
        self.assertEqual(content, expected_hex)

    def test_end_to_end_hash_and_write(self):
        """Test full workflow: create file, hash it, write hash, verify."""
        data = b"end to end test data"
        self.source_file.write_bytes(data)
        digest = calculate_hash(self.source_file, "sha256")
        write_hash(self.hash_file, digest)

        # Verify the hash file contents
        written_hash = self.hash_file.read_text().strip()
        expected_hash = hashlib.sha256(data).hexdigest()
        self.assertEqual(written_hash, expected_hash)
        self.assertEqual(len(written_hash), 64)  # SHA256 produces 64 hex chars


if __name__ == "__main__":
    unittest.main()
