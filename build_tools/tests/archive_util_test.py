#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import tempfile
import unittest
from pathlib import Path

from _therock_utils.archive_util import (
    open_archive_for_read,
    open_archive_for_write,
)

IS_WINDOWS = platform.system() == "Windows"


class ArchiveRoundtripTest(unittest.TestCase):
    """Test writing and reading archives for each compression type."""

    def _roundtrip(self, suffix: str, compression_type: str):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Create a source file to archive.
            src = tmp / "hello.txt"
            src.write_text("hello world")

            # Write archive.
            archive = tmp / f"test.tar.{suffix}"
            with open_archive_for_write(archive, compression_type) as arc:
                arc.add(str(src), arcname="hello.txt")

            self.assertTrue(archive.exists())

            # Read archive and verify contents.
            with open_archive_for_read(archive) as arc:
                members = arc.getnames()
                self.assertIn("hello.txt", members)

    def test_roundtrip_zstd(self):
        self._roundtrip("zst", "zstd")

    def test_roundtrip_xz(self):
        self._roundtrip("xz", "xz")

    def test_roundtrip_zstd_custom_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "hello.txt"
            src.write_text("hello world")

            archive = tmp / "test.tar.zst"
            with open_archive_for_write(archive, "zstd", compression_level=1) as arc:
                arc.add(str(src), arcname="hello.txt")

            with open_archive_for_read(archive) as arc:
                self.assertIn("hello.txt", arc.getnames())


class HandleLeakTest(unittest.TestCase):
    """Verify that closing a ZstdTarFile releases the OS file handle."""

    @unittest.skipUnless(IS_WINDOWS, "Handle leak only blocks deletion on Windows")
    def test_zstd_read_close_releases_handle(self):
        """After closing a zstd archive opened for read, the file can be deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "hello.txt"
            src.write_text("hello world")

            archive = tmp / "test.tar.zst"
            with open_archive_for_write(archive, "zstd") as arc:
                arc.add(str(src), arcname="hello.txt")

            # Open for read, close, then delete.
            tf = open_archive_for_read(archive)
            tf.getnames()
            tf.close()

            # This would raise PermissionError if the handle leaked.
            os.unlink(archive)
            self.assertFalse(archive.exists())

    @unittest.skipUnless(IS_WINDOWS, "Handle leak only blocks deletion on Windows")
    def test_zstd_write_close_releases_handle(self):
        """After closing a zstd archive opened for write, the file can be deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "hello.txt"
            src.write_text("hello world")

            archive = tmp / "test.tar.zst"
            tf = open_archive_for_write(archive, "zstd")
            tf.add(str(src), arcname="hello.txt")
            tf.close()

            os.unlink(archive)
            self.assertFalse(archive.exists())


class ErrorHandlingTest(unittest.TestCase):
    def test_read_unknown_extension(self):
        with self.assertRaises(ValueError):
            open_archive_for_read(Path("test.tar.gz"))

    def test_write_unknown_compression(self):
        with self.assertRaises(ValueError):
            open_archive_for_write(Path("test.tar.gz"), "gzip")


if __name__ == "__main__":
    unittest.main()
