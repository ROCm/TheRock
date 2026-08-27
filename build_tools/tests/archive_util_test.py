#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import tarfile
import tempfile
import unittest
from pathlib import Path

from _therock_utils.archive_util import (
    normalize_tarinfo,
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


class NormalizeTarinfoTest(unittest.TestCase):
    """Verify the metadata normalization applied to reproducible archives."""

    def test_build_specific_metadata_is_cleared(self):
        tarinfo = tarfile.TarInfo("lib/libfoo.so.1")
        tarinfo.mtime = 1700000000
        tarinfo.uid = 1000
        tarinfo.gid = 1000
        tarinfo.uname = "builder"
        tarinfo.gname = "builder"

        normalize_tarinfo(tarinfo)

        self.assertEqual(tarinfo.mtime, 0)
        self.assertEqual(tarinfo.uid, 0)
        self.assertEqual(tarinfo.gid, 0)
        self.assertEqual(tarinfo.uname, "root")
        self.assertEqual(tarinfo.gname, "root")

    def test_permissions_and_identity_are_preserved(self):
        tarinfo = tarfile.TarInfo("bin/tool")
        tarinfo.mode = 0o755
        tarinfo.size = 1234
        tarinfo.type = tarfile.REGTYPE

        normalize_tarinfo(tarinfo)

        # Only the varying metadata is touched; anything describing the content
        # must survive.
        self.assertEqual(tarinfo.mode, 0o755)
        self.assertEqual(tarinfo.size, 1234)
        self.assertEqual(tarinfo.type, tarfile.REGTYPE)
        self.assertEqual(tarinfo.name, "bin/tool")

    def test_symlink_target_is_preserved(self):
        tarinfo = tarfile.TarInfo("share/doc/README")
        tarinfo.type = tarfile.SYMTYPE
        tarinfo.linkname = "README.txt"

        normalize_tarinfo(tarinfo)

        self.assertEqual(tarinfo.type, tarfile.SYMTYPE)
        self.assertEqual(tarinfo.linkname, "README.txt")


class ErrorHandlingTest(unittest.TestCase):
    def test_read_unknown_extension(self):
        with self.assertRaises(ValueError):
            open_archive_for_read(Path("test.tar.gz"))

    def test_write_unknown_compression(self):
        with self.assertRaises(ValueError):
            open_archive_for_write(Path("test.tar.gz"), "gzip")


if __name__ == "__main__":
    unittest.main()
