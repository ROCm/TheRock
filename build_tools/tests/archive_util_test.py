#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from _therock_utils import archive_util
from _therock_utils.archive_util import (
    add_tree,
    get_archive_timestamp,
    normalize_tarinfo,
    open_archive_for_read,
    open_archive_for_write,
)
from _therock_utils.source_date import ENV_VAR, STANDARD_ENV_VAR

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


class ArchiveTimestampTest(unittest.TestCase):
    """Verify how the archive mtime is resolved."""

    def setUp(self):
        get_archive_timestamp.cache_clear()
        self.addCleanup(get_archive_timestamp.cache_clear)

    @staticmethod
    def _env(**overrides):
        """Environment with both timestamp vars cleared, then `overrides` applied."""
        env = {
            k: v for k, v in os.environ.items() if k not in (STANDARD_ENV_VAR, ENV_VAR)
        }
        env.update(overrides)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_therock_var_is_used(self):
        with self._env(**{ENV_VAR: "1700000000"}):
            self.assertEqual(get_archive_timestamp(), 1700000000)

    def test_standard_var_outranks_the_therock_var(self):
        # Nothing sets SOURCE_DATE_EPOCH unless explicitly asked to, so its
        # presence is a deliberate decision and beats the computed default.
        with self._env(**{STANDARD_ENV_VAR: "1600000000", ENV_VAR: "1700000000"}):
            self.assertEqual(get_archive_timestamp(), 1600000000)

    def test_non_integer_value_is_rejected(self):
        with self._env(**{ENV_VAR: "yesterday"}):
            with self.assertRaisesRegex(ValueError, ENV_VAR):
                get_archive_timestamp()

    def test_negative_value_is_rejected(self):
        with self._env(**{STANDARD_ENV_VAR: "-1"}):
            with self.assertRaisesRegex(ValueError, "negative"):
                get_archive_timestamp()

    def test_falls_back_to_current_time_when_unset(self):
        # A worker run directly, with no orchestrator to resolve the source
        # timestamp. Must never be the epoch.
        before = int(time.time())
        with self._env():
            timestamp = get_archive_timestamp()
        self.assertGreaterEqual(timestamp, before)

    def test_does_no_git_work(self):
        # Resolving the source timestamp costs several git calls and this runs
        # once per archive process, so it belongs in source_date, called once by
        # an orchestrator.
        self.assertNotIn("subprocess", vars(archive_util))


class NormalizeTarinfoTest(unittest.TestCase):
    """Verify the metadata normalization applied to reproducible archives."""

    def setUp(self):
        get_archive_timestamp.cache_clear()
        self.addCleanup(get_archive_timestamp.cache_clear)

    def test_build_specific_metadata_is_cleared(self):
        tarinfo = tarfile.TarInfo("lib/libfoo.so.1")
        tarinfo.mtime = 1600000000
        tarinfo.uid = 1000
        tarinfo.gid = 1000
        tarinfo.uname = "builder"
        tarinfo.gname = "builder"

        with mock.patch.dict(os.environ, {STANDARD_ENV_VAR: "1700000000"}):
            normalize_tarinfo(tarinfo)

        self.assertEqual(tarinfo.mtime, 1700000000)
        self.assertEqual(tarinfo.uid, 0)
        self.assertEqual(tarinfo.gid, 0)
        self.assertEqual(tarinfo.uname, "root")
        self.assertEqual(tarinfo.gname, "root")

    def test_mtime_is_not_the_epoch(self):
        # Regression guard: an epoch mtime survives extraction and makes fresh
        # SDK inputs look older than a downstream project's existing objects,
        # suppressing rebuilds.
        tarinfo = tarfile.TarInfo("include/foo.h")
        normalize_tarinfo(tarinfo)
        self.assertGreater(tarinfo.mtime, 0)

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


class AddTreeTest(unittest.TestCase):
    """Verify reproducible tree archiving (used for the wheel devel tarball)."""

    def _write_tree(self, root: Path):
        # Deliberately created out of order so that a passing sort assertion
        # cannot be explained by creation order.
        (root / "pkg" / "sub").mkdir(parents=True)
        (root / "pkg" / "z.txt").write_text("z")
        (root / "pkg" / "a.txt").write_text("a")
        (root / "pkg" / "sub" / "m.txt").write_text("m")

    def _archive_tree(self, tmp: Path, archive: Path) -> list[str]:
        with open_archive_for_write(archive, "xz") as tf:
            add_tree(tf, tmp / "pkg", relative_to=tmp)
        with open_archive_for_read(archive) as tf:
            return tf.getnames()

    def test_member_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_tree(tmp)
            names = self._archive_tree(tmp, tmp / "out.tar.xz")

        # Sorted within each directory, directories visited in sorted order.
        # The exact sequence matters less than it being pinned: creation order
        # and filesystem iteration order must not leak into the archive.
        self.assertEqual(names, ["pkg/a.txt", "pkg/sub", "pkg/z.txt", "pkg/sub/m.txt"])

    def test_same_tree_archives_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_tree(tmp)

            first = tmp / "first.tar.xz"
            with open_archive_for_write(first, "xz") as tf:
                add_tree(tf, tmp / "pkg", relative_to=tmp)

            # Simulate the same content produced by a later build.
            for p in (tmp / "pkg").rglob("*"):
                os.utime(p, (1700000000, 1700000000))

            second = tmp / "second.tar.xz"
            with open_archive_for_write(second, "xz") as tf:
                add_tree(tf, tmp / "pkg", relative_to=tmp)

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_member_metadata_is_normalized(self):
        get_archive_timestamp.cache_clear()
        self.addCleanup(get_archive_timestamp.cache_clear)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_tree(tmp)
            archive = tmp / "out.tar.xz"
            with mock.patch.dict(os.environ, {STANDARD_ENV_VAR: "1700000000"}):
                with open_archive_for_write(archive, "xz") as tf:
                    add_tree(tf, tmp / "pkg", relative_to=tmp)
            with open_archive_for_read(archive) as tf:
                members = tf.getmembers()

        self.assertTrue(members)
        for member in members:
            self.assertEqual(member.mtime, 1700000000, member.name)
            self.assertEqual(member.uid, 0, member.name)
            self.assertEqual(member.gid, 0, member.name)

    def test_reports_added_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_tree(tmp)
            added = []
            with open_archive_for_write(tmp / "out.tar.xz", "xz") as tf:
                add_tree(tf, tmp / "pkg", relative_to=tmp, on_add=added.append)

        self.assertIn("pkg/a.txt", [name.replace(os.sep, "/") for name in added])


class ErrorHandlingTest(unittest.TestCase):
    def test_read_unknown_extension(self):
        with self.assertRaises(ValueError):
            open_archive_for_read(Path("test.tar.gz"))

    def test_write_unknown_compression(self):
        with self.assertRaises(ValueError):
            open_archive_for_write(Path("test.tar.gz"), "gzip")


if __name__ == "__main__":
    unittest.main()
