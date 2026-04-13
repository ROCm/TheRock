#!/usr/bin/env python
"""Unit tests for build_tarballs.py."""

import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from build_tarballs import compress_tarball, is_kpack_split


class TestIsKpackSplit(unittest.TestCase):
    def _write_manifest(self, tmpdir: Path, flags: dict):
        manifest_dir = tmpdir / "share" / "therock"
        manifest_dir.mkdir(parents=True)
        manifest = {"flags": flags}
        (manifest_dir / "therock_manifest.json").write_text(json.dumps(manifest))

    def test_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._write_manifest(tmpdir, {"KPACK_SPLIT_ARTIFACTS": True})
            self.assertTrue(is_kpack_split(tmpdir))

    def test_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._write_manifest(tmpdir, {"KPACK_SPLIT_ARTIFACTS": False})
            self.assertFalse(is_kpack_split(tmpdir))

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_kpack_split(Path(tmpdir)))


class TestCompressTarball(unittest.TestCase):
    def test_creates_tarball(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src = tmpdir / "src"
            src.mkdir()
            (src / "bin").mkdir()
            (src / "bin" / "hello").write_text("hello world")
            (src / "lib").mkdir()
            (src / "lib" / "libfoo.so").write_bytes(b"\x00" * 1024)

            tarball_path = tmpdir / "output" / "test.tar.gz"
            compress_tarball(source_dir=src, tarball_path=tarball_path)

            self.assertTrue(tarball_path.exists())
            self.assertGreater(tarball_path.stat().st_size, 0)

            with tarfile.open(tarball_path, "r:gz") as tf:
                names = tf.getnames()
                self.assertIn("./bin/hello", names)
                self.assertIn("./lib/libfoo.so", names)


if __name__ == "__main__":
    unittest.main()
