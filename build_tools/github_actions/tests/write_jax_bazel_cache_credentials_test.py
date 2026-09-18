#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for write_jax_bazel_cache_credentials.py"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from write_jax_bazel_cache_credentials import (
    CERT_NAME,
    KEY_NAME,
    main,
    materialize_credentials,
)


class MaterializeCredentialsTest(unittest.TestCase):
    """Tests that runner-local certs are copied privately, or skipped."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source_dir = Path(self.tmp.name) / "source"
        self.dest_dir = Path(self.tmp.name) / "dest"
        self.source_dir.mkdir()

    def _write_source(self) -> None:
        # Existence and size are what matter; real PEM headers trip scanners.
        (self.source_dir / CERT_NAME).write_text("test certificate")
        (self.source_dir / KEY_NAME).write_text("test key")

    def test_copies_both_files_when_present(self):
        self._write_source()
        copied = materialize_credentials(self.source_dir, self.dest_dir)
        self.assertEqual(copied, self.dest_dir)
        self.assertEqual((self.dest_dir / CERT_NAME).read_text(), "test certificate")
        self.assertEqual((self.dest_dir / KEY_NAME).read_text(), "test key")

    def test_returns_none_when_the_certificate_is_missing(self):
        (self.source_dir / KEY_NAME).write_text("test key")
        self.assertIsNone(materialize_credentials(self.source_dir, self.dest_dir))
        self.assertFalse(self.dest_dir.exists())

    def test_returns_none_when_the_key_is_missing(self):
        (self.source_dir / CERT_NAME).write_text("test certificate")
        self.assertIsNone(materialize_credentials(self.source_dir, self.dest_dir))

    def test_returns_none_when_the_source_dir_is_empty(self):
        self.assertIsNone(materialize_credentials(self.source_dir, self.dest_dir))

    def test_returns_none_when_a_source_file_is_empty(self):
        (self.source_dir / CERT_NAME).write_text("test certificate")
        (self.source_dir / KEY_NAME).write_text("")
        self.assertIsNone(materialize_credentials(self.source_dir, self.dest_dir))

    def test_replaces_a_stale_destination(self):
        self._write_source()
        self.dest_dir.mkdir()
        (self.dest_dir / "stale").write_text("leftover")
        materialize_credentials(self.source_dir, self.dest_dir)
        self.assertFalse((self.dest_dir / "stale").exists())
        self.assertTrue((self.dest_dir / CERT_NAME).is_file())

    def test_restricts_destination_permissions(self):
        self._write_source()
        materialize_credentials(self.source_dir, self.dest_dir)
        dir_mode = stat.S_IMODE(self.dest_dir.stat().st_mode)
        cert_mode = stat.S_IMODE((self.dest_dir / CERT_NAME).stat().st_mode)
        key_mode = stat.S_IMODE((self.dest_dir / KEY_NAME).stat().st_mode)
        self.assertEqual(dir_mode, 0o700)
        self.assertEqual(cert_mode, 0o600)
        self.assertEqual(key_mode, 0o600)


class MainTest(unittest.TestCase):
    """Tests that GITHUB_OUTPUT receives the destination path, or empty."""

    def _run(self, source_dir: Path, dest_dir: Path) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            env = {"GITHUB_OUTPUT": str(output), "RUNNER_TEMP": tmp}
            with mock.patch.dict(os.environ, env, clear=False):
                main(["--source-dir", str(source_dir), "--dest-dir", str(dest_dir)])
            return output.read_text()

    def test_main_writes_the_destination_when_credentials_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            dest_dir = Path(tmp) / "dest"
            source_dir.mkdir()
            (source_dir / CERT_NAME).write_text("test certificate")
            (source_dir / KEY_NAME).write_text("test key")
            output = self._run(source_dir, dest_dir)
        self.assertIn(f"credentials_dir={dest_dir}", output)

    def test_main_writes_empty_when_credentials_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            dest_dir = Path(tmp) / "dest"
            source_dir.mkdir()
            output = self._run(source_dir, dest_dir)
        self.assertRegex(output, r"credentials_dir=\s*$")


if __name__ == "__main__":
    unittest.main()
