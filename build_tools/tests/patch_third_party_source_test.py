#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import patch_third_party_source

SOURCE = b"one\ntwo\nthree\n"
PATCHED = b"one\nTWO\nthree\n"
PATCH = b"--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"


def crlf(data):
    return data.replace(b"\n", b"\r\n")


@unittest.skipUnless(shutil.which("patch"), "needs a patch program on PATH")
class ApplyPatchTest(unittest.TestCase):
    """Runs apply_patch() and the patch program over every combination of LF
    and CRLF line endings in the source file and the patch file.  The patch
    file arrives as CRLF from a checkout made with core.autocrlf=true."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _apply(self, source, patch, system):
        """Apply PATCH to a source tree holding SOURCE as f.txt, with
        platform.system() returning SYSTEM.  Return the exit status, or None
        if apply_patch() returned, and f.txt's contents afterwards."""
        work = Path(tempfile.mkdtemp(dir=self.tmp))
        source_dir = work / "source"
        source_dir.mkdir()
        (source_dir / "f.txt").write_bytes(source)
        patch_file = work / "0001-change.patch"
        patch_file.write_bytes(patch)
        status = None
        with mock.patch.object(
            patch_third_party_source.platform, "system", return_value=system
        ):
            try:
                patch_third_party_source.apply_patch(patch_file, source_dir)
            except SystemExit as e:
                status = e.code
        return status, (source_dir / "f.txt").read_bytes()

    def _check_all(self, system):
        for patch in (PATCH, crlf(PATCH)):
            name = "CRLF" if b"\r" in patch else "LF"
            with self.subTest(patch_file=name, source="LF"):
                self.assertEqual(self._apply(SOURCE, patch, system), (None, PATCHED))
            # A CRLF source file cannot be patched.  The CRs are stripped from
            # the patch file's lines, by apply_patch() on Windows and by the
            # patch program itself elsewhere, so those lines no longer equal
            # the source file's, and the patch program rejects the hunk.
            with self.subTest(patch_file=name, source="CRLF"):
                status, result = self._apply(crlf(SOURCE), patch, system)
                self.assertNotIn(status, (None, 0))
                self.assertEqual(result, crlf(SOURCE))

    # On Windows, apply_patch() runs the patch program with --binary, which
    # works on Linux too, so this test runs on every host.
    def test_windows(self):
        self._check_all("Windows")

    # Elsewhere, apply_patch() runs the patch program without --binary, which
    # crashes some patch programs on Windows, so this test is skipped there.
    @unittest.skipIf(
        platform.system() == "Windows",
        "no --binary, which some Windows patch programs crash on",
    )
    def test_linux(self):
        self._check_all("Linux")


if __name__ == "__main__":
    unittest.main()
