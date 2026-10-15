#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the therock_install_windows_dlls.cmake script.  It only copies
files, so these tests run on every host."""

import subprocess
import tempfile
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = THEROCK_ROOT / "cmake" / "therock_install_windows_dlls.cmake"


class InstallWindowsDllsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        # A sub-project installed under lib/rocm_sysdeps of the stage.
        self.stage = tmp / "stage"
        self.prefix = self.stage / "lib" / "rocm_sysdeps"

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, *names):
        for name in names:
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)

    def _run(self, *defines):
        return subprocess.run(
            ["cmake", *defines, "-P", str(SCRIPT_PATH)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _run_script(self):
        result = self._run(f"-DPREFIX={self.prefix}", f"-DSTAGE_DIR={self.stage}")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_copies_dlls(self):
        """The DLLs in the sub-project's bin/ are copied to the stage's bin/,
        where the Windows loader finds them for the executables there.

        Nothing else is copied: not an executable beside the DLLs, nor a DLL
        outside bin/, nor a static library.  The copies have the originals'
        contents, and the originals stay, as the sub-project's own tools load
        them from there, and its exported CMake targets point at them."""
        self._install(
            "bin/a.dll", "bin/b.dll", "bin/tool.exe", "lib/c.dll", "lib/a.lib"
        )
        self._run_script()
        stage_bin = self.stage / "bin"
        self.assertEqual(
            sorted(p.name for p in stage_bin.iterdir()), ["a.dll", "b.dll"]
        )
        self.assertEqual((stage_bin / "a.dll").read_text(), "bin/a.dll")
        self.assertTrue((self.prefix / "bin" / "a.dll").exists())

    def test_no_dlls(self):
        """A sub-project with no DLLs, such as a sysdep that builds only a
        static library, succeeds without creating the stage's bin/."""
        self._install("lib/a.lib")
        self._run_script()
        self.assertFalse((self.stage / "bin").exists())

    def test_missing_argument(self):
        """Without STAGE_DIR the script fails, instead of copying the DLLs to
        /bin."""
        result = self._run(f"-DPREFIX={self.prefix}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("PREFIX and STAGE_DIR are both required", result.stdout)


if __name__ == "__main__":
    unittest.main()
