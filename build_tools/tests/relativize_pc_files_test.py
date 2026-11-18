#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import tempfile
import unittest
from pathlib import Path

from relativize_pc_files import relativize_pc_files


def _write_pc(prefix: Path, name: str, text: str) -> Path:
    pc_dir = prefix / "lib" / "pkgconfig"
    pc_dir.mkdir(parents=True, exist_ok=True)
    pc_file = pc_dir / name
    pc_file.write_text(text)
    return pc_file


class RelativizePcFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.prefix = Path(self._tmp.name) / "stage" / "lib" / "rocm_sysdeps"

    def tearDown(self):
        self._tmp.cleanup()

    def test_windows_drive_prefix(self):
        # As a Windows build writes it: a drive-letter prefix, repeated in
        # libdir and in an -L flag.
        build_prefix = "D:/build/ncurses/stage/lib/rocm_sysdeps"
        pc_file = _write_pc(
            self.prefix,
            "ncursesw.pc",
            f"prefix={build_prefix}\n"
            f"libdir={build_prefix}/lib\n"
            "includedir=${prefix}/include\n"
            f"Libs: -L{build_prefix}/lib -lncursesw\n",
        )
        self.assertEqual(relativize_pc_files(self.prefix), [pc_file])
        text = pc_file.read_text()
        self.assertIn("prefix=${pcfiledir}/../..\n", text)
        self.assertIn("libdir=${prefix}/lib\n", text)
        self.assertIn("Libs: -L${prefix}/lib -lncursesw\n", text)
        self.assertNotIn("D:/", text)

    def test_posix_prefix(self):
        build_prefix = "/build/gmp/stage/lib/rocm_sysdeps"
        pc_file = _write_pc(
            self.prefix,
            "gmp.pc",
            f"prefix={build_prefix}\n"
            "libdir=${prefix}/lib\n"
            "Libs: -L${libdir} -lgmp\n",
        )
        relativize_pc_files(self.prefix)
        text = pc_file.read_text()
        self.assertIn("prefix=${pcfiledir}/../..\n", text)
        self.assertNotIn(build_prefix, text)

    def test_every_pc_file(self):
        names = ["a.pc", "b.pc"]
        for name in names:
            _write_pc(self.prefix, name, "prefix=/build/x\nlibdir=/build/x/lib\n")
        rewritten = relativize_pc_files(self.prefix)
        self.assertEqual([p.name for p in rewritten], names)
        for p in rewritten:
            self.assertIn("libdir=${prefix}/lib\n", p.read_text())

    def test_already_relative_unchanged(self):
        text = "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\n"
        pc_file = _write_pc(self.prefix, "zlib.pc", text)
        relativize_pc_files(self.prefix)
        self.assertEqual(pc_file.read_text(), text)

    def test_no_pkgconfig_dir(self):
        self.prefix.mkdir(parents=True)
        self.assertEqual(relativize_pc_files(self.prefix), [])


if __name__ == "__main__":
    unittest.main()
