#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the therock_check_imports.cmake script.

The script's only input is what llvm-readobj --coff-imports prints, so these
tests stand in a fake llvm-readobj that prints canned output, and run on every
host, with no Windows binary and no LLVM build."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = THEROCK_ROOT / "cmake" / "therock_check_imports.cmake"

# The fake llvm-readobj.  The script calls it as "READOBJ --coff-imports FILE",
# and it prints FILE's contents, so each test writes the output it wants into
# FILE.  A FILE that does not exist makes it fail as llvm-readobj would.
FAKE_READOBJ = """\
import sys

path = sys.argv[2]
try:
    with open(path, encoding="utf-8") as f:
        sys.stdout.write(f.read())
except OSError:
    sys.stderr.write(f"llvm-readobj: error: '{path}': no such file or directory\\n")
    sys.exit(1)
"""

# Output in the format of llvm-readobj --coff-imports, as observed with the
# llvm-readobj TheRock builds: a "Name:" line per imported DLL, in an Import
# record or a DelayImport record, the latter holding nested Import records with
# only "Symbol:" lines.
READOBJ_OUTPUT = """\
File: D:/build/foo.exe
Format: COFF-x86-64
Arch: x86_64
AddressSize: 64bit
Import {
  Name: KERNEL32.dll
  ImportLookupTableRVA: 0xDCF1E0
  ImportAddressTableRVA: 0xDD0090
  Symbol: GetLastError (0)
  Symbol: LoadLibraryW (0)
}
Import {
  Name: rocm-dbgapi-0-84.dll
  ImportLookupTableRVA: 0xDCF200
  ImportAddressTableRVA: 0xDD00B0
  Symbol: amd_dbgapi_initialize (0)
}
DelayImport {
  Name: SHELL32.dll
  Attributes: 0x1
  ModuleHandle: 0x7602808
  ImportAddressTable: 0x75DE180
  ImportNameTable: 0x74E8198
  BoundDelayImportTable: 0x74E8248
  UnloadDelayImportTable: 0x0
  Import {
    Symbol: ShellExecuteW (0)
  }
}
"""

ALLOW = "kernel32\\.dll|shell32\\.dll|rocm-dbgapi-[0-9-]+\\.dll"


class CheckImportsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        script = self.tmp / "fake_readobj.py"
        script.write_text(FAKE_READOBJ, encoding="utf-8")
        # execute_process runs READOBJ directly, so it must be something the OS
        # can start: a batch file on Windows, a script with a shebang elsewhere.
        if os.name == "nt":
            self.readobj = self.tmp / "llvm-readobj.bat"
            self.readobj.write_text(f'@"{sys.executable}" "{script}" %*\n')
        else:
            self.readobj = self.tmp / "llvm-readobj"
            self.readobj.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n'
            )
            self.readobj.chmod(self.readobj.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, output, allow=ALLOW, readobj=None):
        """Run the script on a binary for which llvm-readobj prints OUTPUT, or
        fails if OUTPUT is None, and return cmake's exit status and output,
        with CMake's line wrapping undone."""
        file = self.tmp / "foo.exe"
        if output is not None:
            file.write_text(output, encoding="utf-8")
        result = subprocess.run(
            [
                "cmake",
                f"-DFILE={file.as_posix()}",
                f"-DREADOBJ={(readobj or self.readobj).as_posix()}",
                f"-DALLOW={allow}",
                "-P",
                str(SCRIPT_PATH),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.returncode, " ".join(result.stdout.split())

    def test_allowed(self):
        """A binary whose imports all match ALLOW passes.  The DelayImport
        record counts as an import, its nested Import record and the Symbol
        lines do not, and the DLL names match whatever their case."""
        status, output = self._run(READOBJ_OUTPUT)
        self.assertEqual(status, 0, output)
        self.assertIn("all 3 imported DLLs allowed", output)

    def test_disallowed(self):
        """A binary importing DLLs outside ALLOW fails, naming every one of
        them, the delay-loaded one included, and none of the allowed ones."""
        output_with_bad = READOBJ_OUTPUT.replace(
            "Name: rocm-dbgapi-0-84.dll", "Name: libintl-8.dll"
        ).replace("Name: SHELL32.dll", "Name: libwinpthread-1.dll")
        status, output = self._run(output_with_bad)
        self.assertNotEqual(status, 0, output)
        self.assertIn(
            "imports DLLs outside the allowed set: libintl-8.dll, "
            "libwinpthread-1.dll",
            output,
        )
        self.assertNotIn("kernel32", output)

    def test_whole_name_match(self):
        """ALLOW must match the whole DLL name, not part of it, so that
        "kernel32\\.dll" does not let through a DLL whose name merely
        contains kernel32.dll."""
        status, output = self._run(
            READOBJ_OUTPUT.replace("Name: KERNEL32.dll", "Name: notkernel32.dll")
        )
        self.assertNotEqual(status, 0, output)
        self.assertIn("allowed set: notkernel32.dll", output)

    def test_no_imports(self):
        """Output with no Name: lines fails instead of passing, as every Windows
        binary imports something, so llvm-readobj's output must not have been
        what the script expects."""
        status, output = self._run("File: D:/build/foo.exe\nFormat: ELF64-x86-64\n")
        self.assertNotEqual(status, 0, output)
        self.assertIn("Found no imports", output)

    def test_readobj_fails(self):
        """A failing llvm-readobj fails the check, passing its error on."""
        status, output = self._run(None)
        self.assertNotEqual(status, 0, output)
        self.assertIn("--coff-imports", output)
        self.assertIn("no such file or directory", output)

    def test_readobj_missing(self):
        """A READOBJ that does not exist fails the check, naming it."""
        missing = self.tmp / "no-such-readobj"
        status, output = self._run(READOBJ_OUTPUT, readobj=missing)
        self.assertNotEqual(status, 0, output)
        self.assertIn(f"llvm-readobj not found: {missing.as_posix()}", output)


if __name__ == "__main__":
    unittest.main()
