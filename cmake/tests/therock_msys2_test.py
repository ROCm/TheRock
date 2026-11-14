#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for therock_msys2_host_gnu_toolchain_args() in therock_msys2.cmake."""

import subprocess
import tempfile
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = THEROCK_ROOT / "cmake" / "therock_msys2.cmake"

REQUIRED = ["usr/bin/bash.exe", "ucrt64/bin/gcc.exe", "ucrt64/bin/g++.exe"]


class HostGnuToolchainArgsTest(unittest.TestCase):
    """The function only checks that files exist and returns arguments, so
    these tests run on every host, with WIN32 set in the script and a fake
    MSYS2 made of empty files."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolve(), as the temp directory can come with Windows 8.3 short names
        # (C:/Users/RUNNER~1/...), which CMake may expand in the paths it returns.
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def _make_msys2(self, name, files):
        """Make a fake MSYS2 named NAME holding FILES, and return its root."""
        root = self.tmp / name
        for file in files:
            path = root / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    def _run(self, msys2_root):
        """Call the function for a sub-project named therock-foo, with
        THEROCK_MSYS2_ROOT set to MSYS2_ROOT, and return the exit status and
        output of cmake.  The returned arguments are printed one per line,
        after ARGS:."""
        script = self.tmp / "test.cmake"
        script.write_text(
            f"""\
set(WIN32 TRUE)
include("{MODULE_PATH.as_posix()}")
therock_msys2_host_gnu_toolchain_args(therock-foo args)
message("ARGS:")
foreach(arg IN LISTS args)
  message("${{arg}}")
endforeach()
""",
            encoding="utf-8",
        )
        # On the command line, as at configure time.  The module's cache
        # default would replace a value set() in the script.
        result = subprocess.run(
            [
                "cmake",
                f"-DTHEROCK_MSYS2_ROOT={msys2_root.as_posix()}",
                "-P",
                str(script),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.returncode, result.stdout

    def test_missing_file(self):
        """An MSYS2 lacking bash, gcc or g++ fails configure.

        Each case leaves out one of the three files.  The function must fail,
        and its message must name both the sub-project that needs MSYS2 and
        the file that is missing.  Without the check, the build would get as
        far as running the sub-project's configure, and fail there with an
        error that names neither."""
        for i, missing in enumerate(REQUIRED):
            with self.subTest(missing=missing):
                root = self._make_msys2(
                    f"msys64-{i}", [f for f in REQUIRED if f != missing]
                )
                status, output = self._run(root)
                self.assertNotEqual(status, 0, output)
                # CMake rewraps the message.
                message = " ".join(output.split())
                self.assertIn("therock-foo is built by MSYS2", message)
                self.assertIn((root / missing).as_posix(), message)

    def test_complete_msys2(self):
        """With bash, gcc and g++ present, the function returns the declare
        arguments for the sub-project, and nothing else.

        They name MSYS2's gcc and g++ as the sub-project's compilers, pass
        THEROCK_MSYS2_ROOT on to the sub-project's own configure, so that it
        uses the same MSYS2, and make therock_msys2.cmake part of the
        sub-project's fingerprint and of its rebuild dependencies."""
        msys2_root = self._make_msys2("msys64", REQUIRED)
        status, output = self._run(msys2_root)
        self.assertEqual(status, 0, output)
        words = output.split("ARGS:\n", 1)[1].splitlines()
        args = dict(zip(words[0::2], words[1::2]))
        root = msys2_root.as_posix()
        self.assertEqual(args.pop("EXTERNAL_C_COMPILER"), f"{root}/ucrt64/bin/gcc.exe")
        self.assertEqual(
            args.pop("EXTERNAL_CXX_COMPILER"), f"{root}/ucrt64/bin/g++.exe"
        )
        self.assertEqual(args.pop("CMAKE_ARGS"), f"-DTHEROCK_MSYS2_ROOT={root}")
        # CMake may name the module through a symlink.
        self.assertEqual(Path(args.pop("FPRINT_FILE_GLOBS")).resolve(), MODULE_PATH)
        self.assertEqual(Path(args.pop("EXTRA_DEPENDS")).resolve(), MODULE_PATH)
        self.assertEqual(args, {})


if __name__ == "__main__":
    unittest.main()
