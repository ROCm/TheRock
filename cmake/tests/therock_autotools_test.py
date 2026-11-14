#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for how therock_autotools_shell() in therock_autotools.cmake names
the compiler and its launcher to configure."""

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = THEROCK_ROOT / "cmake" / "therock_autotools.cmake"


class AutotoolsShellCompilerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # resolve(), as the temp directory can come with Windows 8.3 short names
        # (C:/Users/RUNNER~1/...), which CMake may expand in the paths it returns.
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def _launcher(self, dirname="bin"):
        """Make an empty file to stand for a compiler launcher in DIRNAME, and
        return its path.  On Windows, therock_autotools_shell() looks the
        launcher up, so it has to exist."""
        path = self.tmp / dirname / "launcher.exe"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path.as_posix()

    def _run(self, cc, cxx, launcher=None, configure_env=()):
        """Call therock_autotools_shell() with compilers CC and CXX, LAUNCHER,
        a list, as the launcher of both, and CONFIGURE_ENV, a list of
        VAR=VALUE entries.  Return cmake's exit status and the CC= and CXX=
        entries of the configure command prefix, or cmake's output, with
        whitespace collapsed, if it fails."""
        launcher_lines = ""
        if launcher:
            words = ";".join(launcher)
            launcher_lines = (
                f'set(CMAKE_C_COMPILER_LAUNCHER "{words}")\n'
                f'set(CMAKE_CXX_COMPILER_LAUNCHER "{words}")\n'
            )
        script = self.tmp / "test.cmake"
        script.write_text(
            f"""\
set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
include("{MODULE_PATH.as_posix()}")
set(CMAKE_C_COMPILER "{cc}")
set(CMAKE_CXX_COMPILER "{cxx}")
{launcher_lines}therock_autotools_shell(
  BINARY_DIR "{(self.tmp / "build").as_posix()}"
  CONFIGURE_ENV {" ".join(f'"{e}"' for e in configure_env)}
  RUN_CONFIGURE_VAR run_configure)
foreach(word IN LISTS run_configure)
  if(word MATCHES "^(CC|CXX)=")
    message("${{word}}")
  endif()
endforeach()
""",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["cmake", "-P", str(script)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            # CMake rewraps the message.
            return result.returncode, " ".join(result.stdout.split())
        return 0, result.stdout.splitlines()

    def test_compilers(self):
        """With no launcher, configure gets CC and CXX naming the compilers
        exactly as CMake has them, so that it does not pick one off PATH."""
        status, env = self._run("/opt/gcc", "/opt/g++")
        self.assertEqual(status, 0, env)
        self.assertEqual(env, ["CC=/opt/gcc", "CXX=/opt/g++"])

    def test_caller_compiler(self):
        """A CC or CXX the caller passes in CONFIGURE_ENV, such as expat's
        wrapper adding hardening flags, replaces the compiler the function
        would name for that language, and leaves the other language alone.

        Both would otherwise reach the same "cmake -E env" command, which
        does pick the caller's, as the later of the two, but leaves the
        command naming two compilers."""
        status, env = self._run("/opt/gcc", "/opt/g++", configure_env=["CC=wrap.sh"])
        self.assertEqual(status, 0, env)
        self.assertEqual(env, ["CXX=/opt/g++", "CC=wrap.sh"])

    def test_launcher(self):
        """A compiler launcher (ccache and friends) reaches configure.

        CMake holds the launcher as a list, a program and its arguments.
        configure takes CC and CXX as one string that it splits into words at
        spaces, so the launcher's words go in front of the compiler, joined
        with spaces.  The test names the launcher by an absolute path, which
        comes out unchanged on Windows too, where the launcher is looked up."""
        launcher = self._launcher()
        status, env = self._run("/opt/gcc", "/opt/g++", [launcher, "--flag"])
        self.assertEqual(status, 0, env)
        self.assertEqual(
            env,
            [f"CC={launcher} --flag /opt/gcc", f"CXX={launcher} --flag /opt/g++"],
        )

    def test_space_in_compiler(self):
        """A compiler path with a space fails CMake's configure, naming the
        path.

        configure would split CC at the space, and then fail with "C compiler
        cannot create executables", which says nothing about the cause."""
        status, output = self._run("/opt/my gcc", "/opt/g++")
        self.assertNotEqual(status, 0, output)
        self.assertIn("'/opt/my gcc'", output)
        self.assertIn("contains a space", output)

    def test_space_in_launcher(self):
        """A launcher path with a space fails the same way, as the launcher's
        words end up in CC and CXX too.  On Windows that path is the result
        of a lookup, which can land under "C:/Program Files"."""
        launcher = self._launcher("my bin")
        status, output = self._run("/opt/gcc", "/opt/g++", [launcher])
        self.assertNotEqual(status, 0, output)
        self.assertIn(f"'{launcher}'", output)
        self.assertIn("contains a space", output)

    # Elsewhere, the launcher is passed on as named, and found on PATH when
    # configure runs.
    @unittest.skipUnless(
        platform.system() == "Windows", "the launcher is looked up only on Windows"
    )
    def test_missing_launcher(self):
        """On Windows, a launcher that cannot be found fails CMake's
        configure, naming it.

        The MSYS2 shell that runs configure has a PATH of its own, so a plain
        launcher name is looked up when CMake configures, and replaced with
        its absolute path.  A name that cannot be resolved there must not be
        passed on to fail inside configure."""
        status, output = self._run("/opt/gcc", "/opt/g++", ["therock-no-launcher"])
        self.assertNotEqual(status, 0, output)
        self.assertIn("'therock-no-launcher'", output)
        self.assertIn("not found", output)


if __name__ == "__main__":
    unittest.main()
