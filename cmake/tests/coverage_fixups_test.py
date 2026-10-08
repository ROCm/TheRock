#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the per-project coverage workarounds in cmake/coverage/."""

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
HIPBLASLT_FIXUP = THEROCK_ROOT / "cmake" / "coverage" / "hipBLASLt.cmake"

# hipBLASLt under TheRock, reduced to the part that fails: rocRoller arrives as
# an imported target, and the coverage-only test link names it bare. The
# library file is not called librocroller.so, so a bare -lrocroller cannot
# find it by accident.
PROJECT = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.25)
    project(hipblaslt_like LANGUAGES C)
    file(WRITE "${CMAKE_BINARY_DIR}/roller.c" "int roller(void) { return 7; }\\n")
    file(WRITE "${CMAKE_BINARY_DIR}/main.c"
      "int roller(void);\\nint main(void) { return roller() == 7 ? 0 : 1; }\\n")
    add_library(roller_impl SHARED "${CMAKE_BINARY_DIR}/roller.c")
    set_target_properties(roller_impl PROPERTIES OUTPUT_NAME packaged_roller)
    add_library(roc::rocroller SHARED IMPORTED)
    set_target_properties(roc::rocroller PROPERTIES
      IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libpackaged_roller.so")
    add_subdirectory(clients)
    """
)
CLIENTS = textwrap.dedent(
    """\
    add_executable(hipblaslt-test "${CMAKE_BINARY_DIR}/main.c")
    add_dependencies(hipblaslt-test roller_impl)
    target_link_libraries(hipblaslt-test PRIVATE rocroller)
    """
)


@unittest.skipIf(sys.platform == "win32", "builds and links a shared library")
@unittest.skipUnless(shutil.which("cc"), "needs a host C compiler")
class HipblasltFixupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src" / "clients").mkdir(parents=True)
        (self.root / "src" / "CMakeLists.txt").write_text(PROJECT)
        (self.root / "src" / "clients" / "CMakeLists.txt").write_text(CLIENTS)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, *configure_args: str) -> subprocess.CompletedProcess:
        build_dir = self.root / "build"
        shutil.rmtree(build_dir, ignore_errors=True)
        configure = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.root / "src"),
                "-B",
                str(build_dir),
                *configure_args,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)
        return subprocess.run(
            ["cmake", "--build", str(build_dir)], capture_output=True, text=True
        )

    def test_bare_rocroller_name_links_the_imported_target(self):
        build = self.build(f"-DCMAKE_PROJECT_INCLUDE={HIPBLASLT_FIXUP}")
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)

    def test_without_the_fixup_the_bare_name_does_not_link(self):
        # Keeps the test above honest: the reduced project fails as hipBLASLt's
        # coverage build does.
        build = self.build()
        self.assertNotEqual(build.returncode, 0)
        self.assertIn("rocroller", build.stdout + build.stderr)


if __name__ == "__main__":
    unittest.main()
