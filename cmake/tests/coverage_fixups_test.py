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
ROCALUTION_FIXUP = THEROCK_ROOT / "cmake" / "coverage" / "rocALUTION.cmake"
COVERAGE_FLAGS = (
    "-fprofile-instr-generate -fcoverage-mapping -Xarch_device "
    "-fprofile-instr-generate -Xarch_device -fcoverage-mapping"
)

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


# rocALUTION, reduced to what decides the flags its hipcc commands get: its
# policy level, FindHIP creating HIP_CLANG_FLAGS as a cache entry after
# project(), and hip_add_library() reading it in a subdirectory.
ROCALUTION_PROJECT = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.19)
    project(rocalution_like LANGUAGES NONE)
    set(HIP_CLANG_FLAGS "" CACHE STRING "Semicolon delimited flags for CLANG")
    add_subdirectory(src)
    """
)
ROCALUTION_SRC = textwrap.dedent(
    """\
    file(WRITE "${CMAKE_BINARY_DIR}/hip_clang_flags.txt" "${HIP_CLANG_FLAGS}")
    """
)


class RocalutionFixupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src" / "src").mkdir(parents=True)
        (self.root / "src" / "CMakeLists.txt").write_text(ROCALUTION_PROJECT)
        (self.root / "src" / "src" / "CMakeLists.txt").write_text(ROCALUTION_SRC)
        self.build_dir = self.root / "build"

    def tearDown(self):
        self._tmp.cleanup()

    def configure_with(self, include_body: str) -> list[str]:
        include = self.root / "coverage_include.cmake"
        include.write_text(
            f'set(THEROCK_COVERAGE_COMPILE_FLAGS "{COVERAGE_FLAGS}")\n' + include_body
        )
        configure = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.root / "src"),
                "-B",
                str(self.build_dir),
                f"-DCMAKE_PROJECT_INCLUDE={include}",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)
        flags = (self.build_dir / "hip_clang_flags.txt").read_text()
        return [flag for flag in flags.split(";") if flag]

    def test_hipcc_commands_get_the_coverage_flags(self):
        flags = self.configure_with(f'include("{ROCALUTION_FIXUP.as_posix()}")\n')
        self.assertEqual(flags, COVERAGE_FLAGS.split())

    def test_reconfiguring_does_not_repeat_them(self):
        include_body = f'include("{ROCALUTION_FIXUP.as_posix()}")\n'
        self.configure_with(include_body)
        flags = self.configure_with(include_body)
        self.assertEqual(flags, COVERAGE_FLAGS.split())

    def test_a_normal_variable_would_not_reach_them(self):
        # Keeps the tests above honest: FindHIP creating the cache entry drops
        # a normal variable set from the project include.
        flags = self.configure_with(
            "set(HIP_CLANG_FLAGS ${THEROCK_COVERAGE_COMPILE_FLAGS})\n"
        )
        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
