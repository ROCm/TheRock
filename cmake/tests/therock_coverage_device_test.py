#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for cmake/therock_coverage_device.cmake."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = THEROCK_ROOT / "cmake" / "therock_coverage_device.cmake"
HOST_RUNTIME_DIR = "lib/x86_64-unknown-linux-gnu"

# Answers the two runtime lookups from FAKE_RESOURCE_DIR, the way clang does,
# and hands everything else to the host compiler so CMake's checks pass.
FAKE_COMPILER = textwrap.dedent(
    f"""\
    #!/bin/sh
    for arg in "$@"; do
      case "$arg" in
        -print-resource-dir) echo "$FAKE_RESOURCE_DIR"; exit 0 ;;
        -print-file-name=*)
          name="${{arg#-print-file-name=}}"
          if [ -e "$FAKE_RESOURCE_DIR/{HOST_RUNTIME_DIR}/$name" ]; then
            echo "$FAKE_RESOURCE_DIR/{HOST_RUNTIME_DIR}/$name"
          else
            echo "$name"
          fi
          exit 0 ;;
      esac
    done
    exec c++ "$@"
    """
)

# Two project() calls, since the include runs after each one. GENERIC and
# COLLECTOR are the two host profile archives, passed in by the test.
PROJECT = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.25)
    project(outer LANGUAGES CXX)
    project(inner LANGUAGES CXX)
    file(WRITE "${CMAKE_BINARY_DIR}/f.cpp" "int f() { return 0; }\\n")
    # rocSPARSE's order: the generic archive first, in a group.
    add_library(names_runtime SHARED "${CMAKE_BINARY_DIR}/f.cpp")
    target_link_libraries(names_runtime PRIVATE
      -Wl,--start-group "${GENERIC}" "${COLLECTOR}" -Wl,--end-group)
    add_library(right_order SHARED "${CMAKE_BINARY_DIR}/f.cpp")
    target_link_libraries(right_order PRIVATE "${COLLECTOR}" "${GENERIC}")
    add_library(plain SHARED "${CMAKE_BINARY_DIR}/f.cpp")
    add_executable(app "${CMAKE_BINARY_DIR}/f.cpp")
    add_subdirectory(sub)
    function(print_link_libraries)
      foreach(target names_runtime right_order plain app nested)
        get_target_property(libs ${target} LINK_LIBRARIES)
        message(STATUS "LIBS_${target}=${libs}")
      endforeach()
      message(STATUS "EXE_RULE=${CMAKE_CXX_LINK_EXECUTABLE}")
    endfunction()
    # Deferred after the include's own call, so it sees the result.
    cmake_language(DEFER CALL print_link_libraries)
    """
)
SUBDIRECTORY = textwrap.dedent(
    """\
    add_library(nested SHARED "${CMAKE_BINARY_DIR}/f.cpp")
    target_link_libraries(nested PRIVATE "${GENERIC}")
    """
)


@unittest.skipIf(sys.platform == "win32", "the fake compiler is a shell script")
@unittest.skipUnless(shutil.which("c++"), "needs a host C++ compiler")
class CoverageDeviceIncludeTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        self.resource_dir = self.root / "resource"
        self.compiler = self.root / "fake-clang++"
        self.compiler.write_text(FAKE_COMPILER)
        self.compiler.chmod(0o755)
        (self.root / "src" / "sub").mkdir(parents=True)
        (self.root / "src" / "CMakeLists.txt").write_text(PROJECT)
        (self.root / "src" / "sub" / "CMakeLists.txt").write_text(SUBDIRECTORY)
        # What therock_subproject.cmake generates around the include.
        self.include = self.root / "coverage.cmake"
        self.include.write_text(
            'set(THEROCK_COVERAGE_DEVICE_OPTION "ROCSPARSE_ENABLE_DEVICE_COVERAGE")\n'
            f'include("{MODULE_PATH.as_posix()}")\n'
        )

    def provide(self, relpath: str) -> Path:
        path = self.resource_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def configure(self, expect_success: bool = True) -> str:
        runtime_dir = self.resource_dir / HOST_RUNTIME_DIR
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.root / "src"),
                "-B",
                str(self.root / "build"),
                f"-DCMAKE_CXX_COMPILER={self.compiler}",
                f"-DCMAKE_PROJECT_INCLUDE={self.include}",
                f"-DGENERIC={runtime_dir / 'libclang_rt.profile.a'}",
                f"-DCOLLECTOR={runtime_dir / 'libclang_rt.profile_rocm.a'}",
            ],
            env={**os.environ, "FAKE_RESOURCE_DIR": str(self.resource_dir)},
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode == 0, expect_success, result.stdout)
        return result.stdout

    @staticmethod
    def value(output: str, name: str) -> str:
        prefix = f"-- {name}="
        line = next(l for l in output.splitlines() if l.startswith(prefix))
        return line[len(prefix) :]

    @staticmethod
    def flat(output: str) -> str:
        """CMake wraps error messages, so compare against unwrapped text."""
        return " ".join(output.split())

    def test_collector_goes_ahead_of_a_generic_archive_the_project_names(self):
        self.provide("lib/amdgcn-amd-amdhsa/libclang_rt.profile.a")
        generic = self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile.a")
        collector = self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm.a")

        output = self.configure()

        # Once, although the include ran after both project() calls.
        self.assertEqual(
            self.value(output, "LIBS_names_runtime").split(";"),
            [
                "-Wl,--start-group",
                str(collector),
                str(generic),
                str(collector),
                "-Wl,--end-group",
            ],
        )
        self.assertEqual(
            self.value(output, "LIBS_nested").split(";"), [str(collector), str(generic)]
        )

    def test_targets_that_need_no_reordering_are_left_alone(self):
        # Inserting the collector into every link would make it resolve
        # hipLaunchKernel in binaries with no instrumented code at all.
        self.provide("lib/amdgcn-amd-amdhsa/libclang_rt.profile.a")
        generic = self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile.a")
        collector = self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm.a")

        output = self.configure()

        self.assertEqual(
            self.value(output, "LIBS_right_order").split(";"),
            [str(collector), str(generic)],
        )
        for target in ("plain", "app"):
            with self.subTest(target=target):
                self.assertNotIn("profile_rocm", self.value(output, f"LIBS_{target}"))
        self.assertNotIn("profile_rocm", self.value(output, "EXE_RULE"))

    def test_arch_suffixed_host_collector_is_found(self):
        # The layout of compilers built without LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
        self.provide("lib/linux/libclang_rt.profile-amdgcn.a")
        collector = self.provide(
            f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm-{platform.machine()}.a"
        )

        self.assertIn(f"host collector {collector}", self.flat(self.configure()))

    def test_missing_device_runtime_fails_the_configure(self):
        self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm.a")

        output = self.flat(self.configure(expect_success=False))

        self.assertIn("no amdgcn profile runtime", output)
        self.assertIn("-DROCSPARSE_ENABLE_DEVICE_COVERAGE=OFF", output)

    def test_missing_host_collector_fails_the_configure(self):
        self.provide("lib/amdgcn-amd-amdhsa/libclang_rt.profile.a")

        output = self.flat(self.configure(expect_success=False))

        self.assertIn("no libclang_rt.profile_rocm.a", output)


if __name__ == "__main__":
    unittest.main()
