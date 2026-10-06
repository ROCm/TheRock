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

# Two project() calls, since the include runs after each one.
PROJECT = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.25)
    project(outer LANGUAGES CXX)
    project(inner LANGUAGES CXX)
    message(STATUS "SHARED_RULE=${CMAKE_CXX_CREATE_SHARED_LIBRARY}")
    message(STATUS "EXE_RULE=${CMAKE_CXX_LINK_EXECUTABLE}")
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
        (self.root / "src").mkdir()
        (self.root / "src" / "CMakeLists.txt").write_text(PROJECT)
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
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.root / "src"),
                "-B",
                str(self.root / "build"),
                f"-DCMAKE_CXX_COMPILER={self.compiler}",
                f"-DCMAKE_PROJECT_INCLUDE={self.include}",
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
    def rule(output: str, name: str) -> str:
        prefix = f"-- {name}="
        return next(l for l in output.splitlines() if l.startswith(prefix))

    @staticmethod
    def flat(output: str) -> str:
        """CMake wraps error messages, so compare against unwrapped text."""
        return " ".join(output.split())

    def test_host_collector_is_linked_ahead_of_the_projects_libraries(self):
        self.provide("lib/amdgcn-amd-amdhsa/libclang_rt.profile.a")
        collector = self.provide(f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm.a")

        output = self.configure()

        for name in ("SHARED_RULE", "EXE_RULE"):
            with self.subTest(rule=name):
                rule = self.rule(output, name)
                # Once, although the include ran after both project() calls.
                self.assertEqual(rule.count(str(collector)), 1)
                self.assertLess(
                    rule.index(str(collector)), rule.index("<LINK_LIBRARIES>")
                )

    def test_arch_suffixed_host_collector_is_found(self):
        # The layout of compilers built without LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
        self.provide("lib/linux/libclang_rt.profile-amdgcn.a")
        collector = self.provide(
            f"{HOST_RUNTIME_DIR}/libclang_rt.profile_rocm-{platform.machine()}.a"
        )

        self.assertIn(str(collector), self.rule(self.configure(), "SHARED_RULE"))

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
