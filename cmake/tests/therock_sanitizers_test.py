#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise sanitizer configuration without building the compiler or GPU code."""

import subprocess
import tempfile
import unittest
from pathlib import Path


THEROCK_ROOT = Path(__file__).resolve().parents[2]


class SanitizersTest(unittest.TestCase):
    def configure(self, settings: str, checks: str = "") -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.25)
project(sanitizers_test NONE)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_CXX_COMPILER clang++)
set(THEROCK_SANITIZER CSAN)
set(GPU_TARGETS gfx942 gfx1100)
include("{THEROCK_ROOT.as_posix()}/cmake/therock_sanitizers.cmake")
{settings}
therock_sanitizer_configure(stanza selected clang++ amd-clang test)
cmake_language(EVAL CODE "${{stanza}}")
{checks}
""",
                encoding="utf-8",
            )
            return subprocess.run(
                ["cmake", "-S", str(source), "-B", str(source / "build")],
                capture_output=True,
                text=True,
            )

    def test_csan_flags_and_autotools_linking(self):
        result = self.configure(
            "",
            f"""
foreach(lang C CXX HIP)
  if(NOT CMAKE_${{lang}}_FLAGS_INIT MATCHES "-fsanitize=concurrency")
    message(FATAL_ERROR "Missing ${{lang}} instrumentation")
  endif()
endforeach()
if(NOT GPU_TARGETS STREQUAL "gfx942;gfx1100")
  message(FATAL_ERROR "CSAN changed GPU targets")
endif()
get_directory_property(link_options LINK_OPTIONS)
if(NOT link_options MATCHES "LINK_LANGUAGE:C,CXX,HIP.*-fsanitize=concurrency" OR
   NOT link_options MATCHES "-shared-libsan")
  message(FATAL_ERROR "Missing shared CSAN link options")
endif()
if(NOT THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH OR NOT selected STREQUAL "CSAN")
  message(FATAL_ERROR "Missing runtime path or sanitizer selection")
endif()
include("{THEROCK_ROOT.as_posix()}/debug-tools/pre_hook_rocgdb.cmake")
foreach(kind EXE SHARED)
  if(NOT CMAKE_${{kind}}_LINKER_FLAGS MATCHES "-fsanitize=concurrency -shared-libsan")
    message(FATAL_ERROR "Missing Autotools sanitizer link flags")
  endif()
endforeach()
""",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unsupported_platform(self):
        for setting in (
            "set(CMAKE_SYSTEM_NAME Windows)",
            "set(CMAKE_SYSTEM_PROCESSOR aarch64)",
        ):
            with self.subTest(setting=setting):
                result = self.configure(setting)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CSAN requires a Linux x86-64 host", result.stderr)

    def test_per_project_disable(self):
        result = self.configure(
            "set(test_SANITIZER OFF)",
            """
if(stanza OR selected OR CMAKE_CXX_FLAGS_INIT)
  message(FATAL_ERROR "Disabled project was instrumented")
endif()
""",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_existing_asan_target_mapping(self):
        result = self.configure(
            "set(THEROCK_SANITIZER ASAN)",
            """
if(NOT GPU_TARGETS STREQUAL "gfx942:xnack+;gfx1100" OR
   NOT CMAKE_HIP_FLAGS_INIT MATCHES "-fsanitize=address")
  message(FATAL_ERROR "ASAN target mapping or HIP flags regressed")
endif()
""",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
