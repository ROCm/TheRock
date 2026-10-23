#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""-Werror on rocprofiler-sdk's library build-flags interface.

Component CI compiles rocprofiler-sdk with -Werror. TheRock does not enable
ROCPROFILER_BUILD_DEVELOPER (that option also requires format/lint tools and a
wider warning set), so profiler/post_hook_rocprofiler-sdk.cmake adds -Werror to
rocprofiler-sdk-build-flags, the interface the libraries already link.

Removing that flag makes the property check fail. On GCC/Clang, an unhandled
enumerator compiled with -Wall plus that interface fails, and a covered switch
still builds.
"""

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
POST_HOOK = THEROCK_ROOT / "profiler" / "post_hook_rocprofiler-sdk.cmake"

UNHANDLED_SWITCH = """
enum class CopyOp { Linear, Rect };

const char* name(CopyOp op) {
  switch (op) {
    case CopyOp::Linear:
      return "linear";
  }
  return "unhandled";
}
"""

COVERED_SWITCH = """
enum class CopyOp { Linear, Rect };

const char* name(CopyOp op) {
  switch (op) {
    case CopyOp::Linear:
      return "linear";
    case CopyOp::Rect:
      return "rect";
  }
  return "unhandled";
}
"""


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def run_cmake(
    source_dir: Path,
    build_dir: Path,
    *,
    expect_success: bool = True,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-GNinja",
            f"-DPOST_HOOK={POST_HOOK}",
            *extra_args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(result.stdout)
    if not expect_success and result.returncode == 0:
        raise AssertionError("CMake unexpectedly succeeded:\n" + result.stdout)
    return result


def configure_cxx_probe(
    source_dir: Path,
    build_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Configure a C++ probe, falling back to g++ if the default cannot link.

    Some images point ``c++`` at Clang without ``libstdc++``. The property
    check does not need a compiler; this fallback is only for the compile probe.
    """
    result = run_cmake(source_dir, build_dir, expect_success=False)
    if result.returncode == 0:
        return result
    compiler_broken = "is not able to compile a simple test program" in result.stdout
    gxx = shutil.which("g++")
    if compiler_broken and gxx is not None:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        return run_cmake(
            source_dir,
            build_dir,
            extra_args=(f"-DCMAKE_CXX_COMPILER={gxx}",),
        )
    raise AssertionError(result.stdout)


def run_build(
    build_dir: Path,
    target: str,
    *,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", target],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(result.stdout)
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            f"Build of {target} unexpectedly succeeded:\n" + result.stdout
        )
    return result


class RocprofilerSdkWerrorTest(unittest.TestCase):
    def test_post_hook_puts_werror_on_library_flags_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rocprofiler-sdk-werror-") as tmp:
            source_dir = Path(tmp) / "src"
            build_dir = Path(tmp) / "build"
            write_file(
                source_dir / "CMakeLists.txt",
                """
                cmake_minimum_required(VERSION 3.25)
                project(rocprofiler_sdk_werror_probe LANGUAGES NONE)

                add_library(rocprofiler-sdk-build-flags INTERFACE)
                add_library(rocprofiler-sdk-tool INTERFACE)
                add_library(rocprofiler-sdk-tool-kokkosp INTERFACE)
                add_library(rocprofv3-list-avail INTERFACE)

                include("${POST_HOOK}")

                get_target_property(_sdk_options rocprofiler-sdk-build-flags
                                    INTERFACE_COMPILE_OPTIONS)
                if(NOT _sdk_options OR NOT "-Werror" IN_LIST _sdk_options)
                  message(FATAL_ERROR
                    "rocprofiler-sdk-build-flags is missing -Werror "
                    "(options='${_sdk_options}')")
                endif()

                foreach(_other rocprofiler-sdk-tool rocprofiler-sdk-tool-kokkosp
                        rocprofv3-list-avail)
                  get_target_property(_other_options "${_other}"
                                      INTERFACE_COMPILE_OPTIONS)
                  if(_other_options AND "-Werror" IN_LIST _other_options)
                    message(FATAL_ERROR "${_other} should not carry -Werror")
                  endif()
                endforeach()
                """,
            )
            configure = run_cmake(source_dir, build_dir)
            self.assertNotIn("missing -Werror", configure.stdout)

    def test_unhandled_enumerator_fails_and_covered_switch_builds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rocprofiler-sdk-werror-") as tmp:
            source_dir = Path(tmp) / "src"
            build_dir = Path(tmp) / "build"
            write_file(
                source_dir / "CMakeLists.txt",
                """
                cmake_minimum_required(VERSION 3.25)
                project(rocprofiler_sdk_werror_compile LANGUAGES CXX)

                add_library(rocprofiler-sdk-build-flags INTERFACE)
                add_library(rocprofiler-sdk-tool INTERFACE)
                add_library(rocprofiler-sdk-tool-kokkosp INTERFACE)
                add_library(rocprofv3-list-avail INTERFACE)
                include("${POST_HOOK}")

                if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
                  message(STATUS "ROCPROFILER_SDK_WERROR_COMPILE=skipped")
                  return()
                endif()
                message(STATUS "ROCPROFILER_SDK_WERROR_COMPILE=enabled")

                # -Wall is what the real rocprofiler-sdk-build-flags interface
                # already contributes. This probe only owns -Werror, so the
                # warning itself has to be enabled here.
                add_library(unhandled_switch STATIC unhandled_switch.cpp)
                target_compile_options(unhandled_switch PRIVATE -Wall)
                target_link_libraries(unhandled_switch
                                      PRIVATE rocprofiler-sdk-build-flags)

                add_library(covered_switch STATIC covered_switch.cpp)
                target_compile_options(covered_switch PRIVATE -Wall)
                target_link_libraries(covered_switch
                                      PRIVATE rocprofiler-sdk-build-flags)
                """,
            )
            write_file(source_dir / "unhandled_switch.cpp", UNHANDLED_SWITCH)
            write_file(source_dir / "covered_switch.cpp", COVERED_SWITCH)
            configure = configure_cxx_probe(source_dir, build_dir)
            if "ROCPROFILER_SDK_WERROR_COMPILE=skipped" in configure.stdout:
                self.skipTest("host C++ compiler is not GCC or Clang")
            self.assertIn("ROCPROFILER_SDK_WERROR_COMPILE=enabled", configure.stdout)

            failed = run_build(build_dir, "unhandled_switch", expect_success=False)
            # GCC and Clang quote the enumerator differently, including Unicode
            # quotes. The diagnostic that broke component CI is -Werror=switch.
            self.assertIn("-Werror", failed.stdout)
            self.assertRegex(failed.stdout, r"not handled in switch")
            self.assertIn("-Werror=switch", failed.stdout)
            run_build(build_dir, "covered_switch", expect_success=True)


if __name__ == "__main__":
    unittest.main()
