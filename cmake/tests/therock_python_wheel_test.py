#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the reusable Python wheel subproject declaration."""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def run_cmake(
    source_dir: Path, build_dir: Path, *, expect_success: bool
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-GNinja"],
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


def write_harness(source_dir: Path, *, include_source_dir: bool = True) -> None:
    source_argument = (
        'SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package"' if include_source_dir else ""
    )
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_python_wheel_harness NONE)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_PACKAGE_VERSION "10.1.0.dev0+abc123")
        set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")

        function(therock_detect_python_versions out_executables out_versions)
          set(${{out_executables}}
            "/opt/python/cp310/python;/opt/python/cp311/python"
            PARENT_SCOPE)
          set(${{out_versions}} "3.10;3.11" PARENT_SCOPE)
        endfunction()

        function(therock_cmake_subproject_declare target_name)
          cmake_parse_arguments(PARSE_ARGV 1 ARG
            "DISABLE_AMDGPU_TARGETS;BACKGROUND_BUILD"
            "EXTERNAL_SOURCE_DIR;BINARY_DIR;FPRINT_SOURCE_DIR;COMPILER_TOOLCHAIN"
            "FPRINT_FILE_GLOBS;CMAKE_ARGS;BUILD_DEPS;RUNTIME_DEPS"
          )
          file(WRITE "${{CMAKE_BINARY_DIR}}/declaration.txt" "")
          foreach(_arg IN LISTS ARG_CMAKE_ARGS)
            file(APPEND "${{CMAKE_BINARY_DIR}}/declaration.txt"
              "CMAKE_ARG=[${{_arg}}]\n")
          endforeach()
          foreach(_dep IN LISTS ARG_BUILD_DEPS)
            file(APPEND "${{CMAKE_BINARY_DIR}}/declaration.txt"
              "BUILD_DEP=[${{_dep}}]\n")
          endforeach()
          foreach(_dep IN LISTS ARG_RUNTIME_DEPS)
            file(APPEND "${{CMAKE_BINARY_DIR}}/declaration.txt"
              "RUNTIME_DEP=[${{_dep}}]\n")
          endforeach()
        endfunction()

        function(therock_cmake_subproject_activate)
        endfunction()
        function(therock_provide_artifact)
        endfunction()

        add_custom_target(dep_a)
        set_property(TARGET dep_a PROPERTY THEROCK_DIST_DIR "/dist/a")
        add_custom_target(dep_b)
        set_property(TARGET dep_b PROPERTY THEROCK_DIST_DIR "/dist/b")

        include("${{THEROCK_SOURCE_DIR}}/cmake/therock_python_wheel.cmake")
        therock_python_wheel_subproject(example-wheel
          {source_argument}
          FINALIZE_SCRIPT "${{CMAKE_CURRENT_SOURCE_DIR}}/finalize.py"
          ARTIFACT_DESCRIPTOR "artifact-example.toml"
          ENV_FROM_VARS
            TOOLCHAIN_ROOT
            OTHER_BUILD_INPUT
          COMPILER_TOOLCHAIN
            compiler
          BUILD_DEPS
            dep_a
            dep_b
          RUNTIME_DEPS
            runtime
        )
        """,
    )


class PythonWheelSubprojectTest(unittest.TestCase):
    def test_forwards_lists_and_package_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_harness(source_dir)

            run_cmake(source_dir, build_dir, expect_success=True)

            declaration = (build_dir / "declaration.txt").read_text(encoding="utf-8")
            self.assertIn(
                "CMAKE_ARG=[-DTHEROCK_PYTHON_WHEEL_PYTHON_EXECUTABLES:STRING="
                "/opt/python/cp310/python\\;/opt/python/cp311/python]\n",
                declaration,
            )
            self.assertIn(
                "CMAKE_ARG=[-DTHEROCK_PYTHON_WHEEL_PREFIX_PATH:STRING="
                "/dist/a\\;/dist/b]\n",
                declaration,
            )
            self.assertIn(
                "CMAKE_ARG=[-DTHEROCK_PYTHON_WHEEL_ENV_FROM_VARS:STRING="
                "TOOLCHAIN_ROOT\\;OTHER_BUILD_INPUT]\n",
                declaration,
            )
            self.assertIn(
                "CMAKE_ARG=[-DTHEROCK_PACKAGE_VERSION=" "10.1.0.dev0+abc123]\n",
                declaration,
            )
            self.assertIn("BUILD_DEP=[dep_a]\n", declaration)
            self.assertIn("BUILD_DEP=[dep_b]\n", declaration)
            self.assertIn("RUNTIME_DEP=[runtime]\n", declaration)

    def test_requires_source_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_harness(source_dir, include_source_dir=False)

            result = run_cmake(source_dir, build_dir, expect_success=False)
            self.assertIn("SOURCE_DIR is required", result.stdout)


if __name__ == "__main__":
    unittest.main()
