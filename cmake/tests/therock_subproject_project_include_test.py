#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for CMAKE_INCLUDES timing and reconfiguration.

The fixtures verify that TheRock's generated project-include dispatcher
preserves existing hooks, applies TheRock includes once after language
initialization, and invalidates configured subprojects when those inputs
change. Creating imported shared targets before language initialization
fails when CMP0164 is set to NEW.
"""

import tempfile
import unittest
from pathlib import Path

from therock_subproject_test_utils import (
    HARNESS_INCLUDES,
    THEROCK_ROOT,
    run,
    write_file,
)


def write_harness(
    source_dir: Path,
    *,
    project_include: Path,
    preserved_project_include: Path,
) -> None:
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_project_include_harness C CXX)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        set(ROCM_BUILD_FLAGS_STATE_FILE
          "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\n")

        {includes}

        therock_cmake_subproject_declare(main_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/main"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/main"
          CMAKE_ARGS
            "-DCMAKE_PROJECT_INCLUDE:FILEPATH={preserved_project_include.as_posix()}"
          CMAKE_INCLUDES "{project_include.as_posix()}"
        )
        therock_cmake_subproject_activate(main_project)
        """,
    )


def write_subproject(source_dir: Path) -> None:
    write_file(
        source_dir / "main" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(main_project C)
        add_subdirectory(nested)

        if(NOT TARGET therock_test_imported)
          message(FATAL_ERROR "TheRock project include was not applied")
        endif()
        file(WRITE "${CMAKE_BINARY_DIR}/configured-marker.txt"
          "${THEROCK_TEST_INCLUDE_MARKER}\n")
        """,
    )

    write_file(
        source_dir / "main" / "nested" / "CMakeLists.txt",
        """
        project(nested_project C)
        if(NOT TARGET therock_test_imported)
          message(FATAL_ERROR "Nested project cannot see the imported target")
        endif()
        """,
    )


def write_project_include(path: Path, marker: str) -> None:
    write_file(
        path,
        f"""
        # Reject early execution even with older, more permissive CMake policies.
        if(NOT CMAKE_C_COMPILER_LOADED)
          message(FATAL_ERROR "project include ran before C was initialized")
        endif()
        function(therock_test_record_preserved_include)
          file(APPEND "${{CMAKE_BINARY_DIR}}/preserved-project-include.log"
            "${{PROJECT_NAME}}\n")
        endfunction()
        add_library(therock_test_imported SHARED IMPORTED)
        set(THEROCK_TEST_INCLUDE_MARKER "{marker}")
        file(APPEND "${{CMAKE_BINARY_DIR}}/therock-project-include.log"
          "{marker}\n")
        """,
    )


def write_preserved_project_include(path: Path) -> None:
    write_file(
        path,
        """
        # Record caller hooks for the main and nested fixture projects.
        # Compiler checks run separately without the fixture's recording helper.
        get_property(_in_try_compile GLOBAL PROPERTY IN_TRY_COMPILE)
        if(NOT _in_try_compile)
          therock_test_record_preserved_include()
        endif()
        """,
    )


class ProjectIncludeTest(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.source_dir = Path(temp_dir.name) / "source"
        self.build_dir = Path(temp_dir.name) / "build"
        self.subproject_build_dir = self.build_dir / "main" / "build"
        self.first_include = self.source_dir / "first-include.cmake"
        self.second_include = self.source_dir / "second-include.cmake"
        self.preserved_include = self.source_dir / "preserved-include.cmake"

        write_subproject(self.source_dir)
        write_preserved_project_include(self.preserved_include)

    def configure(self, project_include: Path) -> None:
        write_harness(
            self.source_dir,
            project_include=project_include,
            preserved_project_include=self.preserved_include,
        )
        run(
            "cmake",
            "-S",
            str(self.source_dir),
            "-B",
            str(self.build_dir),
            "-GNinja",
        )

    def build(self) -> None:
        run(
            "cmake",
            "--build",
            str(self.build_dir),
            "--target",
            "main_project+build",
        )

    def read_subproject_file(self, name: str) -> str:
        return (self.subproject_build_dir / name).read_text(encoding="utf-8")

    def test_preserves_existing_hook_and_guards_therock_include(self) -> None:
        """Catch early or repeated includes and lost or reordered caller hooks.

        The caller hook uses a helper from TheRock's include and must run in
        both projects. The child also checks imported-target visibility.
        """
        write_project_include(self.first_include, "first")
        self.configure(self.first_include)
        self.build()

        self.assertEqual(
            self.read_subproject_file("preserved-project-include.log").splitlines(),
            ["main_project", "nested_project"],
        )
        self.assertEqual(
            self.read_subproject_file("therock-project-include.log").splitlines(),
            ["first"],
        )
        self.assertEqual(self.read_subproject_file("configured-marker.txt"), "first\n")

    def test_include_inputs_reconfigure_built_subproject(self) -> None:
        """Catch stale configuration after include-list or included-file edits."""
        write_project_include(self.first_include, "first")
        self.configure(self.first_include)
        self.build()
        self.assertEqual(self.read_subproject_file("configured-marker.txt"), "first\n")

        # Reuse the build tree to catch missing include-list invalidation.
        write_project_include(self.second_include, "second")
        self.configure(self.second_include)
        self.build()
        self.assertEqual(self.read_subproject_file("configured-marker.txt"), "second\n")

        # A normal build must detect this edit without a superproject configure.
        write_project_include(self.second_include, "updated")
        self.build()
        self.assertEqual(
            self.read_subproject_file("configured-marker.txt"), "updated\n"
        )


if __name__ == "__main__":
    unittest.main()
