#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared ROCm build flag protocol and consumer helper."""

import json
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = THEROCK_ROOT / "cmake" / "ROCMBuildFlags.cmake"
FLAG_UTILS_PATH = THEROCK_ROOT / "cmake" / "therock_flag_utils.cmake"
AUX_HEADER_TEMPLATE = (
    THEROCK_ROOT / "base" / "aux-overlay" / "aux_overlay_build_flags.h.in"
)


def run_cmake(
    source_dir: Path,
    build_dir: Path,
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-GNinja", *args],
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


def run_build(
    build_dir: Path,
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(result.stdout)
    if not expect_success and result.returncode == 0:
        raise AssertionError("Build unexpectedly succeeded:\n" + result.stdout)
    return result


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def provider_state(
    *,
    protocol: int = 1,
    complete: bool = True,
    bool_type: str = "BOOL",
    bool_value: str = "1",
    integer_value: str = "-17",
    duplicate_bool: bool = False,
) -> str:
    names = "TEST_BOOL;TEST_INTEGER"
    if duplicate_bool:
        names += ";TEST_BOOL"
    completion = "set(ROCM_BUILD_FLAGS_STATE_COMPLETE 1)\n" if complete else ""
    return (
        f"set(ROCM_BUILD_FLAGS_PROTOCOL_VERSION {protocol})\n"
        'set(ROCM_BUILD_FLAGS_PROVIDER "TestProvider")\n'
        f'set(ROCM_BUILD_FLAGS_NAMES "{names}")\n'
        f'set(ROCM_BUILD_FLAG_TEST_BOOL_TYPE "{bool_type}")\n'
        f'set(ROCM_BUILD_FLAG_TEST_BOOL_VALUE "{bool_value}")\n'
        'set(ROCM_BUILD_FLAG_TEST_INTEGER_TYPE "INTEGER")\n'
        f'set(ROCM_BUILD_FLAG_TEST_INTEGER_VALUE "{integer_value}")\n'
        f"{completion}"
    )


class ROCmBuildFlagsTest(unittest.TestCase):
    def test_therock_registry_emits_typed_json_and_state(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.25)
                project(typed_registry NONE)
                set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
                include("{FLAG_UTILS_PATH.as_posix()}")
                therock_declare_flag(
                  NAME TEST_FALSE
                  TYPE BOOL
                  DEFAULT_VALUE OFF
                  DESCRIPTION "false"
                )
                therock_declare_flag(
                  NAME TEST_TRUE
                  DEFAULT_VALUE ON
                  DESCRIPTION "true"
                )
                therock_declare_flag(
                  NAME TEST_INTEGER
                  TYPE INTEGER
                  DEFAULT_VALUE -17
                  VALID_VALUES -17 5
                  DESCRIPTION "integer"
                )
                therock_finalize_flags()
                """,
            )

            run_cmake(
                source_dir,
                build_dir,
                "-DTHEROCK_FLAG_TEST_INTEGER=5",
            )

            settings = json.loads(
                (build_dir / "flag_settings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                settings,
                {"TEST_FALSE": False, "TEST_TRUE": True, "TEST_INTEGER": 5},
            )
            state = (build_dir / "rocm_build_flags_state.cmake").read_text(
                encoding="utf-8"
            )
            self.assertIn('set(ROCM_BUILD_FLAG_TEST_INTEGER_TYPE "INTEGER")', state)
            self.assertIn('set(ROCM_BUILD_FLAG_TEST_INTEGER_VALUE "5")', state)
            self.assertTrue(
                state.rstrip().endswith("set(ROCM_BUILD_FLAGS_STATE_COMPLETE 1)")
            )

    def test_standalone_project_cache_values_win(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(standalone_flags NONE)
                include("{MODULE_PATH.as_posix()}")
                rocm_resolve_build_flag(
                  NAME TEST_BOOL
                  TYPE BOOL
                  CACHE_VARIABLE PROJECT_BOOL
                  DEFAULT_VALUE OFF
                  DESCRIPTION "bool"
                  OUTPUT_VARIABLE resolved_bool
                )
                rocm_resolve_build_flag(
                  NAME TEST_INTEGER
                  TYPE INTEGER
                  CACHE_VARIABLE PROJECT_INTEGER
                  DEFAULT_VALUE -17
                  VALID_VALUES -17 5
                  DESCRIPTION "integer"
                  OUTPUT_VARIABLE resolved_integer
                )
                file(WRITE "${{CMAKE_BINARY_DIR}}/values.txt"
                  "${{resolved_bool}};${{resolved_integer}}")
                """,
            )

            run_cmake(
                source_dir,
                build_dir,
                "-DPROJECT_BOOL=YES",
                "-DPROJECT_INTEGER=5",
            )
            self.assertEqual(
                (build_dir / "values.txt").read_text(encoding="utf-8"), "1;5"
            )

    def test_integrated_provider_values_win(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            state_path = temp_dir / "state.cmake"
            state_path.write_text(provider_state(), encoding="utf-8")
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(integrated_flags NONE)
                set(ROCM_BUILD_FLAGS_STATE_FILE "{state_path.as_posix()}")
                include("{MODULE_PATH.as_posix()}")
                rocm_resolve_build_flag(
                  NAME TEST_BOOL
                  TYPE BOOL
                  CACHE_VARIABLE PROJECT_BOOL
                  DEFAULT_VALUE OFF
                  DESCRIPTION "bool"
                  OUTPUT_VARIABLE resolved_bool
                )
                rocm_resolve_build_flag(
                  NAME TEST_INTEGER
                  TYPE INTEGER
                  CACHE_VARIABLE PROJECT_INTEGER
                  DEFAULT_VALUE 5
                  VALID_VALUES -17 5
                  DESCRIPTION "integer"
                  OUTPUT_VARIABLE resolved_integer
                )
                file(WRITE "${{CMAKE_BINARY_DIR}}/values.txt"
                  "${{resolved_bool}};${{resolved_integer}}")
                """,
            )

            run_cmake(source_dir, build_dir)
            self.assertEqual(
                (build_dir / "values.txt").read_text(encoding="utf-8"), "1;-17"
            )

    def test_integrated_failures_are_eager(self):
        cases = {
            "protocol": (
                provider_state(protocol=2),
                "",
                "unsupported protocol",
            ),
            "incomplete": (
                provider_state(complete=False),
                "",
                "incomplete",
            ),
            "malformed_integer": (
                provider_state(integer_value="017"),
                "",
                "invalid INTEGER",
            ),
            "duplicate": (
                provider_state(duplicate_bool=True),
                "",
                "repeats flag",
            ),
            "type_mismatch": (
                provider_state(bool_type="INTEGER"),
                "",
                "expects type",
            ),
            "cache_collision": (
                provider_state(),
                "-DPROJECT_BOOL=ON",
                "already set",
            ),
        }
        for name, (state, extra_arg, expected_message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                source_dir = temp_dir / "source"
                build_dir = temp_dir / "build"
                state_path = temp_dir / "state.cmake"
                state_path.write_text(state, encoding="utf-8")
                write_file(
                    source_dir / "CMakeLists.txt",
                    f"""
                    cmake_minimum_required(VERSION 3.7)
                    project(invalid_integrated_flags NONE)
                    set(ROCM_BUILD_FLAGS_STATE_FILE "{state_path.as_posix()}")
                    include("{MODULE_PATH.as_posix()}")
                    rocm_resolve_build_flag(
                      NAME TEST_BOOL
                      TYPE BOOL
                      CACHE_VARIABLE PROJECT_BOOL
                      DEFAULT_VALUE OFF
                      DESCRIPTION "bool"
                      OUTPUT_VARIABLE resolved_bool
                    )
                    """,
                )
                args = [extra_arg] if extra_arg else []
                result = run_cmake(source_dir, build_dir, *args, expect_success=False)
                self.assertIn(expected_message, result.stdout)

    def test_missing_state_file_and_provider_flag_are_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            missing_state = temp_dir / "missing-state.cmake"
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(missing_state NONE)
                set(ROCM_BUILD_FLAGS_STATE_FILE "{missing_state.as_posix()}")
                include("{MODULE_PATH.as_posix()}")
                """,
            )
            result = run_cmake(
                source_dir,
                temp_dir / "missing-state-build",
                expect_success=False,
            )
            self.assertIn("state file does not exist", result.stdout)

            state_path = temp_dir / "state.cmake"
            state_path.write_text(provider_state(), encoding="utf-8")
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(missing_flag NONE)
                set(ROCM_BUILD_FLAGS_STATE_FILE "{state_path.as_posix()}")
                include("{MODULE_PATH.as_posix()}")
                rocm_resolve_build_flag(
                  NAME MISSING_FLAG
                  TYPE BOOL
                  CACHE_VARIABLE PROJECT_BOOL
                  DEFAULT_VALUE OFF
                  DESCRIPTION "bool"
                  OUTPUT_VARIABLE resolved_bool
                )
                """,
            )
            result = run_cmake(
                source_dir,
                temp_dir / "missing-flag-build",
                expect_success=False,
            )
            self.assertIn("does not define flag 'MISSING_FLAG'", result.stdout)

    def test_unknown_preprocessor_flag_fails_to_compile(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_file(
                source_dir / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(unknown_flag C)
                set(_canary_bool_false 0)
                set(_canary_bool_true 1)
                set(_canary_integer_negative -17)
                configure_file(
                  "{AUX_HEADER_TEMPLATE.as_posix()}"
                  "${{CMAKE_BINARY_DIR}}/aux_overlay_build_flags.h"
                  @ONLY
                )
                add_library(unknown_flag OBJECT unknown_flag.c)
                target_include_directories(unknown_flag PRIVATE "${{CMAKE_BINARY_DIR}}")
                """,
            )
            write_file(
                source_dir / "unknown_flag.c",
                """
                #include "aux_overlay_build_flags.h"
                _Static_assert(ROCM_BUILD_FLAG(MISSPELLED_FLAG) == 0, "unknown");
                """,
            )
            run_cmake(source_dir, build_dir)
            result = run_build(build_dir, expect_success=False)
            self.assertIn("MISSPELLED_FLAG", result.stdout)

    def test_installed_package_has_no_build_flag_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            producer_source = temp_dir / "producer"
            producer_build = temp_dir / "producer-build"
            install_dir = temp_dir / "install"
            consumer_source = temp_dir / "consumer"
            consumer_build = temp_dir / "consumer-build"

            write_file(
                producer_source / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.7)
                project(flagged C)
                include("{MODULE_PATH.as_posix()}")
                rocm_resolve_build_flag(
                  NAME TEST_BOOL
                  TYPE BOOL
                  CACHE_VARIABLE FLAGGED_TEST_BOOL
                  DEFAULT_VALUE ON
                  DESCRIPTION "bool"
                  OUTPUT_VARIABLE resolved_bool
                )
                configure_file(flagged_private.h.in
                  "${{CMAKE_BINARY_DIR}}/flagged_private.h" @ONLY)
                add_library(flagged STATIC flagged.c)
                target_include_directories(flagged
                  PUBLIC
                    "$<BUILD_INTERFACE:${{CMAKE_CURRENT_SOURCE_DIR}}/include>"
                    "$<INSTALL_INTERFACE:include>"
                  PRIVATE "${{CMAKE_BINARY_DIR}}"
                )
                install(TARGETS flagged EXPORT FlaggedTargets ARCHIVE DESTINATION lib)
                install(FILES include/flagged.h DESTINATION include)
                install(EXPORT FlaggedTargets
                  FILE FlaggedTargets.cmake
                  NAMESPACE Flagged::
                  DESTINATION lib/cmake/Flagged)
                install(FILES FlaggedConfig.cmake DESTINATION lib/cmake/Flagged)
                """,
            )
            write_file(
                producer_source / "flagged_private.h.in",
                """
                #define FLAGGED_TEST_BOOL @resolved_bool@
                """,
            )
            write_file(
                producer_source / "include" / "flagged.h",
                """
                int flagged_value(void);
                """,
            )
            write_file(
                producer_source / "flagged.c",
                """
                #include "flagged.h"
                #include "flagged_private.h"
                int flagged_value(void) { return FLAGGED_TEST_BOOL; }
                """,
            )
            write_file(
                producer_source / "FlaggedConfig.cmake",
                """
                include("${CMAKE_CURRENT_LIST_DIR}/FlaggedTargets.cmake")
                """,
            )

            run_cmake(
                producer_source,
                producer_build,
                f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            )
            run_build(producer_build)
            subprocess.run(
                ["cmake", "--install", str(producer_build)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            installed_files = [
                path for path in install_dir.rglob("*") if path.is_file()
            ]
            self.assertFalse(
                any(
                    path.name
                    in {
                        "ROCMBuildFlags.cmake",
                        "rocm_build_flags_state.cmake",
                    }
                    for path in installed_files
                )
            )
            installed_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in installed_files
            )
            self.assertNotIn("ROCMBuildFlags", installed_text)
            self.assertNotIn("ROCM_BUILD_FLAGS_STATE_FILE", installed_text)
            self.assertFalse(
                any("flagged_private.h" == path.name for path in installed_files)
            )

            shutil.rmtree(producer_source)
            shutil.rmtree(producer_build)
            write_file(
                consumer_source / "CMakeLists.txt",
                """
                cmake_minimum_required(VERSION 3.7)
                project(flagged_consumer C)
                find_package(Flagged CONFIG REQUIRED)
                add_executable(consumer main.c)
                target_link_libraries(consumer PRIVATE Flagged::flagged)
                """,
            )
            write_file(
                consumer_source / "main.c",
                """
                #include "flagged.h"
                int main(void) { return flagged_value() == 1 ? 0 : 1; }
                """,
            )
            run_cmake(
                consumer_source,
                consumer_build,
                f"-DCMAKE_PREFIX_PATH={install_dir}",
            )
            run_build(consumer_build)


if __name__ == "__main__":
    unittest.main()
