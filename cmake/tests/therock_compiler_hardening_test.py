#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused integration tests for TheRock's CMake hardening policy.

The fixtures cover policy validation, target-specific flags and exceptions,
retained-build transitions, nested-build interfaces, and resulting ELF state.
"""

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
HARDENING_MODULE = THEROCK_ROOT / "cmake" / "therock_compiler_hardening.cmake"
EXPAT_WRAPPER = THEROCK_ROOT / "third-party" / "sysdeps" / "common" / "expat"
LIBDRM_WRAPPER = THEROCK_ROOT / "third-party" / "sysdeps" / "linux" / "libdrm"
HIP_TEST_COMPILER = os.environ.get("THEROCK_TEST_HIP_COMPILER")
HIP_TEST_ARCHITECTURE = os.environ.get("THEROCK_TEST_HIP_ARCHITECTURE", "gfx906")
BUILD_ENV_FLAGS = ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "HIPFLAGS", "LDFLAGS")

GENERAL_COMPILE_FLAGS = (
    "-fstack-protector-strong",
    "-fstack-clash-protection",
    "-fcf-protection=full",
    "-Wformat",
    "-Wformat-security",
    "-U_FORTIFY_SOURCE",
    "-Wp,-U_FORTIFY_SOURCE",
)
FORTIFY_FLAG = "-Wp,-D_FORTIFY_SOURCE=3"
BASELINE_COMPILE_FLAGS = GENERAL_COMPILE_FLAGS + (FORTIFY_FLAG,)
BASELINE_LINK_FLAGS = (
    "-Wl,-z,relro",
    "-Wl,-z,now",
    "-Wl,-z,noexecstack",
)
HARDENED_SOURCE_NAMES = (
    "static.c",
    "object.c",
    "shared.cpp",
    "module.c",
    "main.cpp",
)
FIXTURE_SOURCE_NAMES = HARDENED_SOURCE_NAMES + (
    "fortify_disabled.c",
    "hardening_disabled.c",
)


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_fixture(source_dir: Path) -> None:
    """Writes a multi-target project with full and fortify-only exceptions."""
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)

        if(HARDENING_POLICY_ONLY)
          project(therock_hardening_policy_fixture NONE)
        else()
          project(therock_hardening_fixture C CXX)
        endif()

        include("{HARDENING_MODULE.as_posix()}")
        therock_initialize_hardening_policy()

        if(HARDENING_EXCLUSION_PROBE)
          foreach(_subproject
              amd-llvm
              amd-comgr
              hipcc
              hipify
              amd-llvm-tools
              ordinary-consumer)
            therock_is_hardening_excluded_subproject(
              _excluded "${{_subproject}}"
            )
            file(APPEND "${{CMAKE_BINARY_DIR}}/exclusions.txt"
              "${{_subproject}}=${{_excluded}}\n"
            )
          endforeach()
          return()
        endif()

        if(HARDENING_COMMAND_FLAGS_PROBE)
          therock_get_hardening_command_flags(
            _compile_flags
            _shared_link_flags
            _executable_compile_flags
            _executable_link_flags
            LANGUAGE C
            FORTIFY "${{COMMAND_FORTIFY}}"
          )
          file(WRITE "${{CMAKE_BINARY_DIR}}/command-flags.txt"
            "compile=${{_compile_flags}}\n"
            "shared_link=${{_shared_link_flags}}\n"
            "executable_compile=${{_executable_compile_flags}}\n"
            "executable_link=${{_executable_link_flags}}\n"
          )
          return()
        endif()

        if(HARDENING_POLICY_ONLY)
          return()
        endif()

        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
        add_library(fixture_static STATIC static.c)
        add_library(fixture_object OBJECT object.c)
        add_library(fixture_shared SHARED shared.cpp)
        add_library(fixture_module MODULE module.c)
        add_executable(fixture_exe main.cpp)
        target_link_libraries(
          fixture_exe PRIVATE fixture_static fixture_object
        )
        if(HARDENING_INTERFACE_PIC_OFF)
          add_library(fixture_no_pic_dependency INTERFACE)
          set_property(TARGET fixture_no_pic_dependency
            PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE OFF)
          target_link_libraries(fixture_exe PRIVATE fixture_no_pic_dependency)
        endif()
        if(HARDENING_PREEXISTING_FORTIFY)
          target_compile_options(fixture_static PRIVATE -D_FORTIFY_SOURCE=3)
          target_compile_definitions(fixture_static
            PRIVATE THEROCK_EXPECT_FORTIFY=3)
        endif()
        add_library(fixture_fortify_disabled STATIC fortify_disabled.c)
        set_property(TARGET fixture_fortify_disabled
          PROPERTY THEROCK_DISABLE_FORTIFY ON)
        if(DEFINED HARDENING_EXPECT_FORTIFY)
          foreach(_target fixture_static fixture_shared)
            target_compile_definitions(${{_target}} PRIVATE
              THEROCK_EXPECT_FORTIFY=${{HARDENING_EXPECT_FORTIFY}})
          endforeach()
          target_compile_definitions(fixture_fortify_disabled PRIVATE
            THEROCK_EXPECT_FORTIFY=0)
          foreach(_target fixture_static fixture_shared fixture_fortify_disabled)
            target_compile_options(${{_target}} PRIVATE
              -Wp,-U_FORTIFY_SOURCE -Wp,-D_FORTIFY_SOURCE=2)
          endforeach()
        endif()
        add_executable(fixture_hardening_disabled hardening_disabled.c)
        set_property(TARGET fixture_hardening_disabled
          PROPERTY THEROCK_DISABLE_HARDENING ON)

        if(HARDENING_POSITION_INDEPENDENT_CODE_OFF)
          include(CheckPIESupported)
          check_pie_supported(LANGUAGES CXX)
          set_property(TARGET fixture_exe
            PROPERTY POSITION_INDEPENDENT_CODE OFF)
        endif()

        set(_fixture_targets
          fixture_static
          fixture_object
          fixture_shared
          fixture_module
          fixture_exe
          fixture_fortify_disabled
          fixture_hardening_disabled)
        if(STRICT_TOOLCHAIN_CHECKS)
          function(reject_compiler_flag_changes variable access value file)
            if(access STREQUAL "MODIFIED_ACCESS")
              message(FATAL_ERROR "Probe modified watched variable ${{variable}}")
            endif()
          endfunction()
          variable_watch(CMAKE_C_FLAGS reject_compiler_flag_changes)
          variable_watch(CMAKE_CXX_FLAGS reject_compiler_flag_changes)
        endif()
        therock_apply_hardening_to_targets(
          ${{_fixture_targets}}
        )
        """,
    )
    sources = {
        "static.c": (
            '#include "verify_fortify.h"\n'
            "int fixture_static_global = 0;\n"
            "int *fixture_static(void) { return &fixture_static_global; }\n"
        ),
        "object.c": (
            "int fixture_object_global = 0;\n"
            "int *fixture_object(void) { return &fixture_object_global; }\n"
        ),
        "shared.cpp": '#include "verify_fortify.h"\n'
        'extern "C" int fixture_shared(void) { return 0; }\n',
        "module.c": "int fixture_module(void) { return 0; }\n",
        "main.cpp": 'extern "C" int *fixture_static(void);\n'
        'extern "C" int *fixture_object(void);\n'
        "int main() { return *fixture_static() + *fixture_object(); }\n",
        "fortify_disabled.c": '#include "verify_fortify.h"\n'
        "int fortify_disabled(void) { return 0; }\n",
        "hardening_disabled.c": "int main(void) { return 0; }\n",
    }
    for name, contents in sources.items():
        write_file(source_dir / name, contents)
    write_file(
        source_dir / "verify_fortify.h",
        """
        #ifdef THEROCK_EXPECT_FORTIFY
        #include <features.h>
        #if __USE_FORTIFY_LEVEL != THEROCK_EXPECT_FORTIFY
        #error "effective fortify level does not match the policy"
        #endif
        #endif
        """,
    )


def write_project_include_fixture(source_dir: Path) -> None:
    """Writes a project that relies entirely on the project-include entry point."""
    write_file(
        source_dir / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(therock_hardening_project_include_fixture C)
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
        add_executable(project_include_exe main.c)
        """,
    )
    write_file(source_dir / "main.c", "int main(void) { return 0; }\n")


def write_legacy_policy_fixture(source_dir: Path) -> None:
    """Writes a project whose policy version predates the IN_LIST operator."""
    write_file(
        source_dir / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.2)
        project(therock_hardening_legacy_policy_fixture NONE)
        therock_initialize_hardening_policy()
        """,
    )


def write_child_language_fixture(source_dir: Path, language: str) -> None:
    """Writes a child with its own language and optional compiler flags."""
    suffix = {"C": "c", "CXX": "cpp", "HIP": "hip"}[language]
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_hardening_scope_parent C)
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
        include("{HARDENING_MODULE.as_posix()}")
        therock_initialize_hardening_policy()
        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        add_library(scope_parent STATIC parent.c)
        add_subdirectory(child)
        therock_schedule_hardening()
        """,
    )
    write_file(source_dir / "parent.c", "int scope_parent(void) { return 0; }\n")
    write_file(
        source_dir / "child" / "CMakeLists.txt",
        f"""
        project(therock_hardening_scope_child {language})
        set(CMAKE_{language}_FLAGS "${{CMAKE_{language}_FLAGS}} ${{CHILD_EXTRA_FLAGS}}")
        add_library(scope_child STATIC child.{suffix})
        """,
    )
    write_file(
        source_dir / "child" / f"child.{suffix}",
        "int scope_child(void) { return 0; }\n",
    )


def run_command(
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    # Ordinary fixtures must not inherit developer build flags. Inheritance
    # tests supply their inputs explicitly through cmake -E env instead.
    environment = {
        name: value for name, value in os.environ.items() if name not in BUILD_ENV_FLAGS
    }
    result = subprocess.run(
        args,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    command = " ".join(args)
    if expect_success and result.returncode != 0:
        raise AssertionError(f"Command failed: {command}\n{result.stdout}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            f"Command unexpectedly succeeded: {command}\n{result.stdout}"
        )
    return result


def configure(
    source_dir: Path,
    build_dir: Path,
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-GNinja",
        *args,
        expect_success=expect_success,
    )


def configure_fixture(
    source_dir: Path,
    build_dir: Path,
    *args: str,
    policy: str = "BASELINE",
    build_type: str = "Release",
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    return configure(
        source_dir,
        build_dir,
        f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
        f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DTHEROCK_HARDENING_POLICY={policy}",
        *args,
        expect_success=expect_success,
    )


def build(build_dir: Path) -> subprocess.CompletedProcess[str]:
    return run_command("cmake", "--build", str(build_dir), "--verbose")


def read_compile_commands(
    build_dir: Path,
    expected_source_names: tuple[str, ...] = FIXTURE_SOURCE_NAMES,
) -> dict[str, str]:
    """Returns one compile command for every expected fixture source."""
    commands_path = build_dir / "compile_commands.json"
    entries = json.loads(commands_path.read_text(encoding="utf-8"))
    commands = {Path(entry["file"]).name: entry["command"] for entry in entries}
    if len(entries) != len(expected_source_names) or set(commands) != set(
        expected_source_names
    ):
        raise AssertionError(f"Unexpected compile command sources: {sorted(commands)}")
    return commands


def preprocess(*compiler_args: str) -> dict[str, str]:
    """Reads effective macros after the compiler has processed all flag forms."""
    output = run_command(
        *compiler_args, "-dM", "-E", "-x", "c", os.devnull, "-include", "features.h"
    ).stdout
    return dict(re.findall(r"^#define (\w+) (.*)$", output, re.MULTILINE))


def assert_flags_present(
    test_case: unittest.TestCase,
    command: str,
    flags: tuple[str, ...],
) -> None:
    arguments = shlex.split(command)
    for flag in flags:
        test_case.assertIn(flag, arguments)


def assert_flags_absent(
    test_case: unittest.TestCase,
    command: str,
    flags: tuple[str, ...],
) -> None:
    arguments = shlex.split(command)
    for flag in flags:
        test_case.assertNotIn(flag, arguments)


def find_output_line(output: str, artifact_name: str) -> str:
    """Finds one verbose link command by its output artifact."""
    for line in output.splitlines():
        if f"-o {artifact_name}" in line:
            return line
    raise AssertionError(
        f"No verbose link command found for {artifact_name}:\n{output}"
    )


def read_meson_options(build_dir: Path) -> dict[str, str]:
    """Extracts compiler and linker options from the libdrm custom command."""
    command = run_command(
        "ninja", "-C", str(build_dir), "-t", "commands", "meson_build"
    ).stdout
    tokens = shlex.split(command)
    options = {}
    for name in ("c_args", "cpp_args", "c_link_args", "cpp_link_args"):
        prefix = f"-D{name}="
        values = [
            token.removeprefix(prefix) for token in tokens if token.startswith(prefix)
        ]
        if len(values) != 1:
            raise AssertionError(f"Expected one {name} option in:\n{command}")
        options[name] = values[0]
    return options


class FlagAssertionsTest(unittest.TestCase):
    def test_flags_are_matched_as_arguments(self) -> None:
        """A longer flag must not satisfy or invalidate a shorter flag check."""
        command = "cc -Wformat-security"
        with self.assertRaises(AssertionError):
            assert_flags_present(self, command, ("-Wformat",))
        assert_flags_absent(self, command, ("-Wformat",))
        assert_flags_present(self, command, ("-Wformat-security",))
        with self.assertRaises(AssertionError):
            assert_flags_absent(self, command, ("-Wformat-security",))


class TemporaryProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = Path(temp_dir.name)
        self.source_dir = self.temp_dir / "source"
        self.build_dir = self.temp_dir / "build"


@unittest.skipUnless(
    platform.system() == "Linux"
    and platform.machine() == "x86_64"
    and shutil.which("gcc") is not None
    and shutil.which("g++") is not None
    and shutil.which("readelf") is not None,
    "BASELINE hardening is initially supported on Linux x86_64 with GNU/Clang",
)
class HardeningBuildTest(TemporaryProjectTest):
    """Exercises generated flags, rebuild behavior, and ELF properties."""

    def _build_child_language(self, language: str, *args: str) -> str:
        """Builds the scoped-language fixture and returns its child command."""
        write_child_language_fixture(self.source_dir, language)
        configure(
            self.source_dir,
            self.build_dir,
            f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_HARDENING_POLICY=BASELINE",
            *args,
        )
        build(self.build_dir)
        child_source = "child.hip" if language == "HIP" else "child.cpp"
        commands = read_compile_commands(self.build_dir, ("parent.c", child_source))
        assert_flags_present(
            self, commands["parent.c"], BASELINE_COMPILE_FLAGS + ("-fPIC",)
        )
        return commands[child_source]

    def test_child_language_compiler_is_validated_in_its_scope(self) -> None:
        """Checks a language enabled only by a child project."""
        command = self._build_child_language(
            "CXX", f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}"
        )
        assert_flags_present(self, command, BASELINE_COMPILE_FLAGS + ("-fPIC",))

    @unittest.skipUnless(
        HIP_TEST_COMPILER,
        "set THEROCK_TEST_HIP_COMPILER to exercise child-only HIP scope",
    )
    def test_child_hip_compiler_is_validated_in_its_scope(self) -> None:
        """Checks a HIP language enabled only by a child project."""
        command = self._build_child_language(
            "HIP",
            f"-DCMAKE_HIP_COMPILER={HIP_TEST_COMPILER}",
            f"-DCMAKE_HIP_ARCHITECTURES={HIP_TEST_ARCHITECTURE}",
        )
        arguments = shlex.split(command)
        host_flags = {
            flag
            for prefix, flag in zip(arguments, arguments[1:])
            if prefix == "-Xarch_host"
        }
        for flag in BASELINE_COMPILE_FLAGS + ("-fPIC",):
            self.assertIn(flag, host_flags)

    def test_expat_off_and_excluded_preserve_inherited_fortify(self) -> None:
        """Checks that inactive policy paths preserve inherited wrapper flags."""
        self.source_dir.mkdir()
        for policy, excluded in (("OFF", "FALSE"), ("BASELINE", "TRUE")):
            with self.subTest(policy=policy, excluded=excluded):
                build_dir = self.temp_dir / f"build-{policy.lower()}"
                run_command(
                    "cmake",
                    "-E",
                    "env",
                    "CFLAGS=-D_FORTIFY_SOURCE=2 -O2",
                    "--",
                    "cmake",
                    "-S",
                    str(EXPAT_WRAPPER),
                    "-B",
                    str(build_dir),
                    "-GNinja",
                    f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
                    f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={HARDENING_MODULE}",
                    f"-DTHEROCK_SOURCE_DIR={THEROCK_ROOT}",
                    f"-DTHEROCK_HARDENING_POLICY={policy}",
                    f"-DTHEROCK_HARDENING_EXCLUDED={excluded}",
                    f"-DSOURCE_DIR={self.source_dir}",
                    "-DPATCHELF=patchelf",
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    "-DEXTRA_CPPFLAGS=-U_FORTIFY_SOURCE -DEXTRA=1",
                )
                ninja = (build_dir / "build.ninja").read_text(encoding="utf-8")
                self.assertIn(
                    r"CFLAGS=-D_FORTIFY_SOURCE=2\ -O2",
                    ninja,
                )
                self.assertIn(
                    r"CPPFLAGS=-U_FORTIFY_SOURCE\ -DEXTRA=1",
                    ninja,
                )

    def test_nested_build_command_flags(self) -> None:
        """Checks the flag interface used by non-CMake and nested-build wrappers."""
        write_fixture(self.source_dir)
        for fortify in ("ON", "OFF"):
            with self.subTest(fortify=fortify):
                build_dir = self.temp_dir / fortify.lower()
                configure_fixture(
                    self.source_dir,
                    build_dir,
                    "-DHARDENING_COMMAND_FLAGS_PROBE=ON",
                    f"-DCOMMAND_FORTIFY={fortify}",
                )
                flags = dict(
                    line.split("=", 1)
                    for line in (build_dir / "command-flags.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                for name in ("compile", "executable_compile"):
                    assert_flags_present(self, flags[name], GENERAL_COMPILE_FLAGS)
                    self.assertEqual(
                        FORTIFY_FLAG in shlex.split(flags[name]), fortify == "ON"
                    )
                for name in ("shared_link", "executable_link"):
                    assert_flags_present(self, flags[name], BASELINE_LINK_FLAGS)
                assert_flags_present(self, flags["executable_compile"], ("-fPIE",))
                assert_flags_present(self, flags["executable_link"], ("-pie",))

    def test_pic_conflicts_are_rejected(self) -> None:
        """Checks explicit and dependency-provided non-PIE requirements."""
        write_fixture(self.source_dir)
        cases = {
            "POSITION_INDEPENDENT_CODE_OFF": "THEROCK_DISABLE_HARDENING",
            "INTERFACE_PIC_OFF": "INTERFACE_POSITION_INDEPENDENT_CODE",
        }
        for setting, diagnostic in cases.items():
            with self.subTest(setting=setting):
                result = configure_fixture(
                    self.source_dir,
                    self.temp_dir / setting.lower(),
                    f"-DHARDENING_{setting}=ON",
                    expect_success=False,
                )
                self.assertIn("POSITION_INDEPENDENT_CODE", result.stdout)
                self.assertIn(diagnostic, result.stdout)

    def test_static_and_object_code_is_compatible_with_pie(self) -> None:
        """Checks that non-PIE input flags cannot leave linked objects non-PIC."""
        write_fixture(self.source_dir)
        configure_fixture(
            self.source_dir,
            self.build_dir,
            "-DCMAKE_C_FLAGS=-fno-pie",
            "-DCMAKE_CXX_FLAGS=-fno-pie",
        )
        build_result = build(self.build_dir)
        commands = read_compile_commands(self.build_dir)
        for source_name in ("static.c", "object.c"):
            command = commands[source_name]
            self.assertLess(command.rfind("-fno-pie"), command.rfind("-fPIC"))
        self.assertLess(
            commands["main.cpp"].rfind("-fno-pie"),
            commands["main.cpp"].rfind("-fPIE"),
        )
        link = find_output_line(build_result.stdout, "fixture_exe")
        assert_flags_present(self, link, ("-pie",))
        assert_flags_absent(self, link, ("-no-pie",))
        self._assert_hardened_elf(self.build_dir / "fixture_exe", expect_pie=True)

    def test_cmake_project_include_applies_deferred_hardening(self) -> None:
        """Checks that the nested-CMake entry point hardens later-created targets."""
        write_project_include_fixture(self.source_dir)
        configure(
            self.source_dir,
            self.build_dir,
            f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_HARDENING_POLICY=BASELINE",
            f"-DTHEROCK_SOURCE_DIR={THEROCK_ROOT}",
            "-DTHEROCK_COMPILER_HARDENING_PROJECT_INCLUDE=ON",
            f"-DCMAKE_PROJECT_INCLUDE={HARDENING_MODULE}",
        )
        build_result = build(self.build_dir)
        compile_command = read_compile_commands(self.build_dir, ("main.c",))["main.c"]
        assert_flags_present(self, compile_command, BASELINE_COMPILE_FLAGS + ("-fPIE",))
        link_command = find_output_line(build_result.stdout, "project_include_exe")
        assert_flags_present(self, link_command, BASELINE_LINK_FLAGS + ("-pie",))

    def test_existing_fortify_definition_cannot_displace_policy_reset(self) -> None:
        """Checks that CMake keeps the policy reset and definition together."""
        write_fixture(self.source_dir)
        configure_fixture(
            self.source_dir,
            self.build_dir,
            "-DHARDENING_PREEXISTING_FORTIFY=ON",
        )
        build(self.build_dir)
        command = read_compile_commands(self.build_dir)["static.c"]
        self.assertLess(
            command.rfind("-U_FORTIFY_SOURCE"),
            command.rfind("-D_FORTIFY_SOURCE=3"),
        )

    def test_effective_fortify_level_three_is_required(self) -> None:
        """Rejects headers that reduce effective fortification to level 2.

        Child-only header-search flags must reach the probe. Checking the
        parent's headers could accept a child that cannot satisfy the policy.
        """
        fake_include_dir = self.temp_dir / "fake-include"
        write_child_language_fixture(self.source_dir, "C")
        write_file(
            fake_include_dir / "features.h",
            """
            #include_next <features.h>
            #undef __USE_FORTIFY_LEVEL
            #define __USE_FORTIFY_LEVEL 2
            """,
        )
        include_flag = f"-I{fake_include_dir}"
        for scope, variable in (
            ("root", "CMAKE_C_FLAGS"),
            ("child", "CHILD_EXTRA_FLAGS"),
        ):
            with self.subTest(scope=scope):
                result = configure_fixture(
                    self.source_dir,
                    self.temp_dir / scope,
                    f"-D{variable}={include_flag}",
                    expect_success=False,
                )
                self.assertIn("requires effective _FORTIFY_SOURCE=3", result.stdout)

        # Debug-only generators never enable fortify. A multi-config build
        # containing Release still needs capability validation at configure.
        for generator, configurations in (
            ("Ninja", "Debug"),
            ("Ninja", "Release"),
            ("Ninja Multi-Config", "Debug"),
            ("Ninja Multi-Config", "Debug;Release"),
        ):
            with self.subTest(generator=generator, configurations=configurations):
                option = (
                    "CMAKE_CONFIGURATION_TYPES"
                    if generator == "Ninja Multi-Config"
                    else "CMAKE_BUILD_TYPE"
                )
                result = configure_fixture(
                    self.source_dir,
                    self.temp_dir / f"{generator}-{configurations.replace(';', '-')}",
                    f"-G{generator}",
                    f"-D{option}={configurations}",
                    f"-DCMAKE_C_FLAGS={include_flag}",
                    expect_success="Release" not in configurations,
                )
                if "Release" in configurations:
                    self.assertIn("requires effective _FORTIFY_SOURCE=3", result.stdout)

        # An explicit bridge request describes the inner build, so an outer
        # Debug configuration must not bypass its capability check.
        write_fixture(self.source_dir)
        result = configure_fixture(
            self.source_dir,
            self.temp_dir / "optimized-inner-build",
            "-DHARDENING_COMMAND_FLAGS_PROBE=ON",
            "-DCOMMAND_FORTIFY=ON",
            f"-DCMAKE_C_FLAGS={include_flag}",
            build_type="Debug",
            expect_success=False,
        )
        self.assertIn("requires effective _FORTIFY_SOURCE=3", result.stdout)

    def test_fortify_probe_preserves_watched_compiler_flags(self) -> None:
        """Keeps capability probes compatible with ROCMChecks' variable guards.

        Assigning even an unchanged compiler-flags variable triggers its
        watcher. Probes must use the target's flags without assigning them
        in the calling project.
        """
        write_fixture(self.source_dir)
        for policy in ("OFF", "BASELINE"):
            with self.subTest(policy=policy):
                configure_fixture(
                    self.source_dir,
                    self.temp_dir / policy,
                    "-DSTRICT_TOOLCHAIN_CHECKS=ON",
                    policy=policy,
                )

    def test_libdrm_preserves_inherited_compiler_flags(self) -> None:
        """Preserves user flags through OFF -> BASELINE -> OFF transitions.

        Explicit Meson options replace its environment-derived defaults.
        Initial compiler and preprocessor flags must survive reconfiguration
        without that environment, while policy flags follow the selected mode.
        """
        self.source_dir.mkdir()
        configure_args = (
            "cmake",
            "-S",
            str(LIBDRM_WRAPPER),
            "-B",
            str(self.build_dir),
            "-GNinja",
            f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
            f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}",
            f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={HARDENING_MODULE}",
            f"-DTHEROCK_SOURCE_DIR={THEROCK_ROOT}",
            f"-DSOURCE_DIR={self.source_dir}",
            "-DPATCHELF=patchelf",
            "-DMESON_BUILD=meson",
        )
        run_command(
            "cmake",
            "-E",
            "env",
            "CFLAGS=-DUSER_CFLAGS=1 -Wp,-D_FORTIFY_SOURCE=2",
            "CXXFLAGS=-DUSER_CXXFLAGS=1 -Wp,-D_FORTIFY_SOURCE=2",
            "CPPFLAGS=-DUSER_CPPFLAGS=1 -D_FORTIFY_SOURCE=2",
            "--",
            *configure_args,
            "-DTHEROCK_HARDENING_POLICY=OFF",
        )
        for index, policy in enumerate(("OFF", "BASELINE", "OFF")):
            if index:
                run_command(
                    *configure_args,
                    f"-DTHEROCK_HARDENING_POLICY={policy}",
                )
            options = read_meson_options(self.build_dir)
            for name in ("c_args", "c_link_args"):
                assert_flags_present(self, options[name], ("-DUSER_CFLAGS=1",))
            for name in ("cpp_args", "cpp_link_args"):
                assert_flags_present(self, options[name], ("-DUSER_CXXFLAGS=1",))
            for name in ("c_args", "cpp_args"):
                assert_flags_present(self, options[name], ("-DUSER_CPPFLAGS=1",))
                if policy == "BASELINE":
                    assert_flags_present(self, options[name], BASELINE_COMPILE_FLAGS)
                    self.assertLess(
                        options[name].index("-DUSER_CPPFLAGS=1"),
                        options[name].index("-U_FORTIFY_SOURCE"),
                    )
                else:
                    assert_flags_absent(self, options[name], BASELINE_COMPILE_FLAGS)
                compiler = "gcc" if name == "c_args" else "g++"
                macros = preprocess(compiler, "-O2", *shlex.split(options[name]))
                self.assertEqual(
                    macros["__USE_FORTIFY_LEVEL"], "3" if policy == "BASELINE" else "2"
                )
            for name in ("c_link_args", "cpp_link_args"):
                assert_flags_absent(self, options[name], ("-DUSER_CPPFLAGS=1",))
                if policy == "BASELINE":
                    assert_flags_present(self, options[name], BASELINE_LINK_FLAGS)
                else:
                    assert_flags_absent(self, options[name], BASELINE_LINK_FLAGS)

    def test_fortify_configuration_sanitizer_and_exceptions(self) -> None:
        """Checks effective fortify with inherited forwarded definitions.

        Flag presence alone misses driver reordering: -Wp definitions can win
        over plain -U/-D. Compiled assertions cover enablement and resets for
        Debug, sanitizers, and the target exception in both C and C++.
        A preexisting target reset followed by a definition also catches CMake
        deduplicating away the policy's final reset on suppression paths.
        """
        write_fixture(self.source_dir)
        cases = {
            ("Release", ""): True,
            ("RelWithDebInfo", ""): True,
            ("MinSizeRel", ""): True,
            ("Debug", ""): False,
            ("Release", "ASAN"): False,
        }
        for (build_type, sanitizer), expect_fortify in cases.items():
            with self.subTest(build_type=build_type, sanitizer=sanitizer):
                build_dir = self.temp_dir / f"{build_type}-{sanitizer or 'none'}"
                configure_fixture(
                    self.source_dir,
                    build_dir,
                    f"-DTHEROCK_SANITIZER={sanitizer}",
                    f"-DHARDENING_EXPECT_FORTIFY={3 if expect_fortify else 0}",
                    "-DCMAKE_C_FLAGS=-O2 -Wp,-D_FORTIFY_SOURCE=2",
                    "-DCMAKE_CXX_FLAGS=-O2 -Wp,-D_FORTIFY_SOURCE=2",
                    build_type=build_type,
                )
                commands = read_compile_commands(build_dir)
                for source_name in HARDENED_SOURCE_NAMES:
                    command = commands[source_name]
                    assert_flags_present(self, command, GENERAL_COMPILE_FLAGS)
                    self.assertEqual(
                        FORTIFY_FLAG in shlex.split(command), expect_fortify
                    )

                fortify_disabled = commands["fortify_disabled.c"]
                assert_flags_present(self, fortify_disabled, GENERAL_COMPILE_FLAGS)
                assert_flags_absent(self, fortify_disabled, (FORTIFY_FLAG,))
                assert_flags_absent(
                    self,
                    commands["hardening_disabled.c"],
                    BASELINE_COMPILE_FLAGS,
                )

                build_result = build(build_dir)
                if sanitizer:
                    assert_flags_present(self, commands["main.cpp"], ("-fPIE",))
                    link = find_output_line(build_result.stdout, "fixture_exe")
                    assert_flags_present(self, link, BASELINE_LINK_FLAGS + ("-pie",))

    def test_off_preserves_explicit_user_hardening_flags(self) -> None:
        """OFF preserves user choices, including flags also used by BASELINE."""
        write_fixture(self.source_dir)
        configure_fixture(
            self.source_dir,
            self.build_dir,
            "-DCMAKE_C_FLAGS=-fstack-protector-strong",
            policy="OFF",
        )
        command = read_compile_commands(self.build_dir)["static.c"]
        assert_flags_present(self, command, ("-fstack-protector-strong",))
        assert_flags_absent(self, command, (FORTIFY_FLAG, "-Wp,-U_FORTIFY_SOURCE"))

    def test_expat_policy_follows_all_user_preprocessor_options(self) -> None:
        """Tests actual compiler commands with autotools' late user flags.

        Bundled -Wp options must retain unrelated definitions while the final
        policy selects level 3 or resets fortification. The outer wrapper uses
        Debug; its inner build's optimization determines fortify applicability.
        """
        self.source_dir.mkdir()
        cflags = "-O2 -D_FORTIFY_SOURCE=2 -Wp,-D_FORTIFY_SOURCE=2,-DC_KEEP=1"
        cppflags = "-Wp,-U_FORTIFY_SOURCE,-D_FORTIFY_SOURCE=2,-DCPP_KEEP=1"
        for sanitizer, level in (("", "3"), ("ASAN", "0")):
            with self.subTest(sanitizer=sanitizer):
                build_dir = self.temp_dir / f"{sanitizer or 'optimized'} build"
                run_command(
                    "cmake",
                    "-E",
                    "env",
                    f"CFLAGS={cflags}",
                    "--",
                    "cmake",
                    "-S",
                    str(EXPAT_WRAPPER),
                    "-B",
                    str(build_dir),
                    "-GNinja",
                    f"-DCMAKE_C_COMPILER={shutil.which('gcc')}",
                    f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={HARDENING_MODULE}",
                    f"-DTHEROCK_SOURCE_DIR={THEROCK_ROOT}",
                    "-DTHEROCK_HARDENING_POLICY=BASELINE",
                    f"-DTHEROCK_SANITIZER={sanitizer}",
                    "-DCMAKE_BUILD_TYPE=Debug",
                    f"-DSOURCE_DIR={self.source_dir}",
                    "-DPATCHELF=patchelf",
                    f"-DEXTRA_CPPFLAGS={cppflags}",
                )
                commands = run_command(
                    "ninja", "-C", str(build_dir), "-t", "commands", "build"
                ).stdout
                arguments = shlex.split(commands)
                build_path = next(arg for arg in arguments if arg.startswith("PATH="))
                compilers = [
                    arg.removeprefix("CC=")
                    for arg in arguments
                    if arg.startswith("CC=")
                ]
                self.assertEqual(len(compilers), 2)
                for compiler in compilers:
                    macros = preprocess(
                        "cmake",
                        "-E",
                        "env",
                        build_path,
                        "--",
                        *shlex.split(compiler),
                        *shlex.split(cppflags),
                        *shlex.split(cflags),
                    )
                    self.assertEqual(macros["__USE_FORTIFY_LEVEL"], level)
                    self.assertEqual(macros["C_KEEP"], "1")
                    self.assertEqual(macros["CPP_KEEP"], "1")

    def test_off_baseline_off_in_the_same_build_tree(self) -> None:
        """Checks policy reversal in retained CMake and Ninja state and artifacts."""
        write_fixture(self.source_dir)

        # Establish the unmodified command state in the retained build tree.
        configure_fixture(self.source_dir, self.build_dir, policy="OFF")
        build(self.build_dir)
        off_commands = read_compile_commands(self.build_dir)
        for command in off_commands.values():
            assert_flags_absent(self, command, BASELINE_COMPILE_FLAGS + ("-fPIE",))

        # Enable BASELINE and verify compile and target-type distinctions.
        configure_fixture(self.source_dir, self.build_dir)
        baseline_build = build(self.build_dir)
        baseline_commands = read_compile_commands(self.build_dir)
        for source_name in HARDENED_SOURCE_NAMES:
            command = baseline_commands[source_name]
            assert_flags_present(self, command, BASELINE_COMPILE_FLAGS)
            self.assertLess(
                command.rfind("-U_FORTIFY_SOURCE"),
                command.rfind("-D_FORTIFY_SOURCE=3"),
            )
        assert_flags_present(self, baseline_commands["main.cpp"], ("-fPIE",))
        for source_name in ("static.c", "object.c"):
            assert_flags_present(self, baseline_commands[source_name], ("-fPIC",))
        for source_name in (
            "static.c",
            "object.c",
            "shared.cpp",
            "module.c",
            "hardening_disabled.c",
        ):
            assert_flags_absent(self, baseline_commands[source_name], ("-fPIE",))

        # Link hardening applies to linked artifacts; PIE applies only to the
        # executable and the full target exception suppresses both.
        for artifact, expect_pie in (
            ("fixture_exe", True),
            ("libfixture_shared.so", False),
            ("libfixture_module.so", False),
        ):
            with self.subTest(artifact=artifact):
                link = find_output_line(baseline_build.stdout, artifact)
                assert_flags_present(self, link, BASELINE_LINK_FLAGS)
                self.assertEqual("-pie" in shlex.split(link), expect_pie)
                self._assert_hardened_elf(
                    self.build_dir / artifact, expect_pie=expect_pie
                )
        disabled_commands = run_command(
            "ninja",
            "-C",
            str(self.build_dir),
            "-t",
            "commands",
            "fixture_hardening_disabled",
        ).stdout
        disabled_link = find_output_line(
            disabled_commands, "fixture_hardening_disabled"
        )
        assert_flags_absent(self, disabled_link, BASELINE_LINK_FLAGS + ("-pie",))

        # Returning to OFF must invalidate and remove retained policy flags.
        configure_fixture(self.source_dir, self.build_dir, policy="OFF")
        final_off_build = build(self.build_dir)
        self.assertEqual(read_compile_commands(self.build_dir), off_commands)

        # Both policy changes must invalidate the compile and link commands.
        self.assertIn("main.cpp", baseline_build.stdout)
        self.assertIn("main.cpp", final_off_build.stdout)
        assert_flags_absent(
            self,
            final_off_build.stdout,
            BASELINE_COMPILE_FLAGS + BASELINE_LINK_FLAGS + ("-fPIE", "-pie"),
        )

    def _assert_hardened_elf(self, artifact: Path, *, expect_pie: bool) -> None:
        """Checks ELF properties produced by the link policy."""
        program_headers = run_command("readelf", "-lW", str(artifact)).stdout
        self.assertIn("GNU_RELRO", program_headers)
        stack_line = next(
            (line for line in program_headers.splitlines() if "GNU_STACK" in line),
            None,
        )
        self.assertIsNotNone(stack_line)
        self.assertRegex(stack_line or "", r"\sRW\s")
        self.assertNotRegex(stack_line or "", r"\sRWE\s")

        dynamic = run_command("readelf", "-dW", str(artifact)).stdout
        self.assertRegex(dynamic, r"BIND_NOW|FLAGS.*NOW")

        if expect_pie:
            elf_header = run_command("readelf", "-hW", str(artifact)).stdout
            self.assertRegex(elf_header, r"Type:\s+DYN")


class HardeningPolicyTest(TemporaryProjectTest):
    """Exercises cache defaults, accepted policy values, and scope checks."""

    def _configure_policy(
        self,
        *args: str,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Configures the fixture without enabling compiler languages or targets."""
        write_fixture(self.source_dir)
        return configure(
            self.source_dir,
            self.build_dir,
            "-DHARDENING_POLICY_ONLY=ON",
            *args,
            expect_success=expect_success,
        )

    def test_default_policy_is_off(self) -> None:
        self._configure_policy()
        cache = (self.build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
        self.assertIn("THEROCK_HARDENING_POLICY:STRING=OFF", cache)

    def test_module_sets_policy_required_by_legacy_subproject(self) -> None:
        """The module must not inherit an old subproject's policy settings."""
        write_legacy_policy_fixture(self.source_dir)
        configure(
            self.source_dir,
            self.build_dir,
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={HARDENING_MODULE}",
        )

    def test_compiler_subproject_exclusions_are_exact(self) -> None:
        self._configure_policy("-DHARDENING_EXCLUSION_PROBE=ON")
        exclusions = (self.build_dir / "exclusions.txt").read_text(encoding="utf-8")
        self.assertEqual(
            exclusions.splitlines(),
            [
                "amd-llvm=TRUE",
                "amd-comgr=TRUE",
                "hipcc=TRUE",
                "hipify=TRUE",
                "amd-llvm-tools=FALSE",
                "ordinary-consumer=FALSE",
            ],
        )

    def test_strict_is_reserved(self) -> None:
        result = self._configure_policy(
            "-DTHEROCK_HARDENING_POLICY=STRICT",
            expect_success=False,
        )
        self.assertIn("STRICT", result.stdout)
        self.assertRegex(
            result.stdout.lower(),
            r"not (implemented|supported)|unimplemented|planned",
        )

    def test_invalid_policy_is_rejected(self) -> None:
        result = self._configure_policy(
            "-DTHEROCK_HARDENING_POLICY=INVALID",
            expect_success=False,
        )
        for expected_text in ("INVALID", "OFF", "BASELINE", "STRICT"):
            self.assertIn(expected_text, result.stdout)

    def test_baseline_rejects_unsupported_platform(self) -> None:
        result = self._configure_policy(
            "-DTHEROCK_HARDENING_POLICY=BASELINE",
            "-DCMAKE_SYSTEM_NAME=Darwin",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
            expect_success=False,
        )
        self.assertIn("BASELINE", result.stdout)
        self.assertRegex(result.stdout, re.compile(r"Linux|unsupported", re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
