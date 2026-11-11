#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise argument fingerprints using real sub-project activation/configuration."""

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ("args_source", "args_consumer", "args_control")


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


class CMakeArgumentFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("cmake") or not shutil.which("ninja"):
            raise RuntimeError("CMake and Ninja must be on PATH; tests were not run")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-cmake-args-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        includes = "\n".join(
            f"include({name})"
            for name in (
                "therock_globals",
                "therock_sanitizers",
                "therock_flag_utils",
                "therock_default_targets",
                "therock_subproject",
            )
        )
        write(
            self.source / "CMakeLists.txt",
            f"""
            cmake_minimum_required(VERSION 3.25)
            project(argument_fingerprints NONE)
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
            file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")
            {includes}
            set(DECLARED_ARGS "" CACHE STRING "Declared configure arguments")
            option(IMPORT_STAGE "Exercise prebuilt fingerprint calculation" OFF)
            foreach(_name args_source args_consumer args_control)
              set(_args)
              set(_deps)
              if(_name STREQUAL "args_source")
                set(_args ${{DECLARED_ARGS}})
              elseif(_name STREQUAL "args_consumer")
                set(_deps RUNTIME_DEPS args_source)
              endif()
              therock_cmake_subproject_declare(${{_name}}
                EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/${{_name}}"
                BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/${{_name}}"
                FPRINT_SOURCE_HASH
                DISABLE_AMDGPU_TARGETS
                NO_MERGE_COMPILE_COMMANDS
                CMAKE_ARGS ${{_args}}
                ${{_deps}}
              )
              get_target_property(_stage ${{_name}} THEROCK_STAGE_DIR)
              if(_name STREQUAL "args_source" AND IMPORT_STAGE)
                file(WRITE "${{_stage}}.prebuilt" "test stage\\n")
              endif()
              therock_cmake_subproject_activate(${{_name}})
              get_target_property(_fprint ${{_name}} THEROCK_FPRINT)
              get_target_property(_inner ${{_name}} THEROCK_BINARY_DIR)
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{_name}}.fprint" "${{_fprint}}")
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{_name}}.inner" "${{_inner}}")
            endforeach()
            """,
        )
        for name in PROJECTS:
            write(
                self.source / name / "CMakeLists.txt",
                """
                cmake_minimum_required(VERSION 3.25)
                project(argument_output NONE)
                if(DEFINED VALUE)
                  set(_value "${VALUE}")
                else()
                  set(_value "<unset>")
                endif()
                file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.txt" "${_value}\\n")
                install(FILES "${CMAKE_CURRENT_BINARY_DIR}/probe.txt" DESTINATION share)
                """,
            )

    def run_command(self, *args):
        result = subprocess.run(
            [str(arg) for arg in args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, declared=(), overrides=(), *, prebuilt=False):
        self.run_command(
            "cmake",
            "-S",
            self.source,
            "-B",
            self.build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDECLARED_ARGS:STRING=" + ";".join(declared),
            "-Dargs_source_CMAKE_ARGS:STRING=" + ";".join(overrides),
            "-DIMPORT_STAGE:BOOL=" + ("ON" if prebuilt else "OFF"),
        )
        values = {
            name: (self.build / f"{name}.fprint").read_text(encoding="utf-8").strip()
            for name in PROJECTS
        }
        for value in values.values():
            self.assertRegex(value, r"^[0-9a-f]{64}$")
        return values

    def configured_output(self):
        self.run_command(
            "cmake", "--build", self.build, "--target", "args_source+configure"
        )
        inner = Path((self.build / "args_source.inner").read_text(encoding="utf-8"))
        return (inner / "probe.txt").read_text(encoding="utf-8")

    def assert_propagation(self, before, after):
        changed = {name for name in PROJECTS if before[name] != after[name]}
        self.assertEqual(changed, {"args_source", "args_consumer"})

    def test_declared_argument_changes_output_and_dependency_fingerprints(self):
        baseline = self.configure(("-DVALUE=baseline",))
        self.assertEqual(self.configured_output(), "baseline\n")
        changed = self.configure(("-DVALUE=changed",))
        self.assertEqual(self.configured_output(), "changed\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(changed, self.configure(("-DVALUE=changed",)))
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",)))

    def test_per_project_override_reaches_output_and_fingerprint(self):
        declared = ("-DVALUE=declared",)
        baseline = self.configure(declared, ("-DVALUE=first",))
        self.assertEqual(self.configured_output(), "first\n")
        changed = self.configure(declared, ("-DVALUE=second",))
        self.assertEqual(self.configured_output(), "second\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(declared, ("-DVALUE=first",)))

    def test_duplicate_argument_order_preserves_last_value_precedence(self):
        first = ("-DVALUE=first", "-DVALUE=second")
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "second\n")
        changed = self.configure(tuple(reversed(first)))
        self.assertEqual(self.configured_output(), "first\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(first))

    def test_argument_removal_and_restoration(self):
        baseline = self.configure()
        changed = self.configure(("-DVALUE=added",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_explicit_empty_value_is_distinct_from_no_argument(self):
        baseline = self.configure()
        changed = self.configure(("-DVALUE=",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_prebuilt_stage_still_accounts_for_requested_arguments(self):
        baseline = self.configure(("-DVALUE=baseline",), prebuilt=True)
        changed = self.configure(("-DVALUE=changed",), prebuilt=True)
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",), prebuilt=True))

    def test_configure_time_values_in_generator_expressions(self):
        first = ("-DVALUE=$<IF:$<BOOL:1>,first,second>",)
        second = ("-DVALUE=$<IF:$<BOOL:0>,first,second>",)
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "first\n")
        changed = self.configure(second)
        self.assertEqual(self.configured_output(), "second\n")
        self.assert_propagation(baseline, changed)


if __name__ == "__main__":
    unittest.main()
