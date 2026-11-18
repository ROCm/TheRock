#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise argument fingerprints using real sub-project activation/configuration."""

import unittest
from pathlib import Path

from utils import (
    FingerprintTestCase,
    ProjectFingerprints,
    cmake_project_setup,
    write,
)

PROJECTS = ("args_source", "args_consumer", "args_control")


class CMakeArgumentFingerprintTest(FingerprintTestCase):
    required_tools = ("cmake", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.source = self.root / "source"
        self.build = self.root / "build"
        write(
            self.source / "CMakeLists.txt",
            f"""
            {cmake_project_setup("argument_fingerprints")}
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

    def configure(
        self,
        declared: tuple[str, ...] = (),
        overrides: tuple[str, ...] = (),
        *,
        prebuilt: bool = False,
    ) -> ProjectFingerprints:
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
        fingerprints = {
            name: (self.build / f"{name}.fprint").read_text(encoding="utf-8").strip()
            for name in PROJECTS
        }
        for fingerprint in fingerprints.values():
            self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        return fingerprints

    def configured_output(self) -> str:
        self.run_command(
            "cmake", "--build", self.build, "--target", "args_source+configure"
        )
        inner_build = Path(
            (self.build / "args_source.inner").read_text(encoding="utf-8")
        )
        return (inner_build / "probe.txt").read_text(encoding="utf-8")

    def assert_propagation(
        self, before: ProjectFingerprints, after: ProjectFingerprints
    ) -> None:
        changed = {name for name in PROJECTS if before[name] != after[name]}
        self.assertEqual(changed, {"args_source", "args_consumer"})

    def test_declared_argument_changes_output_and_dependency_fingerprints(self) -> None:
        baseline = self.configure(("-DVALUE=baseline",))
        self.assertEqual(self.configured_output(), "baseline\n")
        changed = self.configure(("-DVALUE=changed",))
        self.assertEqual(self.configured_output(), "changed\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(changed, self.configure(("-DVALUE=changed",)))
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",)))

    def test_per_project_override_reaches_output_and_fingerprint(self) -> None:
        declared = ("-DVALUE=declared",)
        baseline = self.configure(declared=declared, overrides=("-DVALUE=first",))
        self.assertEqual(self.configured_output(), "first\n")
        changed = self.configure(declared=declared, overrides=("-DVALUE=second",))
        self.assertEqual(self.configured_output(), "second\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(
            baseline, self.configure(declared=declared, overrides=("-DVALUE=first",))
        )

    def test_duplicate_argument_order_preserves_last_value_precedence(self) -> None:
        first = ("-DVALUE=first", "-DVALUE=second")
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "second\n")
        changed = self.configure(tuple(reversed(first)))
        self.assertEqual(self.configured_output(), "first\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(first))

    def test_argument_removal_and_restoration(self) -> None:
        baseline = self.configure()
        changed = self.configure(("-DVALUE=added",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_explicit_empty_value_is_distinct_from_no_argument(self) -> None:
        baseline = self.configure()
        changed = self.configure(("-DVALUE=",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_prebuilt_stage_still_accounts_for_requested_arguments(self) -> None:
        baseline = self.configure(("-DVALUE=baseline",), prebuilt=True)
        changed = self.configure(("-DVALUE=changed",), prebuilt=True)
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",), prebuilt=True))

    def test_configure_time_values_in_generator_expressions(self) -> None:
        first = ("-DVALUE=$<IF:$<BOOL:1>,first,second>",)
        second = ("-DVALUE=$<IF:$<BOOL:0>,first,second>",)
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "first\n")
        changed = self.configure(second)
        self.assertEqual(self.configured_output(), "second\n")
        self.assert_propagation(baseline, changed)


if __name__ == "__main__":
    unittest.main()
