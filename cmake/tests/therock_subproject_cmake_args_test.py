#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test that a subproject's CMake arguments are part of its fingerprint.

A subproject is configured with the arguments in its `CMAKE_ARGS`, plus any
`<name>_CMAKE_ARGS` override. Those arguments change what is built, so a
change to them must change the fingerprint of the subproject and of the
subprojects that depend on it.

A generator expression (`$<...>`) in an argument is evaluated after the
fingerprint is computed. Expressions whose value depends only on their own text
(such as `$<IF:$<BOOL:1>,a,b>`) are hashed as text. Any other expression (such
as `$<TARGET_PROPERTY:...>`) makes the fingerprint invalid (empty), because
its text does not identify its value.

The fixture has three subprojects: `args_source` gets the arguments,
`args_consumer` has a runtime dependency on it, and `args_control` is
unrelated. The tests drive `cmake/therock_subproject.cmake` through this small
superproject, so they need no submodules and no compiler.
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ("args_source", "args_consumer", "args_control")
ProjectFingerprints = dict[str, str]


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


class CMakeArgumentFingerprintTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="therock-cmake-args-")
        self.addCleanup(temporary.cleanup)
        # Resolve Windows 8.3 temporary paths before passing them to CMake.
        root = Path(temporary.name).resolve()
        self.source = root / "source"
        self.build = root / "build"
        write(
            self.source / "CMakeLists.txt",
            f"""
            cmake_minimum_required(VERSION 3.25)
            project(argument_fingerprints NONE)
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            # Use the interpreter running the tests; `FindPython` is slow,
            # especially on Windows.
            set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
            # therock_subproject.cmake fingerprints this file. The fixture
            # declares no flags, so an empty state file is enough.
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
            file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")
            include(therock_globals)
            include(therock_sanitizers)
            include(therock_flag_utils)
            include(therock_default_targets)
            include(therock_subproject)
            set(DECLARED_ARGS "" CACHE STRING "Declared configure arguments")
            option(IMPORT_STAGE "Exercise prebuilt fingerprint calculation" OFF)
            set(TEST_ARG_VALUE "" CACHE STRING "Value of a target property")
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
              # Only a generator expression reads this, so it is not in the
              # argument text.
              set_target_properties(${{_name}} PROPERTIES
                TEST_ARG_VALUE "${{TEST_ARG_VALUE}}")
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

    def run_command(self, *args: str | Path) -> None:
        result = subprocess.run(
            [str(arg) for arg in args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Command failed: {args}\n{result.stdout}\n{result.stderr}",
        )

    def configure(
        self,
        declared: tuple[str, ...] = (),
        overrides: tuple[str, ...] = (),
        *,
        prebuilt: bool = False,
        property_value: str = "",
        invalid: tuple[str, ...] = (),
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
            "-DTEST_ARG_VALUE:STRING=" + property_value,
        )
        fingerprints = {
            name: (self.build / f"{name}.fprint").read_text(encoding="utf-8").strip()
            for name in PROJECTS
        }
        for name, fingerprint in fingerprints.items():
            if name in invalid:
                self.assertEqual(fingerprint, "", name)
            else:
                self.assertRegex(fingerprint, r"^[0-9a-f]{64}$", name)
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
        """A changed argument changes the subproject and its dependent, not others."""
        baseline = self.configure(("-DVALUE=baseline",))
        self.assertEqual(self.configured_output(), "baseline\n")
        changed = self.configure(("-DVALUE=changed",))
        self.assertEqual(self.configured_output(), "changed\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(changed, self.configure(("-DVALUE=changed",)))
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",)))

    def test_per_project_override_reaches_output_and_fingerprint(self) -> None:
        """A `<name>_CMAKE_ARGS` override counts like a declared argument."""
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
        """Reordering repeated arguments counts, because the last value wins."""
        first = ("-DVALUE=first", "-DVALUE=second")
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "second\n")
        changed = self.configure(tuple(reversed(first)))
        self.assertEqual(self.configured_output(), "first\n")
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(first))

    def test_argument_removal_and_restoration(self) -> None:
        """Adding an argument changes fingerprints; removing it restores them."""
        baseline = self.configure()
        changed = self.configure(("-DVALUE=added",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_explicit_empty_value_is_distinct_from_no_argument(self) -> None:
        """`-DVALUE=` is a different argument from no argument at all."""
        baseline = self.configure()
        changed = self.configure(("-DVALUE=",))
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure())

    def test_prebuilt_stage_still_accounts_for_requested_arguments(self) -> None:
        """Arguments count even when the subproject is imported as prebuilt."""
        baseline = self.configure(("-DVALUE=baseline",), prebuilt=True)
        changed = self.configure(("-DVALUE=changed",), prebuilt=True)
        self.assert_propagation(baseline, changed)
        self.assertEqual(baseline, self.configure(("-DVALUE=baseline",), prebuilt=True))

    def test_configure_time_values_in_generator_expressions(self) -> None:
        """Text-only generator expressions are hashed as text."""
        first = ("-DVALUE=$<IF:$<BOOL:1>,first,second>",)
        second = ("-DVALUE=$<IF:$<BOOL:0>,first,second>",)
        baseline = self.configure(first)
        self.assertEqual(self.configured_output(), "first\n")
        changed = self.configure(second)
        self.assertEqual(self.configured_output(), "second\n")
        self.assert_propagation(baseline, changed)

    def test_generator_expression_with_unknown_value_invalidates_fingerprint(
        self,
    ) -> None:
        """Other generator expressions make the fingerprint invalid.

        The argument text stays the same while the property it reads changes,
        and CMake evaluates it after fingerprints are computed, so the text
        cannot identify the value.
        """
        unknown = ("-DVALUE=$<TARGET_PROPERTY:args_source,TEST_ARG_VALUE>",)
        invalid = ("args_source", "args_consumer")
        baseline = self.configure()
        first = self.configure(unknown, property_value="first", invalid=invalid)
        self.assertEqual(self.configured_output(), "first\n")
        self.configure(unknown, property_value="second", invalid=invalid)
        self.assertEqual(self.configured_output(), "second\n")
        self.assertEqual(first["args_control"], baseline["args_control"])
        self.assertEqual(baseline, self.configure())


if __name__ == "__main__":
    unittest.main()
