#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test the shared/ctest input that monorepo sub-projects get automatically.

monorepo_lib is inside the rocm-libraries superrepo. standalone_lib is in a
separate checkout with the same layout, so it must not get the input.
"""

import unittest
from pathlib import Path

from therock_fingerprint_test_utils import (
    FingerprintTestCase,
    cmake_project_setup,
    write,
)


class SuperrepoTestInputsTest(FingerprintTestCase):
    required_tools = ("cmake", "git", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.superrepo = self.root / "rocm-libraries"
        self.build = self.root / "build"
        write(
            self.superrepo / "projects/lib/CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.25)\nproject(lib NONE)\n",
        )
        write(self.superrepo / "shared/ctest/categories.txt", "categories\n")
        write(self.superrepo / "shared/other/input.txt", "other\n")
        self.git("init", "-q")
        self.commit("initial")
        self.standalone = self.root / "standalone"
        write(
            self.standalone / "projects/lib/CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.25)\nproject(lib NONE)\n",
        )
        self.run_command("git", "-C", self.standalone, "init", "-q")
        self.run_command("git", "-C", self.standalone, "add", "--all")
        self.run_command("git", "-C", self.standalone, "commit", "-qm", "initial")
        write(
            self.root / "harness/CMakeLists.txt",
            f"""
            {cmake_project_setup("superrepo_inputs")}
            set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "{self.superrepo.as_posix()}")
            set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${{CMAKE_BINARY_DIR}}/no-rocm-systems")
            therock_cmake_subproject_declare(monorepo_lib
              EXTERNAL_SOURCE_DIR "{(self.superrepo / "projects/lib").as_posix()}"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/monorepo_lib"
              DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
            )
            therock_cmake_subproject_activate(monorepo_lib)
            therock_cmake_subproject_declare(standalone_lib
              EXTERNAL_SOURCE_DIR "{(self.standalone / "projects/lib").as_posix()}"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/standalone_lib"
              DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
            )
            therock_cmake_subproject_activate(standalone_lib)
            foreach(_name monorepo_lib standalone_lib)
              get_target_property(_fprint ${{_name}} THEROCK_FPRINT)
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{_name}}.fprint" "${{_fprint}}")
            endforeach()
            """,
        )

    def git(self, *args: str | Path) -> None:
        self.run_command("git", "-C", self.superrepo, *args)

    def commit(self, message: str) -> None:
        self.git("add", "--all")
        self.git("commit", "-qm", message)

    def fingerprint(self, *, testing: bool, name: str = "monorepo_lib") -> str:
        self.run_command(
            "cmake",
            "-S",
            self.root / "harness",
            "-B",
            self.build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_BUILD_TESTING:BOOL=" + ("ON" if testing else "OFF"),
        )
        return (self.build / f"{name}.fprint").read_text(encoding="utf-8").strip()

    def edit_and_commit(self, relative: str) -> None:
        write(self.superrepo / relative, "changed\n")
        self.commit("edit " + relative)

    def test_shared_ctest_change_affects_fingerprint_when_testing(self) -> None:
        baseline = self.fingerprint(testing=True)
        self.assertRegex(baseline, r"^[0-9a-f]{64}$")
        self.edit_and_commit("shared/other/input.txt")
        self.assertEqual(baseline, self.fingerprint(testing=True))
        self.edit_and_commit("shared/ctest/categories.txt")
        self.assertNotEqual(baseline, self.fingerprint(testing=True))

    def test_shared_ctest_change_is_ignored_without_testing(self) -> None:
        baseline = self.fingerprint(testing=False)
        self.edit_and_commit("shared/ctest/categories.txt")
        self.assertEqual(baseline, self.fingerprint(testing=False))

    def test_shared_ctest_is_ignored_outside_the_superrepo(self) -> None:
        baseline = self.fingerprint(testing=True, name="standalone_lib")
        self.assertRegex(baseline, r"^[0-9a-f]{64}$")
        self.edit_and_commit("shared/ctest/categories.txt")
        self.assertEqual(
            baseline, self.fingerprint(testing=True, name="standalone_lib")
        )

    def test_missing_shared_ctest_directory_is_not_an_error(self) -> None:
        self.git("rm", "-rq", "shared/ctest")
        self.commit("remove shared/ctest")
        self.assertRegex(self.fingerprint(testing=True), r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
