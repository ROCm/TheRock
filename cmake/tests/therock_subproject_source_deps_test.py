#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure-only ADDITIONAL_SOURCE_INPUTS tests using Git source fixtures.

Fixture subprojects:

- consumer: declares ADDITIONAL_SOURCE_INPUTS. Its CMake lists
  are in consumer/cmake-root, so its fingerprint covers the whole consumer
  checkout.
- downstream: RUNTIME_DEPS consumer.
- control: FPRINT_SOURCE_HASH, no dependencies.

The inputs are a whole separate checkout (whole-repo) and a directory inside a
larger checkout (monorepo/inputs/first).

Real declarations are exercised on a full checkout by
cmake/tests/therock_source_inputs_real_tree_test.py.
"""

import unittest
from collections.abc import Collection
from pathlib import Path

from therock_fingerprint_test_utils import (
    FingerprintTestCase,
    ProjectFingerprints,
    cmake_project_setup,
    write,
)

PROJECTS = ("consumer", "downstream", "control")
INPUT_USERS = {"consumer", "downstream"}
MINIMAL_PROJECT = "cmake_minimum_required(VERSION 3.25)\nproject(fixture NONE)\n"


class SourceDependencyFingerprintTest(FingerprintTestCase):
    required_tools = ("git", "cmake", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.harness = self.root / "harness"
        self.sources = self.root / "sources with spaces"
        self.build = self.root / "build"
        self.consumer = self.sources / "consumer"
        self.whole_repo = self.sources / "whole-repo"
        self.monorepo = self.sources / "monorepo"
        self.subdir_input = self.monorepo / "inputs/first"

        write(self.consumer / "cmake-root/CMakeLists.txt", MINIMAL_PROJECT)
        write(self.consumer / "sibling/input.txt", "sibling\n")
        write(self.whole_repo / "input.txt", "whole\n")
        write(self.subdir_input / "input.txt", "same content\n")
        write(self.monorepo / "inputs/second/input.txt", "same content\n")
        write(self.harness / "downstream/CMakeLists.txt", MINIMAL_PROJECT)
        write(self.harness / "control/CMakeLists.txt", MINIMAL_PROJECT)
        for repository in (self.consumer, self.whole_repo, self.monorepo):
            self.git(repository, "init", "-q")
            self.commit(repository, "initial")

        write(
            self.harness / "CMakeLists.txt",
            f"""
            {cmake_project_setup("source_dependencies")}
            set(_consumer_hash_option)
            if(HASH_PRIMARY_SOURCE)
              set(_consumer_hash_option FPRINT_SOURCE_HASH)
            endif()

            therock_cmake_subproject_declare(consumer
              EXTERNAL_SOURCE_DIR "{self.consumer.as_posix()}"
              CMAKE_LISTS_RELPATH cmake-root
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/consumer"
              DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
              ADDITIONAL_SOURCE_INPUTS ${{TEST_SOURCE_INPUTS}}
              ${{_consumer_hash_option}}
            )
            get_target_property(_stage consumer THEROCK_STAGE_DIR)
            if(IMPORT_STAGE)
              file(WRITE "${{_stage}}.prebuilt" "fixture\\n")
            else()
              file(REMOVE "${{_stage}}.prebuilt")
            endif()
            therock_cmake_subproject_activate(consumer)

            therock_cmake_subproject_declare(downstream
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/downstream"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/downstream"
              FPRINT_SOURCE_HASH DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
              RUNTIME_DEPS consumer
            )
            therock_cmake_subproject_activate(downstream)

            therock_cmake_subproject_declare(control
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/control"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/control"
              FPRINT_SOURCE_HASH DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
            )
            therock_cmake_subproject_activate(control)

            get_target_property(_paths consumer THEROCK_ADDITIONAL_SOURCE_INPUTS)
            file(WRITE "${{CMAKE_BINARY_DIR}}/source-deps.txt" "${{_paths}}")
            foreach(_name {' '.join(PROJECTS)})
              get_target_property(_fprint ${{_name}} THEROCK_FPRINT)
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{_name}}.fprint" "${{_fprint}}")
            endforeach()
            """,
        )

    def git(self, repository: Path, *args: str) -> str:
        return self.run_command("git", "-C", repository, *args).stdout.strip()

    def commit(self, repository: Path, message: str) -> None:
        self.git(repository, "add", "--all")
        self.git(repository, "commit", "-qm", message)

    def edit_and_commit(self, repository: Path, relative: str) -> None:
        write(repository / relative, "changed\n")
        self.commit(repository, "edit " + relative)

    def configure(
        self,
        inputs: tuple[Path, ...] | None = None,
        *,
        prebuilt: bool = False,
        hash_primary: bool = False,
    ) -> ProjectFingerprints:
        if inputs is None:
            inputs = (self.whole_repo, self.subdir_input)
        self.run_command(
            "cmake",
            "-S",
            self.harness,
            "-B",
            self.build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_VERBOSE=ON",
            "-DTEST_SOURCE_INPUTS:STRING=" + ";".join(p.as_posix() for p in inputs),
            "-DIMPORT_STAGE:BOOL=" + ("ON" if prebuilt else "OFF"),
            "-DHASH_PRIMARY_SOURCE:BOOL=" + ("ON" if hash_primary else "OFF"),
        )
        return {
            name: (self.build / f"{name}.fprint").read_text(encoding="utf-8").strip()
            for name in PROJECTS
        }

    def assert_valid(
        self, fingerprints: ProjectFingerprints, invalid: Collection[str] = ()
    ) -> None:
        for name, fingerprint in fingerprints.items():
            if name in invalid:
                self.assertEqual(fingerprint, "", name)
            else:
                self.assertRegex(fingerprint, r"^[0-9a-f]{64}$", name)

    def assert_changed(
        self,
        before: ProjectFingerprints,
        after: ProjectFingerprints,
        expected: Collection[str],
    ) -> None:
        self.assert_valid(before)
        self.assert_valid(after)
        self.assertEqual(
            {name for name in PROJECTS if before[name] != after[name]}, set(expected)
        )

    def test_repeat_and_relative_inputs_are_stable(self) -> None:
        baseline = self.configure()
        self.assert_valid(baseline)
        self.assertEqual(baseline, self.configure())
        # Relative inputs resolve against the declaring directory (the harness).
        relative_inputs = (
            Path("../sources with spaces/whole-repo"),
            Path("../sources with spaces/monorepo/inputs/first"),
        )
        self.assertEqual(baseline, self.configure(relative_inputs))
        # CMake may expand Windows short paths; compare resolved paths.
        recorded = (
            (self.build / "source-deps.txt").read_text(encoding="utf-8").split(";")
        )
        self.assertEqual(
            [Path(path).resolve() for path in recorded],
            [self.whole_repo.resolve(), self.subdir_input.resolve()],
        )

    def test_primary_scope_includes_directories_beside_cmake_lists(self) -> None:
        baseline = self.configure()
        self.edit_and_commit(self.consumer, "sibling/input.txt")
        self.assert_changed(baseline, self.configure(), INPUT_USERS)

    def test_subdirectory_input_change_and_revert(self) -> None:
        baseline = self.configure()
        self.edit_and_commit(self.monorepo, "inputs/first/input.txt")
        changed = self.configure()
        self.assert_changed(baseline, changed, INPUT_USERS)
        self.assertEqual(changed, self.configure())
        self.git(self.monorepo, "revert", "--no-edit", "HEAD")
        self.assertEqual(baseline, self.configure())

    def test_input_path_is_part_of_the_fingerprint(self) -> None:
        # inputs/first and inputs/second have identical trees, so only the
        # declared path tells them apart.
        self.assertEqual(
            self.git(self.monorepo, "rev-parse", "HEAD:inputs/first"),
            self.git(self.monorepo, "rev-parse", "HEAD:inputs/second"),
        )
        first = self.configure((self.monorepo / "inputs/first",))
        second = self.configure((self.monorepo / "inputs/second",))
        self.assert_changed(first, second, INPUT_USERS)

    def test_dirty_and_staged_inputs_invalidate_consumers(self) -> None:
        baseline = self.configure()
        write(self.whole_repo / "input.txt", "dirty\n")
        self.assert_valid(self.configure(), invalid=INPUT_USERS)
        self.git(self.whole_repo, "add", "input.txt")
        self.assert_valid(self.configure(), invalid=INPUT_USERS)
        self.git(
            self.whole_repo,
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "input.txt",
        )
        self.assertEqual(baseline, self.configure())

    def test_missing_directory_is_invalid_even_with_a_stale_marker(self) -> None:
        self.configure()
        # Make the declared path absent without deleting read-only Git objects
        # on Windows. TemporaryDirectory cleanup owns the moved repository.
        self.whole_repo.rename(self.root / "removed-whole-repo")
        write(self.sources / ".whole-repo.smrev", "stale\n")
        self.assert_valid(self.configure(), invalid=INPUT_USERS)

    def test_file_input_is_rejected_as_not_a_directory(self) -> None:
        self.assert_valid(
            self.configure((self.whole_repo / "input.txt",)), invalid=INPUT_USERS
        )

    def test_inputs_apply_when_primary_source_is_file_hashed(self) -> None:
        baseline = self.configure(hash_primary=True)
        self.edit_and_commit(self.whole_repo, "input.txt")
        self.assert_changed(baseline, self.configure(hash_primary=True), INPUT_USERS)

    def test_inputs_apply_to_prebuilt_fingerprints(self) -> None:
        baseline = self.configure(prebuilt=True)
        self.edit_and_commit(self.whole_repo, "input.txt")
        self.assert_changed(baseline, self.configure(prebuilt=True), INPUT_USERS)
        write(self.whole_repo / "input.txt", "dirty\n")
        self.assert_valid(self.configure(prebuilt=True), invalid=INPUT_USERS)


if __name__ == "__main__":
    unittest.main()
