#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure-only ADDITIONAL_SOURCE_INPUTS tests using Git source fixtures."""

import unittest
from collections.abc import Collection
from pathlib import Path

from utils import (
    FingerprintTestCase,
    ProjectFingerprints,
    cmake_project_setup,
    write,
)

SCOPES = {
    "rocm-cmake": "base/rocm-cmake",
    "rocm-half": "base/half",
    "amd-llvm": "compiler/amd-llvm",
    "amd-comgr": "compiler/amd-llvm/amd/comgr",
    "hipcc": "compiler/amd-llvm/amd/hipcc",
    "hipify": "compiler/hipify",
}
ROOTS = (
    "base/rocm-cmake",
    "base/half",
    "compiler/amd-llvm",
    "compiler/hipify",
    "compiler/spirv-llvm-translator",
    "rocm-systems",
)
LLVM_USERS = {
    "amd-llvm",
    "amd-comgr",
    "hipcc",
    "hipify",
}
PROJECTS = (*SCOPES, "control")


class SourceDependencyFingerprintTest(FingerprintTestCase):
    required_tools = ("git", "cmake", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.source = self.root / "harness"
        self.sources = self.root / "sources with spaces"
        self.build = self.root / "build"
        self.spirv = self.sources / "compiler/spirv-llvm-translator"
        self.systems = self.sources / "rocm-systems"
        self.hsa = self.systems / "projects/rocr-runtime/runtime/hsa-runtime/inc"
        for relative in ROOTS:
            repository = self.sources / relative
            repository.mkdir(parents=True)
            self.git(repository, "init", "-q")
        for name, relative in SCOPES.items():
            directory = self.sources / relative
            entry = directory / "llvm" if name == "amd-llvm" else directory
            write(
                entry / "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.25)\nproject(fixture NONE)\n",
            )
            write(directory / "input.txt", name + "\n")
        for directory in (self.spirv, self.hsa):
            write(directory / "input.txt", directory.name + "\n")
        write(self.sources / "compiler/amd-llvm/clang/input.txt", "clang\n")
        write(self.systems / "unrelated/input.txt", "unrelated\n")
        write(
            self.source / "control/CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.25)\nproject(control NONE)\n",
        )
        for relative in ROOTS:
            self.commit(self.sources / relative, "initial")
        self.write_harness()

    def git(self, repository: str | Path, *args: str) -> str:
        return self.run_command("git", "-C", repository, *args).stdout.strip()

    def commit(self, repository: str | Path, message: str) -> None:
        self.git(repository, "add", "--all")
        self.git(repository, "commit", "-qm", message)

    def write_harness(self) -> None:
        declarations = []
        for name, relative in SCOPES.items():
            directory = (self.sources / relative).as_posix()
            args = []
            if name != "rocm-cmake":
                args.append("BUILD_DEPS rocm-cmake")
            if name in LLVM_USERS - {"amd-llvm"}:
                args.append("RUNTIME_DEPS amd-llvm")
            if name == "amd-llvm":
                args.extend(
                    (
                        "CMAKE_LISTS_RELPATH llvm",
                        "ADDITIONAL_SOURCE_INPUTS ${TEST_LLVM_SOURCE_INPUTS}",
                        "${_main_hash_option}",
                    )
                )
            declarations.append(
                f"""
                therock_cmake_subproject_declare({name}
                  EXTERNAL_SOURCE_DIR "{directory}"
                  BINARY_DIR "${{CMAKE_BINARY_DIR}}/{name}"
                  DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
                  {' '.join(args)}
                )
                get_target_property(_stage {name} THEROCK_STAGE_DIR)
                if(IMPORT_STAGE AND "{name}" STREQUAL "amd-llvm")
                  file(WRITE "${{_stage}}.prebuilt" "fixture\\n")
                else()
                  file(REMOVE "${{_stage}}.prebuilt")
                endif()
                therock_cmake_subproject_activate({name})
            """
            )
        write(
            self.source / "CMakeLists.txt",
            f"""
            {cmake_project_setup("source_dependencies")}
            set(_main_hash_option)
            if(HASH_PRIMARY_SOURCE)
              set(_main_hash_option FPRINT_SOURCE_HASH)
            endif()
            {''.join(declarations)}
            therock_cmake_subproject_declare(control
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/control"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/control"
              FPRINT_SOURCE_HASH DISABLE_AMDGPU_TARGETS NO_MERGE_COMPILE_COMMANDS
            )
            therock_cmake_subproject_activate(control)
            get_target_property(_paths amd-llvm THEROCK_ADDITIONAL_SOURCE_INPUTS)
            file(WRITE "${{CMAKE_BINARY_DIR}}/source-deps.txt" "${{_paths}}")
            foreach(_name {' '.join(PROJECTS)})
              get_target_property(_fprint ${{_name}} THEROCK_FPRINT)
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{_name}}.fprint" "${{_fprint}}")
            endforeach()
        """,
        )

    def configure(
        self,
        deps: tuple[str, ...] | None = None,
        *,
        prebuilt: bool = False,
        hash_primary: bool = False,
    ) -> ProjectFingerprints:
        if deps is None:
            deps = (self.spirv.as_posix(), self.hsa.as_posix())
        self.run_command(
            "cmake",
            "-S",
            self.source,
            "-B",
            self.build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_VERBOSE=ON",
            "-DTEST_LLVM_SOURCE_INPUTS:STRING=" + ";".join(deps),
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

    def test_repeat_and_relative_source_dependencies(self) -> None:
        baseline = self.configure()
        self.assert_valid(baseline)
        self.assertEqual(baseline, self.configure())
        # The harness and source directories share the same temporary parent.
        relative = tuple(
            (Path("..") / directory.relative_to(self.source.parent)).as_posix()
            for directory in (self.spirv, self.hsa)
        )
        self.assertEqual(baseline, self.configure(relative))
        # CMake may expand Windows short paths; compare resolved paths.
        recorded = (
            (self.build / "source-deps.txt").read_text(encoding="utf-8").split(";")
        )
        self.assertEqual(
            [Path(path).resolve() for path in recorded],
            [self.spirv.resolve(), self.hsa.resolve()],
        )

    def test_source_changes_and_restoration_for_all_projects(self) -> None:
        baseline = self.configure()
        for name, relative in SCOPES.items():
            with self.subTest(project=name):
                directory = self.sources / relative
                repository = self.git(directory, "rev-parse", "--show-toplevel")
                write(directory / "input.txt", "changed\n")
                self.commit(repository, "source edit")
                if name == "rocm-cmake":
                    expected = set(SCOPES)
                elif name in {"amd-llvm", "amd-comgr", "hipcc"}:
                    expected = LLVM_USERS
                else:
                    expected = {name}
                self.assert_changed(baseline, self.configure(), expected)
                self.git(repository, "revert", "--no-edit", "HEAD")
                self.assertEqual(baseline, self.configure())

    def test_each_llvm_external_source_changes_only_llvm_users_and_restores(
        self,
    ) -> None:
        baseline = self.configure()
        for directory in (self.spirv, self.hsa):
            with self.subTest(directory=directory):
                repository = self.git(directory, "rev-parse", "--show-toplevel")
                write(directory / "input.txt", "changed\n")
                self.commit(repository, "external source edit")
                changed = self.configure()
                self.assert_changed(baseline, changed, LLVM_USERS)
                self.assertEqual(changed, self.configure())
                self.git(repository, "revert", "--no-edit", "HEAD")
                self.assertEqual(baseline, self.configure())

    def test_llvm_root_already_includes_clang(self) -> None:
        baseline = self.configure()
        llvm = self.sources / "compiler/amd-llvm"
        write(llvm / "clang/input.txt", "changed\n")
        self.commit(llvm, "Clang edit")
        self.assert_changed(baseline, self.configure(), LLVM_USERS)

    def test_optional_hsa_source_dependency_can_be_omitted(self) -> None:
        deps = (self.spirv.as_posix(),)
        baseline = self.configure(deps)
        self.assert_valid(baseline)
        write(self.hsa / "input.txt", "changed\n")
        self.commit(self.systems, "HSA edit")
        self.assertEqual(baseline, self.configure(deps))

    def test_unrelated_dirty_and_committed_changes_leave_fingerprints_stable(
        self,
    ) -> None:
        baseline = self.configure()
        write(self.systems / "unrelated/input.txt", "unrelated edit\n")
        self.assertEqual(baseline, self.configure())
        self.commit(self.systems, "unrelated edit")
        self.assertEqual(baseline, self.configure())

    def test_dirty_and_staged_external_sources_invalidate_consumers(self) -> None:
        baseline = self.configure()
        write(self.spirv / "input.txt", "dirty\n")
        self.assert_valid(self.configure(), invalid=LLVM_USERS)
        self.git(self.spirv, "add", "input.txt")
        self.assert_valid(self.configure(), invalid=LLVM_USERS)
        self.git(
            self.spirv,
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
        self.spirv.rename(self.root / "removed-spirv-llvm-translator")
        self.assertFalse(self.spirv.exists())
        write(self.spirv.with_name(".spirv-llvm-translator.smrev"), "stale\n")
        self.assert_valid(self.configure(), invalid=LLVM_USERS)

    def test_non_git_directory_requires_a_marker(self) -> None:
        directory = self.root / "unversioned"
        directory.mkdir()
        self.assert_valid(self.configure((directory.as_posix(),)), invalid=LLVM_USERS)
        marker = directory.with_name(".unversioned.smrev")
        write(marker, "source;recipe\nfirst\n")
        baseline = self.configure((directory.as_posix(),))
        write(marker, "source;recipe\nsecond\n")
        self.assert_changed(
            baseline, self.configure((directory.as_posix(),)), LLVM_USERS
        )

    def test_file_dependency_is_rejected_as_not_a_directory(self) -> None:
        self.assert_valid(
            self.configure(((self.spirv / "input.txt").as_posix(),)), invalid=LLVM_USERS
        )

    def test_dependencies_apply_when_primary_source_is_file_hashed(self) -> None:
        baseline = self.configure(hash_primary=True)
        write(self.spirv / "input.txt", "changed\n")
        self.commit(self.spirv, "source edit")
        self.assert_changed(baseline, self.configure(hash_primary=True), LLVM_USERS)

    def test_dependencies_apply_to_prebuilt_fingerprints(self) -> None:
        baseline = self.configure(prebuilt=True)
        write(self.spirv / "input.txt", "changed\n")
        self.commit(self.spirv, "source edit")
        self.assert_changed(baseline, self.configure(prebuilt=True), LLVM_USERS)
        write(self.spirv / "input.txt", "dirty\n")
        self.assert_valid(self.configure(prebuilt=True), invalid=LLVM_USERS)


if __name__ == "__main__":
    unittest.main()
