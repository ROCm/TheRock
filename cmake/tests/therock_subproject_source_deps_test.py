#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure-only ADDITIONAL_SOURCE_INPUTS tests for seven Git sub-projects."""

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
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


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


class SourceDependencyFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ("git", "cmake", "ninja") if not shutil.which(name)]
        if missing:
            raise RuntimeError(
                f"Required tools unavailable: {', '.join(missing)}; tests were not run"
            )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-source-deps-")
        self.addCleanup(temporary.cleanup)
        # Use the long path when Windows provides an 8.3 temporary path.
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "harness"
        self.sources = self.root / "sources with spaces"
        self.build = self.root / "build"
        self.spirv = self.sources / "compiler/spirv-llvm-translator"
        self.systems = self.sources / "rocm-systems"
        self.hsa = self.systems / "projects/rocr-runtime/runtime/hsa-runtime/inc"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_AUTHOR_NAME="Source dependency test",
            GIT_AUTHOR_EMAIL="test@example.invalid",
            GIT_COMMITTER_NAME="Source dependency test",
            GIT_COMMITTER_EMAIL="test@example.invalid",
        )
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

    def run_command(self, *args):
        result = subprocess.run(
            [str(arg) for arg in args],
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def git(self, repository, *args):
        return self.run_command("git", "-C", repository, *args)

    def commit(self, repository, message):
        self.git(repository, "add", "--all")
        self.git(repository, "commit", "-qm", message)

    def write_harness(self):
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
            cmake_minimum_required(VERSION 3.25)
            project(source_dependencies NONE)
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
            file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")
            {includes}
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

    def configure(self, deps=None, *, prebuilt=False, hash_primary=False):
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
            name: (self.build / f"{name}.fprint").read_text().strip()
            for name in PROJECTS
        }

    def assert_valid(self, values, invalid=()):
        for name, value in values.items():
            if name in invalid:
                self.assertEqual(value, "", name)
            else:
                self.assertRegex(value, r"^[0-9a-f]{64}$", name)

    def assert_changed(self, before, after, expected):
        self.assert_valid(before)
        self.assert_valid(after)
        self.assertEqual(
            {name for name in PROJECTS if before[name] != after[name]}, set(expected)
        )

    def test_repeat_and_relative_source_dependencies(self):
        baseline = self.configure()
        self.assert_valid(baseline)
        self.assertEqual(baseline, self.configure())
        relative = tuple(
            Path(os.path.relpath(p, self.source)).as_posix()
            for p in (self.spirv, self.hsa)
        )
        self.assertEqual(baseline, self.configure(relative))
        # CMake may expand Windows short paths; compare resolved paths.
        recorded = (self.build / "source-deps.txt").read_text().split(";")
        self.assertEqual(
            [Path(path).resolve() for path in recorded],
            [self.spirv.resolve(), self.hsa.resolve()],
        )

    def test_source_changes_and_restoration_for_all_seven_projects(self):
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

    def test_each_llvm_external_source_changes_only_llvm_users_and_restores(self):
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

    def test_llvm_root_already_includes_clang(self):
        baseline = self.configure()
        llvm = self.sources / "compiler/amd-llvm"
        write(llvm / "clang/input.txt", "changed\n")
        self.commit(llvm, "Clang edit")
        self.assert_changed(baseline, self.configure(), LLVM_USERS)

    def test_optional_hsa_source_dependency_can_be_omitted(self):
        deps = (self.spirv.as_posix(),)
        baseline = self.configure(deps)
        self.assert_valid(baseline)
        write(self.hsa / "input.txt", "changed\n")
        self.commit(self.systems, "HSA edit")
        self.assertEqual(baseline, self.configure(deps))

    def test_unrelated_dirty_and_committed_changes_leave_fingerprints_stable(self):
        baseline = self.configure()
        write(self.systems / "unrelated/input.txt", "unrelated edit\n")
        self.assertEqual(baseline, self.configure())
        self.commit(self.systems, "unrelated edit")
        self.assertEqual(baseline, self.configure())

    def test_dirty_and_staged_external_sources_invalidate_consumers(self):
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

    def test_missing_directory_is_invalid_even_with_a_stale_marker(self):
        self.configure()
        # Make the declared path absent without deleting read-only Git objects
        # on Windows. TemporaryDirectory cleanup owns the moved repository.
        self.spirv.rename(self.root / "removed-spirv-llvm-translator")
        self.assertFalse(self.spirv.exists())
        write(self.spirv.with_name(".spirv-llvm-translator.smrev"), "stale\n")
        self.assert_valid(self.configure(), invalid=LLVM_USERS)

    def test_non_git_directory_requires_a_marker(self):
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

    def test_file_dependency_is_rejected_as_not_a_directory(self):
        self.assert_valid(
            self.configure(((self.spirv / "input.txt").as_posix(),)), invalid=LLVM_USERS
        )

    def test_empty_marker_uses_git_and_dirty_source_invalidates_marker(self):
        baseline = self.configure()
        marker = self.spirv.with_name(".spirv-llvm-translator.smrev")
        write(marker, "")
        self.assertEqual(baseline, self.configure())
        write(marker, "fixed marker\n")
        self.assert_valid(self.configure())
        write(self.spirv / "input.txt", "dirty\n")
        self.assert_valid(self.configure(), invalid=LLVM_USERS)

    def test_same_tree_commit_keeps_external_dependency_identity(self):
        baseline = self.configure()
        self.git(self.spirv, "commit", "--allow-empty", "-qm", "same tree")
        self.assertEqual(baseline, self.configure())

    def test_dependencies_apply_when_primary_source_is_file_hashed(self):
        baseline = self.configure(hash_primary=True)
        write(self.spirv / "input.txt", "changed\n")
        self.commit(self.spirv, "source edit")
        self.assert_changed(baseline, self.configure(hash_primary=True), LLVM_USERS)

    def test_dependencies_apply_to_prebuilt_fingerprints(self):
        baseline = self.configure(prebuilt=True)
        write(self.spirv / "input.txt", "changed\n")
        self.commit(self.spirv, "source edit")
        self.assert_changed(baseline, self.configure(prebuilt=True), LLVM_USERS)
        write(self.spirv / "input.txt", "dirty\n")
        self.assert_valid(self.configure(prebuilt=True), invalid=LLVM_USERS)


if __name__ == "__main__":
    unittest.main()
