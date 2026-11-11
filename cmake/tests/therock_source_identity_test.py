#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test the source-directory helper with disposable Git repositories.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


THEROCK_ROOT = Path(__file__).resolve().parents[2]


class SourceIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("cmake") or not shutil.which("git"):
            raise RuntimeError("CMake and Git must be on PATH; tests were not run")
        module = (THEROCK_ROOT / "cmake/therock_subproject.cmake").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"(?ms)^function\(_therock_subproject_fprint_source_dir\b.*?"
            r"^endfunction\(\)", module
        )
        if not match:
            raise RuntimeError("Cannot find the source-directory helper")
        cls.helper = match.group(0)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-source-identity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "amd-llvm"
        self.repo.mkdir()
        # Keep tests independent of the user's Git settings and credentials.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
            GIT_AUTHOR_NAME="Source identity test",
            GIT_AUTHOR_EMAIL="test@example.invalid",
            GIT_COMMITTER_NAME="Source identity test",
            GIT_COMMITTER_EMAIL="test@example.invalid",
        )
        self.git("init", "-q")
        for directory in ("llvm", "clang", "lld", "compiler-rt", "cmake", "amd/comgr", "amd/hipcc"):
            self.write(directory + "/input.txt", directory + "\n")
        self.write("outside.txt", "outside\n")
        self.commit("initial")
        self.script = self.root / "read-identity.cmake"
        self.script.write_text(
            "cmake_minimum_required(VERSION 3.20)\n" + self.helper + "\n"
            '_therock_subproject_fprint_source_dir(identity "${SOURCE}")\n'
            'file(WRITE "${RESULT}" "${identity}")\n', encoding="utf-8"
        )

    def run_command(self, *args, check=True):
        result = subprocess.run(
            [str(arg) for arg in args], env=self.env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if check and result.returncode:
            self.fail(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
        return result

    def git(self, *args):
        return self.run_command("git", "-C", self.repo, *args).stdout.strip()

    def write(self, relative, contents):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-qm", message)

    def identity(self, directory=None):
        result = self.root / "result.txt"
        result.unlink(missing_ok=True)
        self.run_command(
            "cmake", f"-DSOURCE={(directory or self.repo).as_posix()}",
            f"-DRESULT={result.as_posix()}", "-P", self.script,
        )
        return result.read_text(encoding="utf-8")

    def marker(self, contents):
        path = self.repo.with_name("." + self.repo.name + ".smrev")
        path.write_text(contents, encoding="utf-8")
        return path

    def test_repository_root_uses_complete_tree(self):
        self.assertEqual(self.identity(), self.git("rev-parse", "HEAD^{tree}"))

    def test_same_tree_commit_keeps_identity(self):
        before, head = self.identity(), self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "new commit, same tree")
        self.assertNotEqual(head, self.git("rev-parse", "HEAD"))
        self.assertEqual(before, self.identity())

    def test_llvm_root_includes_sibling_source_directories(self):
        for directory in ("llvm", "clang", "lld", "compiler-rt", "cmake", "amd/comgr", "amd/hipcc"):
            with self.subTest(directory=directory):
                before = self.identity()
                self.write(directory + "/input.txt", "changed\n")
                self.commit("change " + directory)
                self.assertNotEqual(before, self.identity())

    def test_subdirectory_ignores_unrelated_commits(self):
        directory = self.repo / "llvm"
        before = self.identity(directory)
        self.write("outside.txt", "changed\n")
        self.commit("outside selected directory")
        self.assertEqual(before, self.identity(directory))

    def test_subdirectory_ignores_unrelated_tracked_edits(self):
        directory = self.repo / "llvm"
        before = self.identity(directory)
        self.write("outside.txt", "changed\n")
        self.assertEqual(before, self.identity(directory))
        self.git("add", "outside.txt")
        self.assertEqual(before, self.identity(directory))

    def test_tracked_edits_inside_scope_are_invalid(self):
        directory = self.repo / "llvm"
        self.write("llvm/input.txt", "dirty\n")
        self.assertEqual("", self.identity(directory))
        self.git("add", "llvm/input.txt")
        self.assertEqual("", self.identity(directory))
        self.assertEqual("", self.identity())

    def test_tracked_deletion_is_invalid(self):
        (self.repo / "llvm/input.txt").unlink()
        self.assertEqual("", self.identity())

    def test_content_revert_restores_identity(self):
        before = self.identity()
        self.write("llvm/input.txt", "changed\n")
        self.commit("change")
        self.assertNotEqual(before, self.identity())
        self.git("revert", "--no-edit", "HEAD")
        self.assertEqual(before, self.identity())

    def test_marker_retains_priority_and_exact_contents(self):
        contents = "https://example.invalid/llvm.git\nrevision+PATCHED:digest\n"
        self.marker(contents)
        self.assertEqual(contents, self.identity())

    def test_dirty_tracked_source_invalidates_marker(self):
        self.marker("fixed source identity\n")
        self.write("clang/input.txt", "dirty\n")
        self.assertEqual("", self.identity())

    def test_empty_marker_uses_git_tree(self):
        self.marker("")
        self.assertEqual(self.git("rev-parse", "HEAD^{tree}"), self.identity())

    def test_non_git_source_uses_marker(self):
        source = self.root / "fetched"
        source.mkdir()
        (self.root / ".fetched.smrev").write_text("archive-and-recipe", encoding="utf-8")
        self.assertEqual("archive-and-recipe", self.identity(source))

    def test_non_git_source_without_marker_is_invalid(self):
        source = self.root / "unversioned"
        source.mkdir()
        self.assertEqual("", self.identity(source))

    def test_unborn_repository_is_invalid(self):
        source = self.root / "unborn"
        source.mkdir()
        self.run_command("git", "-C", source, "init", "-q")
        self.assertEqual("", self.identity(source))

    def test_linked_worktree_uses_tree_identity(self):
        worktree = self.root / "linked"
        self.git("worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertEqual(self.identity(), self.identity(worktree))

    def test_existing_policy_does_not_detect_untracked_files(self):
        # Documents a limitation; it is not a cache-safety acceptance test.
        before = self.identity()
        self.write("clang/untracked.cpp", "untracked\n")
        self.assertEqual(before, self.identity())


if __name__ == "__main__":
    unittest.main()
