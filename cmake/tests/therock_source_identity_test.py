#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test the source-directory helper with disposable Git repositories."""

import re
import unittest
from pathlib import Path

from therock_fingerprint_test_utils import THEROCK_ROOT, FingerprintTestCase


class SourceIdentityTest(FingerprintTestCase):
    required_tools = ("cmake", "git")
    command_timeout_seconds = 60

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        module = (THEROCK_ROOT / "cmake/therock_subproject.cmake").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"(?ms)^function\(_therock_subproject_fprint_source_dir\b.*?"
            r"^endfunction\(\)",
            module,
        )
        if not match:
            raise RuntimeError("Cannot find the source-directory helper")
        cls.helper = match.group(0)

    def setUp(self) -> None:
        super().setUp()
        self.repo = self.root / "amd-llvm"
        self.repo.mkdir()
        self.git("init", "-q")
        for directory in (
            "llvm",
            "clang",
            "lld",
            "compiler-rt",
            "cmake",
            "amd/comgr",
            "amd/hipcc",
        ):
            self.write(directory + "/input.txt", directory + "\n")
        self.write("outside.txt", "outside\n")
        self.commit("initial")
        self.script = self.root / "read-identity.cmake"
        self.script.write_text(
            "cmake_minimum_required(VERSION 3.20)\n" + self.helper + "\n"
            '_therock_subproject_fprint_source_dir(identity "${SOURCE}")\n'
            'file(WRITE "${RESULT}" "${identity}")\n',
            encoding="utf-8",
        )

    def git(self, *args: str | Path) -> str:
        return self.run_command("git", "-C", self.repo, *args).stdout.strip()

    def write(self, relative: str, contents: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def commit(self, message: str) -> None:
        self.git("add", "--all")
        self.git("commit", "-qm", message)

    def identity(self, directory: Path | None = None) -> str:
        result = self.root / "result.txt"
        result.unlink(missing_ok=True)
        self.run_command(
            "cmake",
            f"-DSOURCE={(directory or self.repo).as_posix()}",
            f"-DRESULT={result.as_posix()}",
            "-P",
            self.script,
        )
        return result.read_text(encoding="utf-8")

    def marker(self, contents: str) -> Path:
        path = self.repo.with_name("." + self.repo.name + ".smrev")
        path.write_text(contents, encoding="utf-8")
        return path

    def test_repository_root_uses_complete_tree(self) -> None:
        self.assertEqual(self.identity(), self.git("rev-parse", "HEAD^{tree}"))

    def test_source_identities_survive_checkout_relocation(self) -> None:
        before = {scope: self.identity(self.repo / scope) for scope in (".", "llvm")}
        relocated = self.root / "relocated checkout with spaces"
        self.repo.rename(relocated)
        for scope, identity in before.items():
            with self.subTest(scope=scope):
                self.assertRegex(identity, r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
                self.assertEqual(identity, self.identity(relocated / scope))

    def test_same_tree_commit_keeps_identity(self) -> None:
        before = self.identity()
        previous_commit = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "new commit, same tree")
        self.assertNotEqual(previous_commit, self.git("rev-parse", "HEAD"))
        self.assertEqual(before, self.identity())

    def test_subdirectory_ignores_unrelated_commits(self) -> None:
        directory = self.repo / "llvm"
        before = self.identity(directory)
        self.write("outside.txt", "changed\n")
        self.commit("outside selected directory")
        self.assertEqual(before, self.identity(directory))

    def test_subdirectory_ignores_unrelated_tracked_edits(self) -> None:
        directory = self.repo / "llvm"
        before = self.identity(directory)
        self.write("outside.txt", "changed\n")
        self.assertEqual(before, self.identity(directory))
        self.git("add", "outside.txt")
        self.assertEqual(before, self.identity(directory))

    def test_tracked_edits_inside_scope_are_invalid(self) -> None:
        directory = self.repo / "llvm"
        self.write("llvm/input.txt", "dirty\n")
        self.assertEqual("", self.identity(directory))
        self.git("add", "llvm/input.txt")
        self.assertEqual("", self.identity(directory))
        self.assertEqual("", self.identity())

    def test_tracked_deletion_is_invalid(self) -> None:
        (self.repo / "llvm/input.txt").unlink()
        self.assertEqual("", self.identity())

    def test_marker_retains_priority_and_exact_contents(self) -> None:
        contents = "https://example.invalid/llvm.git\nrevision+PATCHED:digest\n"
        self.marker(contents)
        self.assertEqual(contents, self.identity())

    def test_dirty_tracked_source_invalidates_marker(self) -> None:
        self.marker("fixed source identity\n")
        self.write("clang/input.txt", "dirty\n")
        self.assertEqual("", self.identity())

    def test_empty_marker_uses_git_tree(self) -> None:
        self.marker("")
        self.assertEqual(self.git("rev-parse", "HEAD^{tree}"), self.identity())

    def test_non_git_source_uses_marker(self) -> None:
        source = self.root / "fetched"
        source.mkdir()
        (self.root / ".fetched.smrev").write_text(
            "archive-and-recipe", encoding="utf-8"
        )
        self.assertEqual("archive-and-recipe", self.identity(source))

    def test_non_git_source_without_marker_is_invalid(self) -> None:
        source = self.root / "unversioned"
        source.mkdir()
        self.assertEqual("", self.identity(source))

    def test_unborn_repository_is_invalid(self) -> None:
        source = self.root / "unborn"
        source.mkdir()
        self.run_command("git", "-C", source, "init", "-q")
        self.assertEqual("", self.identity(source))

    def test_linked_worktree_uses_tree_identity(self) -> None:
        worktree = self.root / "linked"
        self.git("worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertEqual(self.identity(), self.identity(worktree))


if __name__ == "__main__":
    unittest.main()
