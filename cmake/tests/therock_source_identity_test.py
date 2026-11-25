#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test the source-directory identity helper with disposable Git repositories."""

import unittest
from pathlib import Path

from therock_fingerprint_test_utils import (
    FingerprintTestCase,
    cmake_project_setup,
    write,
)


class SourceIdentityTest(FingerprintTestCase):
    required_tools = ("cmake", "git", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.git("init", "-q")
        write(self.repo / "main/input.txt", "main\n")
        write(self.repo / "outside.txt", "outside\n")
        self.commit("initial")
        self.harness = self.root / "harness"
        self.build = self.root / "build"
        self.result = self.root / "result.txt"
        # Loads the real therock_subproject.cmake, so the helper under test is
        # the one subprojects use.
        write(
            self.harness / "CMakeLists.txt",
            f"""
            {cmake_project_setup("source_identity")}
            _therock_subproject_fprint_source_dir(identity "${{SOURCE}}")
            file(WRITE "${{RESULT}}" "${{identity}}")
            """,
        )

    def git(self, *args: str | Path) -> str:
        return self.run_command("git", "-C", self.repo, *args).stdout.strip()

    def commit(self, message: str) -> None:
        self.git("add", "--all")
        self.git("commit", "-qm", message)

    def identity(self, directory: Path | None = None) -> str:
        self.result.unlink(missing_ok=True)
        self.run_command(
            "cmake",
            "-S",
            self.harness,
            "-B",
            self.build,
            "-G",
            "Ninja",
            f"-DSOURCE={(directory or self.repo).as_posix()}",
            f"-DRESULT={self.result.as_posix()}",
        )
        return self.result.read_text(encoding="utf-8")

    def marker(self, contents: str) -> None:
        path = self.repo.with_name("." + self.repo.name + ".smrev")
        path.write_text(contents, encoding="utf-8")

    def test_repository_root_uses_complete_tree(self) -> None:
        self.assertEqual(self.identity(), self.git("rev-parse", "HEAD^{tree}"))

    def test_subdirectory_uses_its_own_tree(self) -> None:
        self.assertEqual(
            self.identity(self.repo / "main"), self.git("rev-parse", "HEAD:main")
        )

    def test_same_tree_commit_keeps_identity(self) -> None:
        before = self.identity()
        previous_commit = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "new commit, same tree")
        self.assertNotEqual(previous_commit, self.git("rev-parse", "HEAD"))
        self.assertEqual(before, self.identity())

    def test_subdirectory_ignores_unrelated_commits(self) -> None:
        before = self.identity(self.repo / "main")
        write(self.repo / "outside.txt", "changed\n")
        self.commit("outside selected directory")
        self.assertEqual(before, self.identity(self.repo / "main"))

    def test_subdirectory_ignores_unrelated_tracked_edits(self) -> None:
        before = self.identity(self.repo / "main")
        write(self.repo / "outside.txt", "changed\n")
        self.assertEqual(before, self.identity(self.repo / "main"))
        self.git("add", "outside.txt")
        self.assertEqual(before, self.identity(self.repo / "main"))

    def test_tracked_edits_inside_scope_are_invalid(self) -> None:
        write(self.repo / "main/input.txt", "dirty\n")
        self.assertEqual("", self.identity(self.repo / "main"))
        self.git("add", "main/input.txt")
        self.assertEqual("", self.identity(self.repo / "main"))
        self.assertEqual("", self.identity())

    def test_marker_retains_priority_and_exact_contents(self) -> None:
        contents = "https://example.invalid/checkout.git\nrevision+PATCHED:digest\n"
        self.marker(contents)
        self.assertEqual(contents, self.identity())

    def test_linked_worktree_uses_tree_identity(self) -> None:
        worktree = self.root / "linked"
        self.git("worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertEqual(self.identity(), self.identity(worktree))


if __name__ == "__main__":
    unittest.main()
