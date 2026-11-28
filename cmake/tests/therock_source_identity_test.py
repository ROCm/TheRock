#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test how the identity of a source directory is computed.

A subproject's fingerprint includes an identity for each source directory it
reads. The identity is a string that changes when the directory's contents
change. It is the Git tree of the directory at HEAD, or the contents of a
`.<dirname>.smrev` file next to the directory if one exists.

An empty identity is invalid: the directory's contents cannot be described by a
commit, so the fingerprint is invalid and the subproject's outputs cannot be
reused. This happens when tracked files in the directory have uncommitted
changes.

The tests run the real helper from `therock_subproject.cmake` on a small Git
repository with a `main` directory and a file `outside.txt` next to it.
"""

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
        self.reuse_fixture(self.create_checkout)
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

    def create_checkout(self) -> None:
        self.repo.mkdir()
        self.git("init", "-q")
        write(self.repo / "main/input.txt", "main\n")
        write(self.repo / "outside.txt", "outside\n")
        self.commit("initial")

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
        """A repository root's identity is the tree of the whole HEAD commit."""
        self.assertEqual(self.identity(), self.git("rev-parse", "HEAD^{tree}"))

    def test_subdirectory_uses_its_own_tree(self) -> None:
        """The identity of a subdirectory is the tree of only that directory."""
        self.assertEqual(
            self.identity(self.repo / "main"), self.git("rev-parse", "HEAD:main")
        )

    def test_same_tree_commit_keeps_identity(self) -> None:
        """A commit that changes no files keeps the identity.

        The identity follows the files, not the commit hash, so a new commit
        with the same tree is the same source.
        """
        before = self.identity()
        previous_commit = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "new commit, same tree")
        self.assertNotEqual(previous_commit, self.git("rev-parse", "HEAD"))
        self.assertEqual(before, self.identity())

    def test_subdirectory_ignores_unrelated_commits(self) -> None:
        """Commits outside a directory keep that directory's identity.

        This is why the identity is scoped to the directory: in a monorepo,
        commits to other projects must not change this project's fingerprint.
        """
        before = self.identity(self.repo / "main")
        write(self.repo / "outside.txt", "changed\n")
        self.commit("outside selected directory")
        self.assertEqual(before, self.identity(self.repo / "main"))

    def test_subdirectory_ignores_unrelated_tracked_edits(self) -> None:
        """Uncommitted edits outside a directory keep its identity valid.

        The check for uncommitted changes only looks inside the directory, so
        edits to `outside.txt`, staged or not, neither change nor invalidate
        `main`.
        """
        before = self.identity(self.repo / "main")
        write(self.repo / "outside.txt", "changed\n")
        self.assertEqual(before, self.identity(self.repo / "main"))
        self.git("add", "outside.txt")
        self.assertEqual(before, self.identity(self.repo / "main"))

    def test_tracked_edits_inside_scope_are_invalid(self) -> None:
        """Uncommitted edits inside a directory make its identity invalid.

        The Git tree describes HEAD, not the edited files that would be built.
        This holds for unstaged and staged edits, and for every directory that
        contains the edit, here `main` and the repository root.
        """
        write(self.repo / "main/input.txt", "dirty\n")
        self.assertEqual("", self.identity(self.repo / "main"))
        self.git("add", "main/input.txt")
        self.assertEqual("", self.identity(self.repo / "main"))
        self.assertEqual("", self.identity())

    def test_marker_retains_priority_and_exact_contents(self) -> None:
        """A `.smrev` file next to the directory is the identity, used verbatim.

        Some source fetch steps record the revision in this file instead.
        When it exists it takes priority over Git, and its exact contents,
        including newlines, are the identity.
        """
        contents = "https://example.invalid/checkout.git\nrevision+PATCHED:digest\n"
        self.marker(contents)
        self.assertEqual(contents, self.identity())

    def test_linked_worktree_uses_tree_identity(self) -> None:
        """A linked Git worktree at the same commit has the same identity.

        A worktree has a `.git` file instead of a `.git` directory; the identity
        still comes from the tree, so it matches the main checkout.
        """
        worktree = self.root / "linked"
        self.git("worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertEqual(self.identity(), self.identity(worktree))


if __name__ == "__main__":
    unittest.main()
