#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for resolving the source timestamp used by reproducible archives.

Each test builds a real superproject + submodule fixture, because the signals
this relies on are not interchangeable and the distinctions only show up against
real git. In particular `git submodule status` compares commits only: a
submodule whose working tree is dirty but whose commit still matches the pin
reports ' ', not '+'.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _therock_utils import source_date
from _therock_utils.source_date import (
    compute_source_date_epoch,
    is_worktree_dirty,
    submodule_entries,
)

JAN_2026 = 1767225600  # 2026-01-01T00:00:00Z, the submodule's first commit.
FEB_2026 = 1769990400  # 2026-02-02T00:00:00Z, the superproject pinning it.
JUN_2026 = 1781481600  # 2026-06-15T00:00:00Z, later work inside the submodule.


def git(*args: str, cwd: Path, when: int | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        stamp = f"@{when} +0000"
        env["GIT_AUTHOR_DATE"] = stamp
        env["GIT_COMMITTER_DATE"] = stamp
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@unittest.skipUnless(shutil.which("git"), "git is required")
class SourceDateTest(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_context.cleanup)
        root = Path(self.temp_context.name)

        self.sub = root / "sub"
        self.sub.mkdir()
        git("init", "-q", "-b", "main", cwd=self.sub)
        git("config", "user.email", "t@e.st", cwd=self.sub)
        git("config", "user.name", "Test", cwd=self.sub)
        (self.sub / "f.txt").write_text("v1")
        git("add", "f.txt", cwd=self.sub)
        git("commit", "-q", "-m", "sub v1", cwd=self.sub, when=JAN_2026)

        self.super = root / "super"
        self.super.mkdir()
        git("init", "-q", "-b", "main", cwd=self.super)
        git("config", "user.email", "t@e.st", cwd=self.super)
        git("config", "user.name", "Test", cwd=self.super)
        (self.super / "README.md").write_text("top")
        git("add", "README.md", cwd=self.super)
        git("commit", "-q", "-m", "init", cwd=self.super, when=FEB_2026)
        git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(self.sub),
            "deps/sub",
            cwd=self.super,
        )
        git("add", "-A", cwd=self.super)
        git("commit", "-q", "-m", "pin sub", cwd=self.super, when=FEB_2026)

        self.sub_checkout = self.super / "deps" / "sub"

    # -- the states --------------------------------------------------------

    def test_clean_tree_uses_the_superproject_commit(self):
        # A submodule commit must exist before it can be pinned, so on a clean
        # tree the superproject is already the newest and the max is a no-op.
        self.assertEqual(compute_source_date_epoch(self.super), FEB_2026)

    def test_submodule_past_its_pin_advances_the_timestamp(self):
        (self.sub_checkout / "f.txt").write_text("v2")
        git("add", "f.txt", cwd=self.sub_checkout)
        git("commit", "-q", "-m", "sub v2", cwd=self.sub_checkout, when=JUN_2026)

        entries = [e for e in submodule_entries(self.super) if e.differs_from_pin]
        self.assertEqual([e.state for e in entries], ["+"])
        self.assertEqual(compute_source_date_epoch(self.super), JUN_2026)

    def test_dirty_submodule_worktree_advances_past_every_commit(self):
        (self.sub_checkout / "f.txt").write_text("uncommitted")

        # The commit still matches the pin, so this is invisible to
        # `git submodule status` -- the dirty check is what catches it.
        self.assertEqual([e.state for e in submodule_entries(self.super)], [" "])
        self.assertTrue(is_worktree_dirty(self.super))
        self.assertGreater(compute_source_date_epoch(self.super), JUN_2026)

    def test_dirty_superproject_advances_past_every_commit(self):
        (self.super / "README.md").write_text("edited")
        self.assertTrue(is_worktree_dirty(self.super))
        self.assertGreater(compute_source_date_epoch(self.super), FEB_2026)

    def test_untracked_file_counts_as_dirty(self):
        (self.super / "scratch.txt").write_text("scratch")
        self.assertTrue(is_worktree_dirty(self.super))

    def test_outside_a_git_checkout_falls_back_to_now(self):
        with tempfile.TemporaryDirectory() as plain:
            self.assertEqual(submodule_entries(Path(plain)), [])
            self.assertFalse(is_worktree_dirty(Path(plain)))
            self.assertGreater(compute_source_date_epoch(Path(plain)), FEB_2026)

    # -- parsing -----------------------------------------------------------

    def test_submodule_status_is_parsed(self):
        (entry,) = submodule_entries(self.super)
        self.assertEqual(entry.state, " ")
        self.assertEqual(len(entry.sha), 40)
        self.assertEqual(entry.path, self.sub_checkout)
        self.assertTrue(entry.is_populated)
        self.assertFalse(entry.differs_from_pin)

    def test_uninitialized_submodule_is_skipped(self):
        # No trailing "(describe)" field on these lines, and no checkout to ask.
        shutil.rmtree(self.sub_checkout)
        self.sub_checkout.mkdir()
        (entry,) = submodule_entries(self.super)
        self.assertEqual(entry.state, "-")
        self.assertFalse(entry.is_populated)
        self.assertEqual(entry.path, self.sub_checkout)

    # -- environment plumbing ---------------------------------------------

    def test_child_env_sets_only_the_namespaced_var_by_default(self):
        env = source_date.child_env(repo_dir=self.super, base_env={})
        self.assertEqual(env[source_date.ENV_VAR], str(FEB_2026))
        self.assertNotIn(source_date.STANDARD_ENV_VAR, env)

    def test_child_env_opt_in_also_exports_the_standard_var(self):
        env = source_date.child_env(
            export_standard_var=True, repo_dir=self.super, base_env={}
        )
        self.assertEqual(env[source_date.ENV_VAR], str(FEB_2026))
        self.assertEqual(env[source_date.STANDARD_ENV_VAR], str(FEB_2026))


if __name__ == "__main__":
    unittest.main()
