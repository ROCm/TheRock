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

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from _therock_utils import source_date
from _therock_utils.source_date import (
    compute_source_date_epoch,
    dirty_paths,
    is_worktree_dirty,
    newest_dirty_mtime,
    submodule_entries,
)

JAN_2026 = 1767225600  # 2026-01-01T00:00:00Z, the submodule's first commit.
FEB_2026 = 1769990400  # 2026-02-02T00:00:00Z, the superproject pinning it.
JUN_2026 = 1781481600  # 2026-06-15T00:00:00Z, later work inside the submodule.
DEC_2026 = 1798761600  # 2026-12-31T00:00:00Z, an uncommitted edit.
JAN_2020 = 1577836800  # 2020-01-01T00:00:00Z, deliberately stale.


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

    def test_dirty_submodule_worktree_uses_the_file_mtime(self):
        edited = self.sub_checkout / "f.txt"
        edited.write_text("uncommitted")
        os.utime(edited, (DEC_2026, DEC_2026))

        # The commit still matches the pin, so this is invisible to
        # `git submodule status` -- the dirty scan is what catches it.
        self.assertEqual([e.state for e in submodule_entries(self.super)], [" "])
        self.assertTrue(is_worktree_dirty(self.super))
        self.assertEqual(compute_source_date_epoch(self.super), DEC_2026)

    def test_dirty_superproject_uses_the_file_mtime(self):
        edited = self.super / "README.md"
        edited.write_text("edited")
        os.utime(edited, (DEC_2026, DEC_2026))
        self.assertTrue(is_worktree_dirty(self.super))
        self.assertEqual(compute_source_date_epoch(self.super), DEC_2026)

    def test_untracked_file_counts_as_dirty(self):
        untracked = self.super / "scratch.txt"
        untracked.write_text("scratch")
        os.utime(untracked, (DEC_2026, DEC_2026))
        self.assertTrue(is_worktree_dirty(self.super))
        self.assertEqual(compute_source_date_epoch(self.super), DEC_2026)

    def test_newest_dirty_file_wins(self):
        older = self.super / "older.txt"
        older.write_text("a")
        os.utime(older, (JUN_2026, JUN_2026))
        newer = self.super / "newer.txt"
        newer.write_text("b")
        os.utime(newer, (DEC_2026, DEC_2026))
        self.assertEqual(newest_dirty_mtime(self.super), DEC_2026)
        self.assertEqual(compute_source_date_epoch(self.super), DEC_2026)

    def test_an_old_dirty_file_cannot_drag_the_result_backwards(self):
        # Touching a file to 2020 must not make the archive look older than the
        # commit it sits on.
        stale = self.super / "README.md"
        stale.write_text("edited")
        os.utime(stale, (JAN_2020, JAN_2020))
        self.assertEqual(newest_dirty_mtime(self.super), JAN_2020)
        self.assertEqual(compute_source_date_epoch(self.super), FEB_2026)

    def test_repeated_calls_on_an_untouched_dirty_tree_agree(self):
        # The whole reason this is a file mtime rather than the current time:
        # an unchanged tree must resolve identically every time, or every
        # reconfigure would invalidate everything downstream.
        (self.super / "README.md").write_text("edited")
        first = compute_source_date_epoch(self.super)
        time.sleep(1.1)
        self.assertEqual(compute_source_date_epoch(self.super), first)

    def test_a_path_with_spaces_is_handled(self):
        # `git status --porcelain` quotes these; the -z form does not.
        spaced = self.super / "a file with spaces.txt"
        spaced.write_text("x")
        os.utime(spaced, (DEC_2026, DEC_2026))
        self.assertIn(spaced, dirty_paths(self.super))
        self.assertEqual(compute_source_date_epoch(self.super), DEC_2026)

    def test_a_deleted_file_does_not_break_resolution(self):
        (self.super / "README.md").unlink()
        self.assertTrue(is_worktree_dirty(self.super))
        # Nothing to stat, so it falls back to the commit times.
        self.assertEqual(compute_source_date_epoch(self.super), FEB_2026)

    def test_outside_a_git_checkout_falls_back_to_now(self):
        before = int(time.time())
        with tempfile.TemporaryDirectory() as plain:
            self.assertEqual(submodule_entries(Path(plain)), [])
            self.assertFalse(is_worktree_dirty(Path(plain)))
            self.assertGreaterEqual(compute_source_date_epoch(Path(plain)), before)

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

    # -- the manifest handoff ---------------------------------------------

    def _write_manifest(self, root: Path, **fields) -> Path:
        """Writes a manifest in the flattened artifact layout."""
        manifest = {
            "the_rock_commit": "a" * 40,
            "source_date_epoch": DEC_2026,
            "source_dirty": False,
            "submodules": [],
        }
        manifest.update(fields)
        path = root / "share" / "therock" / "therock_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))
        return path

    def test_manifest_timestamp_is_preferred_over_this_checkout(self):
        # The artifacts were built elsewhere; their recorded value is the one
        # that describes them.
        artifacts = Path(self.temp_context.name) / "artifacts"
        self._write_manifest(artifacts)
        self.assertEqual(
            source_date.resolve(manifest_dir=artifacts, repo_dir=self.super),
            DEC_2026,
        )

    def test_exploded_artifact_layout_is_found_too(self):
        artifacts = Path(self.temp_context.name) / "exploded"
        path = artifacts / source_date.MANIFEST_RELPATHS[0]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"source_date_epoch": JUN_2026}))
        self.assertEqual(source_date.timestamp_from_manifest(artifacts), JUN_2026)

    def test_missing_manifest_falls_back_to_this_checkout(self):
        empty = Path(self.temp_context.name) / "empty"
        empty.mkdir()
        self.assertIsNone(source_date.timestamp_from_manifest(empty))
        self.assertEqual(
            source_date.resolve(manifest_dir=empty, repo_dir=self.super), FEB_2026
        )

    def test_manifest_without_the_field_falls_back(self):
        # Artifacts built before the field existed, or with no git metadata.
        artifacts = Path(self.temp_context.name) / "old"
        self._write_manifest(artifacts, source_date_epoch=None)
        self.assertIsNone(source_date.timestamp_from_manifest(artifacts))
        self.assertEqual(
            source_date.resolve(manifest_dir=artifacts, repo_dir=self.super), FEB_2026
        )

    # -- drift ------------------------------------------------------------

    def test_matching_commit_and_clean_trees_report_no_drift(self):
        head = git("rev-parse", "HEAD", cwd=self.super).strip()
        artifacts = Path(self.temp_context.name) / "matching"
        self._write_manifest(artifacts, the_rock_commit=head)
        self.assertEqual(source_date.describe_source_drift(artifacts, self.super), [])

    def test_a_different_commit_is_reported(self):
        artifacts = Path(self.temp_context.name) / "other_run"
        self._write_manifest(artifacts, the_rock_commit="b" * 40)
        reasons = source_date.describe_source_drift(artifacts, self.super)
        self.assertTrue(any("but this checkout is at" in r for r in reasons))

    def test_artifacts_built_dirty_are_reported(self):
        head = git("rev-parse", "HEAD", cwd=self.super).strip()
        artifacts = Path(self.temp_context.name) / "built_dirty"
        self._write_manifest(artifacts, the_rock_commit=head, source_dirty=True)
        reasons = source_date.describe_source_drift(artifacts, self.super)
        self.assertTrue(any("uncommitted changes" in r for r in reasons))

    def test_a_missing_manifest_is_itself_drift(self):
        empty = Path(self.temp_context.name) / "nothing"
        empty.mkdir()
        reasons = source_date.describe_source_drift(empty, self.super)
        self.assertTrue(any("no therock_manifest.json" in r for r in reasons))

    def test_fail_on_drift_raises_and_otherwise_only_warns(self):
        artifacts = Path(self.temp_context.name) / "drifted"
        self._write_manifest(artifacts, the_rock_commit="c" * 40)

        warnings = []
        value = source_date.resolve_checked(
            manifest_dir=artifacts, repo_dir=self.super, report=warnings.append
        )
        self.assertEqual(value, DEC_2026)
        self.assertTrue(warnings)

        with self.assertRaisesRegex(RuntimeError, "drift"):
            source_date.resolve_checked(
                manifest_dir=artifacts, repo_dir=self.super, fail_on_drift=True
            )

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
