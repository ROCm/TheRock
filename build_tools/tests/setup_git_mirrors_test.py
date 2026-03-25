#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for setup_git_mirrors.py."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from setup_git_mirrors import (
    SubmoduleInfo,
    MirrorResult,
    _parse_ref_lines,
    needs_update,
    create_mirror,
    update_mirror,
    prune_stale_mirrors,
    discover_submodules,
)


def _make_submodule(
    name: str = "llvm-project",
    path: str = "compiler/amd-llvm",
    url: str = "https://github.com/ROCm/llvm-project.git",
    mirror_dir: Path | None = None,
) -> SubmoduleInfo:
    """Helper to build a SubmoduleInfo for tests."""
    if mirror_dir is None:
        mirror_dir = Path("/tmp/test-mirrors")
    return SubmoduleInfo(
        name=name,
        path=path,
        url=url,
        mirror_path=mirror_dir / "ROCm" / f"{name}.git",
    )


class ParseRefLinesTest(unittest.TestCase):
    """Tests for _parse_ref_lines."""

    def test_normal_output(self):
        output = (
            "abc123 refs/heads/main\n"
            "def456 refs/heads/dev\n"
            "789aaa refs/tags/v1.0\n"
        )
        result = _parse_ref_lines(output)
        self.assertEqual(
            result,
            {
                "refs/heads/main": "abc123",
                "refs/heads/dev": "def456",
                "refs/tags/v1.0": "789aaa",
            },
        )

    def test_skips_head(self):
        output = "abc123 HEAD\n" "abc123 refs/heads/main\n"
        result = _parse_ref_lines(output)
        self.assertEqual(result, {"refs/heads/main": "abc123"})
        self.assertNotIn("HEAD", result)

    def test_empty_string(self):
        self.assertEqual(_parse_ref_lines(""), {})

    def test_whitespace_only(self):
        self.assertEqual(_parse_ref_lines("  \n\n  "), {})

    def test_malformed_lines_skipped(self):
        output = "abc123\nvalid_sha refs/heads/main\n"
        result = _parse_ref_lines(output)
        self.assertEqual(result, {"refs/heads/main": "valid_sha"})

    def test_tab_separated(self):
        """ls-remote uses tabs; show-ref uses spaces. Both should work."""
        output = "abc123\trefs/heads/main\ndef456\trefs/tags/v1.0\n"
        result = _parse_ref_lines(output)
        self.assertEqual(
            result,
            {
                "refs/heads/main": "abc123",
                "refs/tags/v1.0": "def456",
            },
        )


class NeedsUpdateTest(unittest.TestCase):
    """Tests for needs_update."""

    def _mock_run(self, local_stdout: str, remote_stdout: str, remote_rc: int = 0):
        """Return a side_effect callable for subprocess.run."""

        def side_effect(cmd, **kwargs):
            if "show-ref" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=local_stdout, stderr=""
                )
            if "ls-remote" in cmd:
                return subprocess.CompletedProcess(
                    cmd, remote_rc, stdout=remote_stdout, stderr=""
                )
            raise ValueError(f"Unexpected command: {cmd}")

        return side_effect

    @mock.patch("setup_git_mirrors.subprocess.run")
    def test_refs_match_returns_false(self, mock_run):
        refs = "abc123 refs/heads/main\ndef456 refs/tags/v1.0\n"
        mock_run.side_effect = self._mock_run(refs, refs)
        self.assertFalse(
            needs_update(Path("/mirrors/repo.git"), "https://example.com/repo.git")
        )

    @mock.patch("setup_git_mirrors.subprocess.run")
    def test_refs_differ_returns_true(self, mock_run):
        local = "abc123 refs/heads/main\n"
        remote = "abc123 refs/heads/main\nnew999 refs/heads/feature\n"
        mock_run.side_effect = self._mock_run(local, remote)
        self.assertTrue(
            needs_update(Path("/mirrors/repo.git"), "https://example.com/repo.git")
        )

    @mock.patch("setup_git_mirrors.subprocess.run")
    def test_ls_remote_failure_returns_true(self, mock_run):
        mock_run.side_effect = self._mock_run(
            "abc123 refs/heads/main\n", "", remote_rc=1
        )
        self.assertTrue(
            needs_update(Path("/mirrors/repo.git"), "https://example.com/repo.git")
        )

    @mock.patch("setup_git_mirrors.subprocess.run", side_effect=OSError("no git"))
    def test_exception_returns_true(self, _mock_run):
        self.assertTrue(
            needs_update(Path("/mirrors/repo.git"), "https://example.com/repo.git")
        )


class CreateMirrorTest(unittest.TestCase):
    """Tests for create_mirror."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sub = _make_submodule(mirror_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("setup_git_mirrors.run_git")
    def test_success(self, mock_git):
        result = create_mirror(self.sub, retries=1)
        self.assertTrue(result.success)
        self.assertEqual(result.action, "created")
        self.assertEqual(result.submodule.name, "llvm-project")
        mock_git.assert_called_once()

    @mock.patch(
        "setup_git_mirrors.run_git", side_effect=subprocess.CalledProcessError(1, "git")
    )
    def test_all_retries_exhausted(self, mock_git):
        result = create_mirror(self.sub, retries=2)
        self.assertFalse(result.success)
        self.assertEqual(result.action, "failed")
        self.assertEqual(mock_git.call_count, 2)

    @mock.patch("setup_git_mirrors.time.sleep")
    @mock.patch("setup_git_mirrors.run_git")
    def test_retries_with_backoff(self, mock_git, mock_sleep):
        mock_git.side_effect = [
            subprocess.CalledProcessError(1, "git"),
            None,
        ]
        result = create_mirror(self.sub, retries=2)
        self.assertTrue(result.success)
        self.assertEqual(result.action, "created")
        mock_sleep.assert_called_once()

    @mock.patch(
        "setup_git_mirrors.run_git", side_effect=subprocess.CalledProcessError(1, "git")
    )
    def test_cleans_up_partial_dir_on_failure(self, _mock_git):
        self.sub.mirror_path.mkdir(parents=True)
        (self.sub.mirror_path / "HEAD").touch()
        create_mirror(self.sub, retries=1)
        self.assertFalse(self.sub.mirror_path.exists())


class UpdateMirrorTest(unittest.TestCase):
    """Tests for update_mirror."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sub = _make_submodule(mirror_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("setup_git_mirrors.create_mirror")
    def test_creates_if_missing(self, mock_create):
        mock_create.return_value = MirrorResult(
            submodule=self.sub, success=True, action="created", elapsed_seconds=1.0
        )
        result = update_mirror(self.sub)
        self.assertEqual(result.action, "created")
        mock_create.assert_called_once()

    @mock.patch("setup_git_mirrors.needs_update", return_value=False)
    @mock.patch("setup_git_mirrors.run_git")
    def test_skips_when_up_to_date(self, mock_git, _mock_needs):
        self.sub.mirror_path.mkdir(parents=True)
        result = update_mirror(self.sub, skip_up_to_date=True)
        self.assertTrue(result.success)
        self.assertEqual(result.action, "skipped")
        mock_git.assert_not_called()

    @mock.patch("setup_git_mirrors.needs_update", return_value=False)
    @mock.patch("setup_git_mirrors.run_git")
    def test_force_overrides_skip(self, mock_git, _mock_needs):
        self.sub.mirror_path.mkdir(parents=True)
        result = update_mirror(self.sub, force=True)
        self.assertTrue(result.success)
        self.assertEqual(result.action, "updated")
        mock_git.assert_called_once()

    @mock.patch("setup_git_mirrors.needs_update", return_value=True)
    @mock.patch("setup_git_mirrors.run_git")
    def test_updates_when_refs_differ(self, mock_git, _mock_needs):
        self.sub.mirror_path.mkdir(parents=True)
        result = update_mirror(self.sub)
        self.assertTrue(result.success)
        self.assertEqual(result.action, "updated")


class PruneStaleMirrorsTest(unittest.TestCase):
    """Tests for prune_stale_mirrors."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mirror_dir = self.temp_dir / "mirrors"
        self.mirror_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_removes_stale_mirror(self):
        org_dir = self.mirror_dir / "ROCm"
        org_dir.mkdir()
        stale = org_dir / "old-repo.git"
        stale.mkdir()
        (stale / "HEAD").touch()

        active_sub = _make_submodule(name="llvm-project", mirror_dir=self.mirror_dir)
        prune_stale_mirrors(self.mirror_dir, [active_sub])

        self.assertFalse(stale.exists())

    def test_keeps_active_mirror(self):
        active_sub = _make_submodule(name="llvm-project", mirror_dir=self.mirror_dir)
        active_sub.mirror_path.mkdir(parents=True)
        (active_sub.mirror_path / "HEAD").touch()

        prune_stale_mirrors(self.mirror_dir, [active_sub])

        self.assertTrue(active_sub.mirror_path.exists())

    def test_removes_empty_org_dir(self):
        org_dir = self.mirror_dir / "StaleOrg"
        org_dir.mkdir()
        stale = org_dir / "gone-repo.git"
        stale.mkdir()

        prune_stale_mirrors(self.mirror_dir, [])

        self.assertFalse(org_dir.exists())

    def test_nonexistent_mirror_dir_is_noop(self):
        prune_stale_mirrors(self.mirror_dir / "nonexistent", [])


class DiscoverSubmodulesTest(unittest.TestCase):
    """Tests for discover_submodules."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mirror_dir = self.temp_dir / "mirrors"
        self.mirror_dir.mkdir()
        self.gitmodules_path = self.temp_dir / ".gitmodules"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_gitmodules(self, content: str):
        self.gitmodules_path.write_text(content)

    @mock.patch("setup_git_mirrors.subprocess.run")
    def test_parses_submodules(self, mock_run):
        self._write_gitmodules(
            '[submodule "llvm-project"]\n'
            "\tpath = compiler/amd-llvm\n"
            "\turl = https://github.com/ROCm/llvm-project.git\n"
        )

        def side_effect(cmd, **kwargs):
            if "--get-regexp" in cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="submodule.llvm-project.url https://github.com/ROCm/llvm-project.git\n",
                    stderr="",
                )
            if "--get" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="compiler/amd-llvm\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        mock_run.side_effect = side_effect

        subs = discover_submodules(
            self.mirror_dir, gitmodules_path=self.gitmodules_path
        )
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].name, "llvm-project")
        self.assertEqual(subs[0].path, "compiler/amd-llvm")
        self.assertEqual(subs[0].url, "https://github.com/ROCm/llvm-project.git")
        self.assertEqual(
            subs[0].mirror_path,
            self.mirror_dir / "ROCm" / "llvm-project.git",
        )

    def test_missing_gitmodules_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_submodules(
                self.mirror_dir,
                gitmodules_path=self.temp_dir / "nonexistent",
            )


if __name__ == "__main__":
    unittest.main()
