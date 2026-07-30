# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for dev-build version tagging in build_prod_wheels.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent / "external-builds" / "pytorch"))

from build_prod_wheels import compute_build_version, get_source_commit_short


class ComputeBuildVersionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = Path(self.temp_dir.name)
        (self.source_dir / "version.txt").write_text("2.12.0a0\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_dev_release_uses_base_plus_suffix(self):
        version = compute_build_version(self.source_dir, "+rocm7.10.0", "ci")
        self.assertEqual(version, "2.12.0a0+rocm7.10.0")

    def test_dev_release_tags_git_commit_in_local_segment(self):
        with mock.patch(
            "build_prod_wheels.get_source_commit_short", return_value="1a2b3c4d"
        ):
            version = compute_build_version(self.source_dir, "+rocm7.10.0", "dev")
        self.assertEqual(version, "2.12.0a0+git1a2b3c4d.rocm7.10.0")

    def test_dev_release_without_commit_falls_back_to_suffix(self):
        with mock.patch(
            "build_prod_wheels.get_source_commit_short", return_value=""
        ):
            version = compute_build_version(self.source_dir, "+rocm7.10.0", "dev")
        self.assertEqual(version, "2.12.0a0+rocm7.10.0")


class GetSourceCommitShortTest(unittest.TestCase):
    def test_reads_short_commit_from_git_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.check_call(["git", "init"], cwd=repo)
            subprocess.check_call(
                ["git", "config", "user.email", "test@example.com"], cwd=repo
            )
            subprocess.check_call(
                ["git", "config", "user.name", "Test User"], cwd=repo
            )
            (repo / "version.txt").write_text("2.12.0a0\n")
            subprocess.check_call(["git", "add", "version.txt"], cwd=repo)
            subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)
            commit = get_source_commit_short(repo, length=8)
            self.assertEqual(len(commit), 8)


if __name__ == "__main__":
    unittest.main()
