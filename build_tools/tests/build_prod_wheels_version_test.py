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

from packaging.version import Version

sys.path.insert(
    0, os.fspath(Path(__file__).parent.parent.parent / "external-builds" / "pytorch")
)

from build_prod_wheels import compute_build_version


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

    def test_versions_sort_by_release_type(self):
        # Composite PyTorch versions must preserve the intended ROCm release
        # ordering for pip install --upgrade: stable > prerelease > nightly > dev.
        stable = Version(compute_build_version(self.source_dir, "+rocm7.14.0", "ci"))
        prerelease = Version(
            compute_build_version(self.source_dir, "+rocm7.14.0rc1", "prerelease")
        )
        nightly = Version(
            compute_build_version(self.source_dir, "+rocm7.14.0a20260811", "nightly")
        )
        dev = Version(
            compute_build_version(
                self.source_dir,
                "+devrocm7.14.0.dev0-1a2b3c4d",
                "ci",
            )
        )

        self.assertGreater(stable, prerelease)
        self.assertGreater(prerelease, nightly)
        self.assertGreater(nightly, dev)

    def test_rocm_major_versions_have_known_sorting_issue(self):
        # `rocm7` currently sorts above `rocm10` as local-version strings.
        # This is tracked by https://github.com/ROCm/TheRock/issues/7183.
        rocm_7_14 = Version(
            compute_build_version(self.source_dir, "+rocm7.14.0a20260728", "nightly")
        )
        rocm_10_0 = Version(
            compute_build_version(self.source_dir, "+rocm10.0.0a20260806", "nightly")
        )

        self.assertGreater(rocm_7_14, rocm_10_0)

    def test_dev_release_tags_git_commit_in_local_segment(self):
        with mock.patch(
            "build_prod_wheels.get_source_commit_short", return_value="1a2b3c4d"
        ):
            version = compute_build_version(self.source_dir, "+rocm7.10.0", "dev")
        self.assertEqual(version, "2.12.0a0+git1a2b3c4d.rocm7.10.0")

    def test_dev_release_without_commit_falls_back_to_suffix(self):
        with mock.patch("build_prod_wheels.get_source_commit_short", return_value=""):
            version = compute_build_version(self.source_dir, "+rocm7.10.0", "dev")
        self.assertEqual(version, "2.12.0a0+rocm7.10.0")


if __name__ == "__main__":
    unittest.main()
