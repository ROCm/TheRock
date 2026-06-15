#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for compute_pytorch_cache_type.py"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import compute_pytorch_cache_type
from compute_pytorch_cache_type import compute_cache_type, main


class TestComputeCacheType(unittest.TestCase):
    """Tests for the compute_cache_type decision function."""

    def _run(self, cache_type, repo="ROCm/TheRock", is_fork=False):
        with mock.patch.object(
            compute_pytorch_cache_type,
            "_is_current_run_pr_from_fork",
            return_value=is_fork,
        ):
            return compute_cache_type(cache_type, repo)

    def test_sccache_in_org_pr_stays_sccache(self):
        self.assertEqual(self._run("sccache", is_fork=False), "sccache")

    def test_sccache_fork_pr_downgrades_to_none(self):
        self.assertEqual(self._run("sccache", is_fork=True), "none")

    def test_sccache_non_rocm_repo_downgrades_to_none(self):
        self.assertEqual(
            self._run("sccache", repo="someone/TheRock", is_fork=False), "none"
        )

    def test_ccache_passes_through_even_on_fork(self):
        # Only sccache depends on the OIDC role; ccache is local.
        self.assertEqual(self._run("ccache", is_fork=True), "ccache")

    def test_none_passes_through(self):
        self.assertEqual(self._run("none", is_fork=True), "none")

    def test_fork_check_skipped_when_not_sccache(self):
        # _is_current_run_pr_from_fork must not even be consulted for ccache/none.
        with mock.patch.object(
            compute_pytorch_cache_type,
            "_is_current_run_pr_from_fork",
            side_effect=AssertionError("should not be called"),
        ):
            self.assertEqual(compute_cache_type("ccache", "ROCm/TheRock"), "ccache")
            self.assertEqual(compute_cache_type("none", "ROCm/TheRock"), "none")


class TestMain(unittest.TestCase):
    """Tests for main() writing the effective cache_type to GITHUB_OUTPUT."""

    def _main_with_output(self, cache_type, repo, is_fork):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gh_output.txt"
            env = {"GITHUB_REPOSITORY": repo, "GITHUB_OUTPUT": str(out)}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                compute_pytorch_cache_type,
                "_is_current_run_pr_from_fork",
                return_value=is_fork,
            ):
                main(["--cache-type", cache_type])
            return out.read_text()

    def test_main_in_org_writes_sccache(self):
        self.assertIn(
            "cache_type=sccache",
            self._main_with_output("sccache", "ROCm/TheRock", is_fork=False),
        )

    def test_main_fork_writes_none(self):
        self.assertIn(
            "cache_type=none",
            self._main_with_output("sccache", "ROCm/TheRock", is_fork=True),
        )


if __name__ == "__main__":
    unittest.main()
