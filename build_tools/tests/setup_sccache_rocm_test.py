# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_tools/setup_sccache_rocm.py (sccache resolution)."""

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import setup_sccache_rocm as s


def _printed(p) -> str:
    return " ".join(str(c) for c in p.call_args_list)


class ResolveSccacheTest(unittest.TestCase):
    def test_not_found_under_gha_warns_and_returns_none(self):
        with mock.patch.object(s, "find_sccache", return_value=None):
            with mock.patch("builtins.print") as p:
                self.assertIsNone(s.resolve_sccache(None, gha=True))
        self.assertIn("::warning::", _printed(p))

    def test_not_found_locally_raises(self):
        with mock.patch.object(s, "find_sccache", return_value=None):
            with self.assertRaises(RuntimeError):
                s.resolve_sccache(None, gha=False)

    def test_explicit_missing_under_gha_warns(self):
        with mock.patch("builtins.print") as p:
            self.assertIsNone(s.resolve_sccache(Path("/nonexistent/sccache"), gha=True))
        self.assertIn("::warning::", _printed(p))

    def test_explicit_missing_locally_raises(self):
        with self.assertRaises(RuntimeError):
            s.resolve_sccache(Path("/nonexistent/sccache"), gha=False)

    def test_valid_path_is_returned(self):
        real = Path(sys.executable)  # a path that exists
        with mock.patch.object(s.subprocess, "run") as run:
            self.assertEqual(s.resolve_sccache(real, gha=True), real)
            run.assert_called_once()  # validated via --version

    def test_version_failure_under_gha_warns(self):
        real = Path(sys.executable)
        err = s.subprocess.CalledProcessError(1, "sccache --version")
        with mock.patch.object(s.subprocess, "run", side_effect=err):
            with mock.patch("builtins.print") as p:
                self.assertIsNone(s.resolve_sccache(real, gha=True))
        self.assertIn("::warning::", _printed(p))

    def test_version_failure_locally_raises(self):
        real = Path(sys.executable)
        with mock.patch.object(s.subprocess, "run", side_effect=OSError("boom")):
            with self.assertRaises(RuntimeError):
                s.resolve_sccache(real, gha=False)


if __name__ == "__main__":
    unittest.main()
