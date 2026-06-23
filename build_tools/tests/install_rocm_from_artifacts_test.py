# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_from_artifacts


class MirageArtifactSelectionTest(unittest.TestCase):
    """Verifies the --mirage flag wires through to fetch_artifacts."""

    def _run_main(self, extra_args):
        """Run main() with fetch_artifacts mocked, returning the captured argv."""
        captured = {}

        def fake_fetch(argv):
            captured["argv"] = argv

        with mock.patch.object(
            install_rocm_from_artifacts, "fetch_artifacts_main", fake_fetch
        ):
            install_rocm_from_artifacts.main(
                [
                    "--run-id",
                    "12345",
                    "--artifact-group",
                    "gfx942",
                    "--amdgpu-targets",
                    "gfx942",
                    "--dry-run",
                ]
                + extra_args
            )
        return captured["argv"]

    def test_mirage_flag_includes_mirage_artifacts(self):
        argv = self._run_main(["--mirage"])
        self.assertIn("mirage_run", argv)

    def test_no_mirage_flag_excludes_mirage_artifacts(self):
        argv = self._run_main(["--rocjitsu"])
        self.assertNotIn("mirage_run", argv)

    def test_mirage_with_tests_includes_test_artifact(self):
        argv = self._run_main(["--mirage", "--tests"])
        self.assertIn("mirage_run", argv)
        self.assertIn("mirage_test", argv)


if __name__ == "__main__":
    unittest.main()
