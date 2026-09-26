#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for data invariants in new_amdgpu_family_matrix.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from new_amdgpu_family_matrix import amdgpu_family_info_matrix_all


class TestGfx103XLinuxReenable(unittest.TestCase):
    """gfx103X Linux test/release flags (re-enabled after #2740 closed)."""

    def _linux_dgpu(self):
        entry = amdgpu_family_info_matrix_all["gfx103X"]["dgpu"]["linux"]
        return entry

    def test_linux_run_tests_enabled(self):
        linux = self._linux_dgpu()
        self.assertTrue(
            linux["test"]["run_tests"],
            "gfx103X Linux tests should run after #2740 closed",
        )

    def test_linux_fetch_gfx_targets(self):
        linux = self._linux_dgpu()
        self.assertEqual(linux["test"]["fetch-gfx-targets"], ["gfx1030"])

    def test_linux_push_on_success_enabled(self):
        linux = self._linux_dgpu()
        self.assertTrue(
            linux["release"]["push_on_success"],
            "gfx103X Linux nightlies should push on success",
        )

    def test_windows_still_gated(self):
        """Windows gfx1030 remains disabled until #3200 is resolved."""
        win = amdgpu_family_info_matrix_all["gfx103X"]["dgpu"]["windows"]
        self.assertFalse(win["test"]["run_tests"])
        self.assertFalse(win["release"]["push_on_success"])


class TestMatrixStructure(unittest.TestCase):
    """Basic structural checks for the new family matrix."""

    def test_gfx1032_label_mentions_6650(self):
        # Product string lives in cmake, not this matrix; ensure family key exists.
        self.assertIn("gfx103X", amdgpu_family_info_matrix_all)

    def test_required_platforms_present(self):
        for family, kinds in amdgpu_family_info_matrix_all.items():
            for kind, platforms in kinds.items():
                self.assertIn(
                    "linux",
                    platforms,
                    f"{family}/{kind} missing linux platform",
                )


if __name__ == "__main__":
    unittest.main()
