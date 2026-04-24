# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import expand_amdgpu_families
from _therock_utils.cmake_amdgpu_targets import expand_families


_SAMPLE_MAP: dict[str, list[str]] = {
    "gfx942": ["gfx942"],
    "gfx1100": ["gfx1100"],
    "gfx1101": ["gfx1101"],
    "gfx94X-dcgpu": ["gfx942"],
    "gfx110X-all": ["gfx1100", "gfx1101"],
    "dcgpu-all": ["gfx942"],
}


class ExpandFamiliesTest(unittest.TestCase):
    def test_single_family_expands(self):
        self.assertEqual(expand_families(["gfx94X-dcgpu"], _SAMPLE_MAP), ["gfx942"])

    def test_multiple_families_preserve_order(self):
        self.assertEqual(
            expand_families(["gfx110X-all", "gfx94X-dcgpu"], _SAMPLE_MAP),
            ["gfx1100", "gfx1101", "gfx942"],
        )

    def test_overlap_deduplicates_keeping_first_occurrence(self):
        # Both dcgpu-all and gfx94X-dcgpu expand to gfx942; the second occurrence
        # is dropped.
        self.assertEqual(
            expand_families(["dcgpu-all", "gfx94X-dcgpu"], _SAMPLE_MAP),
            ["gfx942"],
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(expand_families([], _SAMPLE_MAP), [])

    def test_unknown_family_raises_by_default(self):
        with self.assertRaisesRegex(ValueError, "Unknown AMD GPU families"):
            expand_families(["gfx94X-dcgpu", "gfxNOPE"], _SAMPLE_MAP)

    def test_unknown_family_silently_skipped_when_not_strict(self):
        self.assertEqual(
            expand_families(
                ["gfx94X-dcgpu", "gfxNOPE", "gfx110X-all"],
                _SAMPLE_MAP,
                strict=False,
            ),
            ["gfx942", "gfx1100", "gfx1101"],
        )


class ExpandAmdgpuFamiliesMainTest(unittest.TestCase):
    """End-to-end test of the script's main() against the real CMake file."""

    def _run_and_capture(self, *argv: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = expand_amdgpu_families.main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue().strip()

    def test_main_expands_single_known_family(self):
        # gfx94X-dcgpu should always contain gfx942.
        out = self._run_and_capture("--amdgpu-families", "gfx94X-dcgpu")
        self.assertIn("gfx942", out.split(","))

    def test_main_empty_input_prints_empty(self):
        out = self._run_and_capture("--amdgpu-families", "")
        self.assertEqual(out, "")

    def test_main_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            expand_amdgpu_families.main(["--amdgpu-families", "gfxNOPE"])


if __name__ == "__main__":
    unittest.main()
