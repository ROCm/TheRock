# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for build_tools/hack/env_check/AMDGPU_LLVM_TARGET.py."""

import sys
import unittest
from pathlib import Path

# Import the table module on its own: env_check/utils.py rewraps sys.stdout on
# Windows at import time, which breaks pytest's output capture.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hack" / "env_check"))
import AMDGPU_LLVM_TARGET


def _target_of(name):
    return [gfx for gfx, names in AMDGPU_LLVM_TARGET._amdgpu.items() if name in names]


class AmdgpuLlvmTargetTest(unittest.TestCase):
    def test_navi14_parts_are_gfx1012(self):
        # Navi 14, per LLVM's AMDGPUUsage processor table.
        for name in ("AMD Radeon RX 5500 XT", "AMD Radeon RX 5500"):
            with self.subTest(name=name):
                self.assertEqual(_target_of(name), ["gfx1012"])

    def test_navi12_parts_are_gfx1011(self):
        # Navi 12, per LLVM's AMDGPUUsage processor table.
        for name in ("AMD Radeon Pro V520", "AMD Radeon Pro 5600M"):
            with self.subTest(name=name):
                self.assertEqual(_target_of(name), ["gfx1011"])

    def test_no_name_listed_under_two_targets(self):
        for names in AMDGPU_LLVM_TARGET._amdgpu.values():
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(len(_target_of(name)), 1, _target_of(name))


if __name__ == "__main__":
    unittest.main()
