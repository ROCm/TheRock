# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import should_build_ck_sdpa_gfx11


class ShouldBuildCkSdpaGfx11Test(unittest.TestCase):
    def _check(
        self,
        pytorch_rocm_arch: str = "gfx1151",
        *,
        pytorch_version: str = "2.16.0a0",
        use_flash_attention: bool = True,
        is_windows: bool = False,
    ) -> bool:
        return should_build_ck_sdpa_gfx11(
            pytorch_version,
            pytorch_rocm_arch,
            use_flash_attention=use_flash_attention,
            is_windows=is_windows,
        )

    def test_enabled_for_gfx11_targets(self):
        for arch in ("gfx1151", "gfx1100;gfx1101;gfx1102;gfx1103", "gfx942;gfx1150"):
            with self.subTest(arch=arch):
                self.assertTrue(self._check(arch))

    def test_disabled_without_gfx11_target(self):
        for arch in ("gfx942", "gfx942;gfx950", "gfx1200;gfx1201", ""):
            with self.subTest(arch=arch):
                self.assertFalse(self._check(arch))

    def test_enabled_for_2_16_and_later(self):
        for version in (
            "2.16.0a0",
            "2.16.0a0+gitdf21fc2d.devrocm10.3.0.dev0",
            "2.16.0+rocm10.3.0",
            "2.17.0a0",
        ):
            with self.subTest(version=version):
                self.assertTrue(self._check(pytorch_version=version))

    def test_disabled_before_2_16(self):
        for version in ("2.13.0", "2.15.0+rocm10.3.0", "2.15.1"):
            with self.subTest(version=version):
                self.assertFalse(self._check(pytorch_version=version))

    def test_disabled_without_flash_attention(self):
        self.assertFalse(self._check(use_flash_attention=False))

    def test_disabled_on_windows(self):
        self.assertFalse(self._check(is_windows=True))


if __name__ == "__main__":
    unittest.main()
