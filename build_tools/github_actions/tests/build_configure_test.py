# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import build_configure


class ResolveDistAmdgpuFamiliesTest(unittest.TestCase):
    def test_explicit_dist_families_win(self):
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families(
                "gfx94X-dcgpu", "gfx110X-all;gfx1151"
            ),
            "gfx110X-all;gfx1151",
        )

    def test_explicit_dist_canonicalizes_classic_aliases(self):
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families(
                "gfx110X-all", "gfx94X,gfx950"
            ),
            "gfx94X-dcgpu;gfx950-dcgpu",
        )

    def test_gfx94x_shard_expands_to_classic_union(self):
        # Nightly therock-ci-linux.yml passes fetch_package_targets' family
        # field (gfx94X-dcgpu / gfx950-dcgpu), not the AMDGPU_FAMILIES token.
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families("gfx94X-dcgpu"),
            "gfx94X-dcgpu;gfx950-dcgpu",
        )

    def test_gfx94x_alias_expands_to_classic_union(self):
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families("gfx94X"),
            "gfx94X-dcgpu;gfx950-dcgpu",
        )

    def test_gfx950_alias_expands_to_classic_union(self):
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families("gfx950"),
            "gfx94X-dcgpu;gfx950-dcgpu",
        )

    def test_nightly_family_list_expands_to_classic_union(self):
        self.assertEqual(
            build_configure.resolve_dist_amdgpu_families("gfx94X, gfx950"),
            "gfx94X-dcgpu;gfx950-dcgpu",
        )

    def test_unrelated_shard_keeps_cmake_default(self):
        self.assertIsNone(build_configure.resolve_dist_amdgpu_families("gfx110X-all"))

    def test_empty_shard_keeps_cmake_default(self):
        self.assertIsNone(build_configure.resolve_dist_amdgpu_families(""))
        self.assertIsNone(build_configure.resolve_dist_amdgpu_families(None))


if __name__ == "__main__":
    unittest.main()
