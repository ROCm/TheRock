#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for data invariants in amdgpu_family_matrix.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from amdgpu_family_matrix import get_all_families_for_trigger_types

ALL_FAMILIES = get_all_families_for_trigger_types(
    ["presubmit", "postsubmit", "nightly"]
)


class TestFamilyMatrixInvariants(unittest.TestCase):
    """Validate structural invariants on the family matrix data."""

    def test_no_duplicate_family_names_per_platform(self):
        """Each (platform, family) pair must be unique across target names.

        Two target names mapping to the same amdgpu_family on the same
        platform would cause silent data loss in matrix expansion.
        """
        for platform in ("linux", "windows"):
            seen: dict[str, str] = {}  # family → target_name
            for target_name, entry in ALL_FAMILIES.items():
                if platform not in entry:
                    continue
                family = entry[platform]["family"]
                if family in seen:
                    self.fail(
                        f"Duplicate family {family!r} on {platform}: "
                        f"target {target_name!r} and {seen[family]!r}"
                    )
                seen[family] = target_name

    def test_required_fields_present(self):
        """Every platform entry must have the required fields."""
        required = {"family", "fetch-gfx-targets", "test-runs-on"}
        for target_name, entry in ALL_FAMILIES.items():
            for platform in ("linux", "windows"):
                if platform not in entry:
                    continue
                platform_info = entry[platform]
                missing = required - platform_info.keys()
                if missing:
                    self.fail(
                        f"{target_name}/{platform} missing required fields: {missing}"
                    )

    def test_variant_families_exist(self):
        """Every family in a variant's allow-list must exist in the matrix."""
        from amdgpu_family_matrix import all_build_variants

        all_family_names = set(ALL_FAMILIES.keys())
        for platform, variants in all_build_variants.items():
            for variant_name, variant_config in variants.items():
                allowed = variant_config.get("families")
                if allowed is None:
                    continue
                for family in allowed:
                    if family not in all_family_names:
                        self.fail(
                            f"Variant {variant_name} on {platform} references "
                            f"unknown family {family!r}"
                        )


if __name__ == "__main__":
    unittest.main()
