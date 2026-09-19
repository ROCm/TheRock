# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the SDK target ownership helpers."""

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.sdk_targets import (
    canonical_target,
    group_package_targets,
    package_owner,
)


class SdkTargetsTest(unittest.TestCase):
    def test_owner_defaults_to_target(self):
        for target in ("gfx942", "gfx1250", "gfx1250-unknown", "gfx1250:unknown+"):
            with self.subTest(target=target):
                self.assertEqual(package_owner(target), target)

    def test_features_preserve_target_identity(self):
        for feature in ("", ":xnack+", "-xnack-", ":sramecc+:xnack-"):
            with self.subTest(feature=feature):
                self.assertEqual(
                    canonical_target("gfx1250-strict" + feature), "gfx1250-strict"
                )
                self.assertEqual(package_owner("gfx1250-strict" + feature), "gfx1250")

    def test_grouping_retains_supplied_members_and_order(self):
        self.assertEqual(
            group_package_targets(
                ["gfx1250-strict", "gfx1250", "gfx1250-strict", "gfx950:xnack+"]
            ),
            {"gfx1250": ["gfx1250-strict", "gfx1250"], "gfx950": ["gfx950:xnack+"]},
        )


if __name__ == "__main__":
    unittest.main()
