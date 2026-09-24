# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the SDK target ownership helpers."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.sdk_targets import (
    canonical_target,
    group_package_targets,
    package_owner,
    ownership_data,
    render_dist_info,
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


class RenderDistInfoTest(unittest.TestCase):
    def test_embedded_ownership_matches_build_metadata(self):
        template = (
            Path(__file__).resolve().parents[1]
            / "packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py"
        )
        source = render_dist_info(template)
        self.assertNotIn("from _therock_utils", source)
        namespace = {}
        exec(source, namespace)
        self.assertEqual(namespace["ownership_data"](), ownership_data())
        self.assertEqual(namespace["package_owner"]("gfx1250-strict"), "gfx1250")

    def test_missing_or_duplicate_markers_are_rejected(self):
        start = "# BEGIN SHARED TARGET METADATA\n"
        end = "# END SHARED TARGET METADATA\n"
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "template.py"
            for contents in ("", start, end, start + start + end, start + end + end):
                with self.subTest(contents=contents):
                    template.write_text(contents)
                    with self.assertRaisesRegex(
                        ValueError, "Invalid target metadata embedding markers"
                    ):
                        render_dist_info(template)


if __name__ == "__main__":
    unittest.main()
