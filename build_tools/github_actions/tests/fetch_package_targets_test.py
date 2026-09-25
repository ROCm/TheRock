# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import fetch_package_targets


class FetchPackageTargetsTest(unittest.TestCase):
    def test_linux_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_linux_multiple_families(self):
        # Note the punctuation that gets stripped and x that gets changed to X.
        args = {
            "AMDGPU_FAMILIES": "gfx94x ,; gfx110x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertGreater(
            len(targets),
            1,
        )

    def test_linux_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # Standard targets have suffixes and may use X for a family.
        self.assertTrue(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-all" == t["amdgpu_family"] for t in targets))

    def test_windows_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx120x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_windows_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "THEROCK_PACKAGE_PLATFORM": "windows",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # dcgpu targets are Linux only.
        self.assertFalse(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-all" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx120X-all" == t["amdgpu_family"] for t in targets))

    def test_gfx94x_multi_label_selects_first(self):
        """Test that first label can be selected via random.choices."""
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }

        first_label = {"label": "linux-gfx942-1gpu-ccs-ossci-rocm", "count": 5}
        with patch("random.choices", return_value=[first_label]):
            targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["test_machine"], "linux-gfx942-1gpu-ccs-ossci-rocm")

    def test_gfx94x_multi_label_selects_second(self):
        """Test that second label can be selected via random.choices."""
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }

        second_label = {"label": "linux-gfx942-1gpu-ccs-csp-ossci-rocm", "count": 28}
        with patch("random.choices", return_value=[second_label]):
            targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)
        self.assertEqual(
            targets[0]["test_machine"], "linux-gfx942-1gpu-ccs-csp-ossci-rocm"
        )

    def test_gfx94x_multi_label_selects_third(self):
        """Test that third label can be selected via random.choices."""
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }

        third_label = {"label": "linux-gfx942-1gpu-ossci-rocm", "count": 5}
        with patch("random.choices", return_value=[third_label]):
            targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["test_machine"], "linux-gfx942-1gpu-ossci-rocm")

    def test_families_without_multi_label_use_primary(self):
        """Families without multi-label config should use primary label."""
        args = {
            "AMDGPU_FAMILIES": "gfx110x",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }

        # Run multiple times to ensure consistency
        for _ in range(5):
            targets = fetch_package_targets.determine_package_targets(args)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0]["test_machine"], "linux-gfx110X-gpu-rocm")


if __name__ == "__main__":
    unittest.main()
