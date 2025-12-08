import json
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import fetch_package_targets

therock_test_runner_dict = {
    "gfx94x": {"linux": "linux-mi325-1gpu-ossci-rocm-frac"},
    "gfx110x": {
        "linux": "linux-gfx110X-gpu-rocm",
        "windows": "windows-gfx110X-gpu-rocm",
    },
    "gfx1151": {
        "linux": "linux-strix-halo-gpu-rocm",
        "windows": "windows-strix-halo-gpu-rocm",
    },
    "gfx950": {"linux": ""},
    "gfx120x": {"linux": "linux-rx9070-gpu-rocm", "windows": ""},
    "gfx90x": {"linux": "", "windows": ""},
    "gfx101x": {"linux": "", "windows": ""},
    "gfx103x": {
        "linux": "linux-rx6950-gpu-rocm",
        "windows": "windows-gfx1030-gpu-rocm",
    },
    "gfx1150": {"linux": "", "windows": ""},
    "gfx1152": {"linux": "", "windows": ""},
    "gfx1153": {"linux": "", "windows": ""},
}

os.environ["ROCM_THEROCK_TEST_RUNNERS"] = json.dumps(therock_test_runner_dict)


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


if __name__ == "__main__":
    unittest.main()
