from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import fetch_package_targets


class FetchPackageTargetsTest(unittest.TestCase):
    def test_linux_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx94x",
            "PYTORCH_DEV_DOCKER": None,
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_linux_multiple_families(self):
        # Note the punctuation that gets stripped and x that gets changed to X.
        args = {
            "AMDGPU_FAMILIES": "gfx94x ,; gfx110x",
            "PYTORCH_DEV_DOCKER": None,
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
            "PYTORCH_DEV_DOCKER": None,
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # Standard targets have suffixes and may use X for a family.
        self.assertTrue(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-dgpu" == t["amdgpu_family"] for t in targets))

    def test_linux_docker_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "PYTORCH_DEV_DOCKER": "true",
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # PyTorch Docker targets do not have suffixes.
        self.assertTrue(any("gfx942" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx1100" == t["amdgpu_family"] for t in targets))

    def test_windows_single_family(self):
        args = {
            "AMDGPU_FAMILIES": "gfx120x",
            "PYTORCH_DEV_DOCKER": None,
            "THEROCK_PACKAGE_PLATFORM": "linux",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertEqual(len(targets), 1)

    def test_windows_no_families(self):
        args = {
            "AMDGPU_FAMILIES": None,
            "PYTORCH_DEV_DOCKER": None,
            "THEROCK_PACKAGE_PLATFORM": "windows",
        }
        targets = fetch_package_targets.determine_package_targets(args)

        self.assertTrue(all("amdgpu_family" in t for t in targets))
        # dcgpu targets are Linux only.
        self.assertFalse(any("gfx94X-dcgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx110X-dgpu" == t["amdgpu_family"] for t in targets))
        self.assertTrue(any("gfx120X-all" == t["amdgpu_family"] for t in targets))


if __name__ == "__main__":
    unittest.main()
