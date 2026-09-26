# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_torch_versions as m

LINUX_WHEELS = (
    "torch-2.14.0+rocm10.2-cp312-cp312-linux_x86_64.whl",
    "torchaudio-2.14.0+rocm10.2-cp312-cp312-linux_x86_64.whl",
    "torchvision-0.29.0+rocm10.2-cp312-cp312-linux_x86_64.whl",
    "triton-3.6.0-cp312-cp312-linux_x86_64.whl",
    "apex-0.1-cp312-cp312-linux_x86_64.whl",
)

ASAN_WHEEL = (
    "torch-2.14.0+git5e015018.rocm10.2.asan.36142888996-cp312-cp312-linux_x86_64.whl"
)


class WriteTorchVersionsTest(unittest.TestCase):
    def dist_dir_with(self, *wheel_names: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        dist_dir = Path(temp_dir.name)
        for wheel_name in wheel_names:
            (dist_dir / wheel_name).touch()
        return dist_dir

    def test_linux_reports_every_wheel_version(self):
        dist_dir = self.dist_dir_with(*LINUX_WHEELS)

        versions = m.get_all_wheel_versions(dist_dir, os="Linux")

        self.assertEqual(versions["torch_version"], "2.14.0+rocm10.2")
        self.assertEqual(versions["torchaudio_version"], "2.14.0+rocm10.2")
        self.assertEqual(versions["torchvision_version"], "0.29.0+rocm10.2")
        self.assertEqual(versions["triton_version"], "3.6.0")
        self.assertEqual(versions["apex_version"], "0.1")

    def test_linux_requires_companion_wheels(self):
        dist_dir = self.dist_dir_with(ASAN_WHEEL)

        with self.assertRaisesRegex(FileNotFoundError, "torchaudio"):
            m.get_all_wheel_versions(dist_dir, os="Linux")

    def test_torch_only_accepts_a_lone_torch_wheel(self):
        dist_dir = self.dist_dir_with(ASAN_WHEEL)

        versions = m.get_all_wheel_versions(dist_dir, os="Linux", torch_only=True)

        self.assertEqual(
            versions["torch_version"],
            "2.14.0+git5e015018.rocm10.2.asan.36142888996",
        )
        self.assertNotIn("torchaudio_version", versions)
        self.assertNotIn("torchvision_version", versions)

    def test_torch_only_still_requires_torch(self):
        dist_dir = self.dist_dir_with()

        with self.assertRaisesRegex(FileNotFoundError, "torch wheel"):
            m.get_all_wheel_versions(dist_dir, os="Linux", torch_only=True)


if __name__ == "__main__":
    unittest.main()
