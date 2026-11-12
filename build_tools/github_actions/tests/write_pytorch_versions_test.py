# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_pytorch_versions as m

LINUX_WHEELS = [
    "torch-2.13.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
    "torchaudio-2.11.1+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
    "torchvision-0.29.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
    "triton-3.7.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
    "apex-1.13.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
]

WINDOWS_WHEELS = [
    "torch-2.13.0+rocm7.14.0-cp312-cp312-win_amd64.whl",
    "torchaudio-2.11.1+rocm7.14.0-cp312-cp312-win_amd64.whl",
    "torchvision-0.29.0+rocm7.14.0-cp312-cp312-win_amd64.whl",
]


def _touch(dist_dir: Path, *names: str) -> None:
    for name in names:
        (dist_dir / name).touch()


class WritePytorchVersionsTest(unittest.TestCase):
    def test_torchcodec_version_is_written_when_the_wheel_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(dist_dir, *LINUX_WHEELS)
            _touch(
                dist_dir, "torchcodec-0.17.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl"
            )
            versions = m.get_all_wheel_versions(dist_dir, os="Linux")
        self.assertEqual(versions["torchcodec_version"], "0.17.0+rocm7.14.0")
        # torch-*.whl must not match the torchcodec wheel.
        self.assertEqual(versions["torch_version"], "2.13.0+rocm7.14.0")

    def test_missing_torchcodec_wheel_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(dist_dir, *LINUX_WHEELS)
            versions = m.get_all_wheel_versions(dist_dir, os="Linux")
        self.assertNotIn("torchcodec_version", versions)

    def test_missing_required_wheel_is_still_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(dist_dir, *LINUX_WHEELS[:-1])
            with self.assertRaisesRegex(FileNotFoundError, "apex"):
                m.get_all_wheel_versions(dist_dir, os="Linux")

    def test_windows_torchcodec_version_is_written_when_the_wheel_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(dist_dir, *WINDOWS_WHEELS)
            _touch(dist_dir, "torchcodec-0.17.0+rocm7.14.0-cp312-cp312-win_amd64.whl")
            versions = m.get_all_wheel_versions(dist_dir, os="Windows")
        self.assertEqual(versions["torchcodec_version"], "0.17.0+rocm7.14.0")
        self.assertEqual(versions["torch_version"], "2.13.0+rocm7.14.0")

    def test_windows_missing_torchcodec_wheel_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(dist_dir, *WINDOWS_WHEELS)
            versions = m.get_all_wheel_versions(dist_dir, os="Windows")
        self.assertNotIn("torchcodec_version", versions)


if __name__ == "__main__":
    unittest.main()
