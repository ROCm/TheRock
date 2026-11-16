# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_pytorch_versions


class WriteTorchVersionsTest(unittest.TestCase):
    def test_torch_only_does_not_require_companion_wheels(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (
                dist_dir
                / "torch-2.14.0+rocm10.1.0rc3.asan-cp312-cp312-linux_x86_64.whl"
            ).write_bytes(b"wheel")
            versions = write_pytorch_versions.get_all_wheel_versions(
                dist_dir, os="Linux", torch_only=True
            )

        self.assertEqual(versions, {"torch_version": "2.14.0+rocm10.1.0rc3.asan"})

    def test_linux_still_requires_companion_wheels(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (
                dist_dir / "torch-2.14.0+rocm10.1.0rc3-cp312-cp312-linux_x86_64.whl"
            ).write_bytes(b"wheel")
            with self.assertRaises(FileNotFoundError):
                write_pytorch_versions.get_all_wheel_versions(dist_dir, os="Linux")


if __name__ == "__main__":
    unittest.main()
