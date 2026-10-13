# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_torch_versions


class WriteTorchVersionsTest(unittest.TestCase):
    def test_torch_only_does_not_require_companion_wheels(self):
        with self._dist_dir(
            "torch-2.14.0+rocm10.1.0rc3.asan-cp312-cp312-linux_x86_64.whl"
        ) as dist_dir:
            versions = write_torch_versions.get_all_wheel_versions(
                dist_dir, os="Linux", torch_only=True
            )

        self.assertEqual(versions, {"torch_version": "2.14.0+rocm10.1.0rc3.asan"})

    def test_linux_still_requires_companion_wheels(self):
        with self._dist_dir(
            "torch-2.14.0+rocm10.1.0rc3-cp312-cp312-linux_x86_64.whl"
        ) as dist_dir:
            with self.assertRaises(FileNotFoundError):
                write_torch_versions.get_all_wheel_versions(dist_dir, os="Linux")

    def _dist_dir(self, *names: str):
        import tempfile

        class _Dir:
            def __enter__(self_inner):
                self_inner.tmp = tempfile.TemporaryDirectory()
                path = Path(self_inner.tmp.name)
                for name in names:
                    (path / name).write_bytes(b"wheel")
                return path

            def __exit__(self_inner, exc_type, exc, tb):
                self_inner.tmp.cleanup()

        return _Dir()


if __name__ == "__main__":
    unittest.main()
