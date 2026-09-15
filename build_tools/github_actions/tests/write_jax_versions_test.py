# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_jax_versions as m


def _touch(dist_dir: Path, *names: str) -> None:
    for name in names:
        (dist_dir / name).touch()


class WriteJaxVersionsTest(unittest.TestCase):
    def test_release_plugin_wheels_report_the_release_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(
                dist_dir,
                "jax_rocm7_plugin-0.11.1+rocm7.14.0a20260908-cp312-cp312-manylinux_2_28_x86_64.whl",
                "jax_rocm7_pjrt-0.11.1+rocm7.14.0a20260908-py3-none-manylinux_2_28_x86_64.whl",
            )
            self.assertEqual(
                m.get_all_jax_wheel_versions(dist_dir),
                {
                    "jax_plugin_version": "0.11.1+rocm7.14.0a20260908",
                    "jax_pjrt_version": "0.11.1+rocm7.14.0a20260908",
                    "jax_version": "0.11.1",
                },
            )

    def test_nightly_plugin_wheels_report_the_dev_version(self):
        # A tip build stamps the plugin with a published JAX nightly's version
        # plus the ROCm local label. Only the local label is split off, so the
        # test side installs jax/jaxlib at exactly the nightly version.
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(
                dist_dir,
                "jax_rocm7_plugin-0.11.2.dev20260914+rocm7.14.0a20260914-cp312-cp312-manylinux_2_28_x86_64.whl",
                "jax_rocm7_pjrt-0.11.2.dev20260914+rocm7.14.0a20260914-py3-none-manylinux_2_28_x86_64.whl",
            )
            self.assertEqual(
                m.get_all_jax_wheel_versions(dist_dir),
                {
                    "jax_plugin_version": "0.11.2.dev20260914+rocm7.14.0a20260914",
                    "jax_pjrt_version": "0.11.2.dev20260914+rocm7.14.0a20260914",
                    "jax_version": "0.11.2.dev20260914",
                },
            )

    def test_jaxlib_wheel_sets_the_base_version_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(
                dist_dir,
                "jaxlib-0.9.0+rocm7.0.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                "jax_rocm7_plugin-0.9.0+rocm7.0.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                "jax_rocm7_pjrt-0.9.0+rocm7.0.0-py3-none-manylinux_2_28_x86_64.whl",
            )
            versions = m.get_all_jax_wheel_versions(dist_dir)
            self.assertEqual(versions["jaxlib_version"], "0.9.0+rocm7.0.0")
            self.assertEqual(versions["jax_version"], "0.9.0")

    def test_missing_plugin_wheel_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(
                dist_dir,
                "jax_rocm7_pjrt-0.11.1+rocm7.14.0-py3-none-manylinux_2_28_x86_64.whl",
            )
            with self.assertRaises(FileNotFoundError):
                m.get_all_jax_wheel_versions(dist_dir)


if __name__ == "__main__":
    unittest.main()
