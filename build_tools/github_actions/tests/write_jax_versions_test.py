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

    def test_a_tip_build_reports_its_own_jax_and_jaxlib(self):
        # A tip build makes all four wheels from one checkout with one suffix.
        # jax_version drops only the local label, which still pins the jax
        # wheel built here since PEP 440 ignores it when the pin has none.
        version = "0.12.0.dev20261002+rocm7.14.0a20261002"
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _touch(
                dist_dir,
                f"jax-{version}-py3-none-any.whl",
                f"jaxlib-{version}-cp312-cp312-manylinux_2_27_x86_64.whl",
                f"jax_rocm7_plugin-{version}-cp312-cp312-manylinux_2_28_x86_64.whl",
                f"jax_rocm7_pjrt-{version}-py3-none-manylinux_2_28_x86_64.whl",
            )
            self.assertEqual(
                m.get_all_jax_wheel_versions(dist_dir),
                {
                    "jaxlib_version": version,
                    "jax_plugin_version": version,
                    "jax_pjrt_version": version,
                    "jax_version": "0.12.0.dev20261002",
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
