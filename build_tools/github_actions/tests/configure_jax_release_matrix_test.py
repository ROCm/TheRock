# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_jax_release_matrix as m


class ConfigureJaxReleaseMatrixTest(unittest.TestCase):
    def test_default_matrix_uses_all_release_python_versions(self):
        matrix = m.generate_jax_matrix(None)

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.11",
                    "jax_ref": "rocm-jaxlib-v0.9.1",
                    "build_jaxlib": False,
                },
                {
                    "python_version": "3.12",
                    "jax_ref": "rocm-jaxlib-v0.9.1",
                    "build_jaxlib": False,
                },
                {
                    "python_version": "3.13",
                    "jax_ref": "rocm-jaxlib-v0.9.1",
                    "build_jaxlib": False,
                },
                {
                    "python_version": "3.14",
                    "jax_ref": "rocm-jaxlib-v0.9.1",
                    "build_jaxlib": False,
                },
            ],
        )

    def test_explicit_python_version_narrows_matrix(self):
        matrix = m.generate_jax_matrix(["3.12"])

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "jax_ref": "rocm-jaxlib-v0.9.1",
                    "build_jaxlib": False,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
