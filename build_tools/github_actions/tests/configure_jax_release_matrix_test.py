# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_jax_release_matrix as m


class ConfigureJaxReleaseMatrixTest(unittest.TestCase):
    def test_default_matrix_includes_multiple_python_versions_and_refs(self):
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
        )
        python_versions = {row["python_version"] for row in matrix}
        jax_refs = {row["jax_ref"] for row in matrix}

        self.assertGreater(len(matrix), 1)
        self.assertGreater(len(python_versions), 1)
        self.assertGreater(len(jax_refs), 1)
        self.assertEqual(
            set(matrix[0]),
            {"python_version", "jax_ref", "jax_repository", "build_mode", "gfx_arch"},
        )

    def test_explicit_python_version_narrows_matrix(self):
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
            python_versions=["3.12"],
        )
        self.assertGreater(len(matrix), 1)
        self.assertEqual(
            {row["python_version"] for row in matrix},
            {"3.12"},
        )

    def test_generate_jax_matrix_uses_requested_refs_only(self):
        matrix = m.generate_jax_matrix(
            jax_refs=["rocm-jaxlib-v0.10.0"],
            python_versions=["3.12"],
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.12")
        self.assertEqual(matrix[0]["jax_ref"], "rocm-jaxlib-v0.10.0")
        self.assertEqual(matrix[0]["jax_repository"], "ROCm/jax")
        self.assertEqual(matrix[0]["build_mode"], "manylinux")
        self.assertEqual(matrix[0]["gfx_arch"], "device-all")

    def test_ci_jax_matrix_excludes_unsupported_build_modes(self):
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="ci",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        self.assertEqual(
            {row["build_mode"] for row in matrix},
            {"manylinux"},
        )
        self.assertNotIn(
            "rocm-jaxlib-v0.9.1",
            {row["jax_ref"] for row in matrix},
        )

    def test_unknown_release_type_raises(self):
        with self.assertRaises(ValueError):
            m.generate_jax_matrix_for_release_type(
                release_type="unknown",
                platform="linux",
            )

    def test_unknown_jax_ref_raises(self):
        with self.assertRaises(ValueError):
            m.generate_jax_matrix_for_release_type(
                release_type="ci",
                platform="linux",
                jax_refs=["unknown-jax-ref"],
            )


if __name__ == "__main__":
    unittest.main()
