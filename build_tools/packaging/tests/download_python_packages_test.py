#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for package recognition in download_python_packages.py.

The JAX plugin/pjrt wheels embed the ROCm major version in their package name,
so these tests cover both the ROCm 7 and ROCm 10 spellings to make sure a ROCm
version bump does not silently drop the wheels from promotion.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from download_python_packages import (
    categorize_package,
    is_allowed_multi_arch_package,
)

JAX_ROCM7_WHEELS = [
    "jax_rocm7_plugin-0.9.2+rocm7.15.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "jax_rocm7_pjrt-0.9.2+rocm7.15.0-py3-none-manylinux_2_28_x86_64.whl",
]

JAX_ROCM10_WHEELS = [
    "jax_rocm10_plugin-0.10.0+rocm10.0.0-cp313-cp313-manylinux_2_27_x86_64.whl",
    "jax_rocm10_pjrt-0.10.0+rocm10.0.0-py3-none-manylinux_2_27_x86_64.whl",
]


class CategorizePackageTest(unittest.TestCase):
    def test_jax_rocm7_wheels_are_promoted(self):
        for filename in JAX_ROCM7_WHEELS:
            with self.subTest(filename=filename):
                self.assertEqual(categorize_package(filename), "promote")

    def test_jax_rocm10_wheels_are_promoted(self):
        for filename in JAX_ROCM10_WHEELS:
            with self.subTest(filename=filename):
                self.assertEqual(categorize_package(filename), "promote")

    def test_jaxlib_is_still_promoted(self):
        self.assertEqual(
            categorize_package("jaxlib-0.9.2-cp312-cp312-manylinux_2_28_x86_64.whl"),
            "promote",
        )

    def test_unrelated_jax_rocm_name_is_unknown(self):
        # Only the plugin/pjrt wheels carry the ROCm major in their name; a
        # majorless or unexpected variant should not be promoted silently.
        self.assertEqual(
            categorize_package("jax_rocm_plugin-0.10.0-py3-none-any.whl"),
            "unknown",
        )


class IsAllowedMultiArchPackageTest(unittest.TestCase):
    def test_jax_rocm7_wheels_are_allowed(self):
        for filename in JAX_ROCM7_WHEELS:
            with self.subTest(filename=filename):
                self.assertTrue(is_allowed_multi_arch_package(filename))

    def test_jax_rocm10_wheels_are_allowed(self):
        for filename in JAX_ROCM10_WHEELS:
            with self.subTest(filename=filename):
                self.assertTrue(is_allowed_multi_arch_package(filename))

    def test_unknown_package_is_not_allowed(self):
        self.assertFalse(
            is_allowed_multi_arch_package("some_other_pkg-1.0-py3-none-any.whl")
        )


if __name__ == "__main__":
    unittest.main()
