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
    core_tarball_prefix,
    is_allowed_multi_arch_package,
    list_packages_structured,
    list_tarball_for_package,
    parse_arguments,
)

JAX_ROCM7_WHEELS = [
    "jax_rocm7_plugin-0.9.2+rocm7.15.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "jax_rocm7_pjrt-0.9.2+rocm7.15.0-py3-none-manylinux_2_28_x86_64.whl",
]

JAX_ROCM10_WHEELS = [
    "jax_rocm10_plugin-0.10.0+rocm10.0.0-cp313-cp313-manylinux_2_27_x86_64.whl",
    "jax_rocm10_pjrt-0.10.0+rocm10.0.0-py3-none-manylinux_2_27_x86_64.whl",
]


class FakePaginator:
    def __init__(self, objects_by_bucket_prefix):
        self.objects_by_bucket_prefix = objects_by_bucket_prefix
        self.calls = []

    def paginate(self, Bucket, Prefix):
        self.calls.append((Bucket, Prefix))
        return [{"Contents": self.objects_by_bucket_prefix.get((Bucket, Prefix), [])}]


class FakeS3Client:
    def __init__(self, objects_by_bucket_prefix):
        self.paginator = FakePaginator(objects_by_bucket_prefix)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator


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


class StructuredDownloadTest(unittest.TestCase):
    def test_structured_lists_product_buckets_and_filters_artifacts(self):
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/index.html",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl.metadata",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/extra/rocm_sdk_core-7.13.0rc1.whl",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
                    "Size": 1,
                },
            ],
            (
                "therock-repo-amd-rc-python",
                "v5/rocm/pytorch/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0rc1-cp312-cp312-linux_x86_64.whl",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0rc1-py3-none-linux_x86_64.whl",
                    "Size": 30,
                },
            ],
            (
                "therock-repo-amd-rc-jax",
                "v5/rocm/jax/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/jax/whl-next/jax-rocm7-plugin/jax_rocm7_plugin-0.9.2+rocm7.13.0rc1-cp312-cp312-manylinux_2_28_x86_64.whl",
                    "Size": 40,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core", "pytorch", "jax"],
            "whl-next",
            "7.13.0rc1",
            architectures=["gfx942"],
        )

        self.assertEqual(
            packages,
            [
                (
                    "therock-repo-amd-rc-core",
                    "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl",
                    10,
                ),
                (
                    "therock-repo-amd-rc-python",
                    "v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0rc1-cp312-cp312-linux_x86_64.whl",
                    20,
                ),
                (
                    "therock-repo-amd-rc-python",
                    "v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0rc1-py3-none-linux_x86_64.whl",
                    30,
                ),
                (
                    "therock-repo-amd-rc-jax",
                    "v5/rocm/jax/whl-next/jax-rocm7-plugin/jax_rocm7_plugin-0.9.2+rocm7.13.0rc1-cp312-cp312-manylinux_2_28_x86_64.whl",
                    40,
                ),
            ],
        )

    def test_parse_structured_args_defaults_to_whl_next_and_all_products(self):
        args = parse_arguments(
            ["--version=7.13.0rc1", "--structured", "--list-multi-arch-packages"]
        )

        self.assertTrue(args.multi_arch)
        self.assertEqual(args.python_index, "whl-next")
        self.assertEqual(args.repo_stream, "rc")
        self.assertEqual(args.product, ["core", "pytorch", "jax"])

    def test_parse_product_accepts_repeated_and_comma_separated_values(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--structured",
                "--list-multi-arch-packages",
                "--product=core,pytorch",
                "--product=jax",
            ]
        )

        self.assertEqual(args.product, ["core", "pytorch", "jax"])

    def test_parse_multi_arch_tarballs_keep_handler_default_output_dir(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--multi-arch",
                "--include-tarballs",
                "--output-dir=downloads",
            ]
        )

        self.assertIsNone(args.tarball_output_dir)

    def test_parse_structured_asan_tarballs_use_asan_output_dir(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--structured",
                "--include-tarballs",
                "--tarball-variant=asan",
                "--output-dir=downloads",
            ]
        )

        self.assertEqual(args.tarball_output_dir, Path("downloads") / "tarball-asan")

    def test_core_tarball_prefixes_reuse_legacy_tarball_listing(self):
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/index.html",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/tarball/other-7.13.0rc1.tar.gz",
                    "Size": 1,
                },
            ]
        }
        client = FakeS3Client(
            {
                ("therock-repo-amd-rc-core", core_tarball_prefix("release")): objects[
                    "Contents"
                ]
            }
        )

        tarballs = list_tarball_for_package(
            client,
            "therock-repo-amd-rc-core",
            core_tarball_prefix("release"),
            None,
            "7.13.0rc1",
        )

        self.assertEqual(
            tarballs,
            [
                (
                    "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    10,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
