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
    PackageEntry,
    categorize_package,
    core_tarball_prefix,
    filter_by_git_sha,
    filter_by_pytorch_version,
    is_allowed_multi_arch_package,
    list_packages_structured,
    list_tarball_for_package,
    parse_arguments,
)

# download_python_packages inserts build_tools onto sys.path on import, so
# _therock_utils is importable here too.
from _therock_utils.storage_location import StorageLocation

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
                "therock-repo-amd-rc-pytorch",
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
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl",
                    ),
                    10,
                ),
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-pytorch",
                        "v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0rc1-cp312-cp312-linux_x86_64.whl",
                    ),
                    20,
                ),
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-pytorch",
                        "v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0rc1-py3-none-linux_x86_64.whl",
                    ),
                    30,
                ),
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-jax",
                        "v5/rocm/jax/whl-next/jax-rocm7-plugin/jax_rocm7_plugin-0.9.2+rocm7.13.0rc1-cp312-cp312-manylinux_2_28_x86_64.whl",
                    ),
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

    def test_version_matching_rejects_numeric_prefix_collision(self):
        # Requesting "3.0.0" must not also match "13.0.0": a plain substring
        # check would incorrectly find "3.0.0" inside "13.0.0".
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-3.0.0-py3-none-linux_x86_64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-13.0.0-py3-none-linux_x86_64.whl",
                    "Size": 20,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core"],
            "whl-next",
            "3.0.0",
        )

        self.assertEqual(
            packages,
            [
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-3.0.0-py3-none-linux_x86_64.whl",
                    ),
                    10,
                ),
            ],
        )

    def test_version_matching_rejects_rc_suffix_collision(self):
        # Requesting "7.13.0rc1" must not also match "7.13.0rc10": a plain
        # substring check would incorrectly find "7.13.0rc1" as a prefix of
        # "7.13.0rc10".
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc10.tar.gz",
                    "Size": 20,
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

    def test_version_matching_rejects_post_and_dev_suffix_collision(self):
        # Requesting "7.13.0" must not also match "7.13.0.post1" or
        # "7.13.0.dev0": the "." separator in these PEP 440 pre/post-release
        # suffixes is not a word-boundary character in regex, so a naive
        # right-boundary check would incorrectly treat "7.13.0" as ending
        # exactly before the ".post1"/".dev0" suffix.
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.post1.tar.gz",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.dev0.tar.gz",
                    "Size": 30,
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
            "7.13.0",
        )

        self.assertEqual(
            tarballs,
            [
                (
                    "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.tar.gz",
                    10,
                )
            ],
        )

    def test_version_matching_rejects_post_and_dev_suffix_collision_for_wheels(self):
        # Same collision as the tarball case above, but for wheel filenames,
        # where the version segment is followed by the platform/abi/python
        # tags (e.g. "-cp310-cp310-win_amd64.whl") instead of ".tar.gz".
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-cp310-cp310-win_amd64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0.post1-cp310-cp310-win_amd64.whl",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0.dev0-py3-none-manylinux_2_27_x86_64.whl",
                    "Size": 30,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core"],
            "whl-next",
            "7.13.0",
        )

        self.assertEqual(
            packages,
            [
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-cp310-cp310-win_amd64.whl",
                    ),
                    10,
                ),
            ],
        )

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


DEV_PYTORCH_FILES = [
    "torch-2.12.0+gita924f282.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "torch-2.12.0+gitb97872d5.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "torch-2.13.0+gitc0691f32.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "amd_torch_device_gfx1100-2.12.0+gitb97872d5.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "amd_torch_device_gfx1100-2.13.0+gitc0691f32.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "torchvision-0.27.0+gitb3d5b111.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "triton-3.8.0+git669b31ac.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "triton-3.8.0+git66b3f9ff.rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
    "apex-1.12.0+rocm10.0.0rc4-cp312-cp312-linux_x86_64.whl",
]


def dev_entries(filenames):
    return [
        PackageEntry(
            StorageLocation(
                "therock-repo-amd-dev-pytorch", f"v5/rocm/pytorch/whl-next/x/{f}"
            ),
            1,
        )
        for f in filenames
    ]


def entry_names(entries):
    return [e.key.split("/")[-1] for e in entries]


# torch, its device wheel, torchvision and triton of the b97872d5 build, plus apex.
B97872D5_BUILD = [DEV_PYTORCH_FILES[i] for i in (1, 3, 5, 6, 8)]


def parse_structured(*extra):
    return parse_arguments(
        ["--version=10.0.0rc4", "--structured", "--list-multi-arch-packages", *extra]
    )


class DevBuildFilterTest(unittest.TestCase):
    def test_git_sha_keeps_matching_and_untagged_wheels(self):
        kept = filter_by_git_sha(
            dev_entries(DEV_PYTORCH_FILES), ["b97872d5", "b3d5b111", "669b31ac"]
        )
        self.assertEqual(entry_names(kept), B97872D5_BUILD)

    def test_git_sha_accepts_full_sha(self):
        kept = filter_by_git_sha(
            dev_entries(DEV_PYTORCH_FILES[:2]),
            ["b97872d55cf78ed4a8dd9f71ce7767e35b39f2fc"],
        )
        self.assertEqual(entry_names(kept), [DEV_PYTORCH_FILES[1]])

    def test_pytorch_version_alone_keeps_every_build_of_that_version(self):
        names = entry_names(
            filter_by_pytorch_version(dev_entries(DEV_PYTORCH_FILES), "2.12")
        )
        # Only the 2.13 torch and device wheels (2, 4) go; non-torch packages are kept.
        self.assertEqual(names, [DEV_PYTORCH_FILES[i] for i in (0, 1, 3, 5, 6, 7, 8)])
        self.assertNotIn(
            DEV_PYTORCH_FILES[0],
            entry_names(
                filter_by_pytorch_version(dev_entries(DEV_PYTORCH_FILES), "2.1")
            ),
        )

    def test_pytorch_version_parses_sdists(self):
        sdists = ["torch-2.12.0.tar.gz", "torch-2.13.0+gitc0691f32.rocm10.0.0rc4.zip"]
        self.assertEqual(
            entry_names(filter_by_pytorch_version(dev_entries(sdists), "2.12")),
            ["torch-2.12.0.tar.gz"],
        )

    def test_git_sha_and_pytorch_version_combine(self):
        kept = filter_by_pytorch_version(
            filter_by_git_sha(
                dev_entries(DEV_PYTORCH_FILES),
                ["b97872d5", "c0691f32", "b3d5b111", "669b31ac"],
            ),
            "2.12",
        )
        self.assertEqual(entry_names(kept), B97872D5_BUILD)

    def test_dev_filters_require_structured(self):
        for flag in ("--git-sha=b97872d5", "--pytorch-version=2.12"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parse_arguments(["--version=10.0.0rc4", flag, "--list-archs"])

    def test_dev_filter_args_are_validated(self):
        args = parse_structured("--git-sha=B97872D5,b3d5b111", "--git-sha=669b31ac")
        self.assertEqual(args.git_sha, ["b97872d5", "b3d5b111", "669b31ac"])
        for bad in (
            "--git-sha=not-a-sha",
            "--git-sha=b97872d",
            "--pytorch-version=2.12x",
        ):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                parse_structured(bad)


if __name__ == "__main__":
    unittest.main()
