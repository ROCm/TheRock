#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for upload_release_packages.py."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from upload_release_packages import (
    infer_structured_product,
    parse_arguments,
    upload_python_files,
    upload_tarball_files,
)


class StructuredUploadTest(unittest.TestCase):
    def test_infer_structured_product_routes_torch_prefixes_to_pytorch(self):
        filenames = [
            "torch-2.10.0+rocm7.13.0-cp312-cp312-linux_x86_64.whl",
            "torchvision-0.25.0+rocm7.13.0-cp312-cp312-linux_x86_64.whl",
            "amd_torch_device_gfx942-2.10.0+rocm7.13.0-py3-none-linux_x86_64.whl",
            "amd_torchvision_device_gfx950-0.25.0+rocm7.13.0-py3-none-linux_x86_64.whl",
        ]

        for filename in filenames:
            with self.subTest(filename=filename):
                self.assertEqual(infer_structured_product(filename), "pytorch")

    def test_infer_structured_product_rejects_unknown_package(self):
        with self.assertRaises(ValueError):
            infer_structured_product("some-unrelated-tool-1.0.0-py3-none-any.whl")

    def test_structured_upload_hard_fails_on_unknown_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels = root / "wheels"
            wheels.mkdir()
            (wheels / "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl").write_text("x")
            (wheels / "some-unrelated-tool-1.0.0-py3-none-any.whl").write_text("x")

            output = io.StringIO()
            # An unrecognized package must abort the whole run rather than
            # being silently skipped, so a dropped package can't be masked by
            # an apparently-successful run.
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as ctx:
                    upload_python_files(
                        root,
                        bucket_name="unused",
                        bucket_prefix="unused",
                        execute=False,
                        structured=True,
                        python_index="whl-next",
                        repo_stream="rc",
                    )

            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("[ERROR]:", output.getvalue())

    def test_structured_python_dry_run_routes_to_product_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels = root / "wheels"
            wheels.mkdir()
            (wheels / "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl").write_text("x")
            (
                wheels / "torch-2.10.0+rocm7.13.0-cp312-cp312-linux_x86_64.whl"
            ).write_text("x")
            (
                wheels
                / "amd_torch_device_gfx942-2.10.0+rocm7.13.0-py3-none-linux_x86_64.whl"
            ).write_text("x")
            (
                wheels
                / "jax_rocm7_plugin-0.9.2+rocm7.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
            ).write_text("x")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                count = upload_python_files(
                    root,
                    bucket_name="unused",
                    bucket_prefix="unused",
                    execute=False,
                    structured=True,
                    python_index="whl-next",
                    repo_stream="rc",
                )

            text = output.getvalue()
            self.assertEqual(count, 4)
            self.assertIn(
                "s3://therock-repo-amd-rc-core/v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
                text,
            )
            self.assertIn(
                "s3://therock-repo-amd-rc-pytorch/v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0-cp312-cp312-linux_x86_64.whl",
                text,
            )
            self.assertIn(
                "s3://therock-repo-amd-rc-pytorch/v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0-py3-none-linux_x86_64.whl",
                text,
            )
            self.assertIn(
                "s3://therock-repo-amd-rc-jax/v5/rocm/jax/whl-next/jax-rocm7-plugin/jax_rocm7_plugin-0.9.2+rocm7.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                text,
            )

    def test_structured_tarball_dry_run_uses_release_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tarballs = root / "tarball"
            tarballs.mkdir()
            (tarballs / "therock-dist-linux-multiarch-7.13.0.tar.gz").write_text("x")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                count = upload_tarball_files(
                    root,
                    bucket_name="therock-repo-amd-rc-core",
                    bucket_prefix="v5/rocm/core/tarball/",
                    execute=False,
                    structured=True,
                    tarball_variant="release",
                )

            self.assertEqual(count, 1)
            self.assertIn(
                "s3://therock-repo-amd-rc-core/v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.tar.gz",
                output.getvalue(),
            )

    def test_structured_tarball_dry_run_uses_asan_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tarballs = root / "tarball-asan"
            tarballs.mkdir()
            (tarballs / "therock-dist-linux-multiarch-7.13.0.tar.gz").write_text("x")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                count = upload_tarball_files(
                    root,
                    bucket_name="therock-repo-amd-rc-core",
                    bucket_prefix="v5/rocm/core/tarball-asan/",
                    execute=False,
                    structured=True,
                    tarball_variant="asan",
                )

            self.assertEqual(count, 1)
            self.assertIn(
                "s3://therock-repo-amd-rc-core/v5/rocm/core/tarball-asan/therock-dist-linux-multiarch-7.13.0.tar.gz",
                output.getvalue(),
            )

    def test_structured_execute_requires_release_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                parse_arguments(
                    [
                        "--input-dir",
                        str(root),
                        "--structured",
                        "--execute",
                    ]
                )

    def test_structured_release_bucket_args_use_repo_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_arguments(
                [
                    "--input-dir",
                    str(root),
                    "--structured",
                    "--use-release-buckets",
                    "--tarball-variant=asan",
                    "--repo-stream=nightly",
                ]
            )

            self.assertTrue(args.multi_arch)
            self.assertEqual(args.bucket, "")
            self.assertEqual(args.tarball_bucket, "therock-repo-amd-nightly-core")
            self.assertEqual(args.tarball_bucket_prefix, "v5/rocm/core/tarball-asan/")


if __name__ == "__main__":
    unittest.main()
