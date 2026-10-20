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
from unittest import mock

from botocore.exceptions import ClientError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from upload_release_packages import (
    infer_structured_product,
    object_exists,
    parse_arguments,
    upload_python_files,
    upload_release_packages,
    upload_tarball_files,
)


class FakeS3Client:
    """`head_object` finds only `existing` keys; `upload_file` records uploads."""

    def __init__(self, existing=(), status=404):
        self.existing = set(existing)
        self.status = status
        self.uploaded = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.existing:
            raise ClientError(
                {
                    "Error": {"Code": str(self.status)},
                    "ResponseMetadata": {"HTTPStatusCode": self.status},
                },
                "HeadObject",
            )
        return {}

    def upload_file(self, filename, bucket, key):
        self.uploaded.append((bucket, key))


class StructuredUploadTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "upload_release_packages.boto3.client", return_value=FakeS3Client()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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


class OverwriteGuardTest(unittest.TestCase):
    TORCH = "torch-2.12.0+rocm10.0.0-cp312-cp312-linux_x86_64.whl"
    TRITON = "triton-3.8.0+git669b31ac.rocm10.0.0-cp312-cp312-linux_x86_64.whl"
    TARBALL = "therock-dist-linux-multiarch-10.0.0.tar.gz"
    TORCH_DEST = (
        "therock-repo-amd-rc-pytorch",
        f"v5/rocm/pytorch/whl-next/torch/{TORCH}",
    )
    TARBALL_DEST = ("therock-repo-amd-rc-core", f"v5/rocm/core/tarball/{TARBALL}")

    def _upload(self, client, execute, allow_overwrite=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wheels").mkdir()
            (root / "tarball").mkdir()
            for name in (self.TORCH, self.TRITON):
                (root / "wheels" / name).write_text("x")
            (root / "tarball" / self.TARBALL).write_text("x")
            output = io.StringIO()
            with mock.patch(
                "upload_release_packages.boto3.client", return_value=client
            ):
                with contextlib.redirect_stdout(output):
                    upload_release_packages(
                        root,
                        upload_tarballs=True,
                        tarball_bucket_name="therock-repo-amd-rc-core",
                        tarball_bucket_prefix="v5/rocm/core/tarball/",
                        execute=execute,
                        multi_arch=True,
                        structured=True,
                        repo_stream="rc",
                        allow_overwrite=allow_overwrite,
                    )
            return output.getvalue()

    def test_errors_are_not_treated_as_missing(self):
        for status, hint in ((403, True), (500, False)):
            with self.subTest(status=status):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(ClientError):
                    object_exists(FakeS3Client(status=status), "b", "k")
                self.assertEqual("s3:ListBucket" in output.getvalue(), hint)

    def test_existing_files_are_skipped(self):
        for execute in (True, False):
            with self.subTest(execute=execute):
                client = FakeS3Client(existing=[self.TORCH_DEST, self.TARBALL_DEST])
                output = self._upload(client, execute=execute)
                for dest in (self.TORCH_DEST, self.TARBALL_DEST):
                    self.assertIn(
                        f"[SKIP] Already exists: s3://{'/'.join(dest)}", output
                    )
                    self.assertNotIn(dest, client.uploaded)
                if execute:
                    self.assertEqual(len(client.uploaded), 1)
                else:
                    self.assertEqual(output.count("[DRY-RUN] Would upload"), 1)

    def test_allow_overwrite_replaces_existing_files(self):
        for execute in (True, False):
            with self.subTest(execute=execute):
                client = FakeS3Client(existing=[self.TORCH_DEST])
                output = self._upload(client, execute=execute, allow_overwrite=True)
                if execute:
                    self.assertIn(f"Overwriting {self.TORCH} ...", output)
                    self.assertIn(self.TORCH_DEST, client.uploaded)
                    self.assertEqual(len(client.uploaded), 3)
                else:
                    self.assertIn(f"[DRY-RUN] Would overwrite: {self.TORCH}", output)
                    self.assertEqual(output.count("[DRY-RUN] Would upload"), 2)
                    self.assertEqual(client.uploaded, [])


if __name__ == "__main__":
    unittest.main()
