# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))

import prepare_pytorch_manifest as m


class PreparePyTorchManifestTest(unittest.TestCase):
    def test_main_passes_through_manifest_url(self) -> None:
        with mock.patch.object(
            m, "gha_set_output"
        ) as gha_set_output, mock.patch.object(
            m.subprocess, "check_call"
        ) as check_call:
            m.main(["--manifest-url", "https://example.com/manifest.json"])

        gha_set_output.assert_called_once_with(
            {"manifest_url": "https://example.com/manifest.json"}
        )
        check_call.assert_not_called()

    def test_main_invokes_generator_with_upload(self) -> None:
        manifest_dir = Path("/tmp/manifests")
        output_dir = Path("/tmp/upload")
        generator_args = [
            "--rocm-version",
            "7.13.0",
            "--run-id",
            "99999",
            "--release-type",
            "dev",
            "--platform",
            "linux",
            "--manifest-dir",
            os.fspath(manifest_dir),
            "--pytorch-git-refs",
            "release/2.12",
            "--projects",
            "pytorch",
            "--output-dir",
            os.fspath(output_dir),
            "--bucket",
            "test",
        ]

        with mock.patch.object(
            m.subprocess, "check_call"
        ) as check_call, mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(generator_args)

        check_call.assert_called_once_with(
            [
                sys.executable,
                str(m.GENERATOR_SCRIPT),
                "--upload",
                "--rocm-version",
                "7.13.0",
                "--run-id",
                "99999",
                "--release-type",
                "dev",
                "--platform",
                "linux",
                "--manifest-dir",
                os.fspath(manifest_dir),
                "--projects",
                "pytorch",
                "--output-dir",
                os.fspath(output_dir),
                "--bucket",
                "test",
                "--pytorch-git-refs",
                "release/2.12",
            ]
        )
        gha_set_output.assert_not_called()

    def test_main_rejects_missing_ref_without_manifest_url(self) -> None:
        with self.assertRaises(SystemExit):
            m.main(
                [
                    "--rocm-version",
                    "7.13.0",
                    "--run-id",
                    "99999",
                    "--manifest-dir",
                    "/tmp/manifests",
                ]
            )

    def test_main_rejects_multiple_refs(self) -> None:
        with self.assertRaises(SystemExit):
            m.main(
                [
                    "--rocm-version",
                    "7.13.0",
                    "--run-id",
                    "99999",
                    "--manifest-dir",
                    "/tmp/manifests",
                    "--pytorch-git-refs",
                    "release/2.9 release/2.10",
                ]
            )


if __name__ == "__main__":
    unittest.main()
