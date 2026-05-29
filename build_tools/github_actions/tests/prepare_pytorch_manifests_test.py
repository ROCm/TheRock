# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))

import prepare_pytorch_manifests as m


class PreparePyTorchManifestsTest(unittest.TestCase):
    def test_uploads_manifest_to_pytorch_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            manifest_url = m.upload_manifest_file(
                manifest_path=manifest_path,
                run_id="99999",
                platform="linux",
                release_type="dev",
                output_dir=Path(staging),
                bucket="test",
            )

            uploaded_manifest = (
                Path(staging)
                / "99999-linux"
                / "manifests"
                / "pytorch"
                / "manifest.json"
            )
            self.assertTrue(uploaded_manifest.is_file())
            self.assertEqual(
                manifest_url,
                "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/manifest.json",
            )

    def test_main_passes_through_manifest_url(self) -> None:
        with mock.patch.object(
            m, "gha_set_output"
        ) as gha_set_output, mock.patch.object(
            m, "gha_append_step_summary"
        ) as gha_append_step_summary:
            m.main(["--manifest-url", "https://example.com/manifest.json"])

        gha_set_output.assert_called_once_with(
            {"manifest_url": "https://example.com/manifest.json"}
        )
        gha_append_step_summary.assert_not_called()

    def test_main_generates_uploads_and_outputs_manifest_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp) / "manifests"
            manifest_path = (
                manifest_dir / "therock-manifest_torch_linux_release-2.12.json"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(
                m.subprocess, "check_call"
            ) as check_call, mock.patch.object(
                m, "gha_set_output"
            ) as gha_set_output, mock.patch.object(
                m, "gha_append_step_summary"
            ) as gha_append_step_summary:
                m.main(
                    [
                        "--rocm-version",
                        "7.13.0",
                        "--run-id",
                        "99999",
                        "--release-type",
                        "dev",
                        "--platform",
                        "linux",
                        "--manifest-dir",
                        str(manifest_dir),
                        "--pytorch-git-ref",
                        "release/2.12",
                        "--projects",
                        "pytorch",
                        "--output-dir",
                        staging,
                        "--bucket",
                        "test",
                    ]
                )

            check_call.assert_called_once_with(
                [
                    sys.executable,
                    str(m.GENERATOR_SCRIPT),
                    "--rocm-version",
                    "7.13.0",
                    "--platform",
                    "linux",
                    "--pytorch-git-refs",
                    "release/2.12",
                    "--output",
                    str(manifest_path),
                    "--projects",
                    "pytorch",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(
            outputs["manifest_url"],
            "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/"
            "therock-manifest_torch_linux_release-2.12.json",
        )
        gha_append_step_summary.assert_called_once()

    def test_main_requires_ref_when_generating_manifest(self) -> None:
        with self.assertRaises(SystemExit) as context:
            m.main(["--rocm-version", "7.13.0", "--run-id", "99999"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
