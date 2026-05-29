# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
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

            upload = m.upload_manifest_file(
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
                upload.manifest_url,
                "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/manifest.json",
            )
            self.assertEqual(
                upload.manifest_dir_s3_uri,
                "s3://test/99999-linux/manifests/pytorch",
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
        self.assertIn(
            "https://example.com/manifest.json",
            gha_append_step_summary.call_args.args[0],
        )

    def test_main_generates_uploads_and_outputs_manifest_url(self) -> None:
        generated_manifest = {
            "pytorch": m.GitSourceInfo(
                repo="https://github.com/ROCm/pytorch",
                commit="1" * 40,
                branch="release/2.12",
                version="2.12.0+rocm7.13.0",
            ),
            "therock": m.GitSourceInfo(
                repo="https://github.com/ROCm/TheRock",
                commit="a" * 40,
                branch="main",
                version="7.13.0",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp) / "manifests"
            with mock.patch.object(
                m, "generate_manifest", return_value=generated_manifest
            ), mock.patch.object(
                m,
                "resolve_therock_source_info",
                return_value=m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                ),
            ), mock.patch.object(
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

            manifest_path = (
                manifest_dir / "therock-manifest_torch_linux_release-2.12.json"
            )
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["schema_version"], 1)
            self.assertEqual(manifest_data["pytorch"]["branch"], "release/2.12")

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(
            outputs["manifest_url"],
            "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/"
            "therock-manifest_torch_linux_release-2.12.json",
        )
        self.assertEqual(
            outputs["manifest_dir_s3_uri"],
            "s3://test/99999-linux/manifests/pytorch",
        )
        gha_append_step_summary.assert_called_once()

    def test_main_requires_ref_when_generating_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "--pytorch-git-ref"):
            m.main(["--rocm-version", "7.13.0", "--run-id", "99999"])


if __name__ == "__main__":
    unittest.main()
