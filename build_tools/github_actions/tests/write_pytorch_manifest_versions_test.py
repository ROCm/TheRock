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

import write_pytorch_manifest_versions as m


class WritePyTorchManifestVersionsTest(unittest.TestCase):
    def test_collects_all_package_versions(self) -> None:
        manifest = {
            "pytorch": {"version": "2.12.0+rocm7.13.0"},
            "pytorch_audio": {"version": "2.12.0+rocm7.13.0"},
            "pytorch_vision": {"version": "0.27.0+rocm7.13.0"},
            "triton": {"version": "3.7.0+rocm7.13.0"},
            "apex": {"version": "1.12.0+rocm7.13.0"},
            "therock": {"rocm_version": "7.13.0"},
        }

        self.assertEqual(
            m.collect_versions(manifest),
            {
                "torch_version": "2.12.0+rocm7.13.0",
                "torchaudio_version": "2.12.0+rocm7.13.0",
                "torchvision_version": "0.27.0+rocm7.13.0",
                "triton_version": "3.7.0+rocm7.13.0",
                "apex_version": "1.12.0+rocm7.13.0",
            },
        )

    def test_allows_filtered_manifests(self) -> None:
        manifest = {
            "pytorch": {"version": "2.12.0+rocm7.13.0"},
            "triton": {"version": "3.7.0+rocm7.13.0"},
        }

        self.assertEqual(
            m.collect_versions(manifest),
            {
                "torch_version": "2.12.0+rocm7.13.0",
                "triton_version": "3.7.0+rocm7.13.0",
            },
        )

    def test_missing_version_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "pytorch"):
            m.collect_versions({"pytorch": {"commit": "1" * 40}})

    def test_main_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"pytorch": {"version": "2.12.0+rocm7.13.0"}}),
                encoding="utf-8",
            )

            with mock.patch.object(m, "gha_set_output") as gha_set_output:
                m.main(["--manifest", str(manifest_path)])

        gha_set_output.assert_called_once_with({"torch_version": "2.12.0+rocm7.13.0"})


if __name__ == "__main__":
    unittest.main()
