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

import generate_pytorch_manifest_matrix as m


class GeneratePyTorchManifestMatrixTest(unittest.TestCase):
    def _write_manifest(self, path: Path, pytorch_ref: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "pytorch": {
                        "repo": "https://github.com/ROCm/pytorch",
                        "commit": "1" * 40,
                        "branch": pytorch_ref,
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_collects_manifest_urls_from_actual_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            self._write_manifest(manifest_dir / "first.json", "release/2.12")
            self._write_manifest(manifest_dir / "second.json", "nightly")

            self.assertEqual(
                m.collect_manifest_urls(
                    manifest_dir=manifest_dir,
                    manifest_dir_url="https://example.com/manifests/",
                ),
                {
                    "release/2.12": "https://example.com/manifests/first.json",
                    "nightly": "https://example.com/manifests/second.json",
                },
            )

    def test_builds_matrix_with_explicit_manifest_urls_and_excludes(self) -> None:
        matrix = m.build_matrix(
            manifest_urls={
                "release/2.8": "https://example.com/release-2.8.json",
                "nightly": "https://example.com/nightly.json",
            },
            python_versions=["3.13", "3.14"],
            pytorch_git_refs=["release/2.8", "nightly"],
            excludes={("release/2.8", "3.14")},
        )

        self.assertEqual(
            matrix,
            {
                "include": [
                    {
                        "python_version": "3.13",
                        "pytorch_git_ref": "release/2.8",
                        "manifest_url": "https://example.com/release-2.8.json",
                    },
                    {
                        "python_version": "3.13",
                        "pytorch_git_ref": "nightly",
                        "manifest_url": "https://example.com/nightly.json",
                    },
                    {
                        "python_version": "3.14",
                        "pytorch_git_ref": "nightly",
                        "manifest_url": "https://example.com/nightly.json",
                    },
                ]
            },
        )

    def test_missing_requested_ref_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "release/2.12"):
            m.build_matrix(
                manifest_urls={"nightly": "https://example.com/nightly.json"},
                python_versions=["3.12"],
                pytorch_git_refs=["release/2.12"],
                excludes=set(),
            )

    def test_main_writes_matrix_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            self._write_manifest(manifest_dir / "manifest.json", "release/2.12")

            with mock.patch.object(m, "gha_set_output") as gha_set_output:
                m.main(
                    [
                        "--manifest-dir",
                        str(manifest_dir),
                        "--manifest-dir-url",
                        "https://example.com/manifests",
                        "--python-versions",
                        "3.12 3.13",
                        "--pytorch-git-refs",
                        "release/2.12",
                    ]
                )

        gha_set_output.assert_called_once()
        output = json.loads(gha_set_output.call_args.args[0]["matrix"])
        self.assertEqual(len(output["include"]), 2)
        self.assertEqual(
            output["include"][0]["manifest_url"],
            "https://example.com/manifests/manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
