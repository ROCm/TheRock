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
PYTORCH_DIR = THIS_DIR.parents[2] / "external-builds" / "pytorch"
sys.path.insert(0, os.fspath(PYTORCH_DIR))

import checkout_from_manifest


class CheckoutFromManifestTest(unittest.TestCase):
    def _write_manifest(self, path: Path, entries: dict) -> Path:
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_main_checks_out_manifest_projects_in_order(self) -> None:
        manifest = {
            "pytorch_audio": {
                "repo": "https://github.com/pytorch/audio",
                "commit": "2" * 40,
            },
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
            },
            "triton": {
                "repo": "https://github.com/ROCm/triton",
                "commit": "4" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)
            checkout_root = tmp_path / "checkouts"

            checkout_from_manifest.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--checkout-root",
                    str(checkout_root),
                ]
            )

        calls = [call.args[0] for call in check_call.call_args_list]
        self.assertEqual(
            [Path(call[1]).name for call in calls],
            [
                "pytorch_torch_repo.py",
                "pytorch_audio_repo.py",
                "pytorch_triton_repo.py",
            ],
        )
        self.assertEqual(
            calls[0][calls[0].index("--checkout-dir") + 1],
            str(checkout_root / "pytorch"),
        )
        self.assertNotIn("--torch-dir", calls[0])
        self.assertEqual(
            calls[1][calls[1].index("--torch-dir") + 1], str(checkout_root / "pytorch")
        )
        self.assertEqual(
            calls[1][calls[1].index("--checkout-dir") + 1],
            str(checkout_root / "pytorch_audio"),
        )
        self.assertEqual(calls[2][calls[2].index("--repo-hashtag") + 1], "4" * 40)

    def test_projects_filter_and_no_hipify(self) -> None:
        manifest = {
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
            },
            "pytorch_audio": {
                "repo": "https://github.com/pytorch/audio",
                "commit": "2" * 40,
            },
            "pytorch_vision": {
                "repo": "https://github.com/pytorch/vision",
                "commit": "3" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)
            checkout_root = tmp_path / "checkouts"

            checkout_from_manifest.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--checkout-root",
                    str(checkout_root),
                    "--projects",
                    "pytorch;pytorch_vision",
                    "--no-hipify",
                ]
            )

        calls = [call.args[0] for call in check_call.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertEqual(Path(calls[0][1]).name, "pytorch_torch_repo.py")
        self.assertEqual(Path(calls[1][1]).name, "pytorch_vision_repo.py")
        self.assertIn("--no-hipify", calls[0])
        self.assertIn("--no-hipify", calls[1])
        self.assertEqual(
            calls[1][calls[1].index("--torch-dir") + 1], str(checkout_root / "pytorch")
        )

    def test_unknown_requested_project_errors(self) -> None:
        manifest = {
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)
            checkout_root = tmp_path / "checkouts"

            with self.assertRaises(SystemExit):
                checkout_from_manifest.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--checkout-root",
                        str(checkout_root),
                        "--projects",
                        "pytorch triton",
                    ]
                )

        check_call.assert_not_called()

    def test_downloads_manifest_url_and_validates_ref(self) -> None:
        manifest = {
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
                "branch": "release/2.12",
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.urllib.request, "urlretrieve"
        ) as urlretrieve, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "downloaded.json"
            checkout_root = tmp_path / "checkouts"

            def fake_urlretrieve(_url, output_path):
                Path(output_path).write_text(json.dumps(manifest), encoding="utf-8")

            urlretrieve.side_effect = fake_urlretrieve

            checkout_from_manifest.main(
                [
                    "--manifest-url",
                    "https://example.com/manifest.json",
                    "--manifest-output",
                    str(manifest_path),
                    "--checkout-root",
                    str(checkout_root),
                    "--expected-pytorch-git-ref",
                    "release/2.12",
                ]
            )

        urlretrieve.assert_called_once_with(
            "https://example.com/manifest.json", manifest_path
        )
        self.assertEqual(len(check_call.call_args_list), 1)

    def test_expected_ref_mismatch_errors_before_checkout(self) -> None:
        manifest = {
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
                "branch": "release/2.12",
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "release/2.11"):
                checkout_from_manifest.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--checkout-root",
                        str(tmp_path / "checkouts"),
                        "--expected-pytorch-git-ref",
                        "release/2.11",
                    ]
                )

        check_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
