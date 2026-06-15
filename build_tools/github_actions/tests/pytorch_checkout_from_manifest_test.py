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
EXAMPLE_MANIFEST = PYTORCH_DIR / "pytorch_manifest_nightly.example.json"
sys.path.insert(0, os.fspath(PYTORCH_DIR))

import checkout_from_manifest


class PyTorchCheckoutFromManifestTest(unittest.TestCase):
    def _write_manifest(self, path: Path, entries: object) -> Path:
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def _commands_by_script(self, check_call: mock.Mock) -> dict[str, list[str]]:
        return {
            Path(call.args[0][1]).name: call.args[0]
            for call in check_call.call_args_list
        }

    def _option_value(self, command: list[str], option: str) -> str:
        return command[command.index(option) + 1]

    def test_main_checks_out_manifest_projects(self) -> None:
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
            resolved_checkout_root = checkout_root.resolve()

            checkout_from_manifest.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--checkout-root",
                    str(checkout_root),
                ]
            )

        commands = self._commands_by_script(check_call)
        self.assertEqual(
            set(commands),
            {
                "pytorch_torch_repo.py",
                "pytorch_audio_repo.py",
                "pytorch_triton_repo.py",
            },
        )

        pytorch_command = commands["pytorch_torch_repo.py"]
        audio_command = commands["pytorch_audio_repo.py"]
        triton_command = commands["pytorch_triton_repo.py"]

        self.assertEqual(
            self._option_value(pytorch_command, "--checkout-dir"),
            str(resolved_checkout_root / "pytorch"),
        )
        self.assertNotIn("--torch-dir", pytorch_command)
        self.assertNotIn("--torch-dir", audio_command)
        self.assertNotIn("--torch-dir", triton_command)
        self.assertEqual(
            self._option_value(audio_command, "--checkout-dir"),
            str(resolved_checkout_root / "pytorch_audio"),
        )
        self.assertEqual(self._option_value(triton_command, "--repo-hashtag"), "4" * 40)

    def test_checked_in_example_manifest_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            checkout_root = Path(tmp) / "checkouts"

            checkout_from_manifest.main(
                [
                    "--manifest",
                    str(EXAMPLE_MANIFEST),
                    "--checkout-root",
                    str(checkout_root),
                    "--projects",
                    "pytorch",
                    "--no-hipify",
                ]
            )

        manifest = json.loads(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
        command = self._commands_by_script(check_call)["pytorch_torch_repo.py"]
        self.assertEqual(
            self._option_value(command, "--gitrepo-origin"), manifest["pytorch"]["repo"]
        )
        self.assertEqual(
            self._option_value(command, "--repo-hashtag"), manifest["pytorch"]["commit"]
        )

    def test_projects_filter_and_checkout_options(self) -> None:
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
                    "--no-submodules",
                    "--no-commit-hipify",
                ]
            )

        commands = self._commands_by_script(check_call)
        self.assertEqual(
            set(commands), {"pytorch_torch_repo.py", "pytorch_vision_repo.py"}
        )
        for command in commands.values():
            self.assertIn("--no-hipify", command)
            self.assertIn("--no-submodules", command)
            self.assertIn("--no-commit-hipify", command)
            self.assertNotIn("--torch-dir", command)

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

    def test_manifest_root_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", [])

            with self.assertRaisesRegex(ValueError, "Manifest root"):
                checkout_from_manifest.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--checkout-root",
                        str(tmp_path / "checkouts"),
                    ]
                )

        check_call.assert_not_called()

    def test_unsupported_manifest_schema_version_errors_before_checkout(self) -> None:
        manifest = {
            "schema_version": 2,
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "schema_version"):
                checkout_from_manifest.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--checkout-root",
                        str(tmp_path / "checkouts"),
                    ]
                )

        check_call.assert_not_called()

    def test_manifest_with_no_supported_projects_errors(self) -> None:
        manifest = {
            "therock": {
                "repo": "https://github.com/ROCm/TheRock",
                "commit": "a" * 40,
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            checkout_from_manifest.subprocess, "check_call"
        ) as check_call:
            tmp_path = Path(tmp)
            manifest_path = self._write_manifest(tmp_path / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "no supported PyTorch projects"):
                checkout_from_manifest.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--checkout-root",
                        str(tmp_path / "checkouts"),
                    ]
                )

        check_call.assert_not_called()

    def test_missing_required_manifest_fields_error_before_checkout(self) -> None:
        required_fields = {
            "repo": "https://github.com/ROCm/pytorch",
            "commit": "1" * 40,
        }
        for missing_field in required_fields:
            with self.subTest(missing_field=missing_field):
                manifest = {"pytorch": dict(required_fields)}
                manifest["pytorch"].pop(missing_field)

                with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                    checkout_from_manifest.subprocess, "check_call"
                ) as check_call:
                    tmp_path = Path(tmp)
                    manifest_path = self._write_manifest(
                        tmp_path / "manifest.json", manifest
                    )

                    with self.assertRaisesRegex(
                        ValueError, f"pytorch.*{missing_field}"
                    ):
                        checkout_from_manifest.main(
                            [
                                "--manifest",
                                str(manifest_path),
                                "--checkout-root",
                                str(tmp_path / "checkouts"),
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
            checkout_root = tmp_path / "checkouts"
            manifest_path = checkout_root.resolve() / "pytorch_manifest.json"

            def fake_urlretrieve(_url, output_path):
                Path(output_path).write_text(json.dumps(manifest), encoding="utf-8")

            urlretrieve.side_effect = fake_urlretrieve

            checkout_from_manifest.main(
                [
                    "--manifest-url",
                    "https://example.com/manifest.json",
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
