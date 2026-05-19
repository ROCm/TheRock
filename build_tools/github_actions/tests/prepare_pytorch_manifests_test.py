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

    def test_generate_manifest_files_uses_ref_specific_defaults(self) -> None:
        def fake_generate_manifest(**kwargs):
            return {
                "pytorch": {
                    "repo": "https://github.com/ROCm/pytorch",
                    "commit": "1" * 40,
                    "branch": kwargs["pytorch_git_ref"],
                }
            }

        def fake_default_projects(platform: str, pytorch_ref: str) -> list[str]:
            return ["pytorch", pytorch_ref]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            m, "generate_manifest", side_effect=fake_generate_manifest
        ) as generate_manifest, mock.patch.object(
            m,
            "default_projects_for_pytorch_ref",
            side_effect=fake_default_projects,
        ):
            m.generate_manifest_files(
                manifest_dir=Path(tmp),
                pytorch_git_refs=["release/2.12", "release/2.13"],
                rocm_version="7.13.0",
                version_suffix="+rocm7.13.0",
                platform="windows",
                projects=None,
                therock_info=m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                ),
            )

        self.assertEqual(
            [call.kwargs["projects"] for call in generate_manifest.call_args_list],
            [
                ["pytorch", "release/2.12"],
                ["pytorch", "release/2.13"],
            ],
        )

    def test_uploads_directory_to_correct_path_and_outputs_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp)
            staging_dir = Path(staging)
            (manifest_dir / "manifest.json").write_text("{}")

            upload = m.upload_manifest_directory(
                manifest_dir=manifest_dir,
                run_id="99999",
                platform="linux",
                release_type="dev",
                output_dir=staging_dir,
                bucket="test",
            )

            dest_dir = staging_dir / "99999-linux" / "manifests" / "pytorch"
            self.assertTrue((dest_dir / "manifest.json").is_file())
            self.assertEqual(
                upload.manifest_dir_url,
                "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch",
            )
            self.assertEqual(
                upload.manifest_dir_s3_uri,
                "s3://test/99999-linux/manifests/pytorch",
            )

    def test_main_passes_through_manifest_url(self) -> None:
        with mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(
                [
                    "--manifest-url",
                    "https://example.com/manifest.json",
                ]
            )

        gha_set_output.assert_called_once_with(
            {"manifest_url": "https://example.com/manifest.json"}
        )

    def test_main_generates_single_manifest_and_writes_url(self) -> None:
        generated = {
            "pytorch": {
                "repo": "https://github.com/ROCm/pytorch",
                "commit": "1" * 40,
                "branch": "release/2.12",
                "version": "2.12.0+rocm7.13.0",
            }
        }
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp) / "manifests"
            with mock.patch.object(
                m, "generate_manifest", return_value=generated
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
            ) as gha_set_output:
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
                        "--pytorch-git-refs",
                        "release/2.12",
                        "--output-dir",
                        staging,
                        "--bucket",
                        "test",
                    ]
                )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(
            outputs["manifest_url"],
            "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/"
            "therock-manifest_torch_linux_release-2.12.json",
        )

    def test_main_generates_release_matrix(self) -> None:
        def fake_generate_manifest(**kwargs):
            return {
                "pytorch": {
                    "repo": "https://github.com/ROCm/pytorch",
                    "commit": "1" * 40,
                    "branch": kwargs["pytorch_git_ref"],
                    "version": "2.12.0+rocm7.13.0",
                }
            }

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp) / "manifests"
            with mock.patch.object(
                m, "generate_manifest", side_effect=fake_generate_manifest
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
            ) as gha_set_output:
                m.main(
                    [
                        "--output-mode",
                        "matrix",
                        "--matrix-preset",
                        "linux-release",
                        "--rocm-version",
                        "7.13.0",
                        "--run-id",
                        "99999",
                        "--release-type",
                        "dev",
                        "--manifest-dir",
                        str(manifest_dir),
                        "--pytorch-git-refs",
                        "release/2.8 nightly",
                        "--python-versions",
                        "3.13 3.14",
                        "--output-dir",
                        staging,
                        "--bucket",
                        "test",
                    ]
                )

        output = gha_set_output.call_args.args[0]
        matrix = json.loads(output["matrix"])
        self.assertEqual(len(matrix["include"]), 3)
        self.assertNotIn(
            {
                "python_version": "3.14",
                "pytorch_git_ref": "release/2.8",
                "manifest_url": (
                    "https://test.s3.amazonaws.com/99999-linux/manifests/pytorch/"
                    "therock-manifest_torch_linux_release-2.8.json"
                ),
            },
            matrix["include"],
        )


if __name__ == "__main__":
    unittest.main()
