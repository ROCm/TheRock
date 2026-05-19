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
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent))

import generate_pytorch_manifest_upfront as m


class GeneratePyTorchManifestUpfrontTest(unittest.TestCase):
    def _patch_github_api(self, *, resolves: dict, files: dict):
        def fake_resolve(repo: str, ref: str) -> str:
            return resolves[(repo, ref)]

        def fake_fetch(repo: str, path: str, ref: str) -> str:
            return files[(repo, path, ref)]

        return mock.patch.multiple(
            m,
            gha_resolve_git_ref=mock.Mock(side_effect=fake_resolve),
            gha_fetch_file_contents=mock.Mock(side_effect=fake_fetch),
        )

    def test_default_refs_match_release_matrix(self) -> None:
        self.assertEqual(
            m.DEFAULT_PYTORCH_GIT_REFS,
            [
                "release/2.8",
                "release/2.9",
                "release/2.10",
                "release/2.11",
                "release/2.12",
                "nightly",
            ],
        )

    def test_default_projects_for_platform(self) -> None:
        self.assertEqual(
            m.default_projects_for_platform("linux"),
            ["pytorch", "pytorch_audio", "pytorch_vision", "triton", "apex"],
        )
        self.assertEqual(
            m.default_projects_for_platform("windows"),
            ["pytorch", "pytorch_audio", "pytorch_vision", "triton"],
        )

    def test_stable_linux_manifest_resolves_related_commits_and_versions(self) -> None:
        shas = {
            "pytorch": "1" * 40,
            "audio": "2" * 40,
            "vision": "3" * 40,
            "triton": "4" * 40,
            "apex": "5" * 40,
        }
        related_commits = "\n".join(
            [
                "centos|src|torchaudio|release/2.10|"
                f"{shas['audio']}|https://github.com/pytorch/audio",
                "centos|src|torchvision|release/2.10|"
                f"{shas['vision']}|https://github.com/pytorch/vision",
                "ubuntu|src|torchvision|ignored|"
                f"{'9' * 40}|https://github.com/pytorch/vision",
                "centos|src|apex|release/2.10|"
                f"{shas['apex']}|https://github.com/ROCm/apex",
            ]
        )
        resolves = {
            ("ROCm/pytorch", "release/2.10"): shas["pytorch"],
        }
        files = {
            ("ROCm/pytorch", "related_commits", shas["pytorch"]): related_commits,
            (
                "ROCm/pytorch",
                ".ci/docker/triton_version.txt",
                shas["pytorch"],
            ): "3.6.0\n",
            (
                "ROCm/pytorch",
                ".ci/docker/ci_commit_pins/triton.txt",
                shas["pytorch"],
            ): shas["triton"],
            ("ROCm/pytorch", "version.txt", shas["pytorch"]): "2.10.0\n",
            ("pytorch/audio", "version.txt", shas["audio"]): "2.10.0\n",
            ("pytorch/vision", "version.txt", shas["vision"]): "0.25.0\n",
            ("ROCm/apex", "version.txt", shas["apex"]): "1.10.0\n",
        }

        with self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="release/2.10",
                rocm_version="7.13.0a20260501",
                version_suffix="+rocm7.13.0a20260501",
                platform="linux",
                projects=m.default_projects_for_platform("linux"),
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="users/example/branch",
            )

        self.assertEqual(
            set(manifest),
            {
                "pytorch",
                "pytorch_audio",
                "pytorch_vision",
                "triton",
                "apex",
                "therock",
            },
        )
        self.assertEqual(manifest["pytorch"]["commit"], shas["pytorch"])
        self.assertEqual(manifest["pytorch"]["repo"], "https://github.com/ROCm/pytorch")
        self.assertEqual(manifest["pytorch"]["branch"], "release/2.10")
        self.assertEqual(manifest["pytorch"]["version"], "2.10.0+rocm7.13.0a20260501")
        self.assertEqual(manifest["pytorch_audio"]["commit"], shas["audio"])
        self.assertEqual(
            manifest["pytorch_audio"]["version"], "2.10.0+rocm7.13.0a20260501"
        )
        self.assertEqual(manifest["pytorch_vision"]["commit"], shas["vision"])
        self.assertEqual(
            manifest["pytorch_vision"]["version"], "0.25.0+rocm7.13.0a20260501"
        )
        self.assertEqual(manifest["triton"]["commit"], shas["triton"])
        self.assertEqual(manifest["triton"]["version"], "3.6.0+rocm7.13.0a20260501")
        self.assertEqual(manifest["apex"]["commit"], shas["apex"])
        self.assertEqual(manifest["therock"]["rocm_version"], "7.13.0a20260501")

    def test_nightly_linux_manifest_resolves_branches(self) -> None:
        shas = {
            "pytorch": "1" * 40,
            "audio": "2" * 40,
            "vision": "3" * 40,
            "triton": "4" * 40,
            "apex": "5" * 40,
        }
        resolves = {
            ("pytorch/pytorch", "nightly"): shas["pytorch"],
            ("pytorch/audio", "nightly"): shas["audio"],
            ("pytorch/vision", "nightly"): shas["vision"],
            ("ROCm/triton", "release/3.6.x"): shas["triton"],
            ("ROCm/apex", "master"): shas["apex"],
        }
        files = {
            (
                "pytorch/pytorch",
                ".ci/docker/triton_version.txt",
                shas["pytorch"],
            ): "3.6.0\n",
            ("pytorch/pytorch", "version.txt", shas["pytorch"]): "2.11.0a0\n",
            ("pytorch/audio", "version.txt", shas["audio"]): "2.11.0a0\n",
            ("pytorch/vision", "version.txt", shas["vision"]): "0.26.0a0\n",
            ("ROCm/apex", "version.txt", shas["apex"]): "1.11.0\n",
        }

        with self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="nightly",
                rocm_version="7.13.0.dev0+abc",
                version_suffix="+devrocm7.13.0.dev0-abc",
                platform="linux",
                projects=m.default_projects_for_platform("linux"),
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertEqual(
            manifest["pytorch"]["repo"], "https://github.com/pytorch/pytorch"
        )
        self.assertEqual(manifest["pytorch"]["branch"], "nightly")
        self.assertEqual(manifest["pytorch_audio"]["branch"], "nightly")
        self.assertEqual(manifest["pytorch_vision"]["branch"], "nightly")
        self.assertEqual(manifest["triton"]["repo"], "https://github.com/ROCm/triton")
        self.assertEqual(manifest["triton"]["branch"], "release/3.6.x")
        self.assertEqual(manifest["apex"]["branch"], "master")
        self.assertEqual(
            manifest["pytorch"]["version"], "2.11.0a0+devrocm7.13.0.dev0-abc"
        )

    def test_windows_manifest_uses_triton_windows_and_excludes_apex(self) -> None:
        shas = {
            "pytorch": "1" * 40,
            "audio": "2" * 40,
            "vision": "3" * 40,
            "triton": "4" * 40,
        }
        related_commits = "\n".join(
            [
                "centos|src|torchaudio|release/2.10|"
                f"{shas['audio']}|https://github.com/pytorch/audio",
                "centos|src|torchvision|release/2.10|"
                f"{shas['vision']}|https://github.com/pytorch/vision",
            ]
        )
        resolves = {
            ("ROCm/pytorch", "release/2.10"): shas["pytorch"],
        }
        files = {
            ("ROCm/pytorch", "related_commits", shas["pytorch"]): related_commits,
            (
                "ROCm/pytorch",
                ".ci/docker/triton_version.txt",
                shas["pytorch"],
            ): "3.6.0\n",
            (
                "ROCm/pytorch",
                ".ci/docker/ci_commit_pins/triton-windows.txt",
                shas["pytorch"],
            ): shas["triton"],
            ("ROCm/pytorch", "version.txt", shas["pytorch"]): "2.10.0\n",
            ("pytorch/audio", "version.txt", shas["audio"]): "2.10.0\n",
            ("pytorch/vision", "version.txt", shas["vision"]): "0.25.0\n",
        }

        with self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="release/2.10",
                rocm_version="7.13.0a20260501",
                version_suffix="+rocm7.13.0a20260501",
                platform="windows",
                projects=m.default_projects_for_platform("windows"),
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertNotIn("apex", manifest)
        self.assertEqual(manifest["triton"]["commit"], shas["triton"])
        self.assertEqual(
            manifest["triton"]["repo"], "https://github.com/triton-lang/triton-windows"
        )

    def test_main_writes_single_output_manifest_with_project_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "manifest.json"
            generated = {
                "pytorch": {
                    "repo": "https://github.com/ROCm/pytorch",
                    "commit": "1" * 40,
                    "version": "2.10.0+rocm7.13.0",
                }
            }
            with mock.patch.object(
                m,
                "detect_therock_source_info",
                return_value=m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                ),
            ), mock.patch.object(
                m, "generate_manifest", return_value=generated
            ) as generate:
                m.main(
                    [
                        "--rocm-version",
                        "7.13.0",
                        "--version-suffix",
                        "+rocm7.13.0",
                        "--platform",
                        "linux",
                        "--output",
                        str(out_path),
                        "--pytorch-git-refs",
                        "release/2.10",
                        "--projects",
                        "pytorch triton",
                    ]
                )

            self.assertEqual(
                json.loads(out_path.read_text(encoding="utf-8")), generated
            )
            generate.assert_called_once()
            self.assertEqual(
                generate.call_args.kwargs["projects"], ["pytorch", "triton"]
            )

    def test_output_requires_single_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                m.main(
                    [
                        "--rocm-version",
                        "7.13.0",
                        "--version-suffix",
                        "+rocm7.13.0",
                        "--output",
                        str(Path(tmp) / "manifest.json"),
                        "--pytorch-git-refs",
                        "release/2.10 nightly",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
