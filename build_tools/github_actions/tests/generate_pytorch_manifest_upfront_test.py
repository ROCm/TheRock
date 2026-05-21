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

    def _stable_windows_github_data(self) -> tuple[dict[str, str], dict, dict]:
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
            ("ROCm/pytorch", "version.txt", shas["pytorch"]): "2.10.0\n",
            ("pytorch/audio", "version.txt", shas["audio"]): "2.10.0\n",
            ("pytorch/vision", "version.txt", shas["vision"]): "0.25.0\n",
        }
        return shas, resolves, files

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
            ["pytorch", "pytorch_audio", "pytorch_vision"],
        )

    def test_windows_triton_default_is_ref_gated(self) -> None:
        windows_projects = ["pytorch", "pytorch_audio", "pytorch_vision"]
        self.assertEqual(
            m.default_projects_for_pytorch_ref("windows", "release/2.13"),
            windows_projects,
        )

        with mock.patch.object(m, "WINDOWS_TRITON_MIN_RELEASE", (2, 13)):
            self.assertEqual(
                m.default_projects_for_pytorch_ref("windows", "release/2.12"),
                windows_projects,
            )
            self.assertEqual(
                m.default_projects_for_pytorch_ref("windows", "release/2.13"),
                windows_projects + ["triton"],
            )
            self.assertEqual(
                m.default_projects_for_pytorch_ref("windows", "123456abcdef"),
                windows_projects,
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

    def test_stable_manifest_requires_related_commit_pins(self) -> None:
        pytorch_sha = "1" * 40
        resolves = {
            ("ROCm/pytorch", "release/2.10"): pytorch_sha,
        }
        files = {
            ("ROCm/pytorch", "related_commits", pytorch_sha): "",
        }

        with self._patch_github_api(resolves=resolves, files=files):
            with self.assertRaisesRegex(ValueError, "torchaudio"):
                m.generate_manifest(
                    pytorch_git_ref="release/2.10",
                    rocm_version="7.13.0a20260501",
                    version_suffix="+rocm7.13.0a20260501",
                    platform="linux",
                    projects=["pytorch", "pytorch_audio"],
                    therock_commit="a" * 40,
                    therock_repo="https://github.com/ROCm/TheRock",
                    therock_branch="main",
                )

    def test_stable_manifest_rejects_repos_without_pin_policy(self) -> None:
        pytorch_sha = "1" * 40
        resolves = {
            ("ROCm/pytorch", "release/2.10"): pytorch_sha,
        }
        files: dict[tuple[str, str, str], str] = {}

        with mock.patch.dict(
            m.REPOS,
            {
                "dummy": m.RepoConfig(
                    stable_repo="example/dummy",
                    nightly_repo="example/dummy",
                    version_file="version.txt",
                )
            },
        ), self._patch_github_api(resolves=resolves, files=files):
            with self.assertRaisesRegex(ValueError, "stable PyTorch release pin"):
                m.resolve_sources(
                    "release/2.10",
                    "+rocm7.13.0",
                    "linux",
                    ["pytorch", "dummy"],
                )

    def test_pytorch_only_manifest_does_not_fetch_related_commits(self) -> None:
        pytorch_sha = "1" * 40
        resolves = {
            ("ROCm/pytorch", "release/2.10"): pytorch_sha,
        }
        files = {
            ("ROCm/pytorch", "version.txt", pytorch_sha): "2.10.0\n",
        }

        with self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="release/2.10",
                rocm_version="7.13.0a20260501",
                version_suffix="+rocm7.13.0a20260501",
                platform="linux",
                projects=["pytorch"],
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertEqual(set(manifest), {"pytorch", "therock"})
        self.assertEqual(manifest["pytorch"]["version"], "2.10.0+rocm7.13.0a20260501")

    def test_malformed_related_commits_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Malformed related_commits"):
            m._parse_related_commits("centos|src|torchaudio")

    def test_nightly_linux_manifest_uses_triton_pin(self) -> None:
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
            ("ROCm/apex", "master"): shas["apex"],
        }
        files = {
            (
                "pytorch/pytorch",
                ".ci/docker/triton_version.txt",
                shas["pytorch"],
            ): "3.6.0\n",
            (
                "pytorch/pytorch",
                ".ci/docker/ci_commit_pins/triton.txt",
                shas["pytorch"],
            ): shas["triton"],
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
        self.assertEqual(manifest["triton"]["commit"], shas["triton"])
        self.assertNotIn("branch", manifest["triton"])
        self.assertEqual(manifest["apex"]["branch"], "master")
        self.assertEqual(
            manifest["pytorch"]["version"], "2.11.0a0+devrocm7.13.0.dev0-abc"
        )

    def test_windows_manifest_excludes_triton_and_apex_by_default(self) -> None:
        _shas, resolves, files = self._stable_windows_github_data()

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
        self.assertNotIn("triton", manifest)

    def test_windows_manifest_triton_opt_in_is_not_enabled(self) -> None:
        _shas, resolves, files = self._stable_windows_github_data()

        with self._patch_github_api(resolves=resolves, files=files):
            with self.assertRaisesRegex(
                NotImplementedError,
                "Windows Triton manifest generation is not enabled",
            ):
                m.generate_manifest(
                    pytorch_git_ref="release/2.10",
                    rocm_version="7.13.0a20260501",
                    version_suffix="+rocm7.13.0a20260501",
                    platform="windows",
                    projects=[
                        "pytorch",
                        "pytorch_audio",
                        "pytorch_vision",
                        "triton",
                    ],
                    therock_commit="a" * 40,
                    therock_repo="https://github.com/ROCm/TheRock",
                    therock_branch="main",
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
            ) as generate, mock.patch.object(
                m, "derive_version_suffix", return_value="+rocm7.13.0"
            ):
                m.main(
                    [
                        "--rocm-version",
                        "7.13.0",
                        "--platform",
                        "linux",
                        "--output",
                        str(out_path),
                        "--pytorch-git-refs",
                        "release/2.10",
                        "--projects",
                        "pytorch;triton",
                    ]
                )

            self.assertEqual(
                json.loads(out_path.read_text(encoding="utf-8")), generated
            )
            generate.assert_called_once()
            self.assertEqual(
                generate.call_args.kwargs["projects"], ["pytorch", "triton"]
            )
            self.assertEqual(generate.call_args.kwargs["version_suffix"], "+rocm7.13.0")

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
