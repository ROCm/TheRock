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

import generate_pytorch_source_manifest as m


class GeneratePyTorchSourceManifestTest(unittest.TestCase):
    def _patch_github_api(self, *, resolves: dict, files: dict):
        def fake_resolve(repo: str, ref: str) -> str:
            return resolves[(repo, ref)]

        def fake_fetch(repo: str, path: str, ref: str) -> str:
            return files[(repo, path, ref)]

        return mock.patch.multiple(
            m,
            gha_resolve_git_ref=mock.Mock(side_effect=fake_resolve),
            gha_fetch_text_file_contents=mock.Mock(side_effect=fake_fetch),
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
                "ubuntu|pytorch|torchaudio|release/2.10|"
                f"{shas['audio']}|https://github.com/pytorch/audio",
                "ubuntu|pytorch|torchvision|release/2.10|"
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

    def _assert_related_commits_error(
        self, related_commits: str, projects: list[str], pattern: str
    ) -> None:
        pytorch_sha = "1" * 40
        resolves = {
            ("ROCm/pytorch", "release/2.10"): pytorch_sha,
        }
        files = {
            ("ROCm/pytorch", "related_commits", pytorch_sha): related_commits,
        }

        with self._patch_github_api(resolves=resolves, files=files):
            with self.assertRaisesRegex(ValueError, pattern):
                m.generate_manifest(
                    pytorch_git_ref="release/2.10",
                    rocm_version="7.13.0a20260501",
                    version_suffix="+rocm7.13.0a20260501",
                    platform="linux",
                    projects=projects,
                    therock_commit="a" * 40,
                    therock_repo="https://github.com/ROCm/TheRock",
                    therock_branch="main",
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
                "ubuntu|pytorch|torchaudio|release/2.10|"
                f"{shas['audio']}|https://github.com/pytorch/audio",
                "centos|pytorch|torchaudio|release/2.10|"
                f"{shas['audio']}|https://github.com/pytorch/audio",
                "ubuntu|pytorch|torchvision|release/2.10|"
                f"{shas['vision']}|https://github.com/pytorch/vision",
                "centos|pytorch|torchvision|release/2.10|"
                f"{shas['vision']}|https://github.com/pytorch/vision",
                "ubuntu|pytorch|apex|release/2.10|"
                f"{shas['apex']}|https://github.com/ROCm/apex",
                "centos|pytorch|apex|release/2.10|"
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
                projects=[
                    "pytorch",
                    "pytorch_audio",
                    "pytorch_vision",
                    "triton",
                    "apex",
                ],
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="users/example/branch",
            )

        self.assertEqual(
            manifest,
            {
                "pytorch": m.GitSourceInfo(
                    repo="https://github.com/ROCm/pytorch",
                    commit=shas["pytorch"],
                    branch="release/2.10",
                    version="2.10.0+rocm7.13.0a20260501",
                ),
                "pytorch_audio": m.GitSourceInfo(
                    repo="https://github.com/pytorch/audio",
                    commit=shas["audio"],
                    version="2.10.0+rocm7.13.0a20260501",
                ),
                "pytorch_vision": m.GitSourceInfo(
                    repo="https://github.com/pytorch/vision",
                    commit=shas["vision"],
                    version="0.25.0+rocm7.13.0a20260501",
                ),
                "triton": m.GitSourceInfo(
                    repo="https://github.com/ROCm/triton",
                    commit=shas["triton"],
                    version="3.6.0+rocm7.13.0a20260501",
                ),
                "apex": m.GitSourceInfo(
                    repo="https://github.com/ROCm/apex",
                    commit=shas["apex"],
                    version="1.10.0+rocm7.13.0a20260501",
                ),
                "therock": m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="users/example/branch",
                    version="7.13.0a20260501",
                ),
            },
        )

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

        self.assertEqual(
            manifest,
            {
                "pytorch": m.GitSourceInfo(
                    repo="https://github.com/ROCm/pytorch",
                    commit=pytorch_sha,
                    branch="release/2.10",
                    version="2.10.0+rocm7.13.0a20260501",
                ),
                "therock": m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                    version="7.13.0a20260501",
                ),
            },
        )

    def test_manifest_rejects_unknown_project_before_resolving_refs(self) -> None:
        unknown_project = "pytorch_vison"  # Missing the second "i" in "vision".
        with mock.patch.object(m, "gha_resolve_git_ref") as resolve_ref:
            with self.assertRaisesRegex(ValueError, unknown_project):
                m.generate_manifest(
                    pytorch_git_ref="release/2.10",
                    rocm_version="7.13.0a20260501",
                    version_suffix="+rocm7.13.0a20260501",
                    platform="linux",
                    projects=["pytorch", unknown_project],
                    therock_commit="a" * 40,
                    therock_repo="https://github.com/ROCm/TheRock",
                    therock_branch="main",
                )

        resolve_ref.assert_not_called()

    def test_malformed_related_commits_error(self) -> None:
        self._assert_related_commits_error(
            "centos|src|torchaudio",
            ["pytorch", "pytorch_audio"],
            "Malformed related_commits",
        )

    def test_conflicting_related_commits_error(self) -> None:
        self._assert_related_commits_error(
            "\n".join(
                [
                    "ubuntu|pytorch|torchvision|release/0.27|"
                    f"{'2' * 40}|https://github.com/pytorch/vision",
                    "centos|pytorch|torchvision|release/0.27|"
                    f"{'3' * 40}|https://github.com/pytorch/vision",
                ]
            ),
            ["pytorch", "pytorch_vision"],
            "Conflicting related_commits",
        )

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
                projects=[
                    "pytorch",
                    "pytorch_audio",
                    "pytorch_vision",
                    "triton",
                    "apex",
                ],
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertEqual(
            manifest,
            {
                "pytorch": m.GitSourceInfo(
                    repo="https://github.com/pytorch/pytorch",
                    commit=shas["pytorch"],
                    branch="nightly",
                    version="2.11.0a0+devrocm7.13.0.dev0-abc",
                ),
                "pytorch_audio": m.GitSourceInfo(
                    repo="https://github.com/pytorch/audio",
                    commit=shas["audio"],
                    branch="nightly",
                    version="2.11.0a0+devrocm7.13.0.dev0-abc",
                ),
                "pytorch_vision": m.GitSourceInfo(
                    repo="https://github.com/pytorch/vision",
                    commit=shas["vision"],
                    branch="nightly",
                    version="0.26.0a0+devrocm7.13.0.dev0-abc",
                ),
                "triton": m.GitSourceInfo(
                    repo="https://github.com/ROCm/triton",
                    commit=shas["triton"],
                    version="3.6.0+devrocm7.13.0.dev0-abc",
                ),
                "apex": m.GitSourceInfo(
                    repo="https://github.com/ROCm/apex",
                    commit=shas["apex"],
                    branch="master",
                    version="1.11.0+devrocm7.13.0.dev0-abc",
                ),
                "therock": m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                    version="7.13.0.dev0+abc",
                ),
            },
        )

    def test_windows_release_manifest_excludes_triton_and_apex_by_default(self) -> None:
        _shas, resolves, files = self._stable_windows_github_data()

        with self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="release/2.10",
                rocm_version="7.13.0a20260501",
                version_suffix="+rocm7.13.0a20260501",
                platform="windows",
                projects=["pytorch", "pytorch_audio", "pytorch_vision"],
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertNotIn("apex", manifest)
        self.assertNotIn("triton", manifest)

    @unittest.skip("Enable when Windows Triton nightly builds are on by default")
    def test_windows_nightly_default_projects_include_triton(self) -> None:
        self.assertEqual(
            m.default_projects_for_pytorch_ref("windows", "nightly"),
            ["pytorch", "pytorch_audio", "pytorch_vision", "triton"],
        )

    def test_windows_nightly_manifest_uses_triton_windows_pin(self) -> None:
        shas = {
            "pytorch": "1" * 40,
            "audio": "2" * 40,
            "vision": "3" * 40,
            "triton_windows": "4" * 40,
        }
        resolves = {
            ("pytorch/pytorch", "nightly"): shas["pytorch"],
            ("pytorch/audio", "nightly"): shas["audio"],
            ("pytorch/vision", "nightly"): shas["vision"],
        }
        files = {
            (
                "pytorch/pytorch",
                ".ci/docker/triton_version.txt",
                shas["pytorch"],
            ): "3.6.0\n",
            ("pytorch/pytorch", "version.txt", shas["pytorch"]): "2.13.0a0\n",
            ("pytorch/audio", "version.txt", shas["audio"]): "2.13.0a0\n",
            ("pytorch/vision", "version.txt", shas["vision"]): "0.28.0a0\n",
        }

        with mock.patch.object(
            m, "read_triton_windows_pin", return_value=shas["triton_windows"]
        ), self._patch_github_api(resolves=resolves, files=files):
            manifest = m.generate_manifest(
                pytorch_git_ref="nightly",
                rocm_version="7.13.0.dev0+abc",
                version_suffix="+devrocm7.13.0.dev0-abc",
                platform="windows",
                projects=["pytorch", "pytorch_audio", "pytorch_vision", "triton"],
                therock_commit="a" * 40,
                therock_repo="https://github.com/ROCm/TheRock",
                therock_branch="main",
            )

        self.assertEqual(
            manifest["triton"],
            m.GitSourceInfo(
                repo="https://github.com/triton-lang/triton-windows",
                commit=shas["triton_windows"],
                version="3.6.0+devrocm7.13.0.dev0-abc",
            ),
        )
        self.assertNotIn("apex", manifest)

    def test_windows_release_manifest_triton_opt_in_errors(self) -> None:
        pytorch_sha = "1" * 40
        resolves = {
            ("ROCm/pytorch", "release/2.10"): pytorch_sha,
        }
        files = {}

        with self._patch_github_api(resolves=resolves, files=files):
            with self.assertRaisesRegex(
                ValueError,
                "Windows Triton.*PyTorch nightly",
            ):
                m.generate_manifest(
                    pytorch_git_ref="release/2.10",
                    rocm_version="7.13.0a20260501",
                    version_suffix="+rocm7.13.0a20260501",
                    platform="windows",
                    projects=[
                        "pytorch",
                        "triton",
                    ],
                    therock_commit="a" * 40,
                    therock_repo="https://github.com/ROCm/TheRock",
                    therock_branch="main",
                )

    def test_main_writes_single_output_manifest_with_project_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "manifest.json"
            pytorch_sha = "1" * 40
            triton_sha = "2" * 40
            pytorch_repo = "ROCm/pytorch"
            pytorch_ref = "release/2.10"

            # Mock GitHub ref resolution for the requested PyTorch ref.
            resolves = {
                (pytorch_repo, pytorch_ref): pytorch_sha,
            }

            # Mock files fetched from the resolved PyTorch commit. Triton gets
            # both its package version and source commit from PyTorch pin files.
            files = {
                (
                    pytorch_repo,
                    ".ci/docker/triton_version.txt",
                    pytorch_sha,
                ): "3.6.0\n",
                (
                    pytorch_repo,
                    ".ci/docker/ci_commit_pins/triton.txt",
                    pytorch_sha,
                ): triton_sha,
                (pytorch_repo, "version.txt", pytorch_sha): "2.10.0\n",
            }

            # Expected output from the CLI after resolving refs, deriving the
            # version suffix, filtering projects, and writing the manifest.
            expected = {
                "schema_version": 1,
                "pytorch": {
                    "repo": "https://github.com/ROCm/pytorch",
                    "commit": pytorch_sha,
                    "branch": pytorch_ref,
                    "version": "2.10.0+rocm7.13.0",
                },
                "triton": {
                    "repo": "https://github.com/ROCm/triton",
                    "commit": triton_sha,
                    "version": "3.6.0+rocm7.13.0",
                },
                "therock": {
                    "repo": "https://github.com/ROCm/TheRock",
                    "commit": "a" * 40,
                    "branch": "main",
                    "version": "7.13.0",
                },
            }
            with mock.patch.object(
                m,
                "detect_therock_source_info",
                return_value=m.GitSourceInfo(
                    repo="https://github.com/ROCm/TheRock",
                    commit="a" * 40,
                    branch="main",
                ),
            ), self._patch_github_api(resolves=resolves, files=files):
                m.main(
                    [
                        "--rocm-version",
                        "7.13.0",
                        "--platform",
                        "linux",
                        "--output",
                        str(out_path),
                        "--pytorch-git-refs",
                        pytorch_ref,
                        "--projects",
                        "pytorch;triton",
                    ]
                )

            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), expected)

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
