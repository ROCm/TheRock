"""
Unit tests for build_tools/github_actions/generate_pytorch_manifest.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR))

import generate_pytorch_manifest as m


class GeneratePyTorchSourcesManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._gha_keys = [
            "GITHUB_SERVER_URL",
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "GITHUB_REF",
        ]

        self._saved_env: dict[str, str] = {}
        for key in self._gha_keys:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]

        for key in self._gha_keys:
            os.environ.pop(key, None)

        os.environ["GITHUB_SERVER_URL"] = "https://github.com"
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["GITHUB_SHA"] = "b3eda956a19d0151cbb4699739eb71f62596c8bb"
        os.environ["GITHUB_REF"] = "refs/heads/main"

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        for key in self._gha_keys:
            os.environ.pop(key, None)
        for key, value in self._saved_env.items():
            os.environ[key] = value

    def _run_main_with_args(self, argv: list[str]) -> None:
        m.main(argv)

    def test_normalize_release_track(self) -> None:
        self.assertEqual(m.normalize_release_track("nightly"), "nightly")
        self.assertEqual(m.normalize_release_track("release/2.7"), "release-2.7")
        self.assertEqual(
            m.normalize_release_track("users/alice/feature"), "users-alice-feature"
        )

    def test_normalize_py(self) -> None:
        self.assertEqual(m.normalize_py("3.11"), "3.11")
        self.assertEqual(m.normalize_py("py3.11"), "3.11")
        self.assertEqual(m.normalize_py(" py3.12 "), "3.12")

    def test_manifest_filename(self) -> None:
        name = m.manifest_filename(python_version="3.11", pytorch_git_ref="release/2.7")
        self.assertEqual(name, "therock-manifest_torch_py3.11_release-2.7.json")

        name = m.manifest_filename(python_version="py3.12", pytorch_git_ref="nightly")
        self.assertEqual(name, "therock-manifest_torch_py3.12_nightly.json")

    def test_sources_only_manifest(self) -> None:
        manifest_dir = self.tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        pytorch_repo = self.tmp_path / "src_pytorch"
        audio_repo = self.tmp_path / "src_audio"
        vision_repo = self.tmp_path / "src_vision"
        triton_repo = self.tmp_path / "src_triton"

        pytorch_head = "1111111111111111111111111111111111111111"
        audio_head = "2222222222222222222222222222222222222222"
        vision_head = "3333333333333333333333333333333333333333"
        triton_head = "4444444444444444444444444444444444444444"

        def fake_git_head(dirpath: Path, *, label: str) -> m.GitSourceInfo:
            p = dirpath.resolve()
            if p == pytorch_repo.resolve():
                return m.GitSourceInfo(
                    commit=pytorch_head, repo="https://github.com/ROCm/pytorch.git"
                )
            if p == audio_repo.resolve():
                return m.GitSourceInfo(
                    commit=audio_head, repo="https://github.com/pytorch/audio.git"
                )
            if p == vision_repo.resolve():
                return m.GitSourceInfo(
                    commit=vision_head, repo="https://github.com/pytorch/vision.git"
                )
            if p == triton_repo.resolve():
                return m.GitSourceInfo(
                    commit=triton_head, repo="https://github.com/ROCm/triton.git"
                )
            raise AssertionError(f"Unexpected repo path: {p}")

        with mock.patch.object(
            m, "git_head", side_effect=fake_git_head
        ), mock.patch.object(m, "git_branch_best_effort", return_value=None):
            self._run_main_with_args(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--python-version",
                    "3.11",
                    "--pytorch-git-ref",
                    "release/2.7",
                    "--pytorch-dir",
                    str(pytorch_repo),
                    "--pytorch-audio-dir",
                    str(audio_repo),
                    "--pytorch-vision-dir",
                    str(vision_repo),
                    "--triton-dir",
                    str(triton_repo),
                ]
            )

        manifest_path = manifest_dir / "therock-manifest_torch_py3.11_release-2.7.json"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(data.keys()),
            {"pytorch", "pytorch_audio", "pytorch_vision", "triton", "therock"},
        )

        self.assertEqual(data["pytorch"]["commit"], pytorch_head)
        self.assertEqual(data["pytorch"]["repo"], "https://github.com/ROCm/pytorch.git")
        self.assertEqual(data["pytorch"]["branch"], "release/2.7")

        self.assertEqual(data["pytorch_audio"]["commit"], audio_head)
        self.assertEqual(
            data["pytorch_audio"]["repo"], "https://github.com/pytorch/audio.git"
        )
        self.assertNotIn("branch", data["pytorch_audio"])

        self.assertEqual(data["pytorch_vision"]["commit"], vision_head)
        self.assertEqual(
            data["pytorch_vision"]["repo"], "https://github.com/pytorch/vision.git"
        )
        self.assertNotIn("branch", data["pytorch_vision"])

        self.assertEqual(data["triton"]["commit"], triton_head)
        self.assertEqual(data["triton"]["repo"], "https://github.com/ROCm/triton.git")
        self.assertNotIn("branch", data["triton"])

        self.assertEqual(data["therock"]["repo"], "https://github.com/ROCm/TheRock.git")
        self.assertEqual(
            data["therock"]["commit"], "b3eda956a19d0151cbb4699739eb71f62596c8bb"
        )
        self.assertEqual(data["therock"]["branch"], "main")

    def test_sources_only_manifest_without_triton(self) -> None:
        manifest_dir = self.tmp_path / "manifests_no_triton"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        pytorch_repo = self.tmp_path / "src_pytorch2"
        audio_repo = self.tmp_path / "src_audio2"
        vision_repo = self.tmp_path / "src_vision2"

        pytorch_head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        audio_head = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        vision_head = "cccccccccccccccccccccccccccccccccccccccc"

        def fake_git_head(dirpath: Path, *, label: str) -> m.GitSourceInfo:
            p = dirpath.resolve()
            if p == pytorch_repo.resolve():
                return m.GitSourceInfo(
                    commit=pytorch_head, repo="https://github.com/ROCm/pytorch.git"
                )
            if p == audio_repo.resolve():
                return m.GitSourceInfo(
                    commit=audio_head, repo="https://github.com/pytorch/audio.git"
                )
            if p == vision_repo.resolve():
                return m.GitSourceInfo(
                    commit=vision_head, repo="https://github.com/pytorch/vision.git"
                )
            raise AssertionError(f"Unexpected repo path: {p}")

        with mock.patch.object(
            m, "git_head", side_effect=fake_git_head
        ), mock.patch.object(m, "git_branch_best_effort", return_value=None):
            self._run_main_with_args(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--python-version",
                    "3.11",
                    "--pytorch-git-ref",
                    "nightly",
                    "--pytorch-dir",
                    str(pytorch_repo),
                    "--pytorch-audio-dir",
                    str(audio_repo),
                    "--pytorch-vision-dir",
                    str(vision_repo),
                ]
            )

        manifest_path = manifest_dir / "therock-manifest_torch_py3.11_nightly.json"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("therock", data)
        self.assertIn("pytorch", data)
        self.assertIn("pytorch_audio", data)
        self.assertIn("pytorch_vision", data)
        self.assertNotIn("triton", data)

        self.assertEqual(data["pytorch"]["branch"], "nightly")
        self.assertNotIn("branch", data["pytorch_audio"])
        self.assertNotIn("branch", data["pytorch_vision"])


if __name__ == "__main__":
    unittest.main()
