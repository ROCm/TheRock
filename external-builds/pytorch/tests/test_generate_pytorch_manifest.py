"""
Unit tests for external-builds/pytorch/generate_pytorch_manifest.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = THIS_DIR.parent
sys.path.insert(0, os.fspath(PYTORCH_DIR))

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
            if key in os.environ:
                del os.environ[key]

        os.environ["GITHUB_SERVER_URL"] = "https://github.com"
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["GITHUB_SHA"] = "b3eda956a19d0151cbb4699739eb71f62596c8bb"
        self.expected_ref = "refs/heads/main"
        os.environ["GITHUB_REF"] = self.expected_ref

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        for key in self._gha_keys:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._saved_env.items():
            os.environ[key] = value

    def _run_main_with_args(self, argv: list[str]) -> None:
        original_argv = sys.argv[:]
        sys.argv = ["generate_pytorch_manifest.py", *argv]
        try:
            m.main()
        finally:
            sys.argv = original_argv

    def _init_git_repo(self, repo_dir: Path, *, remote_url: str) -> str:
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "init", "-q"], cwd=str(repo_dir))
        subprocess.check_call(
            ["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir)
        )
        subprocess.check_call(["git", "config", "user.name", "Test"], cwd=str(repo_dir))
        subprocess.check_call(
            ["git", "remote", "add", "origin", remote_url], cwd=str(repo_dir)
        )

        (repo_dir / "README.txt").write_text("test\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "-A"], cwd=str(repo_dir))
        subprocess.check_call(["git", "commit", "-q", "-m", "init"], cwd=str(repo_dir))

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True
        ).strip()

    def test_sources_only_manifest(self) -> None:
        manifest_dir = self.tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        pytorch_repo = self.tmp_path / "src_pytorch"
        audio_repo = self.tmp_path / "src_audio"
        vision_repo = self.tmp_path / "src_vision"
        triton_repo = self.tmp_path / "src_triton"

        pytorch_head = self._init_git_repo(
            pytorch_repo, remote_url="https://github.com/ROCm/pytorch.git"
        )
        audio_head = self._init_git_repo(
            audio_repo, remote_url="https://github.com/pytorch/audio"
        )
        vision_head = self._init_git_repo(
            vision_repo, remote_url="https://github.com/pytorch/vision"
        )
        triton_head = self._init_git_repo(
            triton_repo, remote_url="https://github.com/ROCm/triton.git"
        )

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

        self.assertEqual(set(data.keys()), {"sources", "therock"})

        sources = data["sources"]
        self.assertEqual(sources["pytorch"]["commit"], pytorch_head)
        self.assertEqual(
            sources["pytorch"]["remote"], "https://github.com/ROCm/pytorch.git"
        )

        self.assertEqual(sources["pytorch_audio"]["commit"], audio_head)
        self.assertEqual(
            sources["pytorch_audio"]["remote"], "https://github.com/pytorch/audio"
        )

        self.assertEqual(sources["pytorch_vision"]["commit"], vision_head)
        self.assertEqual(
            sources["pytorch_vision"]["remote"], "https://github.com/pytorch/vision"
        )

        self.assertEqual(sources["triton"]["commit"], triton_head)
        self.assertEqual(
            sources["triton"]["remote"], "https://github.com/ROCm/triton.git"
        )

        self.assertEqual(data["therock"]["repo"], "https://github.com/ROCm/TheRock")
        self.assertEqual(
            data["therock"]["commit"], "b3eda956a19d0151cbb4699739eb71f62596c8bb"
        )
        self.assertEqual(data["therock"]["ref"], self.expected_ref)

    def test_sources_only_manifest_without_triton(self) -> None:
        manifest_dir = self.tmp_path / "manifests_no_triton"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        pytorch_repo = self.tmp_path / "src_pytorch2"
        audio_repo = self.tmp_path / "src_audio2"
        vision_repo = self.tmp_path / "src_vision2"

        self._init_git_repo(
            pytorch_repo, remote_url="https://github.com/ROCm/pytorch.git"
        )
        self._init_git_repo(audio_repo, remote_url="https://github.com/pytorch/audio")
        self._init_git_repo(vision_repo, remote_url="https://github.com/pytorch/vision")

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
        self.assertNotIn("triton", data["sources"])


if __name__ == "__main__":
    unittest.main()
