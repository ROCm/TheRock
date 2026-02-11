"""
Unit tests for external-builds/pytorch/generate_pytorch_manifest.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        os.environ["GITHUB_REF"] = "refs/heads/main"

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
        m.main(argv)

    def _make_fake_git_checkout(self, d: Path) -> None:
        """Create a minimal directory that passes git_head() checks."""
        d.mkdir(parents=True, exist_ok=True)
        (d / ".git").mkdir(parents=True, exist_ok=True)

    def _mock_capture_for_repos(self, mapping: dict[Path, dict[str, str]]):
        """
        Returns a side_effect function for m.capture().

        mapping[path]["head"]   -> returned for: git rev-parse HEAD
        mapping[path]["origin"] -> returned for: git remote get-url origin
        """

        def _side_effect(cmd: list[str], *, cwd: Path) -> str:
            cwd = cwd.resolve()

            if cwd not in mapping:
                raise AssertionError(f"Unexpected cwd passed to capture(): {cwd}")

            if cmd == ["git", "rev-parse", "HEAD"]:
                return mapping[cwd]["head"]

            if cmd == ["git", "remote", "get-url", "origin"]:
                return mapping[cwd]["origin"]

            raise AssertionError(f"Unexpected git command: {cmd}")

        return _side_effect


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

    def test_parse_branch_from_github_ref(self) -> None:
        self.assertEqual(m.parse_branch_from_github_ref("refs/heads/main"), "main")
        self.assertEqual(
            m.parse_branch_from_github_ref("refs/heads/users/foo/bar"), "users/foo/bar"
        )
        with self.assertRaises(RuntimeError):
            m.parse_branch_from_github_ref("refs/tags/v1.0.0")
        with self.assertRaises(RuntimeError):
            m.parse_branch_from_github_ref("")

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

        self._make_fake_git_checkout(pytorch_repo)
        self._make_fake_git_checkout(audio_repo)
        self._make_fake_git_checkout(vision_repo)
        self._make_fake_git_checkout(triton_repo)

        pytorch_head = "1111111111111111111111111111111111111111"
        audio_head = "2222222222222222222222222222222222222222"
        vision_head = "3333333333333333333333333333333333333333"
        triton_head = "4444444444444444444444444444444444444444"

        capture_map = {
            pytorch_repo.resolve(): {
                "head": pytorch_head,
                "origin": "https://github.com/ROCm/pytorch.git",
            },
            audio_repo.resolve(): {
                "head": audio_head,
                "origin": "https://github.com/pytorch/audio.git",
            },
            vision_repo.resolve(): {
                "head": vision_head,
                "origin": "https://github.com/pytorch/vision.git",
            },
            triton_repo.resolve(): {
                "head": triton_head,
                "origin": "https://github.com/ROCm/triton.git",
            },
        }

        with mock.patch.object(
            m, "capture", side_effect=self._mock_capture_for_repos(capture_map)
        ):
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

        self.assertEqual(data["pytorch_audio"]["commit"], audio_head)
        self.assertEqual(
            data["pytorch_audio"]["repo"], "https://github.com/pytorch/audio.git"
        )

        self.assertEqual(data["pytorch_vision"]["commit"], vision_head)
        self.assertEqual(
            data["pytorch_vision"]["repo"], "https://github.com/pytorch/vision.git"
        )

        self.assertEqual(data["triton"]["commit"], triton_head)
        self.assertEqual(data["triton"]["repo"], "https://github.com/ROCm/triton.git")

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

        self._make_fake_git_checkout(pytorch_repo)
        self._make_fake_git_checkout(audio_repo)
        self._make_fake_git_checkout(vision_repo)

        pytorch_head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        audio_head = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        vision_head = "cccccccccccccccccccccccccccccccccccccccc"

        capture_map = {
            pytorch_repo.resolve(): {
                "head": pytorch_head,
                "origin": "https://github.com/ROCm/pytorch.git",
            },
            audio_repo.resolve(): {
                "head": audio_head,
                "origin": "https://github.com/pytorch/audio.git",
            },
            vision_repo.resolve(): {
                "head": vision_head,
                "origin": "https://github.com/pytorch/vision.git",
            },
        }

        with mock.patch.object(
            m, "capture", side_effect=self._mock_capture_for_repos(capture_map)
        ):
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


if __name__ == "__main__":
    unittest.main()
