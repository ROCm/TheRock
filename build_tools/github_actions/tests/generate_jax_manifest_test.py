"""
Unit tests for build_tools/github_actions/generate_jax_manifest.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, os.fspath(THIS_DIR.parent))

import generate_jax_manifest as m  # noqa: E402


class GenerateJaxManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._gha_keys = [
            "GITHUB_SERVER_URL",
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "GITHUB_REF",
            "GITHUB_RUN_ID",
            "GITHUB_JOB",
        ]

        self._saved_env: dict[str, str] = {}
        for key in self._gha_keys:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]

        for key in self._gha_keys:
            os.environ.pop(key, None)

        os.environ["GITHUB_SERVER_URL"] = "https://github.com"
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["GITHUB_SHA"] = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        os.environ["GITHUB_REF"] = "refs/heads/users/test-branch"
        os.environ["GITHUB_RUN_ID"] = "12345"
        os.environ["GITHUB_JOB"] = "build_jax_wheels"

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

    def test_normalize_py(self) -> None:
        self.assertEqual(m.normalize_py("3.12"), "3.12")
        self.assertEqual(m.normalize_py("py3.12"), "3.12")
        self.assertEqual(m.normalize_py(" py3.13 "), "3.13")

    def test_manifest_filename(self) -> None:
        name = m.manifest_filename(
            python_version="3.12",
            jax_track="release/0.4.28",
        )
        self.assertEqual(name, "therock-manifest_jax_py3.12_release-0.4.28.json")

        name = m.manifest_filename(
            python_version="py3.12",
            jax_track="nightly",
        )
        self.assertEqual(name, "therock-manifest_jax_py3.12_nightly.json")

    def test_sources_only_manifest(self) -> None:
        manifest_dir = self.tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        jax_repo = self.tmp_path / "src_jax"
        jax_head = "1111111111111111111111111111111111111111"

        def fake_git_head(dirpath: Path, *, label: str) -> m.GitSourceInfo:
            p = dirpath.resolve()
            if p == jax_repo.resolve():
                return m.GitSourceInfo(
                    commit=jax_head,
                    repo="https://github.com/ROCm/rocm-jax.git",
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
                    "3.12",
                    "--jax-track",
                    "release/0.0",
                    "--jax-git-ref",
                    "release/0.0",
                    "--jax-dir",
                    str(jax_repo),
                ]
            )

        manifest_path = manifest_dir / "therock-manifest_jax_py3.12_release-0.0.json"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(set(data.keys()), {"jax", "therock"})

        self.assertEqual(data["jax"]["commit"], jax_head)
        self.assertEqual(data["jax"]["repo"], "https://github.com/ROCm/rocm-jax.git")
        self.assertEqual(data["jax"]["branch"], "release/0.0")

        self.assertEqual(data["therock"]["repo"], "https://github.com/ROCm/TheRock.git")
        self.assertEqual(
            data["therock"]["commit"], "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        )
        self.assertEqual(data["therock"]["branch"], "users/test-branch")

    def test_sources_only_manifest_without_jax_git_ref(self) -> None:
        """If branch can't be inferred and no --jax-git-ref is provided, omit branch."""
        manifest_dir = self.tmp_path / "manifests_no_ref"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        jax_repo = self.tmp_path / "src_jax2"
        jax_head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

        def fake_git_head(dirpath: Path, *, label: str) -> m.GitSourceInfo:
            p = dirpath.resolve()
            if p == jax_repo.resolve():
                return m.GitSourceInfo(
                    commit=jax_head,
                    repo="https://github.com/ROCm/rocm-jax.git",
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
                    "3.12",
                    "--jax-track",
                    "nightly",
                    "--jax-dir",
                    str(jax_repo),
                ]
            )

        manifest_path = manifest_dir / "therock-manifest_jax_py3.12_nightly.json"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("jax", data)
        self.assertIn("therock", data)
        self.assertNotIn("branch", data["jax"])


if __name__ == "__main__":
    unittest.main()
