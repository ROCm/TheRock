"""
Unit tests for external-builds/pytorch/generate_pytorch_manifest.py.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = THIS_DIR.parent
sys.path.insert(0, os.fspath(PYTORCH_DIR))

import generate_pytorch_manifest as m  # noqa: E402


class GeneratePyTorchManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        # Save environment state (only keys that exist)
        self._gha_keys = [
            "GITHUB_ACTIONS",
            "GITHUB_RUN_ID",
            "GITHUB_JOB",
            "GITHUB_SERVER_URL",
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "GITHUB_REF",
        ]
        self._saved_env: dict[str, str] = {}
        for key in self._gha_keys:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]

        # Clean environment for tests
        for key in self._gha_keys:
            if key in os.environ:
                del os.environ[key]

        # Set test environment
        os.environ["GITHUB_ACTIONS"] = "true"
        os.environ["GITHUB_RUN_ID"] = "123456"
        os.environ["GITHUB_JOB"] = "build_pytorch_wheels"
        os.environ["GITHUB_SERVER_URL"] = "https://github.com"
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["GITHUB_SHA"] = "aabbccdd"
        os.environ["GITHUB_REF"] = "refs/heads/main"

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        # Clean environment after tests
        for key in self._gha_keys:
            if key in os.environ:
                del os.environ[key]

        # Restore saved environment
        for key, value in self._saved_env.items():
            os.environ[key] = value

    def _run_main_with_args(self, argv: list[str]) -> None:
        original_argv = sys.argv[:]
        sys.argv = ["generate_pytorch_manifest.py", *argv]
        try:
            m.main()
        finally:
            sys.argv = original_argv

    def test_parse_wheel_name_no_build_tag(self) -> None:
        meta = m.parse_wheel_name("torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl")
        self.assertEqual(meta.distribution, "torch")
        self.assertEqual(meta.version, "2.7.0")
        self.assertIsNone(meta.build_tag)
        self.assertEqual(meta.python_tag, "cp312")
        self.assertEqual(meta.abi_tag, "cp312")
        self.assertEqual(meta.platform_tag, "manylinux_2_28_x86_64")

    def test_parse_wheel_name_with_build_tag(self) -> None:
        meta = m.parse_wheel_name(
            "torch-2.7.0+rocm7.10.0a20251120-1-cp312-cp312-win_amd64.whl"
        )
        self.assertEqual(meta.distribution, "torch")
        self.assertEqual(meta.version, "2.7.0+rocm7.10.0a20251120")
        self.assertEqual(meta.build_tag, "1")
        self.assertEqual(meta.python_tag, "cp312")
        self.assertEqual(meta.abi_tag, "cp312")
        self.assertEqual(meta.platform_tag, "win_amd64")

    def test_parse_wheel_name_ignores_non_wheel(self) -> None:
        meta = m.parse_wheel_name("torch-2.7.0.tar.gz")
        self.assertIsNone(meta.distribution)
        self.assertIsNone(meta.version)
        self.assertIsNone(meta.build_tag)
        self.assertIsNone(meta.python_tag)
        self.assertIsNone(meta.abi_tag)
        self.assertIsNone(meta.platform_tag)

    def test_parse_wheel_name_too_few_fields(self) -> None:
        meta = m.parse_wheel_name("torch-2.7.0.whl")
        self.assertIsNone(meta.distribution)
        self.assertIsNone(meta.version)
        self.assertIsNone(meta.build_tag)
        self.assertIsNone(meta.python_tag)
        self.assertIsNone(meta.abi_tag)
        self.assertIsNone(meta.platform_tag)

    def test_manifest_generation_end_to_end(self) -> None:
        out_dir = self.tmp_path / "dist"
        out_dir.mkdir(parents=True)

        (out_dir / "torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(
            b"x"
        )

        manifest_dir = out_dir / "manifests"

        argv = [
            "--output-dir",
            str(out_dir),
            "--manifest-dir",
            str(manifest_dir),
            "--artifact-group",
            "pytorch-wheels",
            "--amdgpu-family",
            "gfx110X-all",
            "--rocm-sdk-version",
            "7.10.0a20251120",
            "--pytorch-rocm-arch",
            "gfx94X",
            "--python-version",
            "3.12",
            "--pytorch-git-ref",
            "nightly",
        ]

        self._run_main_with_args(argv)

        manifests = list(manifest_dir.glob("*.json"))
        self.assertEqual(
            len(manifests), 1, f"Expected exactly one manifest, found: {manifests}"
        )

        manifest_path = manifest_dir / "therock-manifest_torch_py3.12_nightly.json"
        self.assertTrue(
            manifest_path.exists(), f"Expected manifest not found: {manifest_path}"
        )

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["project"], "TheRock")
        self.assertEqual(data["component"], "pytorch")
        self.assertEqual(data["artifact_group"], "pytorch-wheels")

        self.assertEqual(data["run_id"], "123456")
        self.assertEqual(data["job_id"], "build_pytorch_wheels")
        self.assertEqual(data["therock"]["repo"], "https://github.com/ROCm/TheRock")
        self.assertEqual(data["therock"]["commit"], "aabbccdd")
        self.assertEqual(data["therock"]["ref"], "refs/heads/main")

        self.assertEqual(len(data["artifacts"]), 1)
        labels = data["artifacts"][0]["labels"]
        self.assertEqual(labels["distribution"], "torch")
        self.assertEqual(labels["version"], "2.7.0")

    def test_manifest_filename_release_28(self) -> None:
        out_dir = self.tmp_path / "dist_release"
        out_dir.mkdir(parents=True)

        (out_dir / "torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(
            b"x"
        )

        manifest_dir = out_dir / "manifests"

        self._run_main_with_args(
            [
                "--output-dir",
                str(out_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--artifact-group",
                "pytorch-wheels",
                "--amdgpu-family",
                "gfx110X-all",
                "--rocm-sdk-version",
                "7.10.0a20251120",
                "--python-version",
                "3.12",
                "--pytorch-git-ref",
                "release/2.8",
            ]
        )

        manifest_path = manifest_dir / "therock-manifest_torch_py3.12_release-2.8.json"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")


if __name__ == "__main__":
    unittest.main()
