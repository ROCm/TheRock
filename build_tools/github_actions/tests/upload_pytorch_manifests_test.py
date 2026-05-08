#!/usr/bin/env python
"""Unit tests for upload_pytorch_manifests.py."""

import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so upload_pytorch_manifests is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import upload_pytorch_manifests


# Override GITHUB_EVENT_NAME so _is_current_run_pr_from_fork() in
# workflow_outputs.py returns False.
@mock.patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"})
class TestMain(unittest.TestCase):
    """Tests for main() end-to-end with LocalStorageBackend."""

    def test_uploads_directory_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp)
            staging_dir = Path(staging)

            (manifest_dir / "manifest_a.json").write_text("{}")
            (manifest_dir / "manifest_b.json").write_text("{}")

            upload_pytorch_manifests.main(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--run-id",
                    "12345",
                    "--amdgpu-family",
                    "gfx94X-dcgpu",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                ]
            )

            dest_dir = (
                staging_dir
                / f"12345-{upload_pytorch_manifests.PLATFORM}"
                / "manifests"
                / "pytorch"
                / "gfx94X-dcgpu"
            )
            self.assertTrue((dest_dir / "manifest_a.json").is_file())
            self.assertTrue((dest_dir / "manifest_b.json").is_file())

    def test_uploads_without_family(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp)
            staging_dir = Path(staging)

            (manifest_dir / "manifest.json").write_text("{}")

            upload_pytorch_manifests.main(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--run-id",
                    "99999",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                ]
            )

            dest_dir = (
                staging_dir
                / f"99999-{upload_pytorch_manifests.PLATFORM}"
                / "manifests"
                / "pytorch"
            )
            self.assertTrue((dest_dir / "manifest.json").is_file())

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            with self.assertRaises(FileNotFoundError):
                upload_pytorch_manifests.main(
                    [
                        "--manifest-dir",
                        str(tmp),
                        "--run-id",
                        "12345",
                        "--output-dir",
                        str(staging),
                        "--bucket",
                        "test",
                    ]
                )

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            manifest_dir = Path(tmp)
            staging_dir = Path(staging)

            (manifest_dir / "manifest.json").write_text("{}")

            upload_pytorch_manifests.main(
                [
                    "--manifest-dir",
                    str(manifest_dir),
                    "--run-id",
                    "12345",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                    "--dry-run",
                ]
            )

            self.assertEqual(list(staging_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
