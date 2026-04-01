#!/usr/bin/env python
"""Unit tests for upload_pytorch_manifest.py.

Tests verify that the manifest upload places files at the correct S3 paths
and that helper functions normalize inputs correctly. Uses LocalStorageBackend
with a temp directory so no mocking of subprocess or boto3 is needed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so upload_pytorch_manifest is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import upload_pytorch_manifest


class TestNormalizePythonVersion(unittest.TestCase):
    """Tests for normalize_python_version_for_filename()."""

    def test_strips_py_prefix(self):
        self.assertEqual(
            upload_pytorch_manifest.normalize_python_version_for_filename("py3.11"),
            "3.11",
        )

    def test_plain_version(self):
        self.assertEqual(
            upload_pytorch_manifest.normalize_python_version_for_filename("3.12"),
            "3.12",
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            upload_pytorch_manifest.normalize_python_version_for_filename("  py3.13  "),
            "3.13",
        )


class TestSanitizeRefForFilename(unittest.TestCase):
    """Tests for sanitize_ref_for_filename()."""

    def test_simple_ref(self):
        self.assertEqual(
            upload_pytorch_manifest.sanitize_ref_for_filename("nightly"), "nightly"
        )

    def test_slashes_replaced(self):
        self.assertEqual(
            upload_pytorch_manifest.sanitize_ref_for_filename("release/2.7"),
            "release-2.7",
        )

    def test_multiple_slashes(self):
        self.assertEqual(
            upload_pytorch_manifest.sanitize_ref_for_filename("users/alice/experiment"),
            "users-alice-experiment",
        )


class TestMain(unittest.TestCase):
    """Tests for main() end-to-end with LocalStorageBackend."""

    def test_uploads_manifest_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)

            manifest_dir = dist_dir / "manifests"
            manifest_dir.mkdir()
            manifest_name = "therock-manifest_torch_py3.11_nightly.json"
            (manifest_dir / manifest_name).write_text("{}")

            upload_pytorch_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "12345",
                    "--amdgpu-family",
                    "gfx94X-dcgpu",
                    "--python-version",
                    "py3.11",
                    "--pytorch-git-ref",
                    "nightly",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                ]
            )

            self.assertTrue(
                (
                    staging_dir
                    / f"12345-{upload_pytorch_manifest.PLATFORM}"
                    / "manifests"
                    / "gfx94X-dcgpu"
                    / manifest_name
                ).is_file()
            )

    def test_release_ref_sanitized_in_filename(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)

            manifest_dir = dist_dir / "manifests"
            manifest_dir.mkdir()
            manifest_name = "therock-manifest_torch_py3.12_release-2.7.json"
            (manifest_dir / manifest_name).write_text("{}")

            upload_pytorch_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "99999",
                    "--amdgpu-family",
                    "gfx110X-all",
                    "--python-version",
                    "3.12",
                    "--pytorch-git-ref",
                    "release/2.7",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                ]
            )

            self.assertTrue(
                (
                    staging_dir
                    / f"99999-{upload_pytorch_manifest.PLATFORM}"
                    / "manifests"
                    / "gfx110X-all"
                    / manifest_name
                ).is_file()
            )

    def test_missing_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            (dist_dir / "manifests").mkdir()

            with self.assertRaises(FileNotFoundError):
                upload_pytorch_manifest.main(
                    [
                        "--dist-dir",
                        str(dist_dir),
                        "--run-id",
                        "12345",
                        "--amdgpu-family",
                        "gfx94X-dcgpu",
                        "--python-version",
                        "3.11",
                        "--pytorch-git-ref",
                        "nightly",
                        "--output-dir",
                        str(staging),
                        "--bucket",
                        "test",
                    ]
                )

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)

            manifest_dir = dist_dir / "manifests"
            manifest_dir.mkdir()
            manifest_name = "therock-manifest_torch_py3.11_nightly.json"
            (manifest_dir / manifest_name).write_text("{}")

            upload_pytorch_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "12345",
                    "--amdgpu-family",
                    "gfx94X-dcgpu",
                    "--python-version",
                    "py3.11",
                    "--pytorch-git-ref",
                    "nightly",
                    "--output-dir",
                    str(staging_dir),
                    "--bucket",
                    "test",
                    "--dry-run",
                ]
            )

            # Staging dir should have no run output files
            self.assertEqual(list(staging_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
