#!/usr/bin/env python
"""Unit tests for upload_jax_manifest.py.

Tests verify that the manifest upload places files at the correct paths
and that helper functions normalize inputs correctly. Uses LocalStorageBackend
with a temp directory so no mocking of subprocess or boto3 is needed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so upload_jax_manifest is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import upload_jax_manifest
from manifest_utils import (
    normalize_python_version_for_filename,
    normalize_ref_for_filename,
)


class TestNormalizePythonVersion(unittest.TestCase):
    """Tests for normalize_python_version_for_filename()."""

    def test_strips_py_prefix(self):
        self.assertEqual(
            normalize_python_version_for_filename("py3.12"),
            "3.12",
        )

    def test_plain_version(self):
        self.assertEqual(
            normalize_python_version_for_filename("3.12"),
            "3.12",
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            normalize_python_version_for_filename("  py3.13  "),
            "3.13",
        )


class TestSanitizeRefForFilename(unittest.TestCase):
    """Tests for normalize_ref_for_filename()."""

    def test_simple_ref(self):
        self.assertEqual(
            normalize_ref_for_filename("nightly"),
            "nightly",
        )

    def test_slashes_replaced(self):
        self.assertEqual(
            normalize_ref_for_filename("release/0.4.28"),
            "release-0.4.28",
        )

    def test_multiple_slashes(self):
        self.assertEqual(
            normalize_ref_for_filename("users/alice/experiment"),
            "users-alice-experiment",
        )


# Override GITHUB_EVENT_NAME so _is_current_run_pr_from_fork() in
# workflow_outputs.py returns False. Without this, tests that call main()
# fail when run in a pull_request CI job because the real event payload
# has fork=true, which changes the S3 prefix used for output paths.
# TODO: These tests are brittle because they depend on the bucket selection
# logic deep in workflow_outputs.py. Consider mocking at the
# WorkflowOutputRoot level instead.
@mock.patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"})
class TestMain(unittest.TestCase):
    """Tests for main() end-to-end with LocalStorageBackend."""

    def test_uploads_manifest_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)

            manifest_dir = dist_dir / "manifests"
            manifest_dir.mkdir()
            manifest_name = "therock-manifest_jax_py3.12_nightly.json"
            (manifest_dir / manifest_name).write_text("{}", encoding="utf-8")

            upload_jax_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "12345",
                    "--amdgpu-family",
                    "gfx94X-dcgpu",
                    "--python-version",
                    "py3.12",
                    "--jax-git-ref",
                    "nightly",
                    "--output-dir",
                    str(staging_dir),
                ]
            )

            self.assertTrue(
                (
                    staging_dir
                    / f"12345-{upload_jax_manifest.PLATFORM}"
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
            manifest_name = "therock-manifest_jax_py3.12_release-0.4.28.json"
            (manifest_dir / manifest_name).write_text("{}", encoding="utf-8")

            upload_jax_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "99999",
                    "--amdgpu-family",
                    "gfx110X-all",
                    "--python-version",
                    "3.12",
                    "--jax-git-ref",
                    "release/0.4.28",
                    "--output-dir",
                    str(staging_dir),
                ]
            )

            self.assertTrue(
                (
                    staging_dir
                    / f"99999-{upload_jax_manifest.PLATFORM}"
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
                upload_jax_manifest.main(
                    [
                        "--dist-dir",
                        str(dist_dir),
                        "--run-id",
                        "12345",
                        "--amdgpu-family",
                        "gfx94X-dcgpu",
                        "--python-version",
                        "3.12",
                        "--jax-git-ref",
                        "nightly",
                        "--output-dir",
                        str(staging),
                    ]
                )

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)

            manifest_dir = dist_dir / "manifests"
            manifest_dir.mkdir()
            manifest_name = "therock-manifest_jax_py3.12_nightly.json"
            (manifest_dir / manifest_name).write_text("{}", encoding="utf-8")

            upload_jax_manifest.main(
                [
                    "--dist-dir",
                    str(dist_dir),
                    "--run-id",
                    "12345",
                    "--amdgpu-family",
                    "gfx94X-dcgpu",
                    "--python-version",
                    "py3.12",
                    "--jax-git-ref",
                    "nightly",
                    "--output-dir",
                    str(staging_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(list(staging_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
