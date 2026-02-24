#!/usr/bin/env python
"""Unit tests for upload_python_packages.py.

Tests verify that upload functions pass correct OutputLocations to the
UploadBackend, producing the expected file layout. Uses LocalUploadBackend
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
# Add github_actions to path so upload_python_packages is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.run_outputs import RunOutputRoot
from _therock_utils.upload_backend import LocalUploadBackend
import upload_python_packages


def _make_run_root(
    run_id="12345",
    platform="linux",
    bucket="therock-ci-artifacts",
    external_repo="",
):
    return RunOutputRoot(
        bucket=bucket,
        external_repo=external_repo,
        run_id=run_id,
        platform=platform,
    )


class TestMakeRunRoot(unittest.TestCase):
    """Tests for _make_run_root()."""

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("upload_python_packages.RunOutputRoot.from_workflow_run")
    def test_default_uses_from_workflow_run(self, mock_factory):
        mock_factory.return_value = _make_run_root()
        result = upload_python_packages._make_run_root("12345")
        mock_factory.assert_called_once_with(
            run_id="12345", platform=upload_python_packages.PLATFORM
        )
        self.assertEqual(result.bucket, "therock-ci-artifacts")

    def test_bucket_override_skips_factory(self):
        result = upload_python_packages._make_run_root(
            "12345", bucket_override="custom-bucket"
        )
        self.assertEqual(result.bucket, "custom-bucket")
        self.assertEqual(result.external_repo, "")
        self.assertEqual(result.run_id, "12345")


class TestFindPackageFiles(unittest.TestCase):
    """Tests for find_package_files()."""

    def test_finds_wheels_sdists_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")
            (dist_dir / "rocm-1.0.tar.gz").write_bytes(b"sdist")
            (dist_dir / "index.html").write_text("<html></html>")
            (dist_dir / "unrelated.txt").write_text("ignore")

            files = upload_python_packages.find_package_files(dist_dir)
            names = sorted(f.name for f in files)
            self.assertEqual(names, ["index.html", "rocm-1.0.tar.gz", "rocm-1.0.whl"])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = upload_python_packages.find_package_files(Path(tmp))
            self.assertEqual(files, [])


class TestUploadPackages(unittest.TestCase):
    """Tests for upload_packages()."""

    def test_uploads_package_files(self):
        run_root = _make_run_root()
        packages_loc = run_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")
            (dist_dir / "rocm-1.0.tar.gz").write_bytes(b"sdist")
            (dist_dir / "index.html").write_text("<html></html>")

            backend = LocalUploadBackend(staging_dir)
            upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

            base = staging_dir / "12345-linux" / "python" / "gfx94X-dcgpu"
            self.assertTrue((base / "rocm-1.0.whl").is_file())
            self.assertTrue((base / "rocm-1.0.tar.gz").is_file())
            self.assertTrue((base / "index.html").is_file())

    def test_no_files_raises(self):
        run_root = _make_run_root()
        packages_loc = run_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalUploadBackend(staging_dir)
            with self.assertRaises(FileNotFoundError):
                upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

    def test_external_repo_prefix(self):
        run_root = _make_run_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        packages_loc = run_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")

            backend = LocalUploadBackend(staging_dir)
            upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

            self.assertTrue(
                (
                    staging_dir
                    / "Fork-TheRock"
                    / "12345-linux"
                    / "python"
                    / "gfx94X-dcgpu"
                    / "rocm-1.0.whl"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
