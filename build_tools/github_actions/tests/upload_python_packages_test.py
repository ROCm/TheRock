#!/usr/bin/env python
"""Unit tests for upload_python_packages.py.

Tests verify that upload functions pass correct StorageLocations to the
StorageBackend, producing the expected file layout. Uses LocalStorageBackend
with a temp directory so no mocking of subprocess or boto3 is needed.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so upload_python_packages is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import LocalStorageBackend
import upload_python_packages


def _make_output_root(
    run_id="12345",
    platform="linux",
    bucket="therock-ci-artifacts",
    external_repo="",
):
    return WorkflowOutputRoot(
        bucket=bucket,
        external_repo=external_repo,
        run_id=run_id,
        platform=platform,
    )


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
        output_root = _make_output_root()
        packages_loc = output_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")
            (dist_dir / "rocm-1.0.tar.gz").write_bytes(b"sdist")
            (dist_dir / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

            base = staging_dir / "12345-linux" / "python" / "gfx94X-dcgpu"
            self.assertTrue((base / "rocm-1.0.whl").is_file())
            self.assertTrue((base / "rocm-1.0.tar.gz").is_file())
            self.assertTrue((base / "index.html").is_file())

    def test_no_files_raises(self):
        output_root = _make_output_root()
        packages_loc = output_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalStorageBackend(staging_dir)
            with self.assertRaises(FileNotFoundError):
                upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

    def test_external_repo_prefix(self):
        output_root = _make_output_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        packages_loc = output_root.python_packages("gfx94X-dcgpu")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")

            backend = LocalStorageBackend(staging_dir)
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
