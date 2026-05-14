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
import unittest.mock
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


class TestMultiArchUploadPath(unittest.TestCase):
    """Tests that multi-arch uploads go to python/ without an artifact_group subdir."""

    def test_multiarch_upload_path(self):
        output_root = _make_output_root()
        packages_loc = output_root.python_packages("")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")
            (dist_dir / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

            base = staging_dir / "12345-linux" / "python"
            self.assertTrue((base / "rocm-1.0.whl").is_file())
            self.assertTrue((base / "index.html").is_file())
            # Confirm no artifact_group subdirectory was created
            self.assertFalse((base / "multi-arch-release").exists())


class TestGenerateIndex(unittest.TestCase):
    """Tests for generate_index() flat vs per-family auto-detection."""

    def test_flat_layout_skips_client_side_index(self):
        """Flat dist/ (no subdirs) skips client index generation — the
        therock-ci-artifacts bucket serves index.html from its own side,
        and a client-generated index would race with the server view
        when multiple jobs append to the same prefix."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            (dist_dir / "rocm_sdk_device_gfx942-1.0.whl").write_bytes(b"device")

            upload_python_packages.generate_index(dist_dir, multiarch=True)

            self.assertFalse((dist_dir / "index.html").exists())
            self.assertFalse(any(d.is_dir() for d in dist_dir.iterdir()))

    def test_per_family_layout_creates_family_indexes(self):
        """Per-family dist/ (with subdirs) creates per-family indexes, not top-level."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (
                dist_dir / "gfx94X-dcgpu" / "rocm_sdk_libraries_gfx94x_dcgpu-1.0.whl"
            ).write_bytes(b"libs")

            upload_python_packages.generate_index(dist_dir, multiarch=True)

            self.assertTrue((dist_dir / "gfx94X-dcgpu" / "index.html").is_file())
            self.assertFalse((dist_dir / "index.html").is_file())

    def test_dry_run_creates_nothing(self):
        """dry_run=True creates no files regardless of layout."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")

            upload_python_packages.generate_index(
                dist_dir, multiarch=True, dry_run=True
            )

            self.assertFalse((dist_dir / "index.html").is_file())


class TestWriteGhaUploadSummary(unittest.TestCase):
    """Tests for write_gha_upload_summary() three-way families branch."""

    def _make_loc(self):
        return _make_output_root().python_packages("")

    @unittest.mock.patch("upload_python_packages.gha_append_step_summary")
    def test_none_families_single_arch(self, mock_summary):
        """families=None → single-arch summary with /index.html URL."""
        upload_python_packages.write_gha_upload_summary(self._make_loc(), families=None)

        text = mock_summary.call_args[0][0]
        self.assertIn("/index.html", text)
        self.assertNotIn("kpack-split", text)
        self.assertNotIn("per-family", text.lower())

    @unittest.mock.patch("upload_python_packages.gha_append_step_summary")
    def test_empty_families_kpack_split(self, mock_summary):
        """families=[] → kpack-split summary with /index.html URL."""
        upload_python_packages.write_gha_upload_summary(self._make_loc(), families=[])

        text = mock_summary.call_args[0][0]
        self.assertIn("kpack-split", text)
        self.assertIn("/index.html", text)

    @unittest.mock.patch("upload_python_packages.gha_append_step_summary")
    def test_nonempty_families_per_family(self, mock_summary):
        """families=[...] → per-family summary with per-family index URLs."""
        upload_python_packages.write_gha_upload_summary(
            self._make_loc(), families=["gfx94X-dcgpu", "gfx120X-all"]
        )

        text = mock_summary.call_args[0][0]
        self.assertIn("gfx94X-dcgpu/index.html", text)
        self.assertIn("gfx120X-all/index.html", text)
        self.assertNotIn("kpack-split", text)


if __name__ == "__main__":
    unittest.main()
