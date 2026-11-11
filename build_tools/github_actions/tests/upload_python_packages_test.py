#!/usr/bin/env python
"""Unit tests for upload_python_packages.py.

Tests verify that upload functions pass correct StorageLocations to the
StorageBackend, producing the expected file layout. Uses LocalStorageBackend
with a temp directory so no mocking of subprocess or boto3 is needed.
"""

import argparse
import hashlib
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
import mirror_python_dependencies as mirror


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
        packages_loc = output_root.python_packages()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            dist_dir = Path(tmp)
            staging_dir = Path(staging)
            (dist_dir / "rocm-1.0.whl").write_bytes(b"whl")
            (dist_dir / "rocm-1.0.tar.gz").write_bytes(b"sdist")
            (dist_dir / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            upload_python_packages.upload_packages(dist_dir, packages_loc, backend)

            base = staging_dir / "12345-linux" / "python"
            self.assertTrue((base / "rocm-1.0.whl").is_file())
            self.assertTrue((base / "rocm-1.0.tar.gz").is_file())
            self.assertTrue((base / "index.html").is_file())

    def test_no_files_raises(self):
        output_root = _make_output_root()
        packages_loc = output_root.python_packages()
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
        packages_loc = output_root.python_packages()
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
                    / "rocm-1.0.whl"
                ).is_file()
            )


class TestDependencySnapshot(unittest.TestCase):
    def _snapshot(self, root: Path) -> tuple[Path, str, bytes]:
        snapshot_dir = root / "snapshot"
        filename = "pandas-2.3.3-py3-none-any.whl"
        content = b"mirrored dependency wheel"
        wheel = mirror.SnapshotWheel(
            package="pandas",
            project="rocm",
            requested_version="2.3.3",
            selected_version="2.3.3",
            filename=filename,
            source_url=f"https://files.pythonhosted.org/{filename}",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            relative_path=f"wheels/pandas/{filename}",
            destination_key=mirror.dependency_destination_key("pandas", filename),
        )
        wheel_path = snapshot_dir / wheel.relative_path
        wheel_path.parent.mkdir(parents=True)
        wheel_path.write_bytes(content)
        snapshot = mirror.DependencySnapshot(
            schema_version=1,
            created_at="2026-11-12T00:00:00+00:00",
            source_revision="test",
            wheels=(wheel,),
        )
        mirror.write_snapshot_manifest(snapshot, snapshot_dir / "manifest.json")
        return snapshot_dir, filename, content

    def test_verified_dependency_is_uploaded_with_rocm_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_dir, filename, content = self._snapshot(root)
            dist_dir = root / "dist"
            dist_dir.mkdir()
            (dist_dir / "rocm-1.0.whl").write_bytes(b"rocm")

            staging = root / "published"
            args = argparse.Namespace(
                input_packages_dir=root,
                run_id="12345",
                output_dir=staging,
                bucket=None,
                dry_run=False,
                dependency_snapshot=snapshot_dir,
            )
            with unittest.mock.patch.object(
                upload_python_packages,
                "_make_output_root",
                return_value=_make_output_root(),
            ):
                upload_python_packages.run(args)
                upload_python_packages.run(args)
            self.assertFalse(any(path.is_dir() for path in dist_dir.iterdir()))
            self.assertEqual(
                (staging / "12345-linux/python" / filename).read_bytes(), content
            )
            self.assertFalse((staging / "12345-linux/python/manifest.json").exists())

    def test_corrupt_snapshot_is_rejected_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_dir, filename, _ = self._snapshot(root)
            (snapshot_dir / "wheels/pandas" / filename).write_bytes(b"corrupt")
            dist_dir = root / "dist"
            dist_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "SHA256|size"):
                upload_python_packages.stage_dependency_snapshot(snapshot_dir, dist_dir)
            self.assertEqual(list(dist_dir.iterdir()), [])

    def test_conflicting_package_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_dir, filename, _ = self._snapshot(root)
            dist_dir = root / "dist"
            dist_dir.mkdir()
            original = b"existing package"
            (dist_dir / filename).write_bytes(original)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                upload_python_packages.stage_dependency_snapshot(snapshot_dir, dist_dir)
            self.assertEqual((dist_dir / filename).read_bytes(), original)

    def test_dry_run_does_not_stage_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_dir, _, _ = self._snapshot(root)
            dist_dir = root / "dist"
            dist_dir.mkdir()
            upload_python_packages.stage_dependency_snapshot(
                snapshot_dir, dist_dir, dry_run=True
            )
            self.assertEqual(list(dist_dir.iterdir()), [])


class TestWriteGhaUploadSummary(unittest.TestCase):
    """Tests for write_gha_upload_summary()."""

    def test_summary_has_flat_index_and_device_install(self):
        packages_loc = _make_output_root().python_packages()
        with tempfile.TemporaryDirectory() as tmp:
            summary_file = Path(tmp) / "summary.md"
            with unittest.mock.patch.dict(
                os.environ,
                {"GITHUB_STEP_SUMMARY": os.fspath(summary_file), "RUNNER_TEMP": tmp},
            ):
                upload_python_packages.write_gha_upload_summary(packages_loc)

            summary = summary_file.read_text(encoding="utf-8")
            self.assertIn(
                "--find-links=https://therock-ci-artifacts.s3.amazonaws.com/"
                "12345-linux/python/index.html",
                summary,
            )
            self.assertIn("rocm[libraries,devel,device-<YOUR_TARGET>]", summary)


if __name__ == "__main__":
    unittest.main()
