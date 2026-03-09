#!/usr/bin/env python
"""Unit tests for post_build_upload.py upload functions.

Tests verify that the upload functions pass correct StorageLocations to the
StorageBackend, producing the expected file layout. Uses LocalStorageBackend
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
# Add github_actions to path so post_build_upload is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import LocalStorageBackend
import post_build_upload


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


class TestUploadArtifacts(unittest.TestCase):
    """Tests for upload_artifacts()."""

    def test_uploads_tar_xz_files(self):
        """Verify only .tar.xz and .tar.xz.sha256sum files are uploaded."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            artifacts = build_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "core_lib_gfx94X.tar.xz").write_bytes(b"data")
            (artifacts / "core_lib_gfx94X.tar.xz.sha256sum").write_text("abc")
            (artifacts / "some_dir").mkdir()
            (artifacts / "some_dir" / "file.txt").write_text("ignore")
            (artifacts / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_artifacts(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            # .tar.xz and .sha256sum should be at the run root
            self.assertTrue(
                (staging_dir / "12345-linux" / "core_lib_gfx94X.tar.xz").is_file()
            )
            self.assertTrue(
                (
                    staging_dir / "12345-linux" / "core_lib_gfx94X.tar.xz.sha256sum"
                ).is_file()
            )
            # index.html goes to the artifact index path
            self.assertTrue(
                (staging_dir / "12345-linux" / "index-gfx94X-dcgpu.html").is_file()
            )
            # Non-matching files should NOT be uploaded
            self.assertFalse(
                (staging_dir / "12345-linux" / "some_dir" / "file.txt").exists()
            )

    def test_external_repo_prefix(self):
        """Verify external_repo propagates into paths."""
        output_root = _make_output_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            artifacts = build_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "lib.tar.xz").write_bytes(b"data")
            (artifacts / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_artifacts(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            self.assertTrue(
                (staging_dir / "Fork-TheRock" / "12345-linux" / "lib.tar.xz").is_file()
            )

    def test_no_artifacts_dir_skips(self):
        """Verify no error when artifacts/ doesn't exist."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalStorageBackend(staging_dir)
            # Should not raise
            post_build_upload.upload_artifacts(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )


class TestUploadLogs(unittest.TestCase):
    """Tests for upload_logs()."""

    def test_uploads_log_files(self):
        """Verify log files end up at the correct paths."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build.log").write_text("build output")
            (log_dir / "ninja_logs.tar.gz").write_bytes(b"gzip")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_logs(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            base = staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu"
            self.assertTrue((base / "build.log").is_file())
            self.assertTrue((base / "ninja_logs.tar.gz").is_file())

    def test_build_observability_uploaded(self):
        """Verify build_observability.html ends up in the log directory."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build_observability.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_logs(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            self.assertTrue(
                (
                    staging_dir
                    / "12345-linux"
                    / "logs"
                    / "gfx94X-dcgpu"
                    / "build_observability.html"
                ).is_file()
            )

    def test_log_index_uploaded(self):
        """Verify log index.html ends up in the log directory."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "index.html").write_text("<html></html>")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_logs(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            self.assertTrue(
                (
                    staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "index.html"
                ).is_file()
            )

    def test_resource_profiler_flattened(self):
        """Verify resource profiler files are uploaded both in subdirectory and flattened."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            prof_dir = log_dir / "therock-build-prof"
            prof_dir.mkdir(parents=True)
            (prof_dir / "comp-summary.html").write_text("<html></html>")
            (prof_dir / "comp-summary.md").write_text("# Summary")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_logs(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            base = staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu"

            # Subdirectory copy (from upload_directory)
            self.assertTrue(
                (base / "therock-build-prof" / "comp-summary.html").is_file()
            )
            self.assertTrue((base / "therock-build-prof" / "comp-summary.md").is_file())

            # Flattened copy (explicit upload_file calls)
            self.assertTrue((base / "comp-summary.html").is_file())
            self.assertTrue((base / "comp-summary.md").is_file())

    def test_no_log_dir_skips(self):
        """Verify no uploads happen when log dir doesn't exist."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalStorageBackend(staging_dir)
            # Should not raise
            post_build_upload.upload_logs(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )


class TestUploadManifest(unittest.TestCase):
    """Tests for upload_manifest()."""

    def test_manifest_uploaded(self):
        """Verify manifest ends up at the correct path."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            manifest_dir = build_dir / "base" / "aux-overlay" / "build"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "therock_manifest.json").write_text("{}")

            backend = LocalStorageBackend(staging_dir)
            post_build_upload.upload_manifest(
                "gfx94X-dcgpu", build_dir, output_root, backend
            )

            self.assertTrue(
                (
                    staging_dir
                    / "12345-linux"
                    / "manifests"
                    / "gfx94X-dcgpu"
                    / "therock_manifest.json"
                ).is_file()
            )

    def test_missing_manifest_raises(self):
        """Verify FileNotFoundError when manifest doesn't exist."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalStorageBackend(staging_dir)
            with self.assertRaises(FileNotFoundError):
                post_build_upload.upload_manifest(
                    "gfx94X-dcgpu", build_dir, output_root, backend
                )


class TestWriteGhaBuildSummary(unittest.TestCase):
    """Tests for write_gha_build_summary()."""

    @mock.patch("post_build_upload.gha_append_step_summary")
    def test_summary_with_observability(self, mock_summary):
        """Verify observability link included when the report exists."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build_observability.html").write_text("<html></html>")

            post_build_upload.write_gha_build_summary(
                "gfx94X-dcgpu", build_dir, output_root, "success"
            )

        calls = [c[0][0] for c in mock_summary.call_args_list]
        self.assertEqual(len(calls), 4)  # logs, observability, artifacts, manifest

        self.assertIn(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/logs/gfx94X-dcgpu/index.html",
            calls[0],
        )
        self.assertIn(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/logs/gfx94X-dcgpu/build_observability.html",
            calls[1],
        )
        self.assertIn(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/index-gfx94X-dcgpu.html",
            calls[2],
        )
        self.assertIn(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
            calls[3],
        )

    @mock.patch("post_build_upload.gha_append_step_summary")
    def test_summary_without_observability(self, mock_summary):
        """Verify observability link omitted when the report was not generated."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)

            post_build_upload.write_gha_build_summary(
                "gfx94X-dcgpu", build_dir, output_root, "success"
            )

        calls = [c[0][0] for c in mock_summary.call_args_list]
        self.assertEqual(len(calls), 3)  # logs, artifacts, manifest

        for call in calls:
            self.assertNotIn("build_observability", call)

    @mock.patch("post_build_upload.gha_append_step_summary")
    def test_summary_failure_skips_artifacts(self, mock_summary):
        """Verify artifact link is skipped when job failed."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build_observability.html").write_text("<html></html>")

            post_build_upload.write_gha_build_summary(
                "gfx94X-dcgpu", build_dir, output_root, "failure"
            )

        calls = [c[0][0] for c in mock_summary.call_args_list]
        self.assertEqual(len(calls), 3)  # logs, observability, manifest (no artifacts)

        for call in calls:
            self.assertNotIn("index-gfx94X-dcgpu.html", call)

    @mock.patch("post_build_upload.gha_append_step_summary")
    def test_summary_with_external_repo(self, mock_summary):
        """Verify external_repo prefix appears in summary URLs."""
        output_root = _make_output_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)

            post_build_upload.write_gha_build_summary(
                "gfx94X-dcgpu", build_dir, output_root, "success"
            )

        calls = [c[0][0] for c in mock_summary.call_args_list]
        for call in calls:
            self.assertIn("therock-ci-artifacts-external", call)
            self.assertIn("Fork-TheRock/12345-linux", call)


if __name__ == "__main__":
    unittest.main()
