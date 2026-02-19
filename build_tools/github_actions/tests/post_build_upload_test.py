#!/usr/bin/env python
"""Unit tests for post_build_upload.py upload functions.

Tests verify that the migrated upload functions construct correct S3 URIs
and HTTPS URLs from RunOutputRoot, and pass them to subprocess/AWS CLI.
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

from _therock_utils.run_outputs import RunOutputRoot
import post_build_upload


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


class TestUploadArtifacts(unittest.TestCase):
    """Tests for upload_artifacts()."""

    @mock.patch("post_build_upload.run_command")
    def test_s3_uris(self, mock_run_cmd):
        """Verify artifact upload uses correct S3 URIs."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            (build_dir / "artifacts").mkdir()
            (build_dir / "artifacts" / "index.html").write_text("<html></html>")

            post_build_upload.upload_artifacts("gfx94X-dcgpu", build_dir, run_root)

        self.assertEqual(mock_run_cmd.call_count, 2)

        # First call: recursive artifact upload
        recursive_cmd = mock_run_cmd.call_args_list[0][0][0]
        self.assertIn("s3://therock-ci-artifacts/12345-linux", recursive_cmd)

        # Second call: index.html upload
        index_cmd = mock_run_cmd.call_args_list[1][0][0]
        self.assertIn(
            "s3://therock-ci-artifacts/12345-linux/index-gfx94X-dcgpu.html",
            index_cmd,
        )

    @mock.patch("post_build_upload.run_command")
    def test_external_repo_prefix(self, mock_run_cmd):
        """Verify external_repo propagates into S3 URIs."""
        run_root = _make_run_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            (build_dir / "artifacts").mkdir()
            (build_dir / "artifacts" / "index.html").write_text("<html></html>")

            post_build_upload.upload_artifacts("gfx94X-dcgpu", build_dir, run_root)

        recursive_cmd = mock_run_cmd.call_args_list[0][0][0]
        self.assertIn(
            "s3://therock-ci-artifacts-external/Fork-TheRock/12345-linux",
            recursive_cmd,
        )


class TestUploadLogsToS3(unittest.TestCase):
    """Tests for upload_logs_to_s3()."""

    @mock.patch("post_build_upload.run_aws_cp")
    def test_log_dir_uri(self, mock_aws_cp):
        """Verify log upload uses log_dir S3 URI."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build.log").write_text("build output")

            post_build_upload.upload_logs_to_s3("gfx94X-dcgpu", build_dir, run_root)

        # The main log upload call
        mock_aws_cp.assert_called_once_with(
            log_dir,
            "s3://therock-ci-artifacts/12345-linux/logs/gfx94X-dcgpu",
            content_type="text/plain",
        )

    @mock.patch("post_build_upload.run_aws_cp")
    def test_build_observability_uri(self, mock_aws_cp):
        """Verify build_observability upload uses correct S3 URI."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build_observability.html").write_text("<html></html>")

            post_build_upload.upload_logs_to_s3("gfx94X-dcgpu", build_dir, run_root)

        # Find the build_observability upload call
        obs_calls = [
            c for c in mock_aws_cp.call_args_list if "build_observability" in str(c)
        ]
        self.assertEqual(len(obs_calls), 1)
        self.assertEqual(
            obs_calls[0][0][1],
            "s3://therock-ci-artifacts/12345-linux/logs/gfx94X-dcgpu/build_observability.html",
        )

    @mock.patch("post_build_upload.run_aws_cp")
    def test_log_index_uri(self, mock_aws_cp):
        """Verify log index upload uses correct S3 URI."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "index.html").write_text("<html></html>")

            post_build_upload.upload_logs_to_s3("gfx94X-dcgpu", build_dir, run_root)

        index_calls = [c for c in mock_aws_cp.call_args_list if "index.html" in str(c)]
        self.assertEqual(len(index_calls), 1)
        self.assertEqual(
            index_calls[0][0][1],
            "s3://therock-ci-artifacts/12345-linux/logs/gfx94X-dcgpu/index.html",
        )

    @mock.patch("post_build_upload.run_aws_cp")
    def test_resource_profiler_uris(self, mock_aws_cp):
        """Verify resource profiler files use log_file S3 URI."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            prof_dir = log_dir / "therock-build-prof"
            prof_dir.mkdir(parents=True)
            (prof_dir / "comp-summary.html").write_text("<html></html>")
            (prof_dir / "comp-summary.md").write_text("# Summary")

            post_build_upload.upload_logs_to_s3("gfx94X-dcgpu", build_dir, run_root)

        html_calls = [
            c for c in mock_aws_cp.call_args_list if "comp-summary.html" in str(c)
        ]
        self.assertEqual(len(html_calls), 1)
        self.assertEqual(
            html_calls[0][0][1],
            "s3://therock-ci-artifacts/12345-linux/logs/gfx94X-dcgpu/comp-summary.html",
        )

    @mock.patch("post_build_upload.run_aws_cp")
    def test_no_log_dir_skips(self, mock_aws_cp):
        """Verify no uploads happen when log dir doesn't exist."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            # No logs/ directory created
            post_build_upload.upload_logs_to_s3("gfx94X-dcgpu", build_dir, run_root)

        mock_aws_cp.assert_not_called()


class TestUploadManifestToS3(unittest.TestCase):
    """Tests for upload_manifest_to_s3()."""

    @mock.patch("post_build_upload.run_aws_cp")
    def test_manifest_uri(self, mock_aws_cp):
        """Verify manifest upload uses correct S3 URI."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            manifest_dir = build_dir / "base" / "aux-overlay" / "build"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "therock_manifest.json").write_text("{}")

            post_build_upload.upload_manifest_to_s3("gfx94X-dcgpu", build_dir, run_root)

        mock_aws_cp.assert_called_once()
        self.assertEqual(
            mock_aws_cp.call_args[0][1],
            "s3://therock-ci-artifacts/12345-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
        )

    def test_missing_manifest_raises(self):
        """Verify FileNotFoundError when manifest doesn't exist."""
        run_root = _make_run_root()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                post_build_upload.upload_manifest_to_s3(
                    "gfx94X-dcgpu", build_dir, run_root
                )


class TestWriteGhaBuildSummary(unittest.TestCase):
    """Tests for write_gha_build_summary()."""

    @mock.patch("post_build_upload.gha_append_step_summary")
    @mock.patch("post_build_upload.PLATFORM", "linux")
    def test_summary_urls_linux(self, mock_summary):
        """Verify all HTTPS URLs in build summary on Linux."""
        run_root = _make_run_root()
        post_build_upload.write_gha_build_summary("gfx94X-dcgpu", run_root, "success")

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
    @mock.patch("post_build_upload.PLATFORM", "windows")
    def test_summary_urls_windows_no_observability(self, mock_summary):
        """Verify build observability is skipped on Windows."""
        run_root = _make_run_root(platform="windows")
        post_build_upload.write_gha_build_summary("gfx115X-all", run_root, "success")

        calls = [c[0][0] for c in mock_summary.call_args_list]
        self.assertEqual(len(calls), 3)  # logs, artifacts, manifest (no observability)

        # No build_observability link
        for call in calls:
            self.assertNotIn("build_observability", call)

    @mock.patch("post_build_upload.gha_append_step_summary")
    @mock.patch("post_build_upload.PLATFORM", "linux")
    def test_summary_failure_skips_artifacts(self, mock_summary):
        """Verify artifact link is skipped when job failed."""
        run_root = _make_run_root()
        post_build_upload.write_gha_build_summary("gfx94X-dcgpu", run_root, "failure")

        calls = [c[0][0] for c in mock_summary.call_args_list]
        self.assertEqual(len(calls), 3)  # logs, observability, manifest (no artifacts)

        for call in calls:
            self.assertNotIn("index-gfx94X-dcgpu.html", call)

    @mock.patch("post_build_upload.gha_append_step_summary")
    @mock.patch("post_build_upload.PLATFORM", "linux")
    def test_summary_with_external_repo(self, mock_summary):
        """Verify external_repo prefix appears in summary URLs."""
        run_root = _make_run_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        post_build_upload.write_gha_build_summary("gfx94X-dcgpu", run_root, "success")

        calls = [c[0][0] for c in mock_summary.call_args_list]
        for call in calls:
            self.assertIn("therock-ci-artifacts-external", call)
            self.assertIn("Fork-TheRock/12345-linux", call)


if __name__ == "__main__":
    unittest.main()
