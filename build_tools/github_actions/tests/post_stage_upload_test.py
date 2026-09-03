#!/usr/bin/env python
"""Unit tests for post_stage_upload.py.

Tests verify ninja log archiving, upload path construction (generic vs per-arch
stages), and CLI argument handling. Uses LocalStorageBackend with a temp
directory so no mocking of subprocess or boto3 is needed.
"""

import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so post_stage_upload is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import LocalStorageBackend
import post_stage_upload


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


class TestCreateNinjaLogArchive(unittest.TestCase):
    """Tests for create_ninja_log_archive()."""

    def test_archives_ninja_logs(self):
        """Verify .ninja_log files are collected into a tar.gz archive."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            # Create ninja log files in nested build subdirectories.
            for subdir in ["subproject_a", "subproject_b"]:
                d = build_dir / subdir
                d.mkdir()
                (d / ".ninja_log").write_text(f"# ninja log for {subdir}\n")

            post_stage_upload.create_ninja_log_archive(build_dir)

            archive = build_dir / "logs" / "ninja_logs.tar.gz"
            self.assertTrue(archive.exists())

            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            self.assertEqual(len(names), 2)

    def test_archive_uses_relative_paths(self):
        """Verify archive members use paths relative to build_dir, not absolute."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            sub = build_dir / "compiler" / "llvm" / "build"
            sub.mkdir(parents=True)
            (sub / ".ninja_log").write_text("# ninja log\n")

            post_stage_upload.create_ninja_log_archive(build_dir)

            archive = build_dir / "logs" / "ninja_logs.tar.gz"
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            self.assertEqual(len(names), 1)
            # Must be relative to build_dir, not absolute.
            self.assertEqual(names[0], "compiler/llvm/build/.ninja_log")
            self.assertFalse(names[0].startswith("/"))

    def test_no_ninja_logs_skips(self):
        """Verify no archive or logs/ directory created when no .ninja_log files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            post_stage_upload.create_ninja_log_archive(build_dir)
            self.assertFalse((build_dir / "logs").exists())


class TestCreateCcacheLogArchive(unittest.TestCase):
    """Tests for create_ccache_log_archive()."""

    def test_archives_ccache_subdirectory(self):
        """Verify ccache log files are archived; originals kept on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            ccache_dir = build_dir / "logs" / "ccache"
            ccache_dir.mkdir(parents=True)
            (ccache_dir / "ccache.log").write_text("x" * 10000)
            (ccache_dir / "ccache_stats.log").write_text("stats")

            post_stage_upload.create_ccache_log_archive(build_dir)

            archive = build_dir / "logs" / "ccache_logs.tar.zst"
            self.assertTrue(archive.exists())

            import pyzstd

            with pyzstd.ZstdFile(archive, "rb") as zst:
                with tarfile.open(mode="r|", fileobj=zst) as tar:
                    names = sorted(m.name for m in tar)
            self.assertEqual(names, ["ccache.log", "ccache_stats.log"])

            # Originals are preserved (idempotent — re-running produces same result).
            self.assertTrue((ccache_dir / "ccache.log").exists())
            self.assertTrue((ccache_dir / "ccache_stats.log").exists())

    def test_no_ccache_dir_skips(self):
        """Verify no archive created when ccache/ subdirectory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "rocBLAS_build.log").write_text("build output")

            post_stage_upload.create_ccache_log_archive(build_dir)

            self.assertFalse((log_dir / "ccache_logs.tar.zst").exists())

    def test_no_log_dir_skips(self):
        """Verify no error when logs/ doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            post_stage_upload.create_ccache_log_archive(build_dir)


class TestUploadStageLogs(unittest.TestCase):
    """Tests for upload_stage_logs()."""

    def test_generic_stage_upload_path(self):
        """Verify generic stages upload to logs/{stage}/ (no family subdir)."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "amd-llvm_build.log").write_text("llvm build")

            backend = LocalStorageBackend(staging_dir)
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                backend=backend,
                stage_name="compiler-runtime",
                amdgpu_family="",
            )

            base = staging_dir / "12345-linux" / "logs" / "compiler-runtime"
            self.assertTrue((base / "amd-llvm_build.log").is_file())
            # Ensure no extra nesting.
            self.assertFalse((base / "generic").exists())

    def test_ccache_subdir_excluded_but_archive_uploaded(self):
        """Verify raw ccache/ logs are excluded but ccache_logs.tar.zst is uploaded."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            ccache_dir = log_dir / "ccache"
            ccache_dir.mkdir(parents=True)
            (log_dir / "build.log").write_text("build output")
            (log_dir / "ccache_logs.tar.zst").write_bytes(b"compressed")
            (ccache_dir / "ccache.log").write_text("verbose trace")

            backend = LocalStorageBackend(staging_dir)
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                backend=backend,
                stage_name="foundation",
                amdgpu_family="",
            )

            base = staging_dir / "12345-linux" / "logs" / "foundation"
            # Regular logs and archive uploaded.
            self.assertTrue((base / "build.log").is_file())
            self.assertTrue((base / "ccache_logs.tar.zst").is_file())
            # Raw ccache logs excluded.
            self.assertFalse((base / "ccache" / "ccache.log").exists())

    def test_per_arch_stage_upload_path(self):
        """Verify per-arch stages upload to logs/{stage}/{family}/."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "rocBLAS_build.log").write_text("build output")
            (log_dir / "rocBLAS_configure.log").write_text("configure output")
            (log_dir / "ninja_logs.tar.gz").write_bytes(b"gzip")

            backend = LocalStorageBackend(staging_dir)
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                backend=backend,
                stage_name="math-libs",
                amdgpu_family="gfx1151",
            )

            base = staging_dir / "12345-linux" / "logs" / "math-libs" / "gfx1151"
            self.assertTrue((base / "rocBLAS_build.log").is_file())
            self.assertTrue((base / "rocBLAS_configure.log").is_file())
            self.assertTrue((base / "ninja_logs.tar.gz").is_file())

    def test_no_log_dir_skips(self):
        """Verify no error when logs/ doesn't exist."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            backend = LocalStorageBackend(staging_dir)
            # Should not raise.
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                backend=backend,
                stage_name="foundation",
                amdgpu_family="",
            )

    def test_external_repo_and_windows_platform(self):
        """Verify external_repo and platform propagate into upload paths."""
        output_root = _make_output_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
            platform="windows",
        )
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build.log").write_text("output")

            backend = LocalStorageBackend(staging_dir)
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                backend=backend,
                stage_name="math-libs",
                amdgpu_family="gfx1151",
            )

            self.assertTrue(
                (
                    staging_dir
                    / "Fork-TheRock"
                    / "12345-windows"
                    / "logs"
                    / "math-libs"
                    / "gfx1151"
                    / "build.log"
                ).is_file()
            )


class TestMainCli(unittest.TestCase):
    """Tests for CLI argument parsing."""

    @mock.patch.dict(os.environ, {}, clear=False)
    def test_missing_run_id_errors(self):
        """Verify error when --run-id is not provided and env var is unset."""
        os.environ.pop("GITHUB_RUN_ID", None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                post_stage_upload.main(["--build-dir", tmp, "--stage", "foundation"])

    def test_missing_build_dir_exits_cleanly(self):
        """Verify clean exit when build directory doesn't exist."""
        # Should not raise — logs a message and returns.
        post_stage_upload.main(
            [
                "--build-dir",
                "/nonexistent/path",
                "--stage",
                "foundation",
                "--run-id",
                "12345",
            ]
        )


if __name__ == "__main__":
    unittest.main()
