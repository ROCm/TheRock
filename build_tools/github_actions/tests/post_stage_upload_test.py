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

            result = post_stage_upload.create_ninja_log_archive(build_dir)

            self.assertIsNotNone(result)
            self.assertTrue(result.exists())
            self.assertEqual(result.name, "ninja_logs.tar.gz")

            with tarfile.open(result, "r:gz") as tar:
                names = tar.getnames()
            self.assertEqual(len(names), 2)

    def test_no_ninja_logs_returns_none(self):
        """Verify None is returned when no .ninja_log files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            result = post_stage_upload.create_ninja_log_archive(build_dir)
            self.assertIsNone(result)

    def test_creates_log_dir_if_missing(self):
        """Verify logs/ directory is created even when no ninja logs exist."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            post_stage_upload.create_ninja_log_archive(build_dir)
            self.assertTrue((build_dir / "logs").is_dir())


class TestUploadStageLogs(unittest.TestCase):
    """Tests for upload_stage_logs()."""

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

            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                stage_name="math-libs",
                amdgpu_family="gfx1151",
                output_dir=staging_dir,
            )

            base = staging_dir / "12345-linux" / "logs" / "math-libs" / "gfx1151"
            self.assertTrue((base / "rocBLAS_build.log").is_file())
            self.assertTrue((base / "rocBLAS_configure.log").is_file())
            self.assertTrue((base / "ninja_logs.tar.gz").is_file())

    def test_generic_stage_upload_path(self):
        """Verify generic stages upload to logs/{stage}/ (no family subdir)."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "amd-llvm_build.log").write_text("llvm build")

            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                stage_name="compiler-runtime",
                amdgpu_family="",
                output_dir=staging_dir,
            )

            base = staging_dir / "12345-linux" / "logs" / "compiler-runtime"
            self.assertTrue((base / "amd-llvm_build.log").is_file())
            # Ensure no extra nesting.
            self.assertFalse((base / "generic").exists())

    def test_no_log_dir_skips(self):
        """Verify no error when logs/ doesn't exist."""
        output_root = _make_output_root()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            # Should not raise.
            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                stage_name="foundation",
                amdgpu_family="",
                output_dir=staging_dir,
            )

    def test_external_repo_prefix(self):
        """Verify external_repo propagates into upload paths."""
        output_root = _make_output_root(
            external_repo="Fork-TheRock/",
            bucket="therock-ci-artifacts-external",
        )
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build.log").write_text("output")

            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                stage_name="foundation",
                amdgpu_family="",
                output_dir=staging_dir,
            )

            self.assertTrue(
                (
                    staging_dir
                    / "Fork-TheRock"
                    / "12345-linux"
                    / "logs"
                    / "foundation"
                    / "build.log"
                ).is_file()
            )

    def test_windows_platform(self):
        """Verify platform appears in the path prefix."""
        output_root = _make_output_root(platform="windows")
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as staging:
            build_dir = Path(tmp)
            staging_dir = Path(staging)
            log_dir = build_dir / "logs"
            log_dir.mkdir()
            (log_dir / "build.log").write_text("output")

            post_stage_upload.upload_stage_logs(
                build_dir=build_dir,
                output_root=output_root,
                stage_name="math-libs",
                amdgpu_family="gfx1151",
                output_dir=staging_dir,
            )

            base = staging_dir / "12345-windows" / "logs" / "math-libs" / "gfx1151"
            self.assertTrue((base / "build.log").is_file())


class TestStageLogDir(unittest.TestCase):
    """Tests for WorkflowOutputRoot.stage_log_dir()."""

    def test_per_arch(self):
        root = _make_output_root()
        loc = root.stage_log_dir("math-libs", "gfx1151")
        self.assertEqual(loc.relative_path, "12345-linux/logs/math-libs/gfx1151")

    def test_generic(self):
        root = _make_output_root()
        loc = root.stage_log_dir("foundation")
        self.assertEqual(loc.relative_path, "12345-linux/logs/foundation")

    def test_generic_empty_string(self):
        root = _make_output_root()
        loc = root.stage_log_dir("compiler-runtime", "")
        self.assertEqual(loc.relative_path, "12345-linux/logs/compiler-runtime")


class TestMainCli(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_missing_run_id_errors(self):
        """Verify error when --run-id is not provided."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                post_stage_upload.main(
                    ["--build-dir", tmp, "--stage-name", "foundation"]
                )

    def test_missing_build_dir_errors(self):
        """Verify FileNotFoundError for nonexistent build directory."""
        with self.assertRaises(FileNotFoundError):
            post_stage_upload.main(
                [
                    "--build-dir",
                    "/nonexistent/path",
                    "--stage-name",
                    "foundation",
                    "--run-id",
                    "12345",
                ]
            )


if __name__ == "__main__":
    unittest.main()
