#!/usr/bin/env python
"""Unit tests for generate_s3_index.py.

Tests use a local staging directory (no S3 needed) and verify that the correct
index.html files are generated and placed at the expected paths.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add build_tools to path so _therock_utils and generate_s3_index are importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import LocalStorageBackend
import generate_s3_index


def _make_output_root(run_id="12345", platform="linux"):
    return WorkflowOutputRoot(
        bucket="therock-ci-artifacts",
        external_repo="",
        run_id=run_id,
        platform=platform,
    )


class TestDiscoverArtifactGroupsLocal(unittest.TestCase):
    """Tests for _discover_artifact_groups_local()."""

    def test_finds_groups_from_log_subdirs(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            (staging_dir / prefix / "logs" / "gfx94X-dcgpu").mkdir(parents=True)
            (staging_dir / prefix / "logs" / "gfx110X-all").mkdir(parents=True)

            groups = generate_s3_index._discover_artifact_groups_local(staging_dir, prefix)
            self.assertEqual(groups, ["gfx110X-all", "gfx94X-dcgpu"])

    def test_empty_when_no_logs_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            groups = generate_s3_index._discover_artifact_groups_local(
                Path(staging), "12345-linux"
            )
            self.assertEqual(groups, [])

    def test_ignores_files_in_logs_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            logs_dir = staging_dir / prefix / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "some_file.txt").write_text("not a group")
            (logs_dir / "gfx94X-dcgpu").mkdir()

            groups = generate_s3_index._discover_artifact_groups_local(staging_dir, prefix)
            self.assertEqual(groups, ["gfx94X-dcgpu"])


class TestBuildLogEntriesLocal(unittest.TestCase):
    """Tests for _build_log_entries_local()."""

    def test_lists_log_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            log_dir = staging_dir / prefix / "logs" / "gfx94X-dcgpu"
            log_dir.mkdir(parents=True)
            (log_dir / "build.log").write_text("build output")
            (log_dir / "ninja_logs.tar.gz").write_bytes(b"gz")

            entries = generate_s3_index._build_log_entries_local(
                staging_dir, prefix, "gfx94X-dcgpu"
            )
            names = [e.name for e in entries]
            self.assertIn("build.log", names)
            self.assertIn("ninja_logs.tar.gz", names)

    def test_excludes_index_html(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            log_dir = staging_dir / prefix / "logs" / "gfx94X-dcgpu"
            log_dir.mkdir(parents=True)
            (log_dir / "index.html").write_text("<html></html>")
            (log_dir / "build.log").write_text("log")

            entries = generate_s3_index._build_log_entries_local(
                staging_dir, prefix, "gfx94X-dcgpu"
            )
            names = [e.name for e in entries]
            self.assertNotIn("index.html", names)
            self.assertIn("build.log", names)

    def test_returns_empty_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            entries = generate_s3_index._build_log_entries_local(
                Path(staging), "12345-linux", "gfx94X-dcgpu"
            )
            self.assertEqual(entries, [])


class TestBuildArtifactEntriesLocal(unittest.TestCase):
    """Tests for _build_artifact_entries_local()."""

    def test_lists_tar_xz_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            root_dir = staging_dir / prefix
            root_dir.mkdir(parents=True)
            (root_dir / "core_lib_gfx94X.tar.xz").write_bytes(b"data")
            (root_dir / "core_lib_gfx94X.tar.xz.sha256sum").write_text("abc123")
            (root_dir / "unrelated.txt").write_text("ignore")

            entries = generate_s3_index._build_artifact_entries_local(staging_dir, prefix)
            names = [e.name for e in entries]
            self.assertIn("core_lib_gfx94X.tar.xz", names)
            self.assertIn("core_lib_gfx94X.tar.xz.sha256sum", names)
            self.assertNotIn("unrelated.txt", names)

    def test_ignores_subdirectory_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            prefix = "12345-linux"
            subdir = staging_dir / prefix / "logs"
            subdir.mkdir(parents=True)
            (subdir / "something.tar.xz").write_bytes(b"data")

            entries = generate_s3_index._build_artifact_entries_local(staging_dir, prefix)
            self.assertEqual(entries, [])

    def test_returns_empty_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            entries = generate_s3_index._build_artifact_entries_local(
                Path(staging), "12345-linux"
            )
            self.assertEqual(entries, [])


class TestGenerateIndexHtml(unittest.TestCase):
    """Tests for _generate_index_html()."""

    def test_contains_file_entries(self):
        entries = [
            generate_s3_index._FileEntry(
                name="build.log",
                href="build.log",
                size_bytes=1024,
                last_modified=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            )
        ]
        html = generate_s3_index._generate_index_html("test dir", entries, parent_href=None)
        self.assertIn("build.log", html)
        self.assertIn("1 KB", html)

    def test_parent_link_included_when_provided(self):
        html = generate_s3_index._generate_index_html("logs/gfx94X", [], parent_href="https://example.com/index.html")
        self.assertIn("https://example.com/index.html", html)
        self.assertIn("..", html)

    def test_no_parent_link_when_none(self):
        html = generate_s3_index._generate_index_html("artifacts", [], parent_href=None)
        self.assertNotIn("..", html)

    def test_escapes_special_chars(self):
        entries = [
            generate_s3_index._FileEntry(
                name="file&name.tar.xz",
                href="file&name.tar.xz",
                size_bytes=0,
                last_modified=None,
            )
        ]
        html = generate_s3_index._generate_index_html("test", entries, parent_href=None)
        self.assertIn("file&amp;name.tar.xz", html)
        self.assertNotIn("file&name", html)


class TestGenerateIndexesForGroup(unittest.TestCase):
    """Integration tests for generate_indexes_for_group() using LocalStorageBackend."""

    def test_generates_log_and_artifact_indexes(self):
        output_root = _make_output_root()
        with (
            tempfile.TemporaryDirectory() as staging,
            tempfile.TemporaryDirectory() as source,
        ):
            staging_dir = Path(staging)
            source_dir = Path(source)

            # Set up source staging tree (simulates what upload jobs produced)
            prefix = output_root.prefix
            log_dir = source_dir / prefix / "logs" / "gfx94X-dcgpu"
            log_dir.mkdir(parents=True)
            (log_dir / "build.log").write_text("build output")
            (log_dir / "ninja_logs.tar.gz").write_bytes(b"gz")

            art_dir = source_dir / prefix
            (art_dir / "core_lib_gfx94X.tar.xz").write_bytes(b"data")

            backend = LocalStorageBackend(staging_dir)
            generate_s3_index.generate_indexes_for_group(
                artifact_group="gfx94X-dcgpu",
                output_root=output_root,
                backend=backend,
                s3_client=None,
                staging_dir=source_dir,
                dry_run=False,
            )

            # Log index should be at logs/gfx94X-dcgpu/index.html
            log_index = (
                staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "index.html"
            )
            self.assertTrue(log_index.is_file(), f"Expected {log_index}")
            log_html = log_index.read_text()
            self.assertIn("build.log", log_html)
            self.assertIn("ninja_logs.tar.gz", log_html)

            # Artifact index should be at index-gfx94X-dcgpu.html
            artifact_index = staging_dir / "12345-linux" / "index-gfx94X-dcgpu.html"
            self.assertTrue(artifact_index.is_file(), f"Expected {artifact_index}")
            artifact_html = artifact_index.read_text()
            self.assertIn("core_lib_gfx94X.tar.xz", artifact_html)

    def test_log_index_links_back_to_artifact_index(self):
        """Verify that the log index parent link points to the artifact index."""
        output_root = _make_output_root()
        with (
            tempfile.TemporaryDirectory() as staging,
            tempfile.TemporaryDirectory() as source,
        ):
            staging_dir = Path(staging)
            source_dir = Path(source)
            prefix = output_root.prefix
            log_dir = source_dir / prefix / "logs" / "gfx94X-dcgpu"
            log_dir.mkdir(parents=True)

            backend = LocalStorageBackend(staging_dir)
            generate_s3_index.generate_indexes_for_group(
                artifact_group="gfx94X-dcgpu",
                output_root=output_root,
                backend=backend,
                s3_client=None,
                staging_dir=source_dir,
                dry_run=False,
            )

            log_index = (
                staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "index.html"
            )
            log_html = log_index.read_text()
            # Should link back to the artifact index
            self.assertIn("index-gfx94X-dcgpu.html", log_html)


if __name__ == "__main__":
    unittest.main()
