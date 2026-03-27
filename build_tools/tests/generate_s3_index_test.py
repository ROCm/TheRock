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

from _therock_utils.storage_backend import LocalStorageBackend
import generate_s3_index


class TestListFilesLocal(unittest.TestCase):
    """Tests for _list_files_local()."""

    def test_lists_immediate_contents(self):
        """Direct files and subdirs are listed; index.html and subdir files are excluded."""
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            root = staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu"
            root.mkdir(parents=True)
            (root / "build.log").write_text("log")
            (root / "ninja_logs.tar.gz").write_bytes(b"gz")
            (root / "index.html").write_text("<html></html>")
            subdir = root / "therock-build-prof"
            subdir.mkdir()
            (subdir / "comp-summary.html").write_text("<html>")

            entries = generate_s3_index._list_files_local(
                staging_dir, "12345-linux/logs/gfx94X-dcgpu"
            )
            names = [e.name for e in entries]
            hrefs = [e.href for e in entries]
            # Direct files are included
            self.assertIn("build.log", names)
            self.assertIn("ninja_logs.tar.gz", names)
            # index.html is excluded
            self.assertNotIn("index.html", names)
            # Subdirectory appears as an entry linking to its index
            self.assertIn("therock-build-prof/", names)
            self.assertIn("therock-build-prof/index.html", hrefs)
            # Subdirectory files are NOT included directly
            self.assertNotIn("therock-build-prof/comp-summary.html", names)
            self.assertNotIn("comp-summary.html", names)

    def test_returns_empty_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            entries = generate_s3_index._list_files_local(
                Path(staging), "12345-linux/logs"
            )
            self.assertEqual(entries, [])

    def test_sorted_order(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            root = staging_dir / "12345-linux" / "logs"
            root.mkdir(parents=True)
            (root / "z.log").write_text("")
            (root / "a.log").write_text("")

            entries = generate_s3_index._list_files_local(
                staging_dir, "12345-linux/logs"
            )
            names = [e.name for e in entries]
            self.assertEqual(names, ["a.log", "z.log"])


class TestDiscoverDirsWithFilesLocal(unittest.TestCase):
    """Tests for _discover_dirs_with_files_local()."""

    def test_finds_leaf_dirs_at_any_depth(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            # Single-arch layout: logs/{group}/
            (staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu").mkdir(parents=True)
            (
                staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "build.log"
            ).write_text("log")
            # Multi-arch layout: logs/{stage}/{family}/
            (staging_dir / "12345-linux" / "logs" / "math-libs" / "gfx1151").mkdir(
                parents=True
            )
            (
                staging_dir
                / "12345-linux"
                / "logs"
                / "math-libs"
                / "gfx1151"
                / "build.log"
            ).write_text("log2")

            dirs = generate_s3_index._discover_dirs_with_files_local(
                staging_dir, "12345-linux"
            )
            self.assertIn("12345-linux/logs/gfx94X-dcgpu", dirs)
            self.assertIn("12345-linux/logs/math-libs/gfx1151", dirs)

    def test_includes_run_root_when_files_present(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            run_dir = staging_dir / "12345-linux"
            run_dir.mkdir(parents=True)
            (run_dir / "core_lib.tar.xz").write_bytes(b"data")

            dirs = generate_s3_index._discover_dirs_with_files_local(
                staging_dir, "12345-linux"
            )
            self.assertIn("12345-linux", dirs)

    def test_empty_when_no_run_dir(self):
        with tempfile.TemporaryDirectory() as staging:
            dirs = generate_s3_index._discover_dirs_with_files_local(
                Path(staging), "12345-linux"
            )
            self.assertEqual(dirs, [])


class TestGenerateIndexHtml(unittest.TestCase):
    """Tests for _generate_index_html()."""

    def test_contains_file_entries(self):
        entries = [
            generate_s3_index._FileEntry(
                name="gfx94X-dcgpu/build.log",
                href="gfx94X-dcgpu/build.log",
                size_bytes=1024,
                last_modified=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            )
        ]
        html = generate_s3_index._generate_index_html("logs", entries, parent_href=None)
        self.assertIn("gfx94X-dcgpu/build.log", html)
        self.assertIn("1 KB", html)

    def test_parent_link_included_when_provided(self):
        html = generate_s3_index._generate_index_html(
            "logs", [], parent_href="https://example.com/index.html"
        )
        self.assertIn("https://example.com/index.html", html)
        self.assertIn("..", html)

    def test_no_parent_link_when_none(self):
        html = generate_s3_index._generate_index_html("logs", [], parent_href=None)
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


class TestGenerateIndexForDirectory(unittest.TestCase):
    """Integration tests for generate_index_for_directory() using LocalStorageBackend."""

    def test_single_arch_flat_listing(self):
        """Single-arch layout: files are flat in logs/{group}/."""
        with (
            tempfile.TemporaryDirectory() as staging,
            tempfile.TemporaryDirectory() as source,
        ):
            staging_dir = Path(staging)
            source_dir = Path(source)

            log_dir = source_dir / "12345-linux" / "logs" / "gfx94X-dcgpu"
            log_dir.mkdir(parents=True)
            (log_dir / "build.log").write_text("build output")
            (log_dir / "ninja_logs.tar.gz").write_bytes(b"gz")

            backend = LocalStorageBackend(staging_dir)
            generate_s3_index.generate_index_for_directory(
                bucket="therock-ci-artifacts",
                dir_prefix="12345-linux/logs/gfx94X-dcgpu",
                backend=backend,
                staging_dir=source_dir,
                dry_run=False,
            )

            index = staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "index.html"
            self.assertTrue(index.is_file(), f"Expected {index}")
            html = index.read_text()
            self.assertIn("build.log", html)
            self.assertIn("ninja_logs.tar.gz", html)

    def test_multi_arch_per_family_listing(self):
        """Multi-arch layout: each family gets its own index."""
        with (
            tempfile.TemporaryDirectory() as staging,
            tempfile.TemporaryDirectory() as source,
        ):
            staging_dir = Path(staging)
            source_dir = Path(source)

            for family in ["gfx94X-dcgpu", "gfx110X-all"]:
                d = source_dir / "12345-linux" / "logs" / "math-libs" / family
                d.mkdir(parents=True)
                (d / "build.log").write_text(f"log for {family}")

            backend = LocalStorageBackend(staging_dir)
            for family in ["gfx94X-dcgpu", "gfx110X-all"]:
                generate_s3_index.generate_index_for_directory(
                    bucket="therock-ci-artifacts",
                    dir_prefix=f"12345-linux/logs/math-libs/{family}",
                    backend=backend,
                    staging_dir=source_dir,
                    dry_run=False,
                )

            for family in ["gfx94X-dcgpu", "gfx110X-all"]:
                index = (
                    staging_dir
                    / "12345-linux"
                    / "logs"
                    / "math-libs"
                    / family
                    / "index.html"
                )
                self.assertTrue(index.is_file(), f"Expected {index}")
                self.assertIn("build.log", index.read_text())

    def test_empty_dir_generates_empty_index(self):
        with (
            tempfile.TemporaryDirectory() as staging,
            tempfile.TemporaryDirectory() as source,
        ):
            staging_dir = Path(staging)
            source_dir = Path(source)
            (source_dir / "12345-linux" / "logs" / "gfx94X-dcgpu").mkdir(parents=True)

            backend = LocalStorageBackend(staging_dir)
            generate_s3_index.generate_index_for_directory(
                bucket="therock-ci-artifacts",
                dir_prefix="12345-linux/logs/gfx94X-dcgpu",
                backend=backend,
                staging_dir=source_dir,
                dry_run=False,
            )

            index = staging_dir / "12345-linux" / "logs" / "gfx94X-dcgpu" / "index.html"
            self.assertTrue(index.is_file())


if __name__ == "__main__":
    unittest.main()
