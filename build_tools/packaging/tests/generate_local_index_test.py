#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for generate_local_index.py.

Tests verify that index generation creates correct HTML structure with proper
relative paths for local and parent files, and that multi-arch mode correctly
processes directory structures.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add build_tools/packaging/python to path so generate_local_index is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "python"))

from generate_local_index import (
    generate_flat_index,
    generate_multiarch_indexes,
    generate_simple_index,
)


class TestGenerateSimpleIndex(unittest.TestCase):
    """Tests for generate_simple_index()."""

    def test_local_files_only(self):
        """Test index with only local files."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            local_files = [
                Path(tmp) / "package1-1.0.whl",
                Path(tmp) / "package2-2.0.tar.gz",
            ]

            generate_simple_index(output, local_files)

            content = output.read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("<title>Package Index</title>", content)
            self.assertIn('<a href="./package1-1.0.whl">package1-1.0.whl</a>', content)
            self.assertIn(
                '<a href="./package2-2.0.tar.gz">package2-2.0.tar.gz</a>', content
            )

    def test_parent_files_only(self):
        """Test index with only parent files."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            parent_files = [
                Path(tmp) / "parent1-1.0.whl",
                Path(tmp) / "parent2-2.0.tar.gz",
            ]

            generate_simple_index(output, [], parent_files=parent_files)

            content = output.read_text()
            self.assertIn('<a href="../parent1-1.0.whl">parent1-1.0.whl</a>', content)
            self.assertIn(
                '<a href="../parent2-2.0.tar.gz">parent2-2.0.tar.gz</a>', content
            )

    def test_mixed_local_and_parent_files(self):
        """Test index with both local and parent files."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            local_files = [Path(tmp) / "local-1.0.whl"]
            parent_files = [Path(tmp) / "parent-1.0.whl"]

            generate_simple_index(output, local_files, parent_files=parent_files)

            content = output.read_text()
            self.assertIn('<a href="./local-1.0.whl">local-1.0.whl</a>', content)
            self.assertIn('<a href="../parent-1.0.whl">parent-1.0.whl</a>', content)

    def test_custom_title(self):
        """Test index with custom title."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            generate_simple_index(output, [], title="Custom Title")

            content = output.read_text()
            self.assertIn("<title>Custom Title</title>", content)
            self.assertIn("<h1>Custom Title</h1>", content)

    def test_empty_lists(self):
        """Test index with no files."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            generate_simple_index(output, [])

            content = output.read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Package Index", content)
            # Should have no <a> tags
            self.assertNotIn("<a href", content)

    def test_files_are_sorted(self):
        """Test that files appear in sorted order in the index."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            # Provide files in unsorted order
            local_files = [
                Path(tmp) / "zzz-3.0.whl",
                Path(tmp) / "aaa-1.0.whl",
                Path(tmp) / "mmm-2.0.whl",
            ]

            generate_simple_index(output, local_files)

            content = output.read_text()
            # Find positions of each link
            pos_aaa = content.find("aaa-1.0.whl")
            pos_mmm = content.find("mmm-2.0.whl")
            pos_zzz = content.find("zzz-3.0.whl")
            # Verify sorted order
            self.assertLess(pos_aaa, pos_mmm)
            self.assertLess(pos_mmm, pos_zzz)

    def test_valid_html_structure(self):
        """Test that generated HTML has correct structure."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            generate_simple_index(output, [Path(tmp) / "test.whl"])

            content = output.read_text()
            # Check for required HTML elements
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("<html>", content)
            self.assertIn("<head>", content)
            self.assertIn('<meta charset="utf-8">', content)
            self.assertIn("</head>", content)
            self.assertIn("<body>", content)
            self.assertIn("</body>", content)
            self.assertIn("</html>", content)
            # Check it ends with newline
            self.assertTrue(content.endswith("\n"))

    def test_plus_in_filename_is_url_encoded(self):
        """Test that '+' in filenames is percent-encoded in hrefs.

        PEP 440 local versions produce filenames like
        'pkg-1.0+localver-py3-none-any.whl'. S3 interprets a literal '+'
        in URL paths as a space, so hrefs must encode '+' as '%2B'.
        """
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            local_files = [
                Path(tmp) / "rocm_sdk_core-7.0.0+abc123-py3-none-linux_x86_64.whl",
            ]
            parent_files = [
                Path(tmp) / "rocm_sdk_libs-7.0.0+abc123-py3-none-linux_x86_64.whl",
            ]

            generate_simple_index(output, local_files, parent_files=parent_files)

            content = output.read_text()
            # href should have %2B, not literal +
            self.assertIn(
                'href="./rocm_sdk_core-7.0.0%2Babc123-py3-none-linux_x86_64.whl"',
                content,
            )
            self.assertIn(
                'href="../rocm_sdk_libs-7.0.0%2Babc123-py3-none-linux_x86_64.whl"',
                content,
            )
            # Display text should still have literal +
            self.assertIn(
                ">rocm_sdk_core-7.0.0+abc123-py3-none-linux_x86_64.whl</a>",
                content,
            )


class TestGenerateFlatIndex(unittest.TestCase):
    """Tests for generate_flat_index()."""

    def test_generates_top_level_index(self):
        """Top-level wheels and sdists appear in index.html at dist/ root."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            (dist_dir / "rocm_sdk_device_gfx942-1.0.whl").write_bytes(b"device")

            generate_flat_index(dist_dir)

            index = dist_dir / "index.html"
            self.assertTrue(index.is_file())
            content = index.read_text()
            self.assertIn('<a href="./rocm_sdk_core-1.0.whl">', content)
            self.assertIn('<a href="./rocm_sdk_device_gfx942-1.0.whl">', content)

    def test_excludes_subdir_contents(self):
        """Files inside subdirectories are not included in the flat index."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            subdir = dist_dir / "gfx94X-dcgpu"
            subdir.mkdir()
            (subdir / "rocm_sdk_libraries_gfx94x_dcgpu-1.0.whl").write_bytes(b"libs")

            generate_flat_index(dist_dir)

            content = (dist_dir / "index.html").read_text()
            self.assertIn("rocm_sdk_core-1.0.whl", content)
            self.assertNotIn("rocm_sdk_libraries_gfx94x_dcgpu", content)

    def test_custom_patterns(self):
        """Only files matching the specified patterns are included."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "package.whl").write_bytes(b"whl")
            (dist_dir / "package.tar.gz").write_bytes(b"sdist")
            (dist_dir / "README.md").write_bytes(b"readme")

            generate_flat_index(dist_dir, patterns=["*.whl"])

            content = (dist_dir / "index.html").read_text()
            self.assertIn("package.whl", content)
            self.assertNotIn("package.tar.gz", content)
            self.assertNotIn("README.md", content)

    def test_empty_dist(self):
        """Empty dist/ produces a valid index.html with no package links."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)

            generate_flat_index(dist_dir)

            index = dist_dir / "index.html"
            self.assertTrue(index.is_file())
            content = index.read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertNotIn("<a href", content)


class TestGenerateMultiarchIndexes(unittest.TestCase):
    """Tests for generate_multiarch_indexes()."""

    def test_generates_per_family_indexes(self):
        """Test that indexes are generated for each family subdirectory."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            # Create generic packages at top level
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            # Create family subdirectories with packages
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (dist_dir / "gfx94X-dcgpu" / "rocm_libs_gfx94x-1.0.whl").write_bytes(
                b"gfx94x"
            )
            (dist_dir / "gfx120X-all").mkdir()
            (dist_dir / "gfx120X-all" / "rocm_libs_gfx120x-1.0.whl").write_bytes(
                b"gfx120x"
            )

            generate_multiarch_indexes(dist_dir)

            # Verify indexes were created
            self.assertTrue((dist_dir / "gfx94X-dcgpu" / "index.html").is_file())
            self.assertTrue((dist_dir / "gfx120X-all" / "index.html").is_file())

    def test_family_index_includes_local_and_parent(self):
        """Test that family indexes include both local and parent files."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            # Create generic packages
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            (dist_dir / "rocm_sdk_devel-1.0.whl").write_bytes(b"devel")
            # Create family directory
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (dist_dir / "gfx94X-dcgpu" / "rocm_libs_gfx94x-1.0.whl").write_bytes(
                b"gfx94x"
            )

            generate_multiarch_indexes(dist_dir)

            content = (dist_dir / "gfx94X-dcgpu" / "index.html").read_text()
            # Should have local file with ./ prefix
            self.assertIn(
                '<a href="./rocm_libs_gfx94x-1.0.whl">rocm_libs_gfx94x-1.0.whl</a>',
                content,
            )
            # Should have parent files with ../ prefix
            self.assertIn(
                '<a href="../rocm_sdk_core-1.0.whl">rocm_sdk_core-1.0.whl</a>', content
            )
            self.assertIn(
                '<a href="../rocm_sdk_devel-1.0.whl">rocm_sdk_devel-1.0.whl</a>',
                content,
            )

    def test_custom_title_per_family(self):
        """Test that each family gets a custom title."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (dist_dir / "gfx120X-all").mkdir()

            generate_multiarch_indexes(dist_dir)

            content_94x = (dist_dir / "gfx94X-dcgpu" / "index.html").read_text()
            content_120x = (dist_dir / "gfx120X-all" / "index.html").read_text()

            self.assertIn("ROCm Python Packages - gfx94X-dcgpu", content_94x)
            self.assertIn("ROCm Python Packages - gfx120X-all", content_120x)

    def test_no_subdirectories_raises(self):
        """Test that missing family subdirectories raises an error."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")

            with self.assertRaises(FileNotFoundError):
                generate_multiarch_indexes(dist_dir)

    def test_empty_family_directory(self):
        """Test family directory with no packages."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "rocm_sdk_core-1.0.whl").write_bytes(b"core")
            (dist_dir / "gfx94X-dcgpu").mkdir()  # Empty directory

            generate_multiarch_indexes(dist_dir)

            # Index should still be created, just with no local files
            content = (dist_dir / "gfx94X-dcgpu" / "index.html").read_text()
            # Should have parent file
            self.assertIn(
                '<a href="../rocm_sdk_core-1.0.whl">rocm_sdk_core-1.0.whl</a>', content
            )
            # Should not have any ./ links (no local files)
            self.assertNotIn('<a href="./', content)

    def test_custom_patterns(self):
        """Test using custom file patterns."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            # Create files with non-default extensions
            (dist_dir / "README.md").write_bytes(b"readme")
            (dist_dir / "LICENSE.txt").write_bytes(b"license")
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (dist_dir / "gfx94X-dcgpu" / "package.whl").write_bytes(b"whl")

            # Only look for .whl files
            generate_multiarch_indexes(dist_dir, patterns=["*.whl"])

            content = (dist_dir / "gfx94X-dcgpu" / "index.html").read_text()
            # Should include .whl
            self.assertIn("package.whl", content)
            # Should not include .md or .txt
            self.assertNotIn("README.md", content)
            self.assertNotIn("LICENSE.txt", content)

    def test_tar_gz_and_whl_both_included(self):
        """Test default patterns include both .whl and .tar.gz."""
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "generic.whl").write_bytes(b"whl")
            (dist_dir / "generic.tar.gz").write_bytes(b"tar")
            (dist_dir / "gfx94X-dcgpu").mkdir()
            (dist_dir / "gfx94X-dcgpu" / "specific.whl").write_bytes(b"whl")
            (dist_dir / "gfx94X-dcgpu" / "specific.tar.gz").write_bytes(b"tar")

            generate_multiarch_indexes(dist_dir)

            content = (dist_dir / "gfx94X-dcgpu" / "index.html").read_text()
            # Local files
            self.assertIn("specific.whl", content)
            self.assertIn("specific.tar.gz", content)
            # Parent files
            self.assertIn("generic.whl", content)
            self.assertIn("generic.tar.gz", content)


if __name__ == "__main__":
    unittest.main()
