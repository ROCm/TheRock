# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pattern_match module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.pattern_match import (
    RecursiveGlobPattern,
    MatchPredicate,
    PatternMatcher,
)


class _DirEntryStub:
    """Minimal stub for os.DirEntry used in pattern matching tests."""

    def __init__(self, name: str, path: str = "", *, is_dir: bool = False):
        self.name = name
        self.path = path or name
        self._is_dir = is_dir

    def is_dir(self, *, follow_symlinks=True):
        return self._is_dir

    def is_symlink(self):
        return False


class RecursiveGlobPatternTest(unittest.TestCase):
    """Tests for RecursiveGlobPattern."""

    def _match(self, pattern: str, path: str, is_dir: bool = False) -> bool:
        p = RecursiveGlobPattern(pattern)
        entry = _DirEntryStub(os.path.basename(path), is_dir=is_dir)
        return p.matches(path, entry)

    # -- Simple wildcards -------------------------------------------------
    def test_exact_match(self):
        self.assertTrue(self._match("lib/libfoo.so", "lib/libfoo.so"))
        self.assertFalse(self._match("lib/libfoo.so", "lib/libbar.so"))

    def test_star_in_filename(self):
        """'*' should match any characters within a single path segment."""
        self.assertTrue(self._match("lib/*.so", "lib/libfoo.so"))
        self.assertTrue(self._match("lib/*.so", "lib/libbar.so"))
        self.assertFalse(self._match("lib/*.so", "lib/libfoo.a"))
        # '*' should not cross directory boundaries
        self.assertFalse(self._match("lib/*.so", "lib/sub/libfoo.so"))

    def test_question_mark(self):
        """'?' should match any characters within a single path segment."""
        self.assertTrue(self._match("lib/lib?.so", "lib/libA.so"))
        self.assertTrue(self._match("lib/lib?.so", "lib/libXY.so"))
        self.assertFalse(self._match("lib/lib?.so", "lib/sub/libA.so"))

    # -- Recursive '**' patterns ------------------------------------------
    def test_double_star_intermediate(self):
        """'**' between segments matches zero or more directories."""
        self.assertTrue(self._match("lib/**/foo.so", "lib/foo.so"))
        self.assertTrue(self._match("lib/**/foo.so", "lib/sub/foo.so"))
        self.assertTrue(self._match("lib/**/foo.so", "lib/a/b/c/foo.so"))
        self.assertFalse(self._match("lib/**/foo.so", "other/foo.so"))

    def test_double_star_leading(self):
        """'**' at the start matches any prefix."""
        self.assertTrue(self._match("**/foo.so", "foo.so"))
        self.assertTrue(self._match("**/foo.so", "lib/foo.so"))
        self.assertTrue(self._match("**/foo.so", "a/b/c/foo.so"))

    def test_double_star_trailing(self):
        """'**' at the end matches any suffix."""
        self.assertTrue(self._match("lib/**", "lib"))
        self.assertTrue(self._match("lib/**", "lib/foo.so"))
        self.assertTrue(self._match("lib/**", "lib/sub/foo.so"))
        self.assertFalse(self._match("lib/**", "other/foo.so"))

    # -- Edge cases -------------------------------------------------------
    def test_no_match(self):
        self.assertFalse(self._match("include/*.h", "lib/libfoo.so"))

    def test_root_file(self):
        self.assertTrue(self._match("CMakeLists.txt", "CMakeLists.txt"))
        self.assertFalse(self._match("CMakeLists.txt", "sub/CMakeLists.txt"))


class MatchPredicateTest(unittest.TestCase):
    """Tests for MatchPredicate."""

    def _entry(self, name: str, is_dir: bool = False) -> _DirEntryStub:
        return _DirEntryStub(name, is_dir=is_dir)

    def test_no_filters_matches_everything(self):
        pred = MatchPredicate()
        self.assertTrue(pred.matches("lib/foo.so", self._entry("foo.so")))

    def test_includes_only(self):
        pred = MatchPredicate(includes=["lib/*.so"])
        self.assertTrue(pred.matches("lib/foo.so", self._entry("foo.so")))
        self.assertFalse(pred.matches("lib/foo.a", self._entry("foo.a")))
        self.assertFalse(pred.matches("bin/tool", self._entry("tool")))

    def test_excludes_only(self):
        pred = MatchPredicate(excludes=["*.pyc"])
        self.assertTrue(pred.matches("module.py", self._entry("module.py")))
        self.assertFalse(pred.matches("module.pyc", self._entry("module.pyc")))

    def test_includes_and_excludes(self):
        """Excludes should override includes."""
        pred = MatchPredicate(includes=["lib/*"], excludes=["lib/*.a"])
        self.assertTrue(pred.matches("lib/foo.so", self._entry("foo.so")))
        self.assertFalse(pred.matches("lib/foo.a", self._entry("foo.a")))

    def test_force_includes_override_excludes(self):
        """Force includes should override excludes."""
        pred = MatchPredicate(
            includes=["lib/*"],
            excludes=["lib/*.a"],
            force_includes=["lib/important.a"],
        )
        # Regular .a files excluded
        self.assertFalse(pred.matches("lib/foo.a", self._entry("foo.a")))
        # But force-included .a file is kept
        self.assertTrue(
            pred.matches("lib/important.a", self._entry("important.a"))
        )

    def test_force_includes_without_includes(self):
        """Force includes should work even without includes specified."""
        pred = MatchPredicate(
            excludes=["*"],
            force_includes=["important.txt"],
        )
        # Everything excluded
        self.assertFalse(pred.matches("other.txt", self._entry("other.txt")))
        # But force-included file is kept
        self.assertTrue(
            pred.matches("important.txt", self._entry("important.txt"))
        )

    def test_multiple_includes(self):
        """Any matching include should allow the file."""
        pred = MatchPredicate(includes=["*.so", "*.h"])
        self.assertTrue(pred.matches("lib.so", self._entry("lib.so")))
        self.assertTrue(pred.matches("header.h", self._entry("header.h")))
        self.assertFalse(pred.matches("code.cpp", self._entry("code.cpp")))

    def test_multiple_excludes(self):
        """Any matching exclude should reject the file."""
        pred = MatchPredicate(excludes=["*.pyc", "*.pyo"])
        self.assertTrue(pred.matches("module.py", self._entry("module.py")))
        self.assertFalse(pred.matches("module.pyc", self._entry("module.pyc")))
        self.assertFalse(pred.matches("module.pyo", self._entry("module.pyo")))


class PatternMatcherTest(unittest.TestCase):
    """Tests for PatternMatcher."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_tree(self, tree: dict, base: Path | None = None):
        """Create a file tree from a dict specification.

        Keys ending with '/' are directories (values are sub-trees).
        Other keys are files (values are content strings).
        """
        if base is None:
            base = Path(self.temp_dir)
        for name, value in tree.items():
            path = base / name
            if isinstance(value, dict):
                path.mkdir(parents=True, exist_ok=True)
                self._create_tree(value, path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)

    def test_add_basedir_scans_files(self):
        """Test that add_basedir discovers files recursively."""
        self._create_tree(
            {
                "lib": {"libfoo.so": "", "libbar.so": ""},
                "include": {"foo.h": ""},
            }
        )
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))
        all_paths = set(pm.all.keys())
        self.assertIn("lib/libfoo.so", all_paths)
        self.assertIn("lib/libbar.so", all_paths)
        self.assertIn("include/foo.h", all_paths)
        # Directories should also be in the scan
        self.assertIn("lib", all_paths)
        self.assertIn("include", all_paths)

    def test_matches_with_includes(self):
        """Test that matches() filters using include patterns."""
        self._create_tree(
            {
                "lib": {"libfoo.so": "", "libbar.a": ""},
                "bin": {"tool": ""},
            }
        )
        pm = PatternMatcher(includes=["lib/*.so"])
        pm.add_basedir(Path(self.temp_dir))
        matched = {path for path, _ in pm.matches()}
        self.assertIn("lib/libfoo.so", matched)
        self.assertNotIn("lib/libbar.a", matched)
        self.assertNotIn("bin/tool", matched)

    def test_matches_with_excludes(self):
        """Test that matches() filters using exclude patterns."""
        self._create_tree(
            {
                "lib": {"libfoo.so": "", "libfoo.a": ""},
            }
        )
        pm = PatternMatcher(excludes=["**/*.a"])
        pm.add_basedir(Path(self.temp_dir))
        matched = {path for path, _ in pm.matches()}
        self.assertIn("lib/libfoo.so", matched)
        self.assertNotIn("lib/libfoo.a", matched)

    def test_copy_to_copies_files(self):
        """Test that copy_to copies matched files to the destination."""
        self._create_tree(
            {
                "lib": {"libfoo.so": "foo content"},
                "bin": {"tool": "tool content"},
            }
        )
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))

        dest = Path(self.temp_dir) / "output"
        pm.copy_to(destdir=dest)

        self.assertTrue((dest / "lib" / "libfoo.so").exists())
        self.assertTrue((dest / "bin" / "tool").exists())
        self.assertEqual((dest / "lib" / "libfoo.so").read_text(), "foo content")

    def test_copy_to_with_destprefix(self):
        """Test that copy_to uses destprefix correctly."""
        self._create_tree({"file.txt": "content"})
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))

        dest = Path(self.temp_dir) / "output"
        pm.copy_to(destdir=dest, destprefix="prefix/")

        self.assertTrue((dest / "prefix" / "file.txt").exists())
        self.assertEqual((dest / "prefix" / "file.txt").read_text(), "content")

    def test_copy_to_with_always_copy(self):
        """Test that always_copy=True copies instead of hardlinking."""
        self._create_tree({"file.txt": "content"})
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))

        dest = Path(self.temp_dir) / "output"
        pm.copy_to(destdir=dest, always_copy=True)

        self.assertTrue((dest / "file.txt").exists())
        self.assertEqual((dest / "file.txt").read_text(), "content")
        # With always_copy, the inode should be different from the source
        src_ino = os.stat(Path(self.temp_dir) / "file.txt").st_ino
        dst_ino = os.stat(dest / "file.txt").st_ino
        self.assertNotEqual(src_ino, dst_ino)

    def test_copy_to_default_hardlinks(self):
        """Test that default behavior attempts to hardlink."""
        self._create_tree({"file.txt": "content"})
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))

        dest = Path(self.temp_dir) / "output"
        pm.copy_to(destdir=dest, always_copy=False)

        self.assertTrue((dest / "file.txt").exists())
        # Default behavior should hardlink (same inode)
        src_ino = os.stat(Path(self.temp_dir) / "file.txt").st_ino
        dst_ino = os.stat(dest / "file.txt").st_ino
        self.assertEqual(src_ino, dst_ino)

    def test_copy_to_removes_existing_dest(self):
        """Test that copy_to removes existing destination directory."""
        src_dir = Path(self.temp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")

        dest = Path(self.temp_dir) / "output"
        dest.mkdir()
        (dest / "old_file.txt").write_text("old")

        pm = PatternMatcher()
        pm.add_basedir(src_dir)
        pm.copy_to(destdir=dest, remove_dest=True)

        self.assertFalse((dest / "old_file.txt").exists())
        self.assertTrue((dest / "file.txt").exists())

    def test_copy_to_preserves_existing_with_remove_dest_false(self):
        """Test that remove_dest=False preserves existing files."""
        dest = Path(self.temp_dir) / "output"
        dest.mkdir()
        (dest / "existing.txt").write_text("keep me")

        self._create_tree({"new.txt": "new content"})
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))
        pm.copy_to(destdir=dest, remove_dest=False)

        self.assertTrue((dest / "existing.txt").exists())
        self.assertTrue((dest / "new.txt").exists())

    def test_copy_to_handles_symlinks(self):
        """Test that copy_to correctly copies symlinks."""
        src_dir = Path(self.temp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "real.txt").write_text("real content")
        os.symlink("real.txt", src_dir / "link.txt")

        pm = PatternMatcher()
        pm.add_basedir(src_dir)

        dest = Path(self.temp_dir) / "output"
        pm.copy_to(destdir=dest)

        self.assertTrue((dest / "link.txt").is_symlink())
        self.assertEqual(os.readlink(dest / "link.txt"), "real.txt")

    def test_add_entry(self):
        """Test that add_entry manually adds entries."""
        self._create_tree({"file.txt": "content"})
        pm = PatternMatcher()
        # Manually scan and add an entry
        with os.scandir(self.temp_dir) as it:
            for entry in it:
                if entry.name == "file.txt":
                    pm.add_entry("file.txt", entry)
        matched = {path for path, _ in pm.matches()}
        self.assertIn("file.txt", matched)

    def test_empty_directory(self):
        """Test PatternMatcher with an empty directory."""
        pm = PatternMatcher()
        pm.add_basedir(Path(self.temp_dir))
        matched = list(pm.matches())
        self.assertEqual(matched, [])

    def test_recursive_glob_with_nested_dirs(self):
        """Test recursive glob pattern with deeply nested directories."""
        self._create_tree(
            {
                "a": {
                    "b": {"c": {"file.txt": "deep"}},
                    "file.txt": "shallow",
                },
            }
        )
        pm = PatternMatcher(includes=["**/file.txt"])
        pm.add_basedir(Path(self.temp_dir))
        matched = {path for path, _ in pm.matches()}
        self.assertIn("a/b/c/file.txt", matched)
        self.assertIn("a/file.txt", matched)


if __name__ == "__main__":
    unittest.main()
