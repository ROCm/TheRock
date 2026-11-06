# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import clean_python_toolcache


def make_entry(toolcache: Path, version: str = "3.12.12", arch: str = "x64") -> Path:
    arch_dir = toolcache / "Python" / version / arch
    (arch_dir / "bin").mkdir(parents=True)
    (arch_dir / "lib").mkdir()
    (arch_dir.parent / f"{arch}.complete").touch()
    return arch_dir


def write_interpreter(arch_dir: Path, body: str, name: str = "python") -> Path:
    interpreter = arch_dir / "bin" / name
    interpreter.write_text(f"#!/bin/sh\n{body}\n")
    interpreter.chmod(interpreter.stat().st_mode | stat.S_IEXEC)
    return interpreter


@unittest.skipIf(sys.platform == "win32", "POSIX interpreter layout")
class CleanPythonToolcacheTest(unittest.TestCase):
    """Tests for clean_python_toolcache.clean_toolcache."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.toolcache = Path(tmp.name)

    def assert_removed(self, arch_dir: Path):
        self.assertFalse(arch_dir.exists())
        self.assertFalse(arch_dir.with_name(f"{arch_dir.name}.complete").exists())

    def test_working_entry_is_kept(self):
        arch_dir = make_entry(self.toolcache)
        write_interpreter(arch_dir, "exit 0")

        self.assertEqual(
            clean_python_toolcache.clean_toolcache(self.toolcache, "3.12"), []
        )
        self.assertTrue(arch_dir.is_dir())
        self.assertTrue(arch_dir.with_name("x64.complete").exists())

    def test_entry_missing_interpreter_is_removed(self):
        arch_dir = make_entry(self.toolcache)

        removed = clean_python_toolcache.clean_toolcache(self.toolcache, "3.12")

        self.assertEqual(removed, [arch_dir])
        self.assert_removed(arch_dir)

    def test_python3_alone_does_not_keep_the_entry(self):
        # setup-python hands out `bin/python`; a surviving `bin/python3` must not
        # mask its absence.
        arch_dir = make_entry(self.toolcache)
        write_interpreter(arch_dir, "exit 0", name="python3")

        removed = clean_python_toolcache.clean_toolcache(self.toolcache, "3.12")

        self.assertEqual(removed, [arch_dir])
        self.assert_removed(arch_dir)

    def test_entry_without_working_pip_is_removed(self):
        arch_dir = make_entry(self.toolcache)
        write_interpreter(arch_dir, "exit 1")

        removed = clean_python_toolcache.clean_toolcache(self.toolcache, "3.12")

        self.assertEqual(removed, [arch_dir])
        self.assert_removed(arch_dir)

    def test_other_feature_versions_are_untouched(self):
        other = make_entry(self.toolcache, version="3.11.9")

        self.assertEqual(
            clean_python_toolcache.clean_toolcache(self.toolcache, "3.12"), []
        )
        self.assertTrue(other.is_dir())

    def test_missing_toolcache_is_not_an_error(self):
        self.assertEqual(
            clean_python_toolcache.clean_toolcache(self.toolcache / "absent", "3.12"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
