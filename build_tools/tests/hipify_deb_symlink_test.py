# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Source checks; actual installed dh_perl reproduction is TheRock-CI-only."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "packaging/linux"
sys.path.insert(0, str(SCRIPTS))
import deb_package as deb

ORIGINAL = b"#!/usr/bin/env perl\nprint qq(fixture\\n);\n"
PREPARED = b"#! /usr/bin/perl\nprint qq(fixture\\n);\n"
LINK = "../libexec/hipify/hipify-perl"


def layout(root: Path) -> tuple[Path, Path]:
    target, alias = root / "libexec/hipify/hipify-perl", root / "bin/hipify-perl"
    target.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    target.write_bytes(ORIGINAL)
    target.chmod(0o755)
    alias.symlink_to(LINK)
    return alias, target


def snapshot(root: Path) -> tuple:
    alias, target = root / "bin/hipify-perl", root / "libexec/hipify/hipify-perl"
    return (
        alias.is_symlink(),
        str(alias.readlink()),
        stat.S_IMODE(alias.lstat().st_mode),
        target.read_bytes(),
        stat.S_IMODE(target.stat().st_mode),
        target.stat().st_mtime_ns,
        alias.lstat().st_mtime_ns,
    )


class HipifySourceTest(unittest.TestCase):
    def test_copy_and_hardlink_inputs_unchanged_alias_and_mode_preserved(self):
        for hardlink in [False, True]:
            with self.subTest(
                hardlink=hardlink
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, copied = root / "input", root / "copied"
                try:
                    _, original_target = layout(source)
                except OSError as error:
                    self.skipTest(f"File symlinks unavailable: {error}")
                original = snapshot(source)
                deb.copy_package_contents(source, copied)
                target = copied / "libexec/hipify/hipify-perl"
                if hardlink:
                    target.unlink()
                    os.link(original_target, target)
                    self.assertTrue(target.samefile(original_target))
                alias_before = snapshot(copied)[:3]
                self.assertTrue(deb.prepare_hipify_perl_shebang(copied))
                self.assertEqual(snapshot(source), original)
                self.assertEqual(snapshot(copied)[:3], alias_before)
                self.assertEqual(target.read_bytes(), PREPARED)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), original[4])
                self.assertFalse(target.samefile(original_target))
                prepared = snapshot(copied)
                self.assertFalse(deb.prepare_hipify_perl_shebang(copied))
                self.assertEqual(snapshot(copied), prepared)

    def test_failed_replace_keeps_input_and_alias_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                _, target = layout(root)
            except OSError as error:
                self.skipTest(f"File symlinks unavailable: {error}")
            before = snapshot(root)
            with patch.object(deb.shutil, "copystat", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    deb.prepare_hipify_perl_shebang(root)
            self.assertEqual(snapshot(root), before)
            self.assertEqual(list(target.parent.glob(".hipify-perl-*")), [])

    def test_unexpected_alias_and_escaping_target_fail_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                alias, target = layout(root / "input")
            except OSError as error:
                self.skipTest(f"File symlinks unavailable: {error}")
            original = target.read_bytes()
            alias.unlink()
            alias.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "layout"):
                deb.prepare_hipify_perl_shebang(root / "input")
            self.assertEqual(target.read_bytes(), original)
            alias.unlink()
            alias.symlink_to(LINK)
            target.unlink()
            outside = root / "outside"
            outside.write_bytes(original)
            target.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "layout"):
                deb.prepare_hipify_perl_shebang(root / "input")
            self.assertEqual(outside.read_bytes(), original)

    def test_absent_optional_perl_payload_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(deb.prepare_hipify_perl_shebang(Path(temporary)))


if __name__ == "__main__":
    unittest.main()
