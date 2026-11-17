#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ensure_lf_checkout() in fetch_sources.py.

They run real git on a small repository, made into a CRLF checkout with
core.autocrlf=true as Git for Windows does by default, which git honors on every
host, so they run everywhere."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_sources import ensure_lf_checkout

PROBE = "doc/manual.texi"
OTHER = "src/main.c"


def _git(repo, *args):
    """Run git in REPO and return its output."""
    return subprocess.check_output(["git", *args], cwd=repo, text=True)


class EnsureLfCheckoutTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        # Keep the user's and the system's git config out of it, as either may
        # set core.autocrlf.
        self._env = mock.patch.dict(
            os.environ,
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.fspath(Path(self._tmp.name) / "gitconfig"),
            },
        )
        self._env.start()
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        for name in (PROBE, OTHER):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"line one @\nline two\n")
        _git(self.repo, "add", ".")
        _git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "files stored with LF",
        )

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _make_crlf_checkout(self):
        """Check the repository out again as Git for Windows does by default."""
        _git(self.repo, "config", "core.autocrlf", "true")
        _git(self.repo, "rm", "--cached", "-r", "-q", ".")
        _git(self.repo, "reset", "--hard", "-q")
        self.assertIn(b"\r\n", (self.repo / PROBE).read_bytes())
        # Files written in the same second as the index are "racily clean":
        # git checks their contents again, and reset --hard would then rewrite
        # them under the new core.autocrlf by itself.  The files of a real
        # checkout are older than its index, so make these so too, and record
        # that in the index, as git then looks at their timestamps alone.
        hour_ago = time.time() - 3600
        for name in (PROBE, OTHER):
            os.utime(self.repo / name, (hour_ago, hour_ago))
        _git(self.repo, "update-index", "-q", "--refresh")

    def _autocrlf(self):
        return _git(self.repo, "config", "--get", "core.autocrlf").strip()

    def test_crlf_checkout(self):
        """A clean CRLF checkout is checked out again with LF, all of it, not
        only the probe, and keeps core.autocrlf=false for later checkouts."""
        self._make_crlf_checkout()
        ensure_lf_checkout(self.repo, PROBE)
        for name in (PROBE, OTHER):
            self.assertEqual((self.repo / name).read_bytes(), b"line one @\nline two\n")
        self.assertEqual(self._autocrlf(), "false")
        self.assertEqual(_git(self.repo, "status", "--porcelain"), "")

    def test_crlf_checkout_with_changes(self):
        """A CRLF checkout with local changes is left alone, changes, line
        endings and config included, with a warning that says how to fix
        it."""
        self._make_crlf_checkout()
        (self.repo / OTHER).write_bytes(b"changed\r\n")
        with mock.patch("fetch_sources.log") as log:
            ensure_lf_checkout(self.repo, PROBE)
        self.assertEqual((self.repo / OTHER).read_bytes(), b"changed\r\n")
        self.assertIn(b"\r\n", (self.repo / PROBE).read_bytes())
        self.assertEqual(self._autocrlf(), "true")
        message = log.call_args.args[0]
        self.assertIn("CRLF line endings", message)
        self.assertIn("git config core.autocrlf false", message)

    def test_lf_checkout(self):
        """An LF checkout keeps its files as they are, local changes included,
        and gets core.autocrlf=false for later checkouts."""
        (self.repo / OTHER).write_bytes(b"changed\n")
        ensure_lf_checkout(self.repo, PROBE)
        self.assertEqual((self.repo / OTHER).read_bytes(), b"changed\n")
        self.assertEqual(self._autocrlf(), "false")

    def test_not_checked_out(self):
        """A submodule that is not checked out, as an empty directory or with
        no probe file, is left as it is."""
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        ensure_lf_checkout(empty, PROBE)
        self.assertEqual(list(empty.iterdir()), [])
        ensure_lf_checkout(self.repo, "no/such/file")
        with self.assertRaises(subprocess.CalledProcessError):
            self._autocrlf()


if __name__ == "__main__":
    unittest.main()
