#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from _therock_utils.os_util import rmtree_with_retry

IS_WINDOWS = platform.system() == "Windows"

# Script that holds a file open WITHOUT FILE_SHARE_DELETE, preventing deletion.
# This simulates how antivirus scanners, search indexers, or loaded DLLs hold
# files on Windows.
_HOLD_FILE_SCRIPT = textwrap.dedent(
    """\
    import ctypes
    import sys
    import time
    from ctypes import wintypes

    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = wintypes.HANDLE
    CloseHandle = ctypes.windll.kernel32.CloseHandle

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    # No FILE_SHARE_DELETE - prevents deletion while handle is open
    OPEN_EXISTING = 3
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    path = sys.argv[1]
    hold_seconds = float(sys.argv[2])

    handle = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, 0, None)
    if handle == INVALID_HANDLE_VALUE:
        sys.exit(f"CreateFileW failed for {path}")

    # Signal readiness
    print("READY", flush=True)
    time.sleep(hold_seconds)
    CloseHandle(handle)
"""
)


def _start_holder(file_path: Path, hold_seconds: float) -> subprocess.Popen:
    """Start a subprocess that holds *file_path* open without FILE_SHARE_DELETE."""
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_FILE_SCRIPT, str(file_path), str(hold_seconds)],
        stdout=subprocess.PIPE,
        text=True,
    )
    line = holder.stdout.readline().strip()
    assert line == "READY", f"holder subprocess failed to start: {line!r}"
    return holder


class RmtreeWithRetryTest(unittest.TestCase):
    def test_removes_directory(self):
        """Basic case: removes a directory with no locks."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "subdir"
            target.mkdir()
            (target / "file.txt").write_text("hello")
            rmtree_with_retry(target)
            self.assertFalse(target.exists())

    def test_raises_on_nonexistent(self):
        """Raises FileNotFoundError for a path that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                rmtree_with_retry(target)

    @unittest.skipUnless(IS_WINDOWS, "Windows file locking behavior")
    def test_retries_on_locked_file(self):
        """rmtree_with_retry succeeds after a transient file lock is released."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "subdir"
            target.mkdir()
            locked_file = target / "locked.txt"
            locked_file.write_text("data")

            holder = _start_holder(locked_file, hold_seconds=1)
            try:
                # Bare shutil.rmtree should fail while the file is locked.
                with self.assertRaises(PermissionError):
                    shutil.rmtree(target)

                # rmtree_with_retry should succeed after the subprocess releases.
                rmtree_with_retry(target, max_attempts=10, retry_delay_seconds=0.2)
                self.assertFalse(target.exists())
            finally:
                holder.wait()

    @unittest.skipUnless(IS_WINDOWS, "Windows file locking behavior")
    def test_gives_up_after_max_attempts(self):
        """Raises PermissionError if the lock outlasts the retry window."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "subdir"
            target.mkdir()
            locked_file = target / "locked.txt"
            locked_file.write_text("data")

            holder = _start_holder(locked_file, hold_seconds=10)
            try:
                with self.assertRaises(PermissionError):
                    rmtree_with_retry(target, max_attempts=2, retry_delay_seconds=0.1)
            finally:
                holder.terminate()
                holder.wait()


if __name__ == "__main__":
    unittest.main()
