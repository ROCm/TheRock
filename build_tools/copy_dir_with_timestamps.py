# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copies a source directory to a destination directory, preserving file timestamps.

This is a drop-in replacement for `cmake -E copy_directory` which does not
preserve file modification timestamps. This script uses shutil.copy2, which
preserves metadata including nanosecond-precision modification timestamps via
os.utime(ns=...). This prevents autotools-based build systems from incorrectly
detecting that generated files (e.g. .info docs) need to be regenerated.

Usage: copy_dir_with_timestamps.py <source_dir> <dest_dir>

The destination directory must not already exist (same behavior as
`cmake -E copy_directory`). Remove it first with `cmake -E rm -rf` if needed.
"""

import sys
import shutil
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <source_dir> <dest_dir>\n")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.is_dir():
        sys.stderr.write(f"ERROR: Source directory does not exist: {src}\n")
        sys.exit(1)

    if dst.exists():
        sys.stderr.write(
            f"ERROR: Destination already exists: {dst}\n"
            f"       Remove it first with `cmake -E rm -rf {dst}`\n"
        )
        sys.exit(1)

    # copy_function=shutil.copy2 preserves file metadata including
    # nanosecond-precision mtime via os.utime(ns=(st_atime_ns, st_mtime_ns)).
    shutil.copytree(src, dst, copy_function=shutil.copy2)


if __name__ == "__main__":
    main()
