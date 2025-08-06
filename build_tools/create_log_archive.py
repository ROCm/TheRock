#!/usr/bin/env python3

import os
import tarfile
from pathlib import Path
import sys
import argparse

THEROCK_DIR = Path(__file__).resolve().parent.parent


def log(*args):
    print(*args)
    sys.stdout.flush()


def create_ninja_log_archive(build_dir: Path):
    log_dir = build_dir / "logs"

    # Python equivalent of `find  ~/TheRock/build -iname .ninja_log``
    found_files = []
    log(f"[*] Create ninja log archive from: {build_dir}")

    glob_pattern_ninja = f"**/.ninja_log"
    log(f"[*] Path glob: {glob_pattern_ninja}")
    found_files = list(build_dir.glob(glob_pattern_ninja))

    files_to_archive = found_files
    archive_name = log_dir / "ninja_logs.tar.gz"
    if archive_name.exists():
        print(f"NOTE: Archive exists: {archive_name}", file=sys.stderr)
    added_count = 0
    with tarfile.open(archive_name, "w:gz") as tar:
        log(f"[+] Create archive: {archive_name}")
        for file_path in files_to_archive:
            tar.add(file_path)
            added_count += 1
            log(f"[+]  Add: {file_path}")
    log(f"[*] Files Added: {added_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ninja log archive.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", "build")),
        help="Build directory containing logs (default: 'build' or $BUILD_DIR)",
    )
    args = parser.parse_args()

    create_ninja_log_archive(args.build_dir)
