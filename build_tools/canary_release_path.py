#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise local, non-publishing equivalents of release upload phases.

This intentionally performs no network operation and reads no credentials. It
streams the same artifact/log bytes into an isolated canary sink so A/B runs
cover local traversal, reads, hashing, and writes without modifying S3.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def mirror(source: Path, destination: Path) -> dict[str, object]:
    files = []
    for path in sorted(
        item for item in source.rglob("*") if item.is_file() and not item.is_symlink()
    ):
        relative = path.relative_to(source)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256()
        with path.open("rb") as input_file, output.open("wb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                checksum.update(chunk)
                output_file.write(chunk)
        shutil.copystat(path, output, follow_symlinks=False)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": checksum.hexdigest(),
            }
        )
    return {
        "schema": "therock.canary_local_release_phase.v1",
        "scope": "local-filesystem-equivalent-no-network-no-credentials",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"source directory does not exist: {args.source}")
    if args.destination.exists():
        parser.error(f"destination already exists: {args.destination}")
    result = mirror(args.source, args.destination)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
