#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create a deterministic checksum manifest for staged build artifacts."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def create_manifest(root: Path) -> dict[str, object]:
    files = []
    if root.is_dir():
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
        ):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    return {"schema": "therock.artifact_equivalence.v1", "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_manifest(args.artifact_dir)
    if not manifest["files"]:
        parser.error(f"no files found under artifact directory: {args.artifact_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
