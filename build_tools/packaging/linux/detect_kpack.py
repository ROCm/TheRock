#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Detect KPACK_SPLIT_ARTIFACTS flag from TheRock manifest files.

Searches for therock_manifest.json files in the artifacts directory and checks
if any of them have KPACK_SPLIT_ARTIFACTS set to true.

Usage:
    python detect_kpack.py --artifacts-dir /path/to/artifacts --output-format github
    python detect_kpack.py --artifacts-dir /path/to/artifacts --output-format json
    python detect_kpack.py --artifacts-dir /path/to/artifacts --output-format env
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def find_manifest_files(artifacts_dir: Path) -> list[Path]:
    """
    Find all therock_manifest.json files in the artifacts directory.

    Args:
        artifacts_dir: Path to the artifacts directory

    Returns:
        List of Path objects pointing to therock_manifest.json files
    """
    return list(artifacts_dir.rglob("therock_manifest.json"))


def check_kpack_enabled(manifest_path: Path) -> bool:
    """
    Check if KPACK_SPLIT_ARTIFACTS is set to true in a manifest file.

    Args:
        manifest_path: Path to therock_manifest.json file

    Returns:
        True if KPACK_SPLIT_ARTIFACTS is true, False otherwise
    """
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            return manifest.get("KPACK_SPLIT_ARTIFACTS", False) is True
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to read {manifest_path}: {e}", file=sys.stderr)
        return False


def detect_kpack(artifacts_dir: Path) -> tuple[bool, Optional[Path]]:
    """
    Detect if KPACK is enabled in any manifest file.

    Args:
        artifacts_dir: Path to the artifacts directory

    Returns:
        Tuple of (kpack_enabled, manifest_path_if_found)
    """
    manifest_files = find_manifest_files(artifacts_dir)

    if not manifest_files:
        print(
            f"No therock_manifest.json files found in {artifacts_dir}",
            file=sys.stderr,
        )
        return False, None

    print(
        f"Found {len(manifest_files)} therock_manifest.json file(s)",
        file=sys.stderr,
    )

    for manifest_path in manifest_files:
        if check_kpack_enabled(manifest_path):
            print(
                f"✓ KPACK_SPLIT_ARTIFACTS=true found in {manifest_path}",
                file=sys.stderr,
            )
            return True, manifest_path

    print("✓ KPACK_SPLIT_ARTIFACTS not enabled in any manifest", file=sys.stderr)
    return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Detect KPACK_SPLIT_ARTIFACTS flag from TheRock manifest files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Path to the artifacts directory",
    )
    parser.add_argument(
        "--output-format",
        choices=["env", "json", "github"],
        default="github",
        help="Output format: 'env' for shell variables, 'json' for JSON, 'github' for GitHub Actions",
    )

    args = parser.parse_args()

    if not args.artifacts_dir.exists():
        print(
            f"Error: Artifacts directory not found: {args.artifacts_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    kpack_enabled, manifest_path = detect_kpack(args.artifacts_dir)

    # Output in requested format
    if args.output_format == "json":
        output = {
            "enable_kpack": kpack_enabled,
            "manifest_path": str(manifest_path) if manifest_path else None,
        }
        print(json.dumps(output, indent=2))
    elif args.output_format == "github":
        # GitHub Actions output format
        print(f"enable_kpack={'true' if kpack_enabled else 'false'}")
        if manifest_path:
            print(f"manifest_path={manifest_path}")
    else:  # env format
        print(f"export ENABLE_KPACK={'true' if kpack_enabled else 'false'}")
        if manifest_path:
            print(f"export KPACK_MANIFEST_PATH={manifest_path}")


if __name__ == "__main__":
    main()
