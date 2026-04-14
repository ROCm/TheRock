#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate simple HTML index files for pip --find-links from local files.

This module provides utilities to generate index.html files that pip can use
with the --find-links option. It supports both simple single-directory indexes
and multi-arch builds where each GPU family needs its own index that includes
both family-specific packages and generic packages from the parent directory.

For generating an index from packages already in S3, see generate_s3_index.py.

Usage as a library:
    from generate_local_index import generate_simple_index

    generate_simple_index(
        output_path=Path("dist/gfx94X-dcgpu/index.html"),
        local_files=[Path("dist/gfx94X-dcgpu/rocm_sdk_libraries_gfx94x_dcgpu-7.2.0.whl")],
        parent_files=[Path("dist/rocm_sdk_core-7.2.0.whl")]
    )

Usage as a script:
    python generate_local_index.py dist/ --output dist/index.html
    python generate_local_index.py dist/gfx94X-dcgpu/ --output dist/gfx94X-dcgpu/index.html --parent-dir dist/
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import quote


def generate_simple_index(
    output_path: Path,
    local_files: list[Path],
    parent_files: list[Path] | None = None,
    title: str = "Package Index",
) -> None:
    """Generate a simple HTML index for pip --find-links.

    Creates an HTML file with links to package files. Local files are referenced
    with ./ prefix, parent files with ../ prefix. This allows a single index to
    reference packages in both the current directory and parent directory.

    Args:
        output_path: Where to write index.html
        local_files: Files in same directory (will use ./ prefix)
        parent_files: Files in parent directory (will use ../ prefix)
        title: HTML page title

    Example:
        For a multi-arch build where generic packages are at the top level and
        family-specific packages are in subdirectories:

        generate_simple_index(
            output_path=Path("dist/gfx94X-dcgpu/index.html"),
            local_files=[
                Path("dist/gfx94X-dcgpu/rocm_sdk_libraries_gfx94x_dcgpu-7.2.0.whl"),
                Path("dist/gfx94X-dcgpu/rocm-gfx94x_dcgpu-7.2.0.tar.gz"),
            ],
            parent_files=[
                Path("dist/rocm_sdk_core-7.2.0.whl"),
            ],
            title="ROCm Python Packages - gfx94X-dcgpu"
        )

        This generates an index with:
        - ./rocm_sdk_libraries_gfx94x_dcgpu-7.2.0.whl
        - ./rocm-gfx94x_dcgpu-7.2.0.tar.gz
        - ../rocm_sdk_core-7.2.0.whl
    """
    parent_files = parent_files or []

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{title}</title>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
    ]

    # Add local files with ./ prefix
    for file_path in sorted(local_files):
        filename = file_path.name
        html_parts.append(f'  <a href="./{quote(filename)}">{filename}</a><br>')

    # Add parent files with ../ prefix
    for file_path in sorted(parent_files):
        filename = file_path.name
        html_parts.append(f'  <a href="../{quote(filename)}">{filename}</a><br>')

    html_parts.extend(["</body>", "</html>", ""])  # Trailing newline

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(
        f"Generated {output_path}: {len(local_files)} local, {len(parent_files)} parent files"
    )


def generate_flat_index(dist_dir: Path, patterns: list[str] | None = None) -> None:
    """Generate a single flat index.html for kpack-split builds.

    In kpack-split mode all packages (core, libraries, device-per-target, devel,
    meta) are placed directly in dist/ at the top level — there are no family
    subdirectories. This function generates a single index.html covering all
    top-level files so that pip --find-links can discover all wheels.

    Only top-level files are scanned; subdirectories are ignored.

    Args:
        dist_dir: dist/ directory containing packages at the top level
        patterns: File patterns to include (default: ["*.whl", "*.tar.gz"])
    """
    if patterns is None:
        patterns = ["*.whl", "*.tar.gz"]

    local_files = []
    for pattern in patterns:
        local_files.extend([f for f in dist_dir.glob(pattern) if f.is_file()])

    print(
        f"Generating flat index for kpack-split build: {len(local_files)} files in {dist_dir}"
    )

    generate_simple_index(
        output_path=dist_dir / "index.html",
        local_files=local_files,
        parent_files=None,
        title="ROCm Python Packages",
    )


def generate_multiarch_indexes(
    dist_dir: Path, patterns: list[str] | None = None
) -> None:
    """Generate per-family index files for multi-arch builds.

    For multi-arch builds, the dist directory structure is:
        dist/
          rocm_sdk_core-*.whl              (generic, top-level)
          gfx94X-dcgpu/
            rocm_sdk_libraries_gfx94x_dcgpu-*.whl
            rocm-gfx94x_dcgpu-*.tar.gz
          gfx120X-all/
            rocm_sdk_libraries_gfx120x_all-*.whl
            ...

    This generates an index.html in each family subdir that includes:
    - Local family-specific packages (./relative)
    - Parent generic packages (../relative)

    Args:
        dist_dir: Root dist directory containing generic packages and family subdirs
        patterns: File patterns to include (default: ["*.whl", "*.tar.gz"])

    Example:
        generate_multiarch_indexes(Path("packages/dist"))
    """
    if patterns is None:
        patterns = ["*.whl", "*.tar.gz"]

    # Find generic packages at top level
    generic_packages = []
    for pattern in patterns:
        generic_packages.extend([f for f in dist_dir.glob(pattern) if f.is_file()])

    print(f"Found {len(generic_packages)} generic packages at top level: {dist_dir}")

    # Process each family subdirectory
    family_dirs = [d for d in dist_dir.iterdir() if d.is_dir()]

    if not family_dirs:
        raise FileNotFoundError(
            f"No family subdirectories found in {dist_dir}. "
            "generate_multiarch_indexes requires a multi-arch dist layout."
        )

    for family_dir in sorted(family_dirs):
        family = family_dir.name

        # Find family-specific files
        family_files = []
        for pattern in patterns:
            family_files.extend(family_dir.glob(pattern))

        print(
            f"Generating index for {family}: {len(family_files)} local + {len(generic_packages)} parent files"
        )

        generate_simple_index(
            output_path=family_dir / "index.html",
            local_files=family_files,
            parent_files=generic_packages,
            title=f"ROCm Python Packages - {family}",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML index files for pip --find-links"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing packages to index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for index.html (default: <directory>/index.html)",
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        help="Parent directory containing additional packages (will use ../ links)",
    )
    parser.add_argument(
        "--multiarch",
        action="store_true",
        help="Multi-arch mode: generate per-family indexes in subdirectories",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Package Index",
        help="HTML page title (default: 'Package Index')",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.whl", "*.tar.gz"],
        help="File patterns to include (default: *.whl *.tar.gz)",
    )

    args = parser.parse_args()

    if args.multiarch:
        # Multi-arch mode: generate indexes for each family subdirectory
        generate_multiarch_indexes(args.directory, patterns=args.patterns)
    else:
        # Single directory mode
        output_path = args.output or (args.directory / "index.html")

        # Find local files
        local_files = []
        for pattern in args.patterns:
            local_files.extend(args.directory.glob(pattern))

        # Find parent files if specified
        parent_files = []
        if args.parent_dir:
            for pattern in args.patterns:
                parent_files.extend(args.parent_dir.glob(pattern))

        generate_simple_index(
            output_path=output_path,
            local_files=local_files,
            parent_files=parent_files,
            title=args.title,
        )


if __name__ == "__main__":
    main()
