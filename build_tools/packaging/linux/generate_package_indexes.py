#!/usr/bin/env python3

"""Generate index.html files for package repositories.

This script contains index generation logic.
TODO(#3329) Enable running index generation at server-side.

It can be used as a standalone tool:

  python3 build_tools/packaging/linux/generate_package_indexes.py \
    --s3-bucket <bucket> \
    --prefix <prefix>

Examples:
  --prefix deb/20260223-12345
  --prefix rpm/20260223-12345
  --top-prefix deb
  --top-prefix rpm
"""

import argparse
import boto3
import os
from pathlib import Path


SVG_DEFS = """<svg xmlns="http://www.w3.org/2000/svg" style="display:none">
<defs>
  <symbol id="file" viewBox="0 0 265 323">
    <path fill="#4582ec" d="M213 115v167a41 41 0 01-41 41H69a41 41 0 01-41-41V39a39 39 0 0139-39h127a39 39 0 0139 39v76z"/>
    <path fill="#77a4ff" d="M176 17v88a19 19 0 0019 19h88"/>
  </symbol>
  <symbol id="folder-shortcut" viewBox="0 0 265 216">
    <path fill="#4582ec" d="M18 54v-5a30 30 0 0130-30h75a28 28 0 0128 28v7h77a30 30 0 0130 30v84a30 30 0 01-30 30H33a30 30 0 01-30-30V54z"/>
  </symbol>
</defs>
</svg>
"""

HTML_HEAD = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>artifacts</title>
</head>
<body>
{SVG_DEFS}
<table>
<tbody>
"""

HTML_FOOT = """
</tbody>
</table>
</body>
</html>
"""


def generate_index_html(directory: str) -> None:
    """Generate a local index.html for a directory on disk."""
    rows: list[str] = []
    try:
        for entry in os.scandir(directory):
            if entry.name.startswith("."):
                continue
            rows.append(f'<tr><td><a href="{entry.name}">{entry.name}</a></td></tr>')
    except PermissionError:
        return

    index_path = os.path.join(directory, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEAD + "\n".join(rows) + HTML_FOOT)


def generate_indexes_recursive(root: str) -> None:
    """Generate local index.html files for all directories under root."""
    for d, _, _ in os.walk(root):
        generate_index_html(d)


def generate_top_index_from_s3(s3, bucket: str, prefix: str) -> None:
    """Generate index.html for top-level directory using S3 Delimiter.

    This is much more efficient than listing all objects recursively,
    as it only retrieves immediate subdirectories and files.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'deb' or 'rpm')
    """
    print(f"Generating top index from S3: s3://{bucket}/{prefix}/")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/")

    rows: list[str] = []

    for page in pages:
        # Add subdirectories (CommonPrefixes returned by Delimiter)
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"][len(prefix) + 1 :].rstrip("/")
            rows.append(
                f'<tr><td><a href="{folder}/index.html">{folder}/</a></td></tr>'
            )

        # Add files at this level only (no nested files)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or key.endswith("index.html"):
                continue
            name = key[len(prefix) + 1 :]
            if "/" not in name:  # Only files at this level
                rows.append(f'<tr><td><a href="{name}">{name}</a></td></tr>')

    index_content = HTML_HEAD + "\n".join(rows) + HTML_FOOT
    index_key = f"{prefix}/index.html"

    print(f"Uploading top index: {index_key}")
    s3.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=index_content.encode("utf-8"),
        ContentType="text/html",
    )
    print("✓ Successfully uploaded top-level index")


def generate_index_from_s3(
    s3, bucket: str, prefix: str, max_depth: int | None = None
) -> None:
    """Generate index.html files based on what's actually in S3.

    This ensures index files accurately reflect the S3 repository state,
    including files from previous uploads that may have been deduplicated.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'deb/20260223-12345')
        max_depth: Maximum directory depth to generate indexes for.
                   None = unlimited (recursive), 0 = only root level, 1 = root + immediate children
    """
    depth_msg = (
        f" (max depth: {max_depth})" if max_depth is not None else " (recursive)"
    )
    print(f"Generating indexes from S3: s3://{bucket}/{prefix}/{depth_msg}")

    # Get all objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    all_objects = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            all_objects.extend(page["Contents"])
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return

    if not all_objects:
        print(f"No objects found in s3://{bucket}/{prefix}/")
        return

    # Group objects by directory
    directories: dict[str, list[str]] = {}
    for obj in all_objects:
        key = obj["Key"]

        # Skip existing index.html files
        if key.endswith("index.html"):
            continue

        # Get the directory path relative to prefix
        if key.startswith(prefix):
            rel_path = key[len(prefix) :].lstrip("/")
        else:
            rel_path = key

        # Determine directory and filename
        if "/" in rel_path:
            dir_path = "/".join(rel_path.split("/")[:-1])
            filename = rel_path.split("/")[-1]
        else:
            dir_path = ""
            filename = rel_path

        directories.setdefault(dir_path, []).append(filename)

        # Track all parent directories (even if they have no files, only subdirs)
        parts = dir_path.split("/") if dir_path else []
        for i in range(len(parts)):
            parent = "/".join(parts[:i])  # Empty string for root, or partial path
            directories.setdefault(parent, [])

    # Ensure root directory exists
    directories.setdefault("", [])

    uploaded_indexes = 0
    for dir_path, files in sorted(
        directories.items(), key=lambda x: (-x[0].count("/") if x[0] else 1, x[0])
    ):
        # Check depth limit
        if max_depth is not None:
            depth = dir_path.count("/") if dir_path else 0
            if depth > max_depth:
                continue

        rows: list[str] = []

        # Add subdirectories first
        subdirs: set[str] = set()
        for other_dir in directories.keys():
            if dir_path == "":
                if other_dir:
                    subdir = other_dir.split("/")[0] if "/" in other_dir else other_dir
                    subdirs.add(subdir)
            else:
                if other_dir.startswith(dir_path + "/") and other_dir != dir_path:
                    remainder = other_dir[len(dir_path) :].lstrip("/")
                    subdir = remainder.split("/")[0] if "/" in remainder else remainder
                    if subdir:
                        subdirs.add(subdir)

        for subdir in sorted(subdirs):
            rows.append(
                f'<tr><td><a href="{subdir}/index.html">{subdir}/</a></td></tr>'
            )

        # Add files
        for filename in sorted(files):
            rows.append(f'<tr><td><a href="{filename}">{filename}</a></td></tr>')

        index_content = HTML_HEAD + "\n".join(rows) + HTML_FOOT

        if dir_path:
            index_key = f"{prefix}/{dir_path}/index.html"
        else:
            index_key = f"{prefix}/index.html"

        try:
            print(f"Uploading index: {index_key}")
            s3.put_object(
                Bucket=bucket,
                Key=index_key,
                Body=index_content.encode("utf-8"),
                ContentType="text/html",
            )
            uploaded_indexes += 1
        except Exception as e:
            print(f"Error uploading index {index_key}: {e}")

    print(f"Generated and uploaded {uploaded_indexes} index files from S3 state")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--prefix", required=True, help="e.g. deb/20260223-12345")
    parser.add_argument(
        "--top-prefix",
        default=None,
        help="Optional top-level prefix to generate top index for (e.g., 'deb' or 'rpm').",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional max depth for per-prefix index generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    s3 = boto3.client("s3")
    generate_index_from_s3(s3, args.s3_bucket, args.prefix, max_depth=args.max_depth)

    if args.top_prefix is not None:
        generate_top_index_from_s3(s3, args.s3_bucket, args.top_prefix)


if __name__ == "__main__":
    main()
