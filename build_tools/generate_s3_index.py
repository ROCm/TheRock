#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""
Usage:
generate_s3_index.py [-h]
  --run-id RUN_ID
  [--output-dir OUTPUT_DIR]
  [--dry-run]

Generate index.html for each first-level subdirectory under a CI run prefix.

For each subdirectory found under {run_id}-{platform}/:
  - {subdir}/index.html  -- recursive listing of all files under that subdir

In CI (no --output-dir): subdirectories are discovered by listing S3 objects
under the run prefix. Index files are uploaded to S3.

In local mode (--output-dir): subdirectories are discovered by scanning the
local staging directory. Index files are written to the same directory tree.

AWS credentials are resolved through boto3's default credential chain.
"""

import argparse
import html
import os
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import tempfile

PLATFORM = platform.system().lower()

# _therock_utils is a sibling package in the same build_tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from _therock_utils.storage_location import StorageLocation


def log(*args):
    print(*args)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


@dataclass
class _FileEntry:
    name: str
    href: str
    size_bytes: int  # -1 if unknown
    last_modified: datetime | None


def _pretty_size(size_bytes: int) -> str:
    if size_bytes < 0:
        return "&mdash;"
    for factor, suffix in [
        (1024**5, " PB"),
        (1024**4, " TB"),
        (1024**3, " GB"),
        (1024**2, " MB"),
        (1024**1, " KB"),
        (1024**0, " B"),
    ]:
        if size_bytes >= factor:
            return f"{int(size_bytes / factor)}{suffix}"
    return f"{size_bytes} B"


_HTML_STYLE = """\
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    * { padding: 0; margin: 0; }
    body { font-family: sans-serif; background-color: #ffffff; }
    a { color: #006ed3; text-decoration: none; }
    a:hover { color: #319cff; }
    header { padding: 25px 5% 15px 5%; background-color: #f2f2f2; }
    h1 { font-size: 20px; font-weight: normal; color: #999; }
    h1 a { color: #000; margin: 0 4px; }
    main { display: block; }
    table { width: 100%; border-collapse: collapse; }
    tr { border-bottom: 1px dashed #dadada; }
    tbody tr:hover { background-color: #ffffec; }
    th, td { text-align: left; padding: 10px 0; }
    th { padding: 15px 0; font-size: 16px; white-space: nowrap; }
    td { font-size: 14px; }
    td:nth-child(1) { padding-left: 5%; width: 60%; word-break: break-all; }
    td:nth-child(2) { width: 15%; padding: 0 20px; }
    td:nth-child(3) { width: 20%; padding-right: 5%; text-align: right; }
    th:nth-child(1) { padding-left: 5%; }
    th:nth-child(3) { text-align: right; padding-right: 5%; }
    </style>
</head>"""


def _generate_index_html(
    title: str, entries: list[_FileEntry], parent_href: str | None
) -> str:
    """Generate an HTML index page for a list of file entries."""
    lines = [
        _HTML_STYLE,
        "<body>",
        f"<header><h1>{html.escape(title)}</h1></header>",
        "<main><table>",
        "<thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead>",
        "<tbody>",
    ]
    if parent_href:
        lines.append(
            f'<tr><td><a href="{parent_href}">..</a></td>'
            f"<td>&mdash;</td><td>&mdash;</td></tr>"
        )
    for entry in entries:
        size_str = _pretty_size(entry.size_bytes)
        if entry.last_modified:
            mod_iso = entry.last_modified.isoformat()
            mod_str = entry.last_modified.strftime("%Y-%m-%d %H:%M UTC")
            mod_cell = f'<time datetime="{html.escape(mod_iso)}">{mod_str}</time>'
        else:
            mod_cell = "&mdash;"
        lines.append(
            f'<tr><td><a href="{quote(entry.href)}">{html.escape(entry.name)}</a></td>'
            f"<td>{size_str}</td><td>{mod_cell}</td></tr>"
        )
    lines += ["</tbody></table></main>", "</body></html>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# S3 listing helpers
# ---------------------------------------------------------------------------


def _list_files_s3(s3_client, bucket: str, dir_prefix: str) -> list[_FileEntry]:
    """List immediate contents under dir_prefix (non-recursive).

    Returns subdirectories (from CommonPrefixes) and files (from Contents),
    excluding index.html itself.
    """
    prefix = f"{dir_prefix}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    entries = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix) :]  # e.g. "gfx94X/"
            entries.append(
                _FileEntry(
                    name=name,
                    href=name + "index.html",
                    size_bytes=-1,
                    last_modified=None,
                )
            )
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix) :]
            # Skip subdirectory entries, empty keys, and index.html itself.
            if not filename or filename == "index.html" or "/" in filename:
                continue
            entries.append(
                _FileEntry(
                    name=filename,
                    href=filename,
                    size_bytes=obj["Size"],
                    last_modified=obj["LastModified"],
                )
            )
    entries.sort(key=lambda e: e.name)
    return entries


def _discover_dirs_with_files_s3(s3_client, bucket: str, run_prefix: str) -> list[str]:
    """Discover all directories under run_prefix that contain files (any depth)."""
    prefix = f"{run_prefix}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    dirs: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix) :]
            if not filename or filename.endswith("/"):
                continue
            # The directory containing this file
            parent = key.rsplit("/", 1)[0]
            dirs.add(parent)
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Local listing helpers (for --output-dir mode)
# ---------------------------------------------------------------------------


def _list_files_local(staging_dir: Path, dir_prefix: str) -> list[_FileEntry]:
    """List immediate contents in {staging_dir}/{dir_prefix} (non-recursive).

    Returns subdirectories and files, excluding index.html itself.
    """
    root = staging_dir / dir_prefix
    if not root.is_dir():
        return []
    entries = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            entries.append(
                _FileEntry(
                    name=p.name + "/",
                    href=p.name + "/index.html",
                    size_bytes=-1,
                    last_modified=None,
                )
            )
        elif p.is_file() and p.name != "index.html":
            stat = p.stat()
            entries.append(
                _FileEntry(
                    name=p.name,
                    href=p.name,
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ),
                )
            )
    return entries


def _discover_dirs_with_files_local(staging_dir: Path, run_prefix: str) -> list[str]:
    """Discover all directories under {staging_dir}/{run_prefix} that contain files."""
    run_dir = staging_dir / run_prefix
    if not run_dir.is_dir():
        return []
    dirs: set[str] = set()
    for p in run_dir.rglob("*"):
        if p.is_file() and p.name != "index.html":
            rel_dir = p.parent.relative_to(staging_dir).as_posix()
            dirs.add(rel_dir)
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Index generation and upload
# ---------------------------------------------------------------------------


def _upload_html(
    html_content: str, dest: StorageLocation, backend: StorageBackend, dry_run: bool
) -> None:
    """Write html to a temp file and upload it to dest."""
    if dry_run:
        return
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(html_content)
        tmp_path = Path(tmp.name)
    try:
        backend.upload_file(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_index_for_directory(
    bucket: str,
    dir_prefix: str,
    backend: StorageBackend,
    *,
    s3_client=None,
    staging_dir: Path | None = None,
    dry_run: bool = False,
    parent_href: str | None = "../index.html",
) -> None:
    """Generate and upload index.html listing all files under dir_prefix.

    Args:
        bucket: S3 bucket name.
        dir_prefix: Path prefix relative to the bucket root (e.g., '12345-linux/logs').
        backend: Storage backend for uploading the index.
        s3_client: Boto3 S3 client (required when staging_dir is None).
        staging_dir: Local staging directory root (used instead of S3 for local testing).
        dry_run: If True, log what would be uploaded without actually uploading.
        parent_href: href for the parent directory link, or None to omit it (e.g.
            when dir_prefix is the run root and has no indexed parent).
    """
    if staging_dir is not None:
        entries = _list_files_local(staging_dir, dir_prefix)
    else:
        entries = _list_files_s3(s3_client, bucket, dir_prefix)

    title = dir_prefix.rsplit("/", 1)[-1]
    html_content = _generate_index_html(
        title=title, entries=entries, parent_href=parent_href
    )
    dest = StorageLocation(bucket=bucket, relative_path=f"{dir_prefix}/index.html")
    log(
        f"[INFO] Uploading index ({len(entries)} files) → "
        f"{dest.s3_uri if staging_dir is None else dest.relative_path}"
    )
    _upload_html(html_content, dest, backend, dry_run)


def run(args) -> None:
    # WorkflowOutputRoot is only needed in the CLI path (to resolve bucket from GHA env).
    from _therock_utils.workflow_outputs import WorkflowOutputRoot

    output_root = (
        WorkflowOutputRoot.for_local(run_id=args.run_id, platform=PLATFORM)
        if args.output_dir is not None
        else WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform=PLATFORM)
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    prefix = output_root.prefix
    bucket = args.bucket if args.bucket else output_root.bucket

    staging_dir = args.output_dir
    s3_client = None

    if staging_dir is None:
        import boto3

        s3_client = boto3.client("s3")
        log(f"[INFO] Discovering directories from S3 prefix: {prefix}/")
        dirs = _discover_dirs_with_files_s3(s3_client, bucket, prefix)
    else:
        log(f"[INFO] Discovering directories from local dir: {staging_dir / prefix}/")
        dirs = _discover_dirs_with_files_local(staging_dir, prefix)

    if not dirs:
        log("[WARN] No directories with files found. Nothing to index.")
        return

    log(f"[INFO] Found directories: {dirs}")
    for dir_prefix in dirs:
        log(f"\n[INFO] Generating index for: {dir_prefix}")
        generate_index_for_directory(
            bucket=bucket,
            dir_prefix=dir_prefix,
            backend=backend,
            s3_client=s3_client,
            staging_dir=staging_dir,
            dry_run=args.dry_run,
            parent_href=None if dir_prefix == prefix else "../index.html",
        )

    log("\n[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate S3 index files after uploads"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID of the workflow run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local staging directory instead of S3 (for testing)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override the S3 bucket (default: resolved from GHA environment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()
    run(args)
