#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Usage:
generate_s3_index.py [-h]
  --run-id RUN_ID
  [--output-dir OUTPUT_DIR]
  [--dry-run]

Generate index.html files for artifact and log directories after all upload
jobs have completed.

For each artifact group found under the run prefix:
  - logs/{artifact_group}/index.html  -- listing of build log files
  - index-{artifact_group}.html       -- listing of .tar.xz artifact files

In CI (no --output-dir): artifact groups are discovered by listing S3 objects
under the run prefix. Index files are uploaded to S3.

In local mode (--output-dir): artifact groups are discovered by scanning the
local staging directory. Index files are written to the same directory tree.

AWS credentials are resolved through boto3's default credential chain.
"""

import argparse
import fnmatch
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import tempfile

PLATFORM = platform.system().lower()

# _therock_utils is a sibling package in the same build_tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
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


def _generate_index_html(title: str, entries: list[_FileEntry], parent_href: str | None) -> str:
    """Generate an HTML index page for a list of file entries."""
    lines = [
        _HTML_STYLE,
        "<body>",
        f"<header><h1>{_escape_html(title)}</h1></header>",
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
            mod_cell = f'<time datetime="{_escape_html(mod_iso)}">{mod_str}</time>'
        else:
            mod_cell = "&mdash;"
        lines.append(
            f'<tr><td><a href="{_escape_html(entry.href)}">{_escape_html(entry.name)}</a></td>'
            f"<td>{size_str}</td><td>{mod_cell}</td></tr>"
        )
    lines += ["</tbody></table></main>", "</body></html>"]
    return "\n".join(lines)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# S3 listing helpers
# ---------------------------------------------------------------------------


def _list_s3_objects(s3_client, bucket: str, prefix: str) -> list[dict]:
    """List all S3 objects under prefix. Returns list of {key, size, last_modified}."""
    objects: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                }
            )
    return objects


def _discover_artifact_groups_s3(s3_client, bucket: str, run_prefix: str) -> list[str]:
    """Discover artifact groups by listing objects under {run_prefix}/logs/."""
    logs_prefix = f"{run_prefix}/logs/"
    objects = _list_s3_objects(s3_client, bucket, logs_prefix)
    groups: set[str] = set()
    for obj in objects:
        # key looks like: {run_prefix}/logs/{group}/filename
        remainder = obj["key"][len(logs_prefix):]
        parts = remainder.split("/", 1)
        if len(parts) >= 1 and parts[0]:
            groups.add(parts[0])
    return sorted(groups)


def _build_log_entries_s3(s3_client, bucket: str, run_prefix: str, artifact_group: str) -> list[_FileEntry]:
    """List files under {run_prefix}/logs/{artifact_group}/ for the log index."""
    dir_prefix = f"{run_prefix}/logs/{artifact_group}/"
    objects = _list_s3_objects(s3_client, bucket, dir_prefix)
    entries = []
    for obj in objects:
        key = obj["key"]
        filename = key[len(dir_prefix):]
        # Skip index.html itself and subdirectory entries (contain "/")
        if not filename or filename == "index.html" or "/" in filename:
            continue
        entries.append(
            _FileEntry(
                name=filename,
                href=filename,
                size_bytes=obj["size"],
                last_modified=obj["last_modified"],
            )
        )
    entries.sort(key=lambda e: e.name)
    return entries


def _build_artifact_entries_s3(s3_client, bucket: str, run_prefix: str) -> list[_FileEntry]:
    """List *.tar.xz* files at the run root for the artifact index."""
    objects = _list_s3_objects(s3_client, bucket, f"{run_prefix}/")
    entries = []
    for obj in objects:
        key = obj["key"]
        filename = key[len(f"{run_prefix}/"):]
        # Only files directly under the run root (no subdirectory) matching *.tar.xz*
        if "/" in filename:
            continue
        if not (fnmatch.fnmatch(filename, "*.tar.xz") or fnmatch.fnmatch(filename, "*.tar.xz.*")):
            continue
        entries.append(
            _FileEntry(
                name=filename,
                href=filename,
                size_bytes=obj["size"],
                last_modified=obj["last_modified"],
            )
        )
    entries.sort(key=lambda e: e.name)
    return entries


# ---------------------------------------------------------------------------
# Local listing helpers (for --output-dir mode)
# ---------------------------------------------------------------------------


def _discover_artifact_groups_local(staging_dir: Path, run_prefix: str) -> list[str]:
    """Discover artifact groups by listing subdirectories of {staging_dir}/{prefix}/logs/."""
    logs_dir = staging_dir / run_prefix / "logs"
    if not logs_dir.is_dir():
        return []
    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir())


def _build_log_entries_local(staging_dir: Path, run_prefix: str, artifact_group: str) -> list[_FileEntry]:
    """List files in {staging_dir}/{prefix}/logs/{artifact_group}/."""
    log_dir = staging_dir / run_prefix / "logs" / artifact_group
    if not log_dir.is_dir():
        return []
    entries = []
    for p in sorted(log_dir.iterdir()):
        if p.is_file() and p.name != "index.html":
            stat = p.stat()
            entries.append(
                _FileEntry(
                    name=p.name,
                    href=p.name,
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
    return entries


def _build_artifact_entries_local(staging_dir: Path, run_prefix: str) -> list[_FileEntry]:
    """List *.tar.xz* files at {staging_dir}/{prefix}/."""
    root_dir = staging_dir / run_prefix
    if not root_dir.is_dir():
        return []
    entries = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_file():
            continue
        if fnmatch.fnmatch(p.name, "*.tar.xz") or fnmatch.fnmatch(p.name, "*.tar.xz.*"):
            stat = p.stat()
            entries.append(
                _FileEntry(
                    name=p.name,
                    href=p.name,
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Index generation and upload
# ---------------------------------------------------------------------------


def _upload_html(html: str, dest: StorageLocation, backend: StorageBackend, dry_run: bool) -> None:
    """Write html to a temp file and upload it to dest."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(html)
        tmp_path = Path(tmp.name)
    try:
        backend.upload_file(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_indexes_for_group(
    artifact_group: str,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
    *,
    s3_client=None,
    staging_dir: Path | None,
    dry_run: bool,
) -> None:
    """Generate and upload log + artifact indexes for one artifact group."""
    prefix = output_root.prefix
    bucket = output_root.bucket

    # --- Log index ---
    if staging_dir is not None:
        log_entries = _build_log_entries_local(staging_dir, prefix, artifact_group)
    else:
        log_entries = _build_log_entries_s3(s3_client, bucket, prefix, artifact_group)

    artifact_index_url = output_root.artifact_index(artifact_group).https_url
    log_html = _generate_index_html(
        title=f"logs / {artifact_group}",
        entries=log_entries,
        parent_href=artifact_index_url,
    )
    log_index_dest = output_root.log_index(artifact_group)
    log(f"[INFO] Uploading log index → {log_index_dest.s3_uri if staging_dir is None else log_index_dest.relative_path}")
    _upload_html(log_html, log_index_dest, backend, dry_run)

    # --- Artifact index ---
    if staging_dir is not None:
        artifact_entries = _build_artifact_entries_local(staging_dir, prefix)
    else:
        artifact_entries = _build_artifact_entries_s3(s3_client, bucket, prefix)

    artifact_html = _generate_index_html(
        title=f"artifacts / {artifact_group}",
        entries=artifact_entries,
        parent_href=None,
    )
    artifact_index_dest = output_root.artifact_index(artifact_group)
    log(f"[INFO] Uploading artifact index → {artifact_index_dest.s3_uri if staging_dir is None else artifact_index_dest.relative_path}")
    _upload_html(artifact_html, artifact_index_dest, backend, dry_run)


def run(args) -> None:
    output_root = (
        WorkflowOutputRoot.for_local(run_id=args.run_id, platform=PLATFORM)
        if args.output_dir is not None
        else WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform=PLATFORM)
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    staging_dir = args.output_dir
    s3_client = None

    if staging_dir is None:
        import boto3
        s3_client = boto3.client("s3")
        log(f"[INFO] Discovering artifact groups from S3 prefix: {output_root.prefix}/")
        artifact_groups = _discover_artifact_groups_s3(
            s3_client, output_root.bucket, output_root.prefix
        )
    else:
        log(f"[INFO] Discovering artifact groups from local dir: {staging_dir / output_root.prefix}/logs/")
        artifact_groups = _discover_artifact_groups_local(staging_dir, output_root.prefix)

    if not artifact_groups:
        log("[WARN] No artifact groups found. Nothing to index.")
        return

    log(f"[INFO] Found artifact groups: {artifact_groups}")
    for group in artifact_groups:
        log(f"\n[INFO] Generating indexes for artifact group: {group}")
        generate_indexes_for_group(
            artifact_group=group,
            output_root=output_root,
            backend=backend,
            s3_client=s3_client,
            staging_dir=staging_dir,
            dry_run=args.dry_run,
        )

    log("\n[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate S3 index files after uploads")
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
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()
    run(args)
