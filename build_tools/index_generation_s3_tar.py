#!/usr/bin/env python
"""
Script to generate an index.html listing .tar.gz files in an S3 bucket, performing the following:
 * Lists .tar.gz files in the specified S3 bucket.
 * Generates HTML page with sorting and filtering options
 * Displays a page generation timestamp
 * Displays per-artifact size and timestamp (S3 LastModified)
 * Saves the HTML locally as index.html
 * Uploads index.html back to the same S3 bucket

Requirements:
 * `boto3` Python package must be installed, e.g.: pip install boto3

Usage:
Running locally without specifying a bucket will use the default bucket "therock-dev-tarball":
 ./index_generation_s3_tar.py

Generate index.html for all tarballs in a bucket to test locally:
 ./index_generation_s3_tar.py --bucket therock-dev-tarball

Generate index.html for all tarballs in a bucket and upload:
 ./index_generation_s3_tar.py --bucket therock-dev-tarball --upload
"""

import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import re
import json
import logging
from datetime import datetime, timezone
from github_actions.github_actions_utils import gha_append_step_summary

log = logging.getLogger(__name__)


def extract_gpu_details(files):
    # Regex: r"gfx(?:\d+[A-Za-z]*|\w+)"
    # Matches "gfx" + digits with optional letters (e.g., gfx90a/gfx103) or a word token (e.g., gfx_ip).
    # Tweaks: require letter -> [A-Za-z]+; uppercase-only -> [A-Z]* or [A-Z]+; digit-led only -> remove |\w+.
    # Case-insensitive ("gfx"/"GFX"): add re.IGNORECASE.
    # Examples: gfx90a, gfx1150, gfx_ip, gfxX.
    gpu_family_pattern = re.compile(r"gfx(?:\d+[A-Za-z]*|\w+)", re.IGNORECASE)
    gpu_families = set()
    # Each entry in `files` is a 3-tuple: (name: str, mtime: int, size: int).
    # We only need the name here, but we must still unpack all three elements to avoid ValueError.
    for filename in files:
        # f is (name, mtime, size)
        file_name = filename[0]
        match = gpu_family_pattern.search(file_name)
        if match:
            gpu_families.add(match.group(0))
    return sorted(list(gpu_families))


def generate_index_s3(s3_client, bucket_name, prefix: str, upload=False):
    # Strip any leading or trailing slash from the prefix to standardize the directory path used to filter object.
    prefix = prefix.lstrip("/").rstrip("/")
    # List all objects and select .tar.gz keys
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    except NoCredentialsError:
        # Preserve specific exception type for callers to handle
        log.exception(
            "AWS credentials not found when accessing bucket '%s'", bucket_name
        )
        raise
    except ClientError as e:
        # Map common S3 errors to standard exceptions with chaining; otherwise re-raise
        code = e.response.get("Error", {}).get("Code")
        if code in {"AccessDenied", "UnauthorizedOperation"}:
            raise PermissionError(f"Access denied to bucket '{bucket_name}'") from e
        if code in {"NoSuchBucket", "404"}:
            raise FileNotFoundError(f"Bucket '{bucket_name}' not found") from e
        log.exception("ClientError while accessing bucket '%s'", bucket_name)
        raise

    files = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Only include files directly under the given prefix "directory"
            if key.endswith(".tar.gz") and os.path.dirname(key) == prefix:
                # Preserve structure relative to the prefix, rather than using basename
                display_name = key.removeprefix(f"{prefix}/") if prefix else key
                # Append a tuple for each .tar.gz file found in the specified directory:
                # (
                #   filename (str, relative to prefix if set),
                #   last modified time as epoch seconds (int, from S3 LastModified),
                #   file size in bytes (int, from S3 Size)
                # )
                files.append(
                    (
                        display_name,
                        int(
                            obj.get(
                                "LastModified", datetime.now(timezone.utc)
                            ).timestamp()
                        ),
                        int(obj.get("Size", 0)),
                    )
                )

    if not files:
        raise FileNotFoundError(f"No .tar.gz files found in bucket {bucket_name}.")

    # Page title
    bucket_lower = bucket_name.lower()
    if "dev" in bucket_lower:
        page_title = "ROCm SDK dev tarballs"
    elif "nightly" in bucket_lower or "nightlies" in bucket_lower:
        page_title = "ROCm SDK nightly tarballs"
    elif "prerelease" in bucket_lower:
        page_title = "ROCm SDK prerelease tarballs"
    else:
        page_title = "ROCm SDK tarballs"

    # Prepare filter options and files array for JS
    gpu_families = extract_gpu_details(files)
    message = (
        f"Detected GPU families ({len(gpu_families)}): "
        f"{', '.join(gpu_families) if gpu_families else 'none'}"
    )
    gha_append_step_summary(message)
    gpu_families_options = "".join(
        [f'<option value="{family}">{family}</option>' for family in gpu_families]
    )
    files_js_array = json.dumps(
        [{"name": f[0], "mtime": f[1], "size": f[2]} for f in files]
    )
    gha_append_step_summary(
        f"Found {len(files)} .tar.gz files in bucket '{bucket_name}'."
    )
    # Generation timestamp (UTC) with 'UTC' suffix
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # HTML content for displaying files
    html_content = f"""
    <html>
    <head>
        <title>{page_title}</title>
        <meta charset="utf-8"/>
        <meta http-equiv="x-ua-compatible" content="ie=edge"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
            :root {{
                --gap: 12px;
                /* Column widths: 1fr for artifact, size ~10ch, time ~22ch */
                --col-artifact: 1fr;
                --col-size: 10ch;
                --col-time: 22ch;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
                color: #333;
            }}
            .header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 16px;
            }}
            h1 {{ color: #0056b3; margin: 0; }}
            .timestamp {{
                font-size: 14px;
                color: #666;
                white-space: nowrap;
            }}
            .controls {{
                display: flex;
                gap: var(--gap);
                align-items: center;
                flex-wrap: wrap;
                margin-bottom: 8px;
            }}
            label {{ font-weight: bold; }}
            select {{ margin-bottom: 10px; padding: 5px; font-size: 16px; }}

            /* Shared grid for header and rows */
            .grid-row {{
                display: grid;
                grid-template-columns: var(--col-artifact) var(--col-size) var(--col-time);
                gap: var(--gap);
                align-items: center;
            }}
            /* Header row styling */
            .list-header {{
                margin: 8px 0 12px 0;
                padding: 10px;
                color: #555;
                font-weight: bold;
                /* Use same font as rows for better alignment */
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-variant-numeric: tabular-nums;
            }}
            ul {{ list-style-type: none; padding: 0; margin: 0; }}
            li {{
                margin-bottom: 5px;
                padding: 10px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            /* Apply grid to list items too */
            li.grid-row {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-variant-numeric: tabular-nums; /* Align digits */
            }}
            .file-link {{
                text-decoration: none;
                color: #0056b3;
                word-break: break-all;
                min-width: 0;
            }}
            .file-link:hover {{ color: #003d82; }}
            .col-size, .col-time {{
                white-space: nowrap;
                text-align: right;
                color: #666;
                font-size: 13px;
            }}
            /* Responsive: stack on narrow screens */
            @media (max-width: 720px) {{
                .grid-row {{
                    grid-template-columns: 1fr;
                }}
                .col-size, .col-time {{
                    text-align: left;
                }}
            }}
        </style>
        <script>
            const files = {files_js_array};
            function toUTCStringFromEpochSec(sec) {{
                // Format as YYYY-MM-DD HH:MM:SS UTC
                const dateObj = new Date(sec * 1000);
                const yyyy = dateObj.getUTCFullYear();
                const mm = String(dateObj.getUTCMonth() + 1).padStart(2, '0');
                const dd = String(dateObj.getUTCDate()).padStart(2, '0');
                const HH = String(dateObj.getUTCHours()).padStart(2, '0');
                const MM = String(dateObj.getUTCMinutes()).padStart(2, '0');
                const SS = String(dateObj.getUTCSeconds()).padStart(2, '0');
                return `${{yyyy}}-${{mm}}-${{dd}} ${{HH}}:${{MM}}:${{SS}} UTC`;
            }}

            function formatBytes(bytes) {{
                // Human-readable size, base-1024
                if (bytes == null || !Number.isFinite(Number(bytes))) return '—';
                bytes = Number(bytes);
                if (bytes === 0) return '0 B';
                const k = 1024;
                const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
                const unitIndex = Math.floor(Math.log(bytes) / Math.log(k));
                const value = bytes / Math.pow(k, unitIndex);
                return (unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)) + ' ' + units[unitIndex];
            }}

            function applyFilter(fileList, filter) {{
                if (filter === 'all') return fileList;
                return fileList.filter(file => file.name.includes(filter));
            }}

            function createFileRow(file) {{
                const listItem = document.createElement('li');
                listItem.className = 'grid-row';
                listItem.setAttribute('role', 'listitem');

                // Artifact link
                const artifactLink = document.createElement('a');
                const encodedHref = encodeURIComponent(file.name).replace(/%2F/g, '/');
                artifactLink.href = encodedHref;
                artifactLink.target = '_blank';
                artifactLink.rel = 'noopener noreferrer';
                artifactLink.className = 'file-link';
                artifactLink.textContent = file.name;

                // Size column
                const sizeColumn = document.createElement('span');
                sizeColumn.className = 'col-size';
                const sizeText = formatBytes(file.size);
                sizeColumn.textContent = sizeText;
                sizeColumn.setAttribute('aria-label', 'Size ' + sizeText);

                // Time column
                const timeColumn = document.createElement('span');
                timeColumn.className = 'col-time';
                const timeText = toUTCStringFromEpochSec(file.mtime);
                const rawSize = (file.size == null || !Number.isFinite(Number(file.size))) ? 'unknown' : String(file.size);
                timeColumn.textContent = timeText;
                timeColumn.title = 'S3 LastModified (UTC): ' + timeText + '\\nRaw size: ' + rawSize + ' bytes';
                timeColumn.setAttribute('aria-label', 'Time generated ' + timeText);

                // Assemble row
                listItem.appendChild(artifactLink);
                listItem.appendChild(sizeColumn);
                listItem.appendChild(timeColumn);

                return {{ element: listItem }};
            }}

            function renderFiles(fileList) {{
                const listElement = document.getElementById('fileList');
                listElement.innerHTML = '';
                const fragment = document.createDocumentFragment();
                fileList.forEach(file => {{
                    const {{ element }} = createFileRow(file);
                    fragment.appendChild(element);
                }});
                listElement.appendChild(fragment);
            }}

            function updateDisplay() {{
                const order = document.getElementById('sortOrder').value;
                const filter = document.getElementById('filter').value;
                let sortedFiles = [...files].sort((a, b) => {{
                    return (order === 'desc') ? b.mtime - a.mtime : a.mtime - b.mtime;
                }});
                sortedFiles = applyFilter(sortedFiles, filter);
                renderFiles(sortedFiles);
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                updateDisplay();
                document.getElementById('sortOrder').addEventListener('change', updateDisplay);
                document.getElementById('filter').addEventListener('change', updateDisplay);
            }});
        </script>
    </head>
    <body>
        <div class="header">
            <h1>{page_title}</h1>
            <div class="timestamp">Generated: {generated_at}</div>
        </div>
        <div class="controls">
            <label for="sortOrder">Sort by:</label>
            <select id="sortOrder">
                <option value="desc">Last Updated (Recent to Old)</option>
                <option value="asc">First Updated (Old to Recent)</option>
            </select>
            <label for="filter">Filter by:</label>
            <select id="filter">
                <option value="all">All</option>
                {gpu_families_options}
            </select>
        </div>

        <!-- Aligned column headers -->
        <div class="list-header grid-row" role="row">
            <span>Artifact</span>
            <span class="col-size">Size</span>
            <span class="col-time">Time generated</span>
        </div>

        <ul id="fileList" role="list" aria-label="Tarball artifacts"></ul>
    </body>
    </html>
    """

    # Write locally
    local_path = "index.html"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    message = f"index.html generated successfully for bucket '{bucket_name}'. File saved as {local_path}"
    gha_append_step_summary(message)
    # Upload to bucket
    # Generate a prefix for the case that the index file should go to a subdirectory. Empty otherwise.
    upload_prefix = f"{prefix}/" if prefix else ""
    if upload:
        try:
            s3_client.upload_file(
                local_path,
                bucket_name,
                f"{upload_prefix}index.html",
                ExtraArgs={"ContentType": "text/html"},
            )

            # URL to the uploaded index.html
            region = s3_client.meta.region_name or "us-east-2"
            if region == "us-east-2":
                bucket_url = (
                    f"https://{bucket_name}.s3.amazonaws.com/{upload_prefix}index.html"
                )
            else:
                bucket_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{upload_prefix}index.html"

            message = f"index.html successfully uploaded. URL: {bucket_url}"
            gha_append_step_summary(message)

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in {"AccessDenied", "UnauthorizedOperation"}:
                raise PermissionError(
                    f"Access denied uploading to bucket '{bucket_name}'"
                ) from e
            if code in {"NoSuchBucket", "404"}:
                raise FileNotFoundError(
                    f"Bucket '{bucket_name}' not found during upload"
                ) from e
            log.error("Failed to upload index.html to bucket '%s': %s", bucket_name, e)
            gha_append_step_summary(
                f"Failed to upload index.html to bucket '{bucket_name}': {e}"
            )
            raise

    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate index.html for S3 bucket .tar.gz files"
    )
    parser.add_argument(
        "--bucket",
        default="therock-dev-tarball",
        help="S3 bucket name (default: therock-dev-tarball)",
    )
    parser.add_argument("--region", default="us-east-2", help="AWS region name")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload index.html back to S3 (default: do not upload)",
    )
    parser.add_argument(
        "--directory",
        default="",
        help="Directory to index. Defaults to the top level directory.",
    )
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    generate_index_s3(
        s3_client=s3, bucket_name=args.bucket, prefix=args.directory, upload=args.upload
    )
