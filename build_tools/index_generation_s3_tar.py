#!/usr/bin/env python
"""
Script to generate an index.html listing .tar.gz files in an S3 bucket, performing the following:
 * Lists .tar.gz files in the specified S3 bucket.
 * Generates HTML page with sorting and filtering options
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
    for file_name, _ in files:
        match = gpu_family_pattern.search(file_name)
        if match:
            gpu_families.add(match.group(0))
    return sorted(list(gpu_families))


def generate_index_s3(s3_client, bucket_name, upload=False):
    # List all objects and select .tar.gz keys
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)
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
            if key.endswith(".tar.gz"):
                files.append((key, obj["LastModified"].timestamp()))

    if not files:
        raise FileNotFoundError(f"No .tar.gz files found in bucket {bucket_name}.")

    # Page title
    bucket_lower = bucket_name.lower()
    if "dev" in bucket_lower:
        page_title = "ROCm SDK dev tarballs"
    elif "nightly" in bucket_lower or "nightlies" in bucket_lower:
        page_title = "ROCm SDK nightly tarballs"
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
    files_js_array = json.dumps([{"name": f[0], "mtime": f[1]} for f in files])
    gha_append_step_summary(f"Found {len(files)} .tar.gz files in bucket '{bucket_name}'.")

    # HTML content for displaying files
    html_content = f"""
    <html>
    <head>
        <title>{page_title}</title>
        <meta charset="utf-8"/>
        <meta http-equiv="x-ua-compatible" content="ie=edge"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            h1 {{ color: #0056b3; }}
            select {{ margin-bottom: 10px; padding: 5px; font-size: 16px; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 5px; padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }}
            a {{ text-decoration: none; color: #0056b3; word-break: break-all; }}
            a:hover {{ color: #003d82; }}
            .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
            label {{ font-weight: bold; }}
        </style>
        <script>
            const files = {files_js_array};
            function applyFilter(fileList, filter) {{
                if (filter === 'all') return fileList;
                return fileList.filter(file => file.name.includes(filter));
            }}
            function renderFiles(fileList) {{
                const ul = document.getElementById('fileList');
                ul.innerHTML = '';
                fileList.forEach(file => {{
                    const li = document.createElement('li');
                    const href = encodeURIComponent(file.name).replace(/%2F/g, '/');
                    li.innerHTML = `<a href="${{href}}" target="_blank" rel="noopener noreferrer">${{file.name}}</a>`;
                    ul.appendChild(li);
                }});
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
        <h1>{page_title}</h1>
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
        <ul id="fileList"></ul>
    </body>
    </html>
    """

    # Write locally
    local_path = "index.html"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    message = f"index.html generated successfully for bucket '{bucket_name}'. File saved as {local_path}"
    gha_append_step_summary(message)
    # Upload to bucket root
    if upload:
        try:
            s3_client.upload_file(
                local_path,
                bucket_name,
                "index.html",
                ExtraArgs={"ContentType": "text/html"},
            )
            
            # Generate a presigned URL (time-limited) using boto3
            presigned_url = s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket_name, "Key": "index.html"},
                ExpiresIn=3600,
            )

            message = f"index.html successfully uploaded. URL: {presigned_url}"
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
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    generate_index_s3(s3_client=s3, bucket_name=args.bucket, upload=args.upload)
