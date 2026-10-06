#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generate an index.html listing files in an S3 bucket.

Usable as both a CLI tool (for local inspection) and an importable library.
The library entry points are `generate_index_s3()` for flat artifact
listings and `generate_directory_index_s3()` for direct child directory
listings.

Requirements:
 * `boto3` Python package must be installed, e.g.: pip install boto3

Usage:
Running locally without specifying a bucket will use the default bucket "therock-dev-tarball":
 ./index_generation_s3.py

Generate index.html for all tarballs in a bucket to test locally:
 ./index_generation_s3.py --bucket therock-dev-tarball

Generate index.html for all tarballs in a bucket and upload:
 ./index_generation_s3.py --bucket therock-dev-tarball --upload

Generate index.html for direct child directories (not files) under a prefix:
 ./index_generation_s3.py --bucket therock-dev-tarball --directory v5/rocm/extras/rocoptiq/windows-installers --directory-index --upload
"""

import argparse
import json
import logging
import os
import re

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

log = logging.getLogger(__name__)


def _paginate_list_objects_v2(s3_client, bucket_name, **paginate_kwargs):
    """Return a list_objects_v2 page iterator, mapping common S3 errors."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        return paginator.paginate(Bucket=bucket_name, **paginate_kwargs)
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


def _publish_html(s3_client, bucket_name, upload_key, html_content, upload) -> str:
    """Upload html_content to s3://bucket_name/upload_key, or write it locally.

    With upload=True the HTML is PUT to s3://bucket_name/upload_key and the
    HTTPS URL is returned. Otherwise the HTML is written to ./index.html and
    the local path is returned.
    """
    if upload:
        # Upload directly from memory; avoids writing to a (potentially
        # read-only) filesystem when called from AWS Lambda.
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=upload_key,
                Body=html_content.encode("utf-8"),
                ContentType="text/html",
            )
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
            raise

        region = s3_client.meta.region_name or "us-east-2"
        if region == "us-east-2":
            bucket_url = f"https://{bucket_name}.s3.amazonaws.com/{upload_key}"
        else:
            bucket_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{upload_key}"
        log.info("index.html successfully uploaded. URL: %s", bucket_url)
        return bucket_url

    # Local-only mode: write next to the caller's working directory.
    local_path = "index.html"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info(
        "index.html generated successfully for bucket '%s'. File saved as %s",
        bucket_name,
        local_path,
    )
    return local_path


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


def generate_index_s3(
    s3_client,
    bucket_name,
    prefix: str,
    upload: bool = False,
    allow_empty: bool = False,
    artifact_suffix: str = ".tar.gz",
    page_title: str = "Tarball index",
    show_filter: bool = True,
    empty_message: str = "No tarballs available.",
) -> str:
    """Generate index.html for direct-child artifacts at s3://bucket_name/prefix.

    With upload=True the index is PUT to s3://bucket_name/<prefix>/index.html
    and the HTTPS URL is returned. Otherwise the index is written to
    ./index.html and the local path is returned.

    artifact_suffix controls which direct-child files are included.

    page_title controls the generated page title.

    show_filter controls whether the GPU-family filter is included in the
    generated HTML. It should remain True for tarballs and can be set to False
    for other artifact types such as Windows installers.

    empty_message controls the message shown when the generated file list is
    empty.

    Raises FileNotFoundError when the prefix has no matching files unless
    allow_empty=True. Empty indexes are useful for event-driven regeneration
    after the final artifact under a product-local prefix is deleted.
    """
    # Strip any leading or trailing slash from the prefix to standardize the directory path used to filter object.
    prefix = prefix.lstrip("/").rstrip("/")
    # List all objects and select matching direct-child files.
    page_iterator = _paginate_list_objects_v2(s3_client, bucket_name, Prefix=prefix)

    files = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(artifact_suffix) and os.path.dirname(key) == prefix:
                # Only append the filename without the full path.
                files.append(
                    (key.removeprefix(f"{prefix}/"), obj["LastModified"].timestamp())
                )

    if not files and not allow_empty:
        raise FileNotFoundError(
            f"No {artifact_suffix} files found in bucket {bucket_name} "
            f"under prefix '{prefix}'."
        )

    # Prepare filter options and files array for JS.
    gpu_families = extract_gpu_details(files) if show_filter else []
    if show_filter:
        log.info(
            "Detected GPU families (%d): %s",
            len(gpu_families),
            ", ".join(gpu_families) if gpu_families else "none",
        )
    gpu_families_options = "".join(
        [f'<option value="{family}">{family}</option>' for family in gpu_families]
    )
    files_js_array = json.dumps([{"name": f[0], "mtime": f[1]} for f in files])
    log.info(
        "Found %d %s files in bucket '%s'.",
        len(files),
        artifact_suffix,
        bucket_name,
    )

    if show_filter:
        filter_html = f"""
        <label for="filter">Filter by:</label>
        <select id="filter">
            <option value="all">All</option>
            <option value="multiarch">Multiarch</option>
            {gpu_families_options}
        </select>
        """
    else:
        filter_html = ""

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

                if (fileList.length === 0) {{
                    const li = document.createElement('li');
                    li.textContent = '{empty_message}';
                    ul.appendChild(li);
                    return;
                }}

                fileList.forEach(file => {{
                    const li = document.createElement('li');
                    const href = encodeURIComponent(file.name).replace(/%2F/g, '/');
                    li.innerHTML = `<a href="${{href}}" target="_blank" rel="noopener noreferrer">${{file.name}}</a>`;
                    ul.appendChild(li);
                }});
            }}
            function updateDisplay() {{
                const order = document.getElementById('sortOrder').value;
                const filterElement = document.getElementById('filter');
                const filter = filterElement ? filterElement.value : 'all';
                let sortedFiles = [...files].sort((a, b) => {{
                    return (order === 'desc') ? b.mtime - a.mtime : a.mtime - b.mtime;
                }});
                sortedFiles = applyFilter(sortedFiles, filter);
                renderFiles(sortedFiles);
            }}
            document.addEventListener('DOMContentLoaded', function() {{
                updateDisplay();
                document.getElementById('sortOrder').addEventListener('change', updateDisplay);
                const filterElement = document.getElementById('filter');
                if (filterElement) {{
                    filterElement.addEventListener('change', updateDisplay);
                }}
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
            {filter_html}
        </div>
        <ul id="fileList"></ul>
    </body>
    </html>
    """

    # Generate a prefix for the case that the index file should go to a subdirectory. Empty otherwise.
    upload_prefix = f"{prefix}/" if prefix else ""
    upload_key = f"{upload_prefix}index.html"

    return _publish_html(s3_client, bucket_name, upload_key, html_content, upload)


def generate_directory_index_s3(
    s3_client,
    bucket_name,
    prefix: str,
    upload: bool = False,
    page_title: str = "Directory index",
    empty_message: str = "No directories available.",
) -> str:
    """Generate index.html listing direct child directories."""

    prefix = prefix.lstrip("/").rstrip("/")

    page_iterator = _paginate_list_objects_v2(
        s3_client, bucket_name, Prefix=f"{prefix}/", Delimiter="/"
    )

    directories = []

    for page in page_iterator:
        for common_prefix in page.get("CommonPrefixes", []):
            directory = common_prefix["Prefix"].removeprefix(f"{prefix}/")
            directories.append(directory.rstrip("/") + "/")

    directories.sort()

    log.info(
        "Found %d child directories in bucket '%s' under prefix '%s'.",
        len(directories),
        bucket_name,
        prefix,
    )

    directories_html = "".join(
        f'<li><a href="{directory}">{directory}</a></li>\n' for directory in directories
    )

    if not directories_html:
        directories_html = f"<li>{empty_message}</li>"

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
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 5px; padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }}
            a {{ text-decoration: none; color: #0056b3; word-break: break-all; }}
            a:hover {{ color: #003d82; }}
        </style>
    </head>
    <body>
        <h1>{page_title}</h1>
        <ul>
            {directories_html}
        </ul>
    </body>
    </html>
    """

    upload_prefix = f"{prefix}/" if prefix else ""
    upload_key = f"{upload_prefix}index.html"

    return _publish_html(s3_client, bucket_name, upload_key, html_content, upload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate index.html for an S3 bucket prefix"
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
    parser.add_argument(
        "--directory-index",
        action="store_true",
        help=(
            "List direct child subdirectories under --directory instead of "
            "artifact files (default: list files)."
        ),
    )
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    if args.directory_index:
        generate_directory_index_s3(
            s3_client=s3,
            bucket_name=args.bucket,
            prefix=args.directory,
            upload=args.upload,
        )
    else:
        generate_index_s3(
            s3_client=s3, bucket_name=args.bucket, prefix=args.directory, upload=args.upload
        )
