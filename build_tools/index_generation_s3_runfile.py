#!/usr/bin/env python
"""
Script copied off of build_tools/index_generation_s3_tar.py

Script to generate a bare-bones index.html listing .run files in an S3 bucket.
 * Lists .run files in the specified S3 bucket.
 * Generates a simple HTML page with links to each file
 * Saves the HTML locally as index.html
 * Uploads index.html back to the same S3 bucket

Requirements:
 * `boto3` Python package must be installed, e.g.: pip install boto3

Usage:
Running locally without specifying a bucket will use the default bucket "therock-dev-runfile":
 ./index_generation_s3_runfile.py

Generate index.html for all runfiles in a bucket to test locally:
 ./index_generation_s3_runfile.py --bucket therock-dev-runfile

Generate index.html for all runfiles in a bucket and upload:
 ./index_generation_s3_runfile.py --bucket therock-dev-runfile --upload
"""

import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from urllib.parse import quote

try:
    from github_actions.github_actions_utils import gha_append_step_summary
except ImportError:
    # Fallback if not running in GitHub Actions context
    def gha_append_step_summary(message):
        print(message)

log = logging.getLogger(__name__)


def generate_index_s3(s3_client, bucket_name, prefix: str, upload=False):
    """Generate a bare-bones index.html for .run files in an S3 bucket.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        prefix: S3 prefix (directory) to list files from
        upload: Whether to upload index.html back to S3

    Returns:
        Path to the generated index.html file
    """
    # Strip any leading or trailing slash from the prefix
    prefix = prefix.lstrip("/").rstrip("/")

    # List all objects and select .run keys
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    except NoCredentialsError:
        log.exception(
            "AWS credentials not found when accessing bucket '%s'", bucket_name
        )
        raise
    except ClientError as e:
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
            if key.endswith(".run"):
                # Get just the filename, or relative path from prefix
                if prefix:
                    filename = key.removeprefix(f"{prefix}/")
                else:
                    filename = key
                # Store filename and last modified date
                last_modified = obj["LastModified"]
                files.append((filename, last_modified))

    if not files:
        raise FileNotFoundError(f"No .run files found in bucket {bucket_name}.")

    # Sort files by last modified date (newest first)
    files.sort(key=lambda x: x[1], reverse=True)

    # Page title based on bucket name
    bucket_lower = bucket_name.lower()
    if "dev" in bucket_lower:
        page_title = "ROCm SDK dev runfile installers"
    elif "nightly" in bucket_lower or "nightlies" in bucket_lower:
        page_title = "ROCm SDK nightly runfile installers"
    elif "prerelease" in bucket_lower:
        page_title = "ROCm SDK prerelease runfile installers"
    else:
        page_title = "ROCm SDK runfile installers"

    gha_append_step_summary(
        f"Found {len(files)} .run files in bucket '{bucket_name}'."
    )

    # Generate bare-bones HTML with links and dates
    links_html = ""
    for filename, last_modified in files:
        href = quote(filename, safe="/")
        date_str = last_modified.strftime("%Y-%m-%d %H:%M UTC")
        links_html += f'    <a href="{href}">{filename}</a> <small>({date_str})</small><br/>\n'

    html_content = f"""<!DOCTYPE html>
<html>
  <body>
    <h1>{page_title}</h1>
{links_html}  </body>
</html>
"""

    # Write locally
    local_path = "index.html"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    message = f"index.html generated successfully for bucket '{bucket_name}'. File saved as {local_path}"
    gha_append_step_summary(message)

    # Upload to bucket
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
            region = s3_client.meta.region_name or "us-east-1"
            if region == "us-east-1":
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
        description="Generate index.html for S3 bucket .run files"
    )
    parser.add_argument(
        "--bucket",
        default="therock-dev-runfile",
        help="S3 bucket name (default: therock-dev-runfile)",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region name")
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
