#!/usr/bin/env python
"""
Script to generate an index.html listing .tar.gz files in an S3 bucket, performing the following:
 * Lists .tar.gz files in the specified S3 bucket.
 * Generates HTML page with sorting and filtering options
 * Saves the HTML locally as index.html
 * Uploads index.html back to the same S3 bucket

Requirements:
 * `boto3` Python package must be installed, e.g.: pip install boto3

Examples:
Generate index.html for all tarballs in a bucket:
./index_generation_s3_tar.py --bucket <bucketname>


Generate index.html for files under a prefix and in a specific region:
./index_generation_s3_tar.py --bucket <bucketname> --prefix nightly/ --region us-east-1

"""

import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import re
import json
from github_actions.github_actions_utils import *


def extract_gpu_details(files):
    gpu_family_pattern = re.compile(r"gfx(?:\d+[A-Za-z]*|\w+)")
    gpu_families = set()

    for file_name, _ in files:
        match = gpu_family_pattern.search(file_name)
        if match:
            gpu_families.add(match.group(0))

    return list(gpu_families)


def generate_index_s3(bucket_name, region_name="us-east-2", prefix=""):
    # Initialize S3 client
    s3 = boto3.client("s3", region_name=region_name)

    try:
        # List objects in the S3 bucket
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except NoCredentialsError:
        raise Exception(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )
    except ClientError as e:
        raise Exception(f"Error accessing bucket {bucket_name}: {e}")

    if "dev" in bucket_name.lower():
        page_title = "ROCm SDK dev tarballs"
    elif "nightly" in bucket_name.lower():
        page_title = "ROCm SDK nightly tarballs"
    else:
        page_title = "ROCm SDK tarballs"

    if "Contents" not in response:
        raise Exception(
            f"No objects found in bucket {bucket_name} with prefix '{prefix}'."
        )

    files = [
        (obj["Key"], obj["LastModified"].timestamp())
        for obj in response["Contents"]
        if obj["Key"].endswith(".tar.gz")
    ]

    if not files:
        raise Exception(
            f"No .tar.gz files found in bucket {bucket_name} with prefix '{prefix}'."
        )

    # Extract GPU family names from files
    gpu_families = extract_gpu_details(files)
    gpu_families_options = "".join(
        [f'<option value="{family}">{family}</option>' for family in gpu_families]
    )

    # Perpare array of file details for HTML rendering
    files_js_array = str([{"name": f[0], "mtime": f[1]} for f in files]).replace(
        "'", '"'
    )

    # HTML content for displaying files
    html_content = f"""
    <html>
    <head>
        <title>{page_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            h1 {{ color: #0056b3; }}
            select {{ margin-bottom: 10px; padding: 5px; font-size: 16px; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 5px; padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }}
            a {{ text-decoration: none; color: #0056b3; }}
            a:hover {{ color: #003d82; }}
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
                    const urlEncodedKey = encodeURIComponent(file.name).replace(/%2F/g, '/');
                    const s3Url = `https://{bucket_name}.s3.{region_name}.amazonaws.com/${{urlEncodedKey}}`;
                    li.innerHTML = `<a href="${{s3Url}}" target="_blank">${{file.name}}</a>`;
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
        <div>
            <label for="sortOrder">Sort by: </label>
            <select id="sortOrder">
                <option value="desc">Last Updated (Recent to Old)</option>
                <option value="asc">First Updated (Old to Recent)</option>
            </select>
            <label for="filter">Filter by: </label>
            <select id="filter">
                <option value="all">All</option>
                {gpu_families_options}
            </select>
        </div>
        <ul id="fileList"></ul>
    </body>
    </html>
    """

    local_path = "index.html"
    with open(local_path, "w") as f:
        f.write(html_content)

    message = f"index.html generated successfully for bucket '{bucket_name}'. File saved as {local_path}"
    gha_append_step_summary(message)
    print(message)

    try:
        # Upload the index.html to S3 bucket
        s3.upload_file(
            local_path,
            bucket_name,
            "index.html",
            ExtraArgs={"ContentType": "text/html"},
        )
        message = f"index.html successfully uploaded to bucket '{bucket_name}'."
        gha_append_step_summary(message)
        print(message)
    except ClientError as e:
        message = f"Failed to upload index.html to bucket '{bucket_name}': {e}"
        gha_append_step_summary(message)
        raise Exception(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate index.html for S3 bucket tar.gz files."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    args = parser.parse_args()

    region_name = "us-east-2"
    prefix = ""
    generate_index_s3(bucket_name=args.bucket, region_name=region_name, prefix=prefix)
