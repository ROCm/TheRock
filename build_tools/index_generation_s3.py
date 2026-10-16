#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generate an index.html page listing the contents of an S3 bucket prefix.

This script can be used in two ways:
 * As a CLI tool, to generate (and optionally upload) an index.html for local
   inspection or for publishing to S3.
 * As an importable library, for callers such as CI jobs that need to
   (re)generate an index after uploading or deleting artifacts.

Two kinds of index pages are supported:
 * Artifact indexes list the files directly under a prefix, such as
   `.tar.gz` tarballs or `.msi` Windows installers, with sorting and,
   optionally, a GPU-family filter.
 * Directory indexes list the direct child "directories" under a prefix
   (S3 has no real directories, so objects are grouped by common prefix).
   This is useful for browsing between different builds or versions.

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

Generate index.html for Windows installers (.msi) under a prefix:
 ./index_generation_s3.py --bucket therock-repo-amd-stable-staging-extras-rocoptiq --directory v5/rocm/extras/rocoptiq/windows-installers/win11 --windows-installer-index --upload
"""

import argparse
import html
import json
import logging
import os
import re
import sys
import urllib.parse

from collections.abc import Iterator

import boto3
import botocore.client
from botocore.exceptions import NoCredentialsError, ClientError

log = logging.getLogger(__name__)


def _map_client_error(e: ClientError, *, bucket_name: str, context: str) -> None:
    """Raise a standard exception for a ClientError based on its AWS error code.

    context is "access" (listing objects) or "upload" (uploading index.html),
    used to build S3-operation-specific messages. Maps AccessDenied/
    UnauthorizedOperation -> PermissionError and NoSuchBucket ->
    FileNotFoundError; any other ClientError is logged and re-raised as-is.
    """
    code = e.response.get("Error", {}).get("Code")
    message = e.response.get("Error", {}).get("Message")
    if context == "upload":
        denied_message = f"Access denied uploading to bucket '{bucket_name}'"
        not_found_message = f"Bucket '{bucket_name}' not found during upload"
    else:
        denied_message = f"Access denied to bucket '{bucket_name}'"
        not_found_message = f"Bucket '{bucket_name}' not found"
    if code in {"AccessDenied", "UnauthorizedOperation"}:
        raise PermissionError(f"{denied_message}: {message}") from e
    if code == "NoSuchBucket":
        raise FileNotFoundError(f"{not_found_message}: {message}") from e
    if context == "upload":
        log.exception("Failed to upload index.html to bucket '%s': %s", bucket_name, e)
    else:
        log.exception("ClientError while accessing bucket '%s': %s", bucket_name, e)
    raise e


def _paginate_list_objects_v2(
    s3_client: botocore.client.BaseClient, bucket_name: str, **paginate_kwargs
) -> Iterator[dict]:
    """Yield list_objects_v2 pages, mapping common S3 errors.

    A generator so that the paginator's lazy, per-page S3 calls happen inside
    this function's try/except, rather than in the caller's for loop.
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, **paginate_kwargs):
            yield page
    except NoCredentialsError:
        # Preserve specific exception type for callers to handle
        log.exception(
            "AWS credentials not found when accessing bucket '%s'", bucket_name
        )
        raise
    except ClientError as e:
        _map_client_error(e, bucket_name=bucket_name, context="access")


def _publish_html(
    s3_client: botocore.client.BaseClient,
    bucket_name: str,
    upload_key: str,
    html_content: str,
    upload: bool,
) -> str:
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
            _map_client_error(e, bucket_name=bucket_name, context="upload")

        region = s3_client.meta.region_name
        bucket_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{upload_key}"
        log.info("index.html successfully uploaded. URL: %s", bucket_url)
        return bucket_url

    # Local-only mode: write next to the caller's working directory.
    local_path = os.path.abspath("index.html")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info(
        "index.html generated successfully for bucket '%s'. File saved as %s",
        bucket_name,
        local_path,
    )
    return local_path


class _S3IndexGenerator:
    """Shared plumbing for building and publishing an S3 index.html page.

    Subclasses collect the entries to list (artifact files, child
    directories, ...) and call `_render_page()` / `_publish()` to produce and
    upload the resulting index.html.
    """

    def __init__(self, s3_client: botocore.client.BaseClient, bucket_name: str) -> None:
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def _paginate(self, **paginate_kwargs) -> Iterator[dict]:
        return _paginate_list_objects_v2(
            self.s3_client, self.bucket_name, **paginate_kwargs
        )

    def _render_page(
        self, page_title: str, body: str, style: str = "", script: str = ""
    ) -> str:
        return f"""
    <html>
    <head>
        <title>{html.escape(page_title)}</title>
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
            {style}
        </style>
        {script}
    </head>
    <body>
        <h1>{html.escape(page_title)}</h1>
        {body}
    </body>
    </html>
    """

    def _publish(self, upload_key: str, html_content: str, upload: bool) -> str:
        return _publish_html(
            self.s3_client, self.bucket_name, upload_key, html_content, upload
        )


class ArtifactIndexGenerator(_S3IndexGenerator):
    """Builds an index.html listing the files (not subfolders) under a prefix.

    For example, it lists every .tar.gz tarball or every .msi Windows
    installer directly under a prefix. The file suffix to match, the page
    title, and whether to show the GPU-family filter are all configurable,
    so the same class is reused for different artifact types.
    """

    def __init__(
        self,
        s3_client: botocore.client.BaseClient,
        bucket_name: str,
        artifact_suffix: str,
        show_filter: bool,
    ) -> None:
        """
        Args:
            s3_client: boto3 S3 client used to list and upload objects.
            bucket_name: Name of the S3 bucket to read from and write to.
            artifact_suffix: File ending that marks a file as a match, e.g.
                ".tar.gz" or ".msi". Only files ending in this suffix are
                listed; everything else under the prefix is ignored.
            show_filter: Whether to show the GPU-family dropdown filter on
                the page. Set to False for artifact types (like Windows
                installers) that aren't split by GPU family.
        """
        super().__init__(s3_client, bucket_name)
        self.artifact_suffix = artifact_suffix
        self.show_filter = show_filter

    @staticmethod
    def _extract_gpu_families(files: list[tuple[str, float]]) -> list[str]:
        """Find the GPU family name (e.g. "gfx90a") in each file name, if any.

        Only matches "gfx" followed by digits and optional trailing letters,
        so a name like "gfxsomething" is not picked up as a family.
        """
        gpu_family_pattern = re.compile(r"gfx\d+[A-Za-z]*", re.IGNORECASE)
        gpu_families = set()
        for file_name, _ in files:
            match = gpu_family_pattern.search(file_name)
            if match:
                gpu_families.add(match.group(0))
        return sorted(gpu_families)

    def generate(
        self,
        prefix: str,
        upload: bool = False,
        allow_empty: bool = False,
        page_title: str = "Artifact index",
        empty_message: str = "No artifacts available.",
    ) -> str:
        """Generate index.html for direct-child artifacts at s3://bucket_name/prefix.

        With upload=True the index is PUT to s3://bucket_name/<prefix>/index.html
        and the HTTPS URL is returned. Otherwise the index is written to
        ./index.html and the local path is returned.

        Raises FileNotFoundError when the prefix has no matching files unless
        allow_empty=True. Empty indexes are useful for event-driven regeneration
        after the final artifact under a product-local prefix is deleted.
        """
        # Strip any leading or trailing slash from the prefix, so it matches
        # the directory path used to build the S3 prefix below.
        prefix = prefix.strip("/")
        list_prefix = f"{prefix}/" if prefix else ""
        # Delimiter="/" asks S3 to only return direct children, instead of
        # the whole subtree, so there's no need to filter out nested keys
        # ourselves.
        page_iterator = self._paginate(Prefix=list_prefix, Delimiter="/")

        files = []
        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(self.artifact_suffix):
                    # Only append the filename without the full path.
                    files.append(
                        (
                            key.removeprefix(list_prefix),
                            obj["LastModified"].timestamp(),
                        )
                    )

        if not files and not allow_empty:
            raise FileNotFoundError(
                f"No {self.artifact_suffix} files found in bucket {self.bucket_name} "
                f"under prefix '{prefix}'."
            )

        # Prepare filter options and files array for JS.
        gpu_families = self._extract_gpu_families(files) if self.show_filter else []
        if self.show_filter:
            log.info(
                "Detected GPU families (%d): %s",
                len(gpu_families),
                ", ".join(gpu_families) if gpu_families else "none",
            )
        gpu_families_options = "".join(
            [
                f'<option value="{html.escape(family)}">{html.escape(family)}</option>'
                for family in gpu_families
            ]
        )
        files_js_array = json.dumps(
            [{"name": f[0], "mtime": f[1]} for f in files]
        ).replace("</", "<\\/")
        empty_message_js = json.dumps(empty_message).replace("</", "<\\/")
        log.info(
            "Found %d %s files in bucket '%s'.",
            len(files),
            self.artifact_suffix,
            self.bucket_name,
        )

        if self.show_filter:
            filter_html = f"""
            <label for="filter">Filter by:</label>
            <select id="filter">
                <option value="all">All</option>
                <option value="multiarch">Combined (all GFX)</option>
                {gpu_families_options}
            </select>
            """
        else:
            filter_html = ""

        style = """
            select { margin-bottom: 10px; padding: 5px; font-size: 16px; }
            .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
            label { font-weight: bold; }
        """

        script = f"""
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
                    li.textContent = {empty_message_js};
                    ul.appendChild(li);
                    return;
                }}

                fileList.forEach(file => {{
                    const li = document.createElement('li');
                    const href = encodeURIComponent(file.name).replace(/%2F/g, '/');
                    const a = document.createElement('a');
                    a.href = href;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    a.textContent = file.name;
                    li.appendChild(a);
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
        """

        body = f"""
        <div class="controls">
            <label for="sortOrder">Sort by:</label>
            <select id="sortOrder">
                <option value="desc">Last Updated (Recent to Old)</option>
                <option value="asc">First Updated (Old to Recent)</option>
            </select>
            {filter_html}
        </div>
        <ul id="fileList"></ul>
        """

        html_content = self._render_page(page_title, body, style=style, script=script)

        upload_key = f"{list_prefix}index.html"

        return self._publish(upload_key, html_content, upload)


class DirectoryIndexGenerator(_S3IndexGenerator):
    """Builds an index.html listing the "directories" immediately under a prefix.

    S3 has no real directories: a key like "v5/rocm/extras/rvs/foo.tar.gz" is
    just one object whose name happens to contain "/". This class groups
    objects by the next path segment after the prefix, so that, for example,
    under prefix "v5/rocm" with objects at "v5/rocm/extras/rvs/foo.tar.gz"
    and "v5/rocm/other/bar.tar.gz", the listed "directories" are "extras/"
    and "other/" — not "extras/rvs/", which is one level deeper (a
    grandchild, not a direct child). Each entry links to a deeper index page
    one level down, letting users navigate the tree one segment at a time.
    """

    def generate(
        self,
        prefix: str,
        upload: bool = False,
        allow_empty: bool = False,
        page_title: str = "Directory index",
        empty_message: str = "No directories available.",
    ) -> str:
        """Generate index.html for direct child directories at s3://bucket_name/prefix.

        Raises FileNotFoundError when the prefix has no child directories unless
        allow_empty=True.
        """

        prefix = prefix.strip("/")
        list_prefix = f"{prefix}/" if prefix else ""

        page_iterator = self._paginate(Prefix=list_prefix, Delimiter="/")

        directories = []

        for page in page_iterator:
            for common_prefix in page.get("CommonPrefixes", []):
                directory = common_prefix["Prefix"].removeprefix(list_prefix)
                directories.append(directory.rstrip("/") + "/")

        directories.sort()

        if not directories and not allow_empty:
            raise FileNotFoundError(
                f"No child directories found in bucket {self.bucket_name} "
                f"under prefix '{prefix}'."
            )

        log.info(
            "Found %d child directories in bucket '%s' under prefix '%s'.",
            len(directories),
            self.bucket_name,
            prefix,
        )

        directories_html = "".join(
            f'<li><a href="{html.escape(urllib.parse.quote(directory))}">'
            f"{html.escape(directory)}</a></li>\n"
            for directory in directories
        )

        if not directories_html:
            directories_html = f"<li>{html.escape(empty_message)}</li>"

        body = f"""
        <ul>
            {directories_html}
        </ul>
        """

        html_content = self._render_page(page_title, body)

        upload_key = f"{list_prefix}index.html"

        return self._publish(upload_key, html_content, upload)


def generate_tarball_index_s3(
    s3_client: botocore.client.BaseClient,
    bucket_name: str,
    prefix: str,
    upload: bool = False,
    allow_empty: bool = False,
    page_title: str = "Tarball index",
    empty_message: str = "No tarballs available.",
) -> str:
    """Generate index.html listing .tar.gz tarballs at s3://bucket_name/prefix."""
    return ArtifactIndexGenerator(
        s3_client, bucket_name, artifact_suffix=".tar.gz", show_filter=True
    ).generate(
        prefix,
        upload=upload,
        allow_empty=allow_empty,
        page_title=page_title,
        empty_message=empty_message,
    )


def generate_windows_installer_index_s3(
    s3_client: botocore.client.BaseClient,
    bucket_name: str,
    prefix: str,
    upload: bool = False,
    allow_empty: bool = False,
    page_title: str = "Windows Installers",
    empty_message: str = "No Windows installers available.",
) -> str:
    """Generate index.html listing .msi Windows installers at s3://bucket_name/prefix."""
    return ArtifactIndexGenerator(
        s3_client, bucket_name, artifact_suffix=".msi", show_filter=False
    ).generate(
        prefix,
        upload=upload,
        allow_empty=allow_empty,
        page_title=page_title,
        empty_message=empty_message,
    )


def generate_directory_index_s3(
    s3_client: botocore.client.BaseClient,
    bucket_name: str,
    prefix: str,
    upload: bool = False,
    allow_empty: bool = False,
    page_title: str = "Directory index",
    empty_message: str = "No directories available.",
) -> str:
    """Generate index.html listing direct child directories at s3://bucket_name/prefix.

    Raises FileNotFoundError when the prefix has no child directories unless
    allow_empty=True.
    """
    return DirectoryIndexGenerator(s3_client, bucket_name).generate(
        prefix,
        upload=upload,
        allow_empty=allow_empty,
        page_title=page_title,
        empty_message=empty_message,
    )


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate index.html for an S3 bucket prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  Generate index.html for all tarballs in a bucket to test locally:
    ./index_generation_s3.py --bucket therock-dev-tarball

  Generate index.html for all tarballs in a bucket and upload:
    ./index_generation_s3.py --bucket therock-dev-tarball --upload

  Generate index.html for direct child directories under a prefix:
    ./index_generation_s3.py --bucket therock-dev-tarball \\
        --directory v5/rocm/extras/rocoptiq/windows-installers \\
        --directory-index --upload

  Generate index.html for Windows installers (.msi) under a prefix:
    ./index_generation_s3.py --bucket therock-repo-amd-stable-staging-extras-rocoptiq \\
        --directory v5/rocm/extras/rocoptiq/windows-installers/win11 \\
        --windows-installer-index --upload
""",
    )
    parser.add_argument(
        "--bucket",
        default="therock-dev-tarball",
        help="S3 bucket name (default: therock-dev-tarball)",
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region name (default: us-east-2, matching therock-dev-tarball)",
    )
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
    index_type_group = parser.add_mutually_exclusive_group()
    index_type_group.add_argument(
        "--directory-index",
        action="store_true",
        help=(
            "List direct child subdirectories under --directory instead of "
            "artifact files (default: list tarballs)."
        ),
    )
    index_type_group.add_argument(
        "--windows-installer-index",
        action="store_true",
        help=(
            "List .msi Windows installers under --directory instead of "
            ".tar.gz tarballs (default: list tarballs)."
        ),
    )
    args = parser.parse_args(argv)
    s3 = boto3.client("s3", region_name=args.region)
    if args.directory_index:
        generate_directory_index_s3(
            s3_client=s3,
            bucket_name=args.bucket,
            prefix=args.directory,
            upload=args.upload,
        )
    elif args.windows_installer_index:
        generate_windows_installer_index_s3(
            s3_client=s3,
            bucket_name=args.bucket,
            prefix=args.directory,
            upload=args.upload,
        )
    else:
        generate_tarball_index_s3(
            s3_client=s3,
            bucket_name=args.bucket,
            prefix=args.directory,
            upload=args.upload,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
