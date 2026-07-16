#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured product-local PEP 503 index generator.

Generates pip-compatible PEP 503 simple indexes for the stream-subdomain
structured layout, where each package lives in its own directory:

    <product>/<index>/<normalized-package>/<filename>
    <product>/<index>/<normalized-package>/index.html
    <product>/<index>/index.html            (product-local root listing)

`<index>` is `whl` or `whl-next`. Release streams are selected by hostname
(e.g. nightly.repo.amd.com), never encoded in the path, so this generator is
stream-agnostic and knows nothing about CloudFront routing, aggregate
`/rocm/whl/` roots, or S3 bucket -> URL mapping.

Unlike the legacy flat generator (`manage.py`), package identity comes from the
directory name (authoritative, PEP 503-normalized), package pages use
same-directory artifact links (no `../`), and no allow-list or version-threshold
filtering is applied: the directory contents are trusted.

Example usage:

    # Render indexes locally without uploading (the default; writes index.html
    # files under the cwd for inspection):
    python manage_structured.py core/whl --bucket my-python-bucket

    # Generate and upload indexes for every package under a product-local root:
    python manage_structured.py core/whl --bucket my-python-bucket --upload

    # Regenerate a single package directory (skips the product root index):
    python manage_structured.py core/whl --bucket my-python-bucket \
        --package numpy --upload

The bucket may also be supplied via the `S3_BUCKET_PY` environment variable;
an explicit `--bucket` takes precedence.
"""

import argparse
import base64
import concurrent.futures
import dataclasses
import html
import re
import time
from os import getenv, makedirs, path
from urllib.parse import quote

import boto3
import botocore.client
import botocore.exceptions
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)


# File extensions surfaced in the generated index. Anything else in a package
# directory (logs, READMEs, PEP 658 .metadata sidecars) is ignored.
ACCEPTED_FILE_EXTENSIONS = ("whl", "zip", "tar.gz")

# S3 multipart uploads produce composite checksums in the form base64==-N.
# These are not a valid SHA256 of the object and must be discarded.
_MULTIPART_CHECKSUM_RE = re.compile(r"^[A-Za-z0-9+/=]+=-[0-9]+$")

# Metadata fetch concurrency for S3 HEAD requests.
_METADATA_MAX_WORKERS = 6


def _package_name_from_filename(filename: str) -> str:
    """Return the PEP 503-normalized package name for a distribution artifact.

    Uses spec-aware parsing (packaging.utils) rather than a first-hyphen
    heuristic so that sdist/zip filenames with hyphens in the project name
    are handled correctly (e.g. ``llnl-hatchet-2024.1.tar.gz`` ->
    ``llnl-hatchet``).

    Raises:
        ValueError: if the filename cannot be parsed as a wheel or sdist.
    """
    if filename.endswith(".whl"):
        try:
            name, _, _, _ = parse_wheel_filename(filename)
            return name
        except InvalidWheelFilename as e:
            raise ValueError(f"Cannot parse wheel filename {filename!r}: {e}") from e
    try:
        name, _ = parse_sdist_filename(filename)
        return name
    except InvalidSdistFilename as e:
        raise ValueError(f"Cannot parse sdist filename {filename!r}: {e}") from e


@dataclasses.dataclass
class PackageFile:
    """A single distribution artifact inside a package directory.

    Attributes:
        key: Full S3 key (raw, unescaped). '+' is escaped to '%2B' only when
            rendered into an href.
        filename: Basename of the key (the same-directory link target).
        checksum: SHA256 hex digest, or None until fetched.
        pep658: SHA256 hex digest of the PEP 658 .metadata sidecar, or None.
        size: Object size in bytes, or None until fetched.
    """

    key: str
    filename: str
    checksum: str | None
    pep658: str | None
    size: int | None


@dataclasses.dataclass
class PackageDir:
    """A discovered package directory and its artifacts.

    Attributes:
        name: PEP 503-normalized package name (the directory name).
        files: Artifacts contained directly in the directory.
    """

    name: str
    files: list[PackageFile]


@dataclasses.dataclass
class IndexPage:
    """A rendered index.html and the S3 key it should be written to.

    Attributes:
        key: Destination S3 key (e.g. "pytorch/whl/torch/index.html").
        html: Rendered HTML body.
    """

    key: str
    html: str


def pep503_normalize(name: str) -> str:
    """Normalize a package name per PEP 503.

    Lowercase and collapse runs of "-", "_", and "." to a single "-".
    """
    return re.sub(r"[-_.]+", "-", name.lower())


def _accepted_file(filename: str) -> bool:
    return filename.endswith(ACCEPTED_FILE_EXTENSIONS)


def discover_packages(
    keys: list[str],
    root: str,
    package: str | None = None,
) -> list[PackageDir]:
    """Group S3 keys under a product root into package directories.

    Only keys of the exact shape `<root>/<package-dir>/<filename>` are
    considered (one level below the root). Files directly under the root,
    more deeply nested files, and index.html pages are ignored.

    Args:
        keys: Raw S3 keys to group. '+' is escaped to '%2B' only at render
            time, so filenames here carry the literal '+' local-version
            separator that the packaging parsers expect.
        root: The `<product>/<index>` prefix (no trailing slash required).
        package: If set, restrict discovery to this package directory only.

    Returns:
        Package directories sorted by name, each with files sorted by filename.

    Raises:
        ValueError: if a package directory name is not PEP 503-normalized, or
            if a contained file's normalized package token does not match its
            directory (a misfiled artifact).
    """
    root = root.rstrip("/")
    prefix = root + "/"
    grouped: dict[str, list[PackageFile]] = {}

    for key in keys:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        parts = remainder.split("/")
        # Require exactly <package-dir>/<filename>: one level below the root.
        if len(parts) != 2:
            continue
        pkg_dir, filename = parts
        if not _accepted_file(filename):
            continue

        normalized = pep503_normalize(pkg_dir)
        if normalized != pkg_dir:
            raise ValueError(
                f"Package directory name is not normalized: {pkg_dir!r} "
                f"(expected {normalized!r}) in key {key!r}"
            )
        if package is not None and pkg_dir != package:
            continue

        # parse_wheel_filename / parse_sdist_filename return names already
        # canonicalized to the PEP 503 form (lowercase, dashes), matching
        # pep503_normalize, so this equality holds for any correctly filed
        # artifact regardless of the filename's original underscores/case.
        file_pkg = _package_name_from_filename(filename)
        if file_pkg != pkg_dir:
            raise ValueError(
                f"File {filename!r} does not match package directory "
                f"{pkg_dir!r} (parsed package name {file_pkg!r}) in key {key!r}"
            )

        grouped.setdefault(pkg_dir, []).append(
            PackageFile(
                key=key,
                filename=filename,
                checksum=None,
                pep658=None,
                size=None,
            )
        )

    return [
        PackageDir(name=name, files=sorted(grouped[name], key=lambda f: f.filename))
        for name in sorted(grouped)
    ]


def _artifact_attributes(file: PackageFile) -> str:
    """Build the PEP 658 / requires-python attributes for an artifact link."""
    attributes = ""
    if file.pep658:
        pep658_sha = f"sha256={file.pep658}"
        # PEP 714 renames the attribute to data-core-metadata; emit both.
        attributes = (
            f' data-dist-info-metadata="{pep658_sha}"'
            f' data-core-metadata="{pep658_sha}"'
        )
    # networkx version/python constraints (pytorch/pytorch#152191):
    # 3.4.2 for Python 3.10, 3.5+ for Python 3.11+.
    if file.filename == "networkx-3.4.2-py3-none-any.whl":
        attributes += ' data-requires-python="&gt;=3.10"'
    elif file.filename.startswith("networkx-") and file.filename.endswith(
        "-py3-none-any.whl"
    ):
        attributes += ' data-requires-python="&gt;=3.11"'
    return attributes


def render_package_page(pkg: PackageDir, skip_checksum: bool = True) -> str:
    """Render the PEP 503 simple page for a single package directory.

    Artifact links are same-directory (just the filename), so the generated
    page never depends on `../filename` or public CloudFront path mapping.
    """
    out: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "  <body>",
        f"    <h1>Links for {pkg.name}</h1>",
    ]
    for file in pkg.files:
        # Same-directory invariant: a link must not escape the package dir.
        if "/" in file.filename:
            raise ValueError(
                f"Artifact link would escape package directory: {file.filename!r}"
            )
        maybe_fragment = ""
        if file.checksum and not skip_checksum:
            maybe_fragment = f"#sha256={file.checksum}"
        attributes = _artifact_attributes(file)
        # Escape only at render: quote() percent-encodes '+' (and any other
        # unsafe char) in the href, while html.escape() guards the display text.
        # keeping '/' as a path separator is safe because we rejected it above.
        href = f"{quote(file.filename)}{maybe_fragment}"
        out.append(
            f'    <a href="{href}"{attributes}>{html.escape(file.filename)}</a><br/>'
        )
    out.append("  </body>")
    out.append("</html>")
    out.append(f"<!--TIMESTAMP {int(time.time())}-->")
    return "\n".join(out)


def render_root_page(packages: list[PackageDir]) -> str:
    """Render the product-local root page listing local package directories.

    This is the product-local root (e.g. pytorch/whl/index.html), not the
    aggregate /rocm/whl/ root, which is owned by separate infrastructure.
    """
    out: list[str] = ["<!DOCTYPE html>", "<html>", "  <body>"]
    for pkg in sorted(packages, key=lambda p: p.name):
        out.append(f'    <a href="{pkg.name}/">{pkg.name}</a><br/>')
    out.append("  </body>")
    out.append("</html>")
    out.append(f"<!--TIMESTAMP {int(time.time())}-->")
    return "\n".join(out)


def build_index_pages(
    packages: list[PackageDir],
    root: str,
    write_root: bool,
    skip_checksum: bool = True,
) -> list[IndexPage]:
    """Build all index pages for a set of package directories.

    Args:
        packages: Discovered package directories.
        root: The `<product>/<index>` prefix.
        write_root: If True, also generate the product-local root index page.
            Must be False for package-scoped regeneration: a scoped run does
            not see the full package set and would clobber the root listing.
        skip_checksum: If True (default), omit the #sha256 fragment from
            artifact links. Pass False (via --include-checksums) to include.

    Returns:
        Index pages, one per package directory plus optionally the root.
    """
    root = root.rstrip("/")
    pages: list[IndexPage] = []
    if write_root:
        pages.append(
            IndexPage(key=f"{root}/index.html", html=render_root_page(packages))
        )
    for pkg in packages:
        pages.append(
            IndexPage(
                key=f"{root}/{pkg.name}/index.html",
                html=render_package_page(pkg, skip_checksum=skip_checksum),
            )
        )
    return pages


# ---------------------------------------------------------------------------
# S3 I/O (CLI path)
# ---------------------------------------------------------------------------


def list_keys(
    client: botocore.client.BaseClient,
    bucket_name: str,
    root: str,
    package: str | None,
) -> list[str]:
    """List raw object keys under a product root (optionally one package dir).

    Keys are returned verbatim (with any literal '+'); percent-encoding is
    applied only when a key is rendered into an href, never at storage time.
    """
    root = root.rstrip("/")
    list_prefix = f"{root}/{package}/" if package else f"{root}/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def fetch_metadata(
    client: botocore.client.BaseClient,
    bucket_name: str,
    files: list[PackageFile],
) -> None:
    """Populate checksum and size for each file via S3 HEAD requests.

    Falls back to checksum-sha256 object metadata for older files without a
    native checksum.

    A single HEAD populates both size and checksum, so a file is fetched
    whenever either field is still missing.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_METADATA_MAX_WORKERS
    ) as executor:
        futures = {
            idx: executor.submit(
                lambda key: client.head_object(
                    Bucket=bucket_name, Key=key, ChecksumMode="Enabled"
                ),
                file.key,
            )
            for idx, file in enumerate(files)
            if file.size is None or file.checksum is None
        }
        for idx, future in futures.items():
            response = future.result()
            raw = response.get("ChecksumSHA256")
            if raw and _MULTIPART_CHECKSUM_RE.match(raw):
                # Composite checksum from a multipart upload — not a valid
                # SHA256 of the object; discard it.
                print(f"WARNING: {files[idx].key} has multipart checksum: {raw}")
                raw = None
            sha256 = base64.b64decode(raw).hex() if raw else None
            if sha256 is None:
                sha256 = response.get("Metadata", {}).get("checksum-sha256")
            if sha256 is None:
                sha256 = response.get("Metadata", {}).get("x-amz-meta-checksum-sha256")
            files[idx].checksum = sha256
            content_length = response.get("ContentLength")
            if content_length is not None:
                files[idx].size = int(content_length)


def fetch_pep658(
    client: botocore.client.BaseClient,
    bucket_name: str,
    files: list[PackageFile],
) -> None:
    """Populate the PEP 658 .metadata sidecar checksum for each file."""

    def _fetch(key: str) -> str | None:
        try:
            response = client.head_object(
                Bucket=bucket_name, Key=f"{key}.metadata", ChecksumMode="Enabled"
            )
            raw = response.get("ChecksumSHA256")
            if raw and _MULTIPART_CHECKSUM_RE.match(raw):
                print(f"WARNING: {key}.metadata has multipart checksum: {raw}")
                return None
            return base64.b64decode(raw).hex() if raw else None
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_METADATA_MAX_WORKERS
    ) as executor:
        futures = {
            idx: executor.submit(_fetch, file.key) for idx, file in enumerate(files)
        }
        for idx, future in futures.items():
            result = future.result()
            if result is not None:
                files[idx].pep658 = result


def upload_pages(
    client: botocore.client.BaseClient,
    bucket_name: str,
    pages: list[IndexPage],
) -> None:
    """Upload rendered index pages to S3."""
    for page in pages:
        print(f"INFO Uploading {page.key} to {bucket_name}")
        client.put_object(
            Bucket=bucket_name,
            Key=page.key,
            CacheControl="no-cache,no-store,must-revalidate",
            ContentType="text/html",
            Body=page.html.encode("utf-8"),
        )


def save_pages(pages: list[IndexPage]) -> None:
    """Write rendered index pages to the local filesystem for inspection."""
    for page in pages:
        print(f"INFO Saving {page.key}")
        makedirs(path.dirname(page.key), exist_ok=True)
        with open(page.key, mode="w", encoding="utf-8") as f:
            f.write(page.html)


def generate_structured_index(
    client: botocore.client.BaseClient,
    bucket_name: str,
    root: str,
    package: str | None = None,
    upload: bool = True,
    skip_checksum: bool = True,
) -> int:
    """Regenerate structured product-local indexes for a product root.

    Args:
        client: A boto3 S3 client.
        bucket_name: Target S3 bucket.
        root: The `<product>/<index>` prefix.
        package: If set, regenerate only this package directory and skip the
            product-local root index (the scoped listing cannot rebuild the
            root without clobbering it).
        upload: If True, upload to S3; otherwise save pages to local disk.
        skip_checksum: If True (default), omit the #sha256 fragment from
            artifact links. Pass False (via --include-checksums) to include.

    Returns:
        Number of package directories processed.
    """
    keys = list_keys(client, bucket_name, root, package)
    packages = discover_packages(keys, root=root, package=package)

    all_files = [file for pkg in packages for file in pkg.files]
    fetch_metadata(client, bucket_name, all_files)
    fetch_pep658(client, bucket_name, all_files)

    pages = build_index_pages(
        packages, root=root, write_root=package is None, skip_checksum=skip_checksum
    )
    if upload:
        upload_pages(client, bucket_name, pages)
    else:
        save_pages(pages)
    return len(packages)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Generate structured product-local PEP 503 S3 indexes"
    )
    parser.add_argument(
        "prefix",
        help="Product-local root prefix, e.g. pytorch/whl or core/whl-next",
    )
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument(
        "--package",
        type=str,
        default=None,
        help=(
            "Regenerate only this package directory and skip the product root "
            "index (used for targeted single-package regeneration)."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Upload the generated index pages to S3 (default: write them to "
            "local disk without uploading)."
        ),
    )
    parser.add_argument(
        "--include-checksums",
        action="store_true",
        help="Include #sha256 fragments in artifact links (default: omitted).",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    bucket_name = args.bucket or getenv("S3_BUCKET_PY")
    if not bucket_name:
        parser.error("Bucket must be provided via --bucket or S3_BUCKET_PY")

    client = boto3.client("s3")
    count = generate_structured_index(
        client=client,
        bucket_name=bucket_name,
        root=args.prefix,
        package=args.package,
        upload=args.upload,
        skip_checksum=not args.include_checksums,
    )
    print(f"INFO Processed {count} package directories under {args.prefix}")


if __name__ == "__main__":
    main()
