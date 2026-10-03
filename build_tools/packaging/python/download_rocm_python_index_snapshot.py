#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download product-local Python index HTML for aggregate validation.

The downloaded directory mirrors the public ``/rocm/...`` paths expected by
``aggregate_index.py --content-root``. Only root and package ``index.html``
files are downloaded; wheel files and other artifacts are not needed.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from aggregate_index import (
    IndexValidationError,
    ManifestError,
    OwnershipManifest,
    load_ownership_manifest,
    parse_product_root_index,
)


DEFAULT_MANIFEST = Path(__file__).with_name("rocm_whl_next_ownership.yaml")


class SnapshotDownloadError(RuntimeError):
    """Raised when a product index snapshot cannot be downloaded safely."""


def _validate_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise SnapshotDownloadError(
            "origin must be an HTTPS origin without credentials, a path, query, "
            "or fragment"
        )
    return origin.rstrip("/")


def _download_html(url: str, destination: Path, *, allow_missing: bool) -> bool:
    try:
        with urlopen(url) as response:
            final_url = response.geturl()
            if urlsplit(final_url).scheme != "https":
                raise SnapshotDownloadError(
                    f"{url}: redirected to non-HTTPS URL {final_url!r}"
                )
            content = response.read()
    except HTTPError as e:
        if allow_missing and e.code == 404:
            return False
        raise SnapshotDownloadError(f"{url}: HTTP {e.code} {e.reason}") from e
    except URLError as e:
        raise SnapshotDownloadError(f"{url}: download failed: {e.reason}") from e

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SnapshotDownloadError(f"{url}: response is not UTF-8: {e}") from e
    if not text.strip():
        raise SnapshotDownloadError(f"{url}: response is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return True


def _download_product_index(
    *,
    origin: str,
    owner_path: str,
    staging_dir: Path,
    allow_missing: bool,
) -> None:
    root_relative_path = Path("rocm") / owner_path / "index.html"
    root_destination = staging_dir / root_relative_path
    if not _download_html(
        f"{origin}/{root_relative_path.as_posix()}",
        root_destination,
        allow_missing=allow_missing,
    ):
        return

    for link in parse_product_root_index(root_destination):
        package_relative_path = (
            Path("rocm") / owner_path / link.package_name / "index.html"
        )
        _download_html(
            f"{origin}/{package_relative_path.as_posix()}",
            staging_dir / package_relative_path,
            allow_missing=False,
        )


def download_snapshot(
    manifest: OwnershipManifest,
    *,
    stream: str,
    origin: str,
    output_dir: Path,
) -> None:
    """Download a complete HTML snapshot for one manifest stream."""
    if stream not in manifest.streams.known:
        known_streams = ", ".join(manifest.streams.known)
        raise SnapshotDownloadError(
            f"unknown stream {stream!r}; expected one of: {known_streams}"
        )
    if output_dir.exists():
        raise SnapshotDownloadError(f"output directory already exists: {output_dir}")

    origin = _validate_origin(origin)
    index = manifest.python_indexes[0]
    active_owner_paths = index.owner_paths(stream)
    all_owner_paths = {package.owner_path for package in index.packages.values()}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        for owner_path in sorted(all_owner_paths):
            _download_product_index(
                origin=origin,
                owner_path=owner_path,
                staging_dir=staging_dir,
                allow_missing=owner_path not in active_owner_paths,
            )

        if output_dir.exists():
            raise SnapshotDownloadError(
                f"output directory was created during download: {output_dir}"
            )
        staging_dir.rename(output_dir)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Ownership manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--stream", required=True, help="Stream represented by the HTTPS origin"
    )
    parser.add_argument("--origin", required=True, help="Public HTTPS origin")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the product index HTML snapshot",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_ownership_manifest(args.manifest)
        download_snapshot(
            manifest,
            stream=args.stream,
            origin=args.origin,
            output_dir=args.output_dir,
        )
    except (ManifestError, IndexValidationError, SnapshotDownloadError, OSError) as e:
        parser.error(str(e))

    print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
