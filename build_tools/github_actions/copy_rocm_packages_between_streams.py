#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy ROCm Python packages between repo.amd.com release streams.

Dev PyTorch builds publish torch to the ``dev`` aggregate index. Their pinned
``rocm==...`` dependency must live on that same index, otherwise
``pip install torch`` fails at test time. Nightly ROCm packages are first
published to ``nightly.repo.amd.com``; this script copies a version into
``dev`` (or another destination stream) so the indexes stay self-contained.

Example:

    python build_tools/github_actions/copy_rocm_packages_between_streams.py \\
        --rocm-version 10.3.0a20261027 \\
        --source-stream nightly \\
        --dest-stream dev
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.python_package_paths import (  # noqa: E402
    REPO_STREAMS,
    plan_local_uploads,
    repo_product_bucket,
)
from _therock_utils.storage_backend import create_storage_backend  # noqa: E402

logger = logging.getLogger(__name__)

_USER_AGENT = "TheRock-copy-rocm-packages/1.0"
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def stream_index_url(stream: str) -> str:
    """Return the aggregate pip index URL for a repo.amd.com stream."""
    return f"https://{stream}.repo.amd.com/rocm/whl-next/"


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def list_index_packages(index_url: str) -> list[str]:
    """Return package directory names from a PEP 503 simple index page."""
    html = _fetch_text(index_url if index_url.endswith("/") else index_url + "/")
    packages: list[str] = []
    for href in _HREF_RE.findall(html):
        name = href.rstrip("/").rsplit("/", 1)[-1]
        if not name or name.startswith("?") or "." in name:
            continue
        packages.append(name)
    return sorted(set(packages))


def list_package_artifacts(package_url: str, version: str) -> list[str]:
    """Return artifact filenames for ``version`` from a package simple page."""
    html = _fetch_text(package_url if package_url.endswith("/") else package_url + "/")
    artifacts: list[str] = []
    for href in _HREF_RE.findall(html):
        filename = href.split("?", 1)[0].rsplit("/", 1)[-1]
        if version not in filename:
            continue
        if not filename.endswith((".whl", ".tar.gz", ".zip")):
            continue
        artifacts.append(filename)
    return sorted(set(artifacts))


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())


def select_core_packages(package_names: list[str]) -> list[str]:
    """Keep ROCm Core package names (exclude torch/jax device extras)."""
    selected: list[str] = []
    for name in package_names:
        if name == "rocm" or name.startswith("rocm-sdk-"):
            selected.append(name)
    return selected


def select_torch_packages(package_names: list[str]) -> list[str]:
    """Keep torch/apex/triton package names for optional torch copies."""
    selected: list[str] = []
    for name in package_names:
        if (
            name == "torch"
            or name.startswith("torch")
            or name.startswith("amd-torch")
            or name == "apex"
            or name.startswith("triton")
        ):
            selected.append(name)
    return selected


def collect_artifacts_for_packages(
    *,
    index_url: str,
    package_names: list[str],
    version: str,
    dest_dir: Path,
    dry_run: bool = False,
) -> list[Path]:
    """Download matching artifacts for ``package_names`` into ``dest_dir``.

    When ``dry_run`` is set, create empty placeholder files so upload planning
    still works without transferring package payloads.
    """
    downloaded: list[Path] = []
    for package in package_names:
        package_url = f"{index_url.rstrip('/')}/{package}/"
        try:
            artifacts = list_package_artifacts(package_url, version)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.info("Skipping missing package page %s", package_url)
                continue
            raise
        for filename in artifacts:
            dest = dest_dir / filename
            if dest.exists():
                downloaded.append(dest)
                continue
            file_url = f"{package_url.rstrip('/')}/{filename}"
            if dry_run:
                logger.info("[DRY RUN] would download %s", file_url)
                dest.write_bytes(b"")
            else:
                logger.info("Downloading %s", file_url)
                download_file(file_url, dest)
            downloaded.append(dest)
    return downloaded


def upload_product_artifacts(
    *,
    source_dir: Path,
    dest_stream: str,
    product: str,
    dry_run: bool,
) -> int:
    """Upload local artifacts into a structured product bucket."""
    dest_bucket = repo_product_bucket(dest_stream, product)
    plans = plan_local_uploads(
        source_dir=source_dir,
        dest_bucket=dest_bucket,
        product=product,
        index="whl-next",
    )
    if not plans:
        return 0

    backend = create_storage_backend(dry_run=dry_run)
    for plan in plans:
        logger.info("Upload %s -> %s", plan.source.name, plan.dest.s3_uri)
        backend.upload_file(plan.source, plan.dest)
    return len(plans)


def copy_packages(
    *,
    rocm_version: str,
    source_stream: str,
    dest_stream: str,
    products: list[str],
    dry_run: bool,
) -> int:
    """Download packages from ``source_stream`` and publish to ``dest_stream``."""
    if source_stream == dest_stream:
        raise ValueError("source_stream and dest_stream must differ")
    unknown = [p for p in products if p not in ("core", "pytorch")]
    if unknown:
        raise ValueError(f"Unsupported products: {unknown}")
    if not products:
        raise ValueError("At least one product is required")

    index_url = stream_index_url(source_stream)
    logger.info("Listing packages from %s", index_url)
    packages = list_index_packages(index_url)

    total = 0
    with tempfile.TemporaryDirectory(prefix="therock-copy-rocm-") as tmp:
        tmp_path = Path(tmp)
        if "core" in products:
            core_dir = tmp_path / "core"
            core_dir.mkdir()
            core_files = collect_artifacts_for_packages(
                index_url=index_url,
                package_names=select_core_packages(packages),
                version=rocm_version,
                dest_dir=core_dir,
                dry_run=dry_run,
            )
            if not core_files:
                raise FileNotFoundError(
                    f"No ROCm Core packages matching {rocm_version!r} found at {index_url}"
                )
            total += upload_product_artifacts(
                source_dir=core_dir,
                dest_stream=dest_stream,
                product="core",
                dry_run=dry_run,
            )

        if "pytorch" in products:
            torch_dir = tmp_path / "pytorch"
            torch_dir.mkdir()
            torch_files = collect_artifacts_for_packages(
                index_url=index_url,
                package_names=select_torch_packages(packages),
                version=rocm_version,
                dest_dir=torch_dir,
                dry_run=dry_run,
            )
            if not torch_files:
                raise FileNotFoundError(
                    f"No torch packages matching {rocm_version!r} found at {index_url}"
                )
            total += upload_product_artifacts(
                source_dir=torch_dir,
                dest_stream=dest_stream,
                product="pytorch",
                dry_run=dry_run,
            )

    logger.info("Copied %d package file(s) to stream %s", total, dest_stream)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm package version to copy (e.g. 10.3.0a20261027)",
    )
    parser.add_argument(
        "--source-stream",
        default="nightly",
        choices=REPO_STREAMS,
        help="Source repo.amd.com stream (default: nightly)",
    )
    parser.add_argument(
        "--dest-stream",
        default="dev",
        choices=REPO_STREAMS,
        help="Destination repo.amd.com stream (default: dev)",
    )
    parser.add_argument(
        "--products",
        default="core",
        help="Comma-separated products to copy: core and/or pytorch (default: core)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log planned uploads without writing to S3",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log progress (default: true)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    products = [p.strip() for p in args.products.split(",") if p.strip()]
    copy_packages(
        rocm_version=args.rocm_version,
        source_stream=args.source_stream,
        dest_stream=args.dest_stream,
        products=products,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
