#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate RPM repodata from multiple sources into a single repository.

Fetches primary.xml.gz (via repomd.xml) from each source, rewrites
<location href> with an owner-prefix, merges all <package> entries,
regenerates repomd.xml with fresh checksums, and writes output locally
and/or uploads to S3.

Usage:
    python aggregate_rpm_metadata.py \\
        --source "core,https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64,core" \\
        --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/rpm,rvs" \\
        [--output-dir /tmp/rpm-metadata] \\
        [--output-bucket therock-deb-rpm-test] \\
        [--output-prefix metadata/rpm] \\
        [--sign-key KEY_ID]

Each --source value is a comma-separated tuple:
    name,base_url,href_prefix

  name        - short label used for collision messages
  base_url    - public baseurl of the RPM repo (same as dnf baseurl=)
  href_prefix - prefix written into <location href> in merged primary.xml
                (e.g. "core" → "core/<pkg>.rpm")
                CloudFront routes requests for <prefix>/* to the real origin.

At least one --source is required.
At least one of --output-dir or --output-bucket must be provided.

--sign-key KEY_ID
    Signs repomd.xml with GPG, producing repomd.xml.asc. DNF verifies this
    when repo_gpgcheck=1 is set in the .repo file. Without this flag, use
    repo_gpgcheck=0 on the client side. Can also be set via
    AGGREGATE_SIGN_KEY environment variable.
"""

import argparse
import gzip
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# XML namespaces used in RPM repodata
NS_COMMON = "http://linux.duke.edu/metadata/common"
NS_RPM = "http://linux.duke.edu/metadata/rpm"
NS_REPO = "http://linux.duke.edu/metadata/repo"


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


def parse_source(value: str) -> dict:
    """Parse a --source argument into a source dict.

    Format: name,base_url,href_prefix
    """
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--source must be 'name,base_url,href_prefix', got: {value!r}"
        )
    name, base_url, href_prefix = parts
    return {
        "name": name.strip(),
        "base_url": base_url.strip(),
        "href_prefix": href_prefix.strip(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_bytes(url: str) -> bytes:
    """Fetch URL, return raw bytes. Raises on HTTP error."""
    print(f"  Fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "amdrocm-aggregate/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_primary_xml(base_url: str) -> bytes:
    """Fetch and decompress primary.xml.gz from an RPM repo via repomd.xml."""
    repomd_url = f"{base_url.rstrip('/')}/repodata/repomd.xml"
    repomd_data = fetch_bytes(repomd_url)
    repomd = ET.fromstring(repomd_data)

    ns = {"r": NS_REPO}
    primary_loc = repomd.find(".//r:data[@type='primary']/r:location", ns)
    if primary_loc is None:
        raise RuntimeError(f"No primary data found in repomd.xml from {base_url}")

    primary_href = primary_loc.get("href")
    primary_url = f"{base_url.rstrip('/')}/{primary_href}"
    compressed = fetch_bytes(primary_url)
    return gzip.decompress(compressed)


def parse_packages(primary_xml: bytes) -> list[ET.Element]:
    """Parse primary.xml and return list of <package> elements."""
    root = ET.fromstring(primary_xml)
    ns = {"c": NS_COMMON}
    return root.findall("c:package", ns)


def rewrite_location_href(pkg: ET.Element, href_prefix: str) -> None:
    """Rewrite <location href> to href_prefix/<basename> in-place.

    DNF resolves this relative path against the repo baseurl.
    CloudFront routes <prefix>/* requests to the correct origin bucket.
    """
    ns = {"c": NS_COMMON}
    location = pkg.find("c:location", ns)
    if location is None:
        pkg_name_el = pkg.find("c:name", ns)
        pkg_name = pkg_name_el.text if pkg_name_el is not None else "<unknown>"
        raise RuntimeError(
            f"Package '{pkg_name}' has no <location> element in primary.xml — "
            f"source repo may be malformed."
        )
    original_href = location.get("href", "")
    basename = (
        original_href.rsplit("/", 1)[-1] if "/" in original_href else original_href
    )
    location.set("href", f"{href_prefix}/{basename}")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_repomd_xml(primary_gz: bytes, primary_xml: bytes, timestamp: int) -> bytes:
    """Generate repomd.xml with SHA256 checksums for merged primary.xml.gz."""
    ET.register_namespace("", NS_REPO)
    ET.register_namespace("rpm", NS_RPM)

    repomd = ET.Element(f"{{{NS_REPO}}}repomd")
    ET.SubElement(repomd, f"{{{NS_REPO}}}revision").text = str(timestamp)

    data = ET.SubElement(repomd, f"{{{NS_REPO}}}data", type="primary")
    ET.SubElement(data, f"{{{NS_REPO}}}checksum", type="sha256").text = sha256_of(
        primary_gz
    )
    ET.SubElement(data, f"{{{NS_REPO}}}open-checksum", type="sha256").text = sha256_of(
        primary_xml
    )
    ET.SubElement(data, f"{{{NS_REPO}}}location").set("href", "repodata/primary.xml.gz")
    ET.SubElement(data, f"{{{NS_REPO}}}timestamp").text = str(timestamp)
    ET.SubElement(data, f"{{{NS_REPO}}}size").text = str(len(primary_gz))
    ET.SubElement(data, f"{{{NS_REPO}}}open-size").text = str(len(primary_xml))

    return ET.tostring(repomd, encoding="UTF-8", xml_declaration=True)


def gpg_sign_repomd(repomd_path: Path, key_id: Optional[str]) -> Path:
    """Generate detached ASCII-armored signature for repomd.xml.

    Returns path to repomd.xml.asc. DNF verifies this when repo_gpgcheck=1
    is set in the .repo file.
    """
    key_args = ["--local-user", key_id] if key_id else []
    asc_path = repomd_path.parent / "repomd.xml.asc"
    subprocess.run(
        ["gpg", "--batch", "--yes", "--armor", "--detach-sign"]
        + key_args
        + ["--output", str(asc_path), str(repomd_path)],
        check=True,
    )
    return asc_path


def write_local(local_path: Path, dest_dir: Path, rel_key: str) -> None:
    """Copy a generated file to dest_dir preserving relative path."""
    dest = dest_dir / rel_key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(local_path.read_bytes())
    print(f"  Written  → {dest}")


def upload_to_s3(local_path: Path, bucket: str, s3_key: str) -> None:
    """Upload a single file to S3."""
    import boto3

    s3 = boto3.client("s3")
    content_type = "application/octet-stream"
    if local_path.suffix == ".gz":
        content_type = "application/x-gzip"
    elif local_path.suffix in (".xml", ".asc"):
        content_type = "text/xml"

    extra = {"ContentType": content_type}
    if local_path.name in ("repomd.xml", "repomd.xml.asc", "primary.xml.gz"):
        extra["CacheControl"] = "max-age=300"
    else:
        extra["CacheControl"] = "max-age=31536000"

    s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra)
    print(f"  Uploaded → s3://{bucket}/{s3_key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RPM repodata from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        metavar="name,base_url,href_prefix",
        required=True,
        help=(
            "RPM source to include (repeatable, no limit on count). "
            "Format: name,base_url,href_prefix — "
            "href_prefix is written into <location href> "
            "(e.g. 'core' → 'core/<pkg>.rpm')."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to write generated metadata files",
    )
    parser.add_argument(
        "--output-bucket",
        default=None,
        help="S3 bucket to upload generated metadata (requires boto3)",
    )
    parser.add_argument(
        "--output-prefix",
        default="metadata/rpm",
        help="S3 key prefix (default: metadata/rpm)",
    )
    parser.add_argument(
        "--sign-key",
        default=None,
        help=(
            "GPG key ID for signing repomd.xml. Produces repomd.xml.asc so DNF "
            "clients can verify metadata integrity with repo_gpgcheck=1. "
            "Can also be set via AGGREGATE_SIGN_KEY env var."
        ),
    )
    args = parser.parse_args()

    # Validate sources
    try:
        sources = [parse_source(s) for s in args.sources]
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    if len(sources) == 0:
        parser.error("At least one --source is required.")

    if args.output_dir is None and args.output_bucket is None:
        parser.error(
            "At least one of --output-dir or --output-bucket must be provided."
        )

    prefix = args.output_prefix.rstrip("/")
    timestamp = int(time.time())

    print("=" * 70)
    print("AMD ROCm RPM Aggregate Metadata Job")
    if args.output_dir:
        print(f"  Output dir:    {args.output_dir}")
    if args.output_bucket:
        print(f"  Output bucket: s3://{args.output_bucket}/{prefix}/")
    if len(sources) == 1:
        print(f"  Sources: {sources[0]['name']} (single source — no merge needed)")
    else:
        print(f"  Sources: {', '.join(s['name'] for s in sources)}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Fetch + parse + rewrite each source
    # ------------------------------------------------------------------
    all_packages: list[ET.Element] = []
    declared: dict[str, str] = {}

    # Register namespaces so output doesn't use ns0/ns1 prefixes
    ET.register_namespace("", NS_COMMON)
    ET.register_namespace("rpm", NS_RPM)

    for source in sources:
        name = source["name"]
        base_url = source["base_url"]
        href_prefix = source["href_prefix"]

        print(f"\n[{name}] Fetching repomd.xml + primary.xml.gz from {base_url}")
        print(f"         href_prefix={href_prefix}")

        try:
            primary_xml = fetch_primary_xml(base_url)
        except Exception as e:
            sys.exit(
                f"ERROR: Failed to fetch primary.xml.gz from [{name}] {base_url}: {e}"
            )

        packages = parse_packages(primary_xml)
        print(f"[{name}] Found {len(packages)} packages")

        ns = {"c": NS_COMMON}
        for pkg in packages:
            pkg_name = pkg.find("c:name", ns)
            pkg_arch = pkg.find("c:arch", ns)
            pkg_ver = pkg.find("c:version", ns)

            name_str = pkg_name.text if pkg_name is not None else "unknown"
            arch_str = pkg_arch.text if pkg_arch is not None else "unknown"
            if pkg_ver is not None:
                ver_str = (
                    f"{pkg_ver.get('epoch', '0')}"
                    f":{pkg_ver.get('ver', '')}"
                    f"-{pkg_ver.get('rel', '')}"
                )
            else:
                ver_str = "unknown"

            # NEVRA collision detection: same exact package from two sources
            nevra_key = f"{name_str}-{ver_str}.{arch_str}"
            if nevra_key in declared and declared[nevra_key] != name:
                sys.exit(
                    f"ERROR: Package collision for '{nevra_key}': "
                    f"present in both '{declared[nevra_key]}' and '{name}'. "
                    f"Each NEVRA must come from exactly one source."
                )
            declared[nevra_key] = name

            rewrite_location_href(pkg, href_prefix)
            all_packages.append(pkg)

    print(f"\nTotal packages after merge: {len(all_packages)}")

    # ------------------------------------------------------------------
    # Step 2: Build merged primary.xml
    # ------------------------------------------------------------------
    # Register namespaces BEFORE building the tree so ET.tostring() uses
    # clean prefixes. Do NOT also call .set("xmlns:rpm") on the root element
    # — that would write the attribute twice and DNF rejects it.
    ET.register_namespace("", NS_COMMON)
    ET.register_namespace("rpm", NS_RPM)

    merged_root = ET.Element(
        f"{{{NS_COMMON}}}metadata",
        attrib={"packages": str(len(all_packages))},
    )
    for pkg in all_packages:
        merged_root.append(pkg)

    primary_xml_bytes = ET.tostring(merged_root, encoding="UTF-8", xml_declaration=True)
    primary_gz_bytes = gzip.compress(primary_xml_bytes, compresslevel=9)
    print(
        f"primary.xml: {len(primary_xml_bytes):,} bytes uncompressed, "
        f"{len(primary_gz_bytes):,} bytes compressed"
    )

    # ------------------------------------------------------------------
    # Step 3: Generate repomd.xml
    # ------------------------------------------------------------------
    repomd_bytes = build_repomd_xml(primary_gz_bytes, primary_xml_bytes, timestamp)
    print(f"repomd.xml SHA256: {sha256_of(repomd_bytes)}")

    # ------------------------------------------------------------------
    # Step 4: Write to temp dir, optionally sign, then output
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        repodata_dir = tmppath / "repodata"
        repodata_dir.mkdir()

        primary_gz_file = repodata_dir / "primary.xml.gz"
        primary_xml_file = repodata_dir / "primary.xml"
        repomd_file = repodata_dir / "repomd.xml"

        primary_gz_file.write_bytes(primary_gz_bytes)
        primary_xml_file.write_bytes(primary_xml_bytes)
        repomd_file.write_bytes(repomd_bytes)

        files: list[tuple[Path, str]] = [
            (primary_gz_file, "repodata/primary.xml.gz"),
            (repomd_file, "repodata/repomd.xml"),
        ]

        # GPG signing
        key_id = args.sign_key or os.environ.get("AGGREGATE_SIGN_KEY")
        if key_id:
            print(f"\nSigning repomd.xml with key: {key_id}")
            asc_path = gpg_sign_repomd(repomd_file, key_id)
            files.append((asc_path, "repodata/repomd.xml.asc"))
        else:
            print("\nNo signing key — skipping GPG signing")
            print("  (Set --sign-key or AGGREGATE_SIGN_KEY to enable)")
            print("  (Clients must use repo_gpgcheck=0 in .repo file)")

        # Local output
        if args.output_dir:
            print(f"\nWriting {len(files)} files to {args.output_dir}/")
            for local_path, rel_key in files:
                write_local(local_path, args.output_dir, rel_key)

        # S3 upload
        if args.output_bucket:
            print(
                f"\nUploading {len(files)} files to "
                f"s3://{args.output_bucket}/{prefix}/"
            )
            for local_path, rel_key in files:
                upload_to_s3(local_path, args.output_bucket, f"{prefix}/{rel_key}")

    print("\n✓ Aggregate RPM metadata job complete")


if __name__ == "__main__":
    main()
