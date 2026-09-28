#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate DEB package metadata from multiple APT sources into a single repo.

Fetches Packages.gz from each source, rewrites the Filename: field with an
owner-prefix so APT resolves package files through the correct CloudFront path,
merges all stanzas, regenerates Release/InRelease, and writes output locally
and/or uploads to S3.

Limitations:
  - Packages index compression: tries .xz, .bz2, .gz, and uncompressed in
    APT preference order. Fails fast if none are available for a source.
  - A single --architecture value is supported per run. To serve multiple
    architectures, run the script once per architecture. Note that each run
    overwrites dists/<suite>/Release, so multi-arch runs must use separate
    --output-prefix values or the Release file must be manually merged.

Usage:
    python aggregate_deb_metadata.py \\
        --source "core,https://nightly.repo.amd.com/rocm/core/packages/deb/20260823-12345678,dists,pool/core" \\
        --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/deb,flat,pool/rvs" \\
        [--output-dir /tmp/deb-metadata] \\
        [--output-bucket therock-deb-rpm-test] \\
        [--output-prefix metadata/deb] \\
        [--suite stable] \\
        [--component main] \\
        [--architecture amd64] \\
        [--sign-key KEY_ID]

Each --source value is a comma-separated tuple:
    name,base_url,style,pool_prefix

  name        - short label used in pool path prefix and collision messages
  base_url    - public base URL of the APT repo
  style       - "dists" (standard dists/suite/component/binary-arch layout)
                "flat"  (Packages.gz at repo root, i.e. deb [...] <url> /)
  pool_prefix - prefix written into the Filename: field in merged Packages
                (e.g. "pool/core" → "pool/core/<pkg>.deb")
                CloudFront routes requests for pool/<prefix>/* to the real origin.

At least one --source is required.
At least one of --output-dir or --output-bucket must be provided.

--sign-key KEY_ID
    Signs the Release file with GPG, producing Release.gpg (detached signature)
    and InRelease (clearsigned). APT clients configured with signed-by= will
    verify these. Without this flag, packages must be configured with
    [trusted=yes] on the client side. Can also be set via AGGREGATE_SIGN_KEY
    environment variable.
"""

import argparse
import bz2
import gzip
import hashlib
import lzma
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


def parse_source(value: str) -> dict:
    """Parse a --source argument into a source dict.

    Format: name,base_url,style,pool_prefix
    """
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--source must be 'name,base_url,style,pool_prefix', got: {value!r}"
        )
    name, base_url, style, pool_prefix = [p.strip() for p in parts]
    if style not in ("dists", "flat"):
        raise argparse.ArgumentTypeError(
            f"style must be 'dists' or 'flat', got: {style!r}"
        )
    return {
        "name": name,
        "base_url": base_url,
        "style": style,
        "pool_prefix": pool_prefix,
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


def fetch_packages(
    base_url: str, style: str, suite: str, component: str, arch: str
) -> bytes:
    """Download and return decompressed Packages content.

    Tries compression formats in APT preference order: .xz, .bz2, .gz,
    then uncompressed. Raises RuntimeError if none are available.

    style="dists" → standard: <base>/dists/<suite>/<component>/binary-<arch>/Packages*
    style="flat"  → flat repo: <base>/Packages*  (deb [...] <url> /)
    """
    if style == "flat":
        base_path = f"{base_url.rstrip('/')}/Packages"
    else:
        base_path = (
            f"{base_url.rstrip('/')}/dists/{suite}/{component}/binary-{arch}/Packages"
        )

    candidates = [
        (f"{base_path}.xz", lzma.decompress),
        (f"{base_path}.bz2", bz2.decompress),
        (f"{base_path}.gz", gzip.decompress),
        (base_path, lambda b: b),  # uncompressed
    ]

    for url, decompress in candidates:
        try:
            compressed = fetch_bytes(url)
            return decompress(compressed)
        except urllib.error.HTTPError as e:
            # Treat 404 (not found) and 403 (forbidden/not available) as
            # "this format is not served — try the next one". Some servers
            # return 403 for non-existent resources instead of 404.
            if e.code in (403, 404):
                print(f"  Not found ({e.code}): {url} — trying next format")
                continue
            raise

    raise RuntimeError(
        f"No Packages index found for [{base_url}] — "
        f"tried .xz, .bz2, .gz, and uncompressed."
    )


def parse_stanzas(packages_text: str) -> list[dict[str, str]]:
    """Parse an APT Packages file into a list of field dicts."""
    stanzas = []
    for block in packages_text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        fields: dict[str, str] = {}
        current_key = None
        for line in block.splitlines():
            if line.startswith(" ") or line.startswith("\t"):
                if current_key:
                    fields[current_key] += "\n" + line
            elif ": " in line:
                key, _, val = line.partition(": ")
                fields[key] = val
                current_key = key
            elif line.endswith(":"):
                key = line[:-1]
                fields[key] = ""
                current_key = key
        if fields:
            stanzas.append(fields)
    return stanzas


def rewrite_filename(
    stanza: dict[str, str],
    base_url: str,
    pool_prefix: str,
) -> dict[str, str]:
    """Rewrite Filename: field to pool_prefix/<basename>.

    APT resolves this relative path against the aggregate baseurl.
    CloudFront routes pool/<prefix>/* requests to the correct origin bucket.
    """
    result = dict(stanza)
    original = result.get("Filename", "")
    basename = original.rsplit("/", 1)[-1] if "/" in original else original
    result["Filename"] = f"{pool_prefix}/{basename}"
    return result


def stanza_to_text(fields: dict[str, str]) -> str:
    """Serialize a stanza back to RFC 2822 format."""
    return "\n".join(f"{key}: {val}" for key, val in fields.items())


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha1_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def generate_release(
    suite: str,
    component: str,
    arch: str,
    packages_bytes: bytes,
    packages_gz_bytes: bytes,
) -> str:
    """Generate an APT Release file with MD5Sum/SHA1/SHA256 sections."""
    # APT requires RFC 2822 date with +0000 offset, not the word "UTC"
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    pkg_path = f"{component}/binary-{arch}/Packages"
    pkg_gz_path = f"{component}/binary-{arch}/Packages.gz"

    def md5_entry(path, data):
        return f" {md5_of(data):<32s} {len(data):>16d} {path}"

    def sha1_entry(path, data):
        return f" {sha1_of(data):<40s} {len(data):>16d} {path}"

    def sha256_entry(path, data):
        return f" {sha256_of(data):<64s} {len(data):>16d} {path}"

    files = [(pkg_path, packages_bytes), (pkg_gz_path, packages_gz_bytes)]
    md5_block = "\n".join(md5_entry(p, d) for p, d in files)
    sha1_block = "\n".join(sha1_entry(p, d) for p, d in files)
    sha256_block = "\n".join(sha256_entry(p, d) for p, d in files)

    lines = [
        "Origin: AMD ROCm Aggregate",
        "Label: AMD ROCm",
        f"Suite: {suite}",
        f"Codename: {suite}",
        f"Components: {component}",
        f"Architectures: {arch}",
        f"Date: {now}",
        "MD5Sum:",
        md5_block,
        "SHA1:",
        sha1_block,
        "SHA256:",
        sha256_block,
        "",  # trailing newline
    ]
    return "\n".join(lines)


def gpg_sign(release_path: Path, key_id: Optional[str]) -> tuple[Path, Path]:
    """Produce Release.gpg (detached) and InRelease (clearsigned).

    With --sign-key, APT clients configured with signed-by= can verify the
    repo metadata. Without signing, clients must use [trusted=yes].
    Returns (release_gpg_path, inrelease_path).
    """
    key_args = ["--local-user", key_id] if key_id else []
    release_gpg = release_path.parent / "Release.gpg"
    inrelease = release_path.parent / "InRelease"

    subprocess.run(
        ["gpg", "--batch", "--yes", "--armor", "--detach-sign"]
        + key_args
        + ["--output", str(release_gpg), str(release_path)],
        check=True,
    )
    subprocess.run(
        ["gpg", "--batch", "--yes", "--armor", "--clearsign"]
        + key_args
        + ["--output", str(inrelease), str(release_path)],
        check=True,
    )
    return release_gpg, inrelease


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
    elif local_path.name in ("Release", "InRelease", "Packages", "Release.gpg"):
        content_type = "text/plain"

    extra = {"ContentType": content_type}
    if local_path.name in ("Release", "InRelease", "Packages.gz", "Release.gpg"):
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
        description="Aggregate DEB metadata from multiple APT sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        metavar="name,base_url,style,pool_prefix",
        required=True,
        help=(
            "APT source to include (repeatable, no limit on count). "
            "Format: name,base_url,style,pool_prefix — "
            "style is 'dists' or 'flat', "
            "pool_prefix is written into Filename: (e.g. 'pool/core')."
        ),
    )
    parser.add_argument("--suite", default="stable", help="APT suite (default: stable)")
    parser.add_argument(
        "--component", default="main", help="APT component (default: main)"
    )
    parser.add_argument(
        "--architecture", default="amd64", help="Package architecture (default: amd64)"
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
        default="metadata/deb",
        help="S3 key prefix (default: metadata/deb)",
    )
    parser.add_argument(
        "--sign-key",
        default=None,
        help=(
            "GPG key ID for signing Release. Produces Release.gpg and InRelease "
            "so APT clients can verify metadata integrity. "
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

    suite = args.suite
    component = args.component
    arch = args.architecture
    prefix = args.output_prefix.rstrip("/")

    print("=" * 70)
    print("AMD ROCm DEB Aggregate Metadata Job")
    if args.output_dir:
        print(f"  Output dir:    {args.output_dir}")
    if args.output_bucket:
        print(f"  Output bucket: s3://{args.output_bucket}/{prefix}/")
    print(f"  Suite: {suite}  Component: {component}  Arch: {arch}")
    if len(sources) == 1:
        print(f"  Sources: {sources[0]['name']} (single source — no merge needed)")
    else:
        print(f"  Sources: {', '.join(s['name'] for s in sources)}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Fetch + parse + rewrite each source
    # ------------------------------------------------------------------
    all_stanzas: list[dict[str, str]] = []
    declared: dict[str, str] = {}

    for source in sources:
        name = source["name"]
        base_url = source["base_url"]
        style = source["style"]
        pool_prefix = source["pool_prefix"]

        print(f"\n[{name}] Fetching Packages.gz from {base_url}")
        print(f"         style={style}  pool_prefix={pool_prefix}")

        try:
            raw = fetch_packages(base_url, style, suite, component, arch)
        except Exception as e:
            sys.exit(
                f"ERROR: Failed to fetch Packages.gz from [{name}] {base_url}: {e}"
            )

        stanzas = parse_stanzas(raw.decode("utf-8"))
        print(f"[{name}] Found {len(stanzas)} packages")

        for stanza in stanzas:
            pkg_name = stanza.get("Package", "")
            pkg_ver = stanza.get("Version", "")
            pkg_arch = stanza.get("Architecture", "")
            nevra_key = f"{pkg_name}_{pkg_ver}_{pkg_arch}"
            if nevra_key in declared and declared[nevra_key] != name:
                sys.exit(
                    f"ERROR: Package collision for '{nevra_key}': "
                    f"present in both '{declared[nevra_key]}' and '{name}'. "
                    f"Each NEVRA must come from exactly one source."
                )
            declared[nevra_key] = name
            all_stanzas.append(rewrite_filename(stanza, base_url, pool_prefix))

    print(f"\nTotal packages after merge: {len(all_stanzas)}")

    # ------------------------------------------------------------------
    # Step 2: Serialize merged Packages
    # ------------------------------------------------------------------
    packages_text = "\n\n".join(stanza_to_text(s) for s in all_stanzas) + "\n"
    packages_bytes = packages_text.encode("utf-8")
    packages_gz_bytes = gzip.compress(packages_bytes, compresslevel=9)
    print(
        f"Packages: {len(packages_bytes):,} bytes uncompressed, "
        f"{len(packages_gz_bytes):,} bytes compressed"
    )

    # ------------------------------------------------------------------
    # Step 3: Generate Release file
    # ------------------------------------------------------------------
    release_text = generate_release(
        suite, component, arch, packages_bytes, packages_gz_bytes
    )
    release_bytes = release_text.encode("utf-8")
    print(f"Release SHA256: {sha256_of(release_bytes)}")

    # ------------------------------------------------------------------
    # Step 4: Write to temp dir, optionally GPG sign, then output
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        dist_dir = tmppath / "dists" / suite
        binary_dir = dist_dir / component / f"binary-{arch}"
        binary_dir.mkdir(parents=True)

        packages_file = binary_dir / "Packages"
        packages_gz_file = binary_dir / "Packages.gz"
        release_file = dist_dir / "Release"

        packages_file.write_bytes(packages_bytes)
        packages_gz_file.write_bytes(packages_gz_bytes)
        release_file.write_bytes(release_bytes)

        pkg_rel = f"dists/{suite}/{component}/binary-{arch}/Packages"
        pkg_gz_rel = f"dists/{suite}/{component}/binary-{arch}/Packages.gz"
        release_rel = f"dists/{suite}/Release"

        files: list[tuple[Path, str]] = [
            (packages_file, pkg_rel),
            (packages_gz_file, pkg_gz_rel),
            (release_file, release_rel),
        ]

        # GPG signing
        key_id = args.sign_key or os.environ.get("AGGREGATE_SIGN_KEY")
        if key_id:
            print(f"\nSigning Release with key: {key_id}")
            release_gpg, inrelease = gpg_sign(release_file, key_id)
            files += [
                (release_gpg, f"dists/{suite}/Release.gpg"),
                (inrelease, f"dists/{suite}/InRelease"),
            ]
        else:
            print("\nNo signing key — skipping GPG signing")
            print("  (Set --sign-key or AGGREGATE_SIGN_KEY env var to enable)")
            print("  (Clients must use [trusted=yes] in sources.list)")

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

    print("\n✓ Aggregate DEB metadata job complete")


if __name__ == "__main__":
    main()
