#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy therock-dist tarballs to their promoted S3 key.

Release and ASAN ``therock-dist-*.tar.gz`` files are renamed during promotion.
Their contents are not modified, so this copies each object to the new key
inside S3. It does not download the archive.

The version token is the last ``-`` segment of the filename, and it is
rewritten with the same rules as ``promote_packages.py``.

SAFETY FEATURES:
  - DRY-RUN by default. ``--execute`` performs the copy.
  - Source and destination buckets are explicit. There is no production default.
  - An existing destination key is skipped unless ``--allow-overwrite`` is set.

TYPICAL USAGE:
  # Preview a release-tarball rename (no copy):
  python ./build_tools/packaging/copy_promoted_tarballs.py \
    --version=7.13.0rc1 \
    --source-bucket=therock-repo-amd-rc-core \
    --dest-bucket=therock-repo-amd-stable-staging-core \
    --tarball-variant=release

  # Copy ASAN tarballs:
  python ./build_tools/packaging/copy_promoted_tarballs.py \
    --version=7.13.0rc1 \
    --source-bucket=therock-repo-amd-rc-core \
    --dest-bucket=therock-repo-amd-stable-staging-core \
    --tarball-variant=asan \
    --execute
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferManager
from botocore.exceptions import ClientError

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _therock_utils.python_package_paths import CORE_TARBALL_PREFIXES, core_tarball_prefix
from promote_packages import SRC_VERSION_PATTERNS, compute_new_version_str

_MISSING = {"404", "NoSuchKey", "NotFound"}


@dataclass(frozen=True)
class PlannedCopy:
    source_key: str
    dest_key: str
    size: int
    filename: str
    new_filename: str


def directory_prefix(prefix: str) -> str:
    """Return prefix with one trailing slash so ``tarball/`` does not match ``tarball-asan/``."""
    return prefix if prefix.endswith("/") else prefix + "/"


def promoted_tarball_name(
    filename: str, src_version_type: str, dest_version: str
) -> str | None:
    """Return the renamed tarball filename, or None when the version does not change."""
    stem = filename.removesuffix(".tar.gz")
    old_version = stem.split("-")[-1]
    new_version = compute_new_version_str(old_version, src_version_type, dest_version)
    if new_version is None:
        return None
    # Same replacement as promote_targz_tarball: the version is the last '-' segment.
    return stem.replace(old_version, new_version) + ".tar.gz"


def select_copies(
    objects: list[tuple[str, int]],
    source_prefix: str,
    dest_prefix: str,
    version: str,
    src_version_type: str,
    dest_version: str,
) -> tuple[list[PlannedCopy], list[str]]:
    """Select direct ``therock-dist-*.tar.gz`` children whose version token matches."""
    planned: list[PlannedCopy] = []
    unchanged: list[str] = []
    for key, size in objects:
        if not key.startswith(source_prefix):
            continue
        name = key[len(source_prefix) :]
        if "/" in name or not (
            name.startswith("therock-dist") and name.endswith(".tar.gz")
        ):
            continue
        old_version = name.removesuffix(".tar.gz").split("-")[-1]
        if old_version != version:
            continue
        new_name = promoted_tarball_name(name, src_version_type, dest_version)
        if new_name is None:
            unchanged.append(name)
            continue
        planned.append(
            PlannedCopy(
                source_key=key,
                dest_key=dest_prefix + new_name,
                size=size,
                filename=name,
                new_filename=new_name,
            )
        )
    return planned, unchanged


def list_object_sizes(client, bucket: str, prefix: str) -> list[tuple[str, int]]:
    paginator = client.get_paginator("list_objects_v2")
    found: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found.append((obj["Key"], obj["Size"]))
    return found


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in _MISSING:
            return False
        raise
    return True


def execute_copies(
    client,
    source_bucket: str,
    dest_bucket: str,
    planned: list[PlannedCopy],
    execute: bool,
    allow_overwrite: bool,
    copy_fn=None,
) -> tuple[int, int]:
    """Copy planned objects. Returns ``(copied, skipped_existing)``."""
    manager = None
    if execute and copy_fn is None:
        manager = TransferManager(client)

        def copy_fn(source_key: str, dest_key: str) -> None:
            manager.copy(
                {"Bucket": source_bucket, "Key": source_key},
                dest_bucket,
                dest_key,
            ).result()

    copied = 0
    skipped = 0
    try:
        for item in planned:
            same_object = (
                source_bucket == dest_bucket and item.source_key == item.dest_key
            )
            exists = object_exists(client, dest_bucket, item.dest_key)
            if same_object or (exists and not allow_overwrite):
                print(f"    [SKIP] Already exists: s3://{dest_bucket}/{item.dest_key}")
                skipped += 1
                continue
            destination = f"s3://{dest_bucket}/{item.dest_key}"
            source = f"s3://{source_bucket}/{item.source_key}"
            if execute:
                print(f"    Copying {source} -> {destination} ...", end="", flush=True)
                copy_fn(item.source_key, item.dest_key)
                print(" done")
            else:
                action = "overwrite" if exists else "copy"
                print(
                    f"    [DRY-RUN] Would {action}: {source} -> {destination} ({item.size} bytes)"
                )
            copied += 1
    finally:
        if manager is not None:
            manager.shutdown()
    return copied, skipped


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy therock-dist tarballs to their promoted S3 key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", required=True, help="Source version token, e.g. 7.13.0rc1")
    parser.add_argument("--source-bucket", required=True, help="Bucket that holds the source tarballs")
    parser.add_argument("--dest-bucket", required=True, help="Bucket that receives the renamed tarballs")
    parser.add_argument(
        "--tarball-variant",
        default="release",
        choices=sorted(CORE_TARBALL_PREFIXES),
        help="release -> v5/rocm/core/tarball/, asan -> v5/rocm/core/tarball-asan/",
    )
    parser.add_argument(
        "--source-prefix",
        default=None,
        help="Source key prefix. Defaults to the prefix for --tarball-variant.",
    )
    parser.add_argument(
        "--dest-prefix",
        default=None,
        help="Destination key prefix. Defaults to --source-prefix.",
    )
    parser.add_argument(
        "--src-version-type",
        default="rc",
        choices=list(SRC_VERSION_PATTERNS),
        help="Prerelease marker to rewrite (default: rc)",
    )
    parser.add_argument(
        "--dest-version",
        default="release",
        help="Version form to apply. 'release' strips the prerelease marker (default: release)",
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Perform the copy. Without this flag the command only prints the plan.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace a destination key that already exists (default: skip it)",
    )
    args = parser.parse_args(argv)
    source_prefix = args.source_prefix or core_tarball_prefix(args.tarball_variant)
    args.source_prefix = directory_prefix(source_prefix)
    args.dest_prefix = directory_prefix(args.dest_prefix or args.source_prefix)
    return args


def run(client, args: argparse.Namespace, copy_fn=None) -> int:
    print("=" * 80)
    print("Copy promoted tarballs")
    print("=" * 80)
    if not args.execute:
        print("DRY-RUN MODE: No copies will be performed")
    print(f"Version: {args.version}")
    print(f"Variant: {args.tarball_variant}")
    print(f"Source: s3://{args.source_bucket}/{args.source_prefix}")
    print(f"Destination: s3://{args.dest_bucket}/{args.dest_prefix}")

    objects = list_object_sizes(client, args.source_bucket, args.source_prefix)
    planned, unchanged = select_copies(
        objects,
        args.source_prefix,
        args.dest_prefix,
        args.version,
        args.src_version_type,
        args.dest_version,
    )
    for name in unchanged:
        print(f"    [SKIP] Version unchanged: {name}")
    if not planned:
        print(f"[ERROR]: No therock-dist tarballs matched version {args.version}")
        return 1

    total_bytes = sum(item.size for item in planned)
    print(f"Matched {len(planned)} tarball(s), {total_bytes} bytes")
    copied, skipped = execute_copies(
        client,
        args.source_bucket,
        args.dest_bucket,
        planned,
        args.execute,
        args.allow_overwrite,
        copy_fn=copy_fn,
    )
    print(f"Copies: {copied}")
    print(f"Skipped existing: {skipped}")
    return 0


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    client = boto3.client("s3")
    try:
        return run(client, args)
    except ClientError as error:
        print(f"[ERROR]: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
