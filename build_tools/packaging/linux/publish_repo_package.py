#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a built amdrocm-repo package as a standalone per-distro file.

The package is uploaded next to the native content packages, as a
``repo/<os-profile>/`` sibling of the per-run ``packages/<format>/`` prefix,
rather than into the repository index. A client can then fetch it directly by
URL to configure the repository. The file becomes publicly reachable once the
release promotion step copies the per-run tree to the public CDN.

The object name is fixed (``amdrocm-repo.<ext>``) so the download URL is stable
regardless of the built package's versioned filename.

The destination is resolved through ``WorkflowOutputRoot``, the single source of
truth for CI path layout, so the bucket and per-run prefix come from the
workflow context rather than being passed in. That needs the CI environment:
``GITHUB_REPOSITORY``, ``RELEASE_TYPE``, and the event payload used for fork
detection. There is no artifacts bucket for the ``release`` line, so that
release type raises; the publishing job is expected to be gated off it.

Usage:
  python build_tools/packaging/linux/publish_repo_package.py \
      --file repo-package-out/amdrocm-repo-7.14.0-1.el10.noarch.rpm \
      --run-id 12345678901 \
      --os-profile rhel10 \
      --pkg-type rpm
"""

import argparse
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _THIS_DIR.parent.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# An os-profile becomes a path segment in the object key, so restrict it to a
# safe set of characters (no slashes, whitespace, or control characters) and
# require a leading alphanumeric. The leading character matters: without it,
# "." and ".." satisfy the character class, and ".." resolves one directory
# above the intended location when the key is joined onto a local staging
# directory. \Z (not $) so a trailing newline is not accepted.
_OS_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# A workflow run id is always numeric, and it becomes the leading path segment
# of the object key via WorkflowOutputRoot.prefix. Unconstrained, a value such
# as "../.." escapes the destination once the key is joined onto a local
# staging directory. \Z (not $) so a trailing newline is not accepted.
_RUN_ID_RE = re.compile(r"^[0-9]+\Z")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish an amdrocm-repo package to the per-run outputs.",
    )
    p.add_argument("--file", required=True, type=Path, help="Built package to upload")
    p.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions workflow run ID, which selects the per-run prefix",
    )
    p.add_argument(
        "--os-profile", required=True, help="Target distro profile (path segment)"
    )
    p.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package type, used as the published file extension",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Stage into this local directory instead of uploading to S3",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the destination without writing anything",
    )
    args = p.parse_args(argv)
    if not _RUN_ID_RE.match(args.run_id):
        p.error(f"--run-id must be numeric: {args.run_id!r}")
    if not _OS_PROFILE_RE.match(args.os_profile):
        p.error(f"--os-profile has an unexpected format: {args.os_profile!r}")
    if not args.file.is_file():
        p.error(f"--file is not a file: {args.file}")
    return args


def publish(args: argparse.Namespace) -> str:
    """Upload the file to its per-run location. Returns the object key."""
    root = WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform="linux")
    dest = root.native_linux_repo_package(args.pkg_type, args.os_profile)
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    # Name the destination the run actually writes to. --output-dir stages to a
    # local tree, so reporting an s3:// URI there would describe an upload that
    # did not happen. The publishing job runs under continue-on-error, which
    # makes this log the only signal that it did anything.
    target = (
        dest.local_path(args.output_dir) if args.output_dir is not None else dest.s3_uri
    )
    action = "[DRY RUN] Would publish" if args.dry_run else "Uploading"
    print(f"{action} {args.file} -> {target}")
    backend.upload_file(args.file, dest)
    if not args.dry_run:
        print(f"Published: {target}")
    return dest.relative_path


def main(argv=None) -> None:
    publish(parse_args(argv))


if __name__ == "__main__":
    main()
