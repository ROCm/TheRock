#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Publish a built amdrocm-repo package to the per-run outputs.

The destination is ``WorkflowOutputRoot.native_linux_repo_package()``, beside
the package index (see that method for the layout). Release promotion
(``publish_rocm_to_release_buckets.py``) copies it with the rest of the per-run
``packages/<format>/`` tree.

The bucket and prefix come from the CI environment: ``GITHUB_REPOSITORY``,
``RELEASE_TYPE`` and the event payload used for fork detection. The ``release``
line has no artifacts bucket and raises; the publishing job does not run for it.

Usage:
  python build_tools/packaging/linux/publish_repo_package.py \
      --file repo-package-out/amdrocm-repo-10.0.0-1.stable.el10.noarch.rpm \
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

# An os-profile becomes a path segment of the object key. The leading
# alphanumeric rejects "." and "..", which would escape a local staging
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
    # Log where the file actually goes: with --output-dir that is a local path,
    # and an s3:// URI would describe an upload that did not happen.
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
