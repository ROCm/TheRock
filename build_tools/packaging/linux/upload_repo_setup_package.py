#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Upload a built amd-repo-setup package to the per-run outputs.

Not to be confused with ``upload_package_repo.py``, which uploads the package
repository itself (the content packages and their index).

The destination is
``{run_id}-linux/amd-repo-setup/{os_profile}/amd-repo-setup-{tier}.{pkg_type}``
(e.g. ``12345678901-linux/amd-repo-setup/rhel10/amd-repo-setup-stablerc.rpm``).
It sits outside ``packages/<format>/``, so release promotion
(``publish_rocm_to_release_buckets.py``) does not copy it. Publishing it to
repo.amd.com is a separate step.

The bucket and prefix come from the CI environment: ``GITHUB_REPOSITORY``,
``RELEASE_TYPE`` and the event payload used for fork detection. The ``release``
line has no artifacts bucket and raises; the upload job does not run for it.

Usage:
  python build_tools/packaging/linux/upload_repo_setup_package.py \
      --dest-dir repo-setup-package-out \
      --tier stablerc \
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
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.github_actions_api import gha_set_output

# An os-profile becomes a path segment of the object key. The leading
# alphanumeric rejects "." and "..", which would escape a local staging
# directory. \Z (not $) so a trailing newline is not accepted.
_OS_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# A workflow run id is always numeric, and it becomes the leading path segment
# of the object key via WorkflowOutputRoot.prefix. Unconstrained, a value such
# as "../.." escapes the destination once the key is joined onto a local
# staging directory. \Z (not $) so a trailing newline is not accepted.
_RUN_ID_RE = re.compile(r"^[0-9]+\Z")

# The tier becomes part of the object's file name (stable, stablerc, nightly).
# Lowercase letters and digits only, so it cannot add a path segment.
_TIER_RE = re.compile(r"^[a-z][a-z0-9]*\Z")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload an amd-repo-setup package to the per-run outputs.",
    )
    p.add_argument(
        "--dest-dir",
        required=True,
        type=Path,
        help="Builder output directory holding exactly one built package",
    )
    p.add_argument(
        "--tier",
        required=True,
        help="Repository tier, used in the uploaded file name (e.g. 'nightly')",
    )
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
        help="Package type, used to find the package and as the file extension",
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
    if not _TIER_RE.match(args.tier):
        p.error(f"--tier has an unexpected format: {args.tier!r}")
    if not args.dest_dir.is_dir():
        p.error(f"--dest-dir is not a directory: {args.dest_dir}")
    return args


def find_package(dest_dir: Path, pkg_type: str) -> Path:
    """Return the single built package in ``dest_dir``.

    One build produces one package, so anything other than exactly one match is
    an error rather than a guess about which file to upload.
    """
    built = sorted(dest_dir.glob(f"*.{pkg_type}"))
    if len(built) != 1:
        raise ValueError(
            f"expected exactly one .{pkg_type} in {dest_dir}, found {len(built)}"
        )
    return built[0]


def upload(args: argparse.Namespace) -> str:
    """Upload the built package to its per-run location. Returns the object key.

    After an upload to S3, the package's https URL is written to
    ``$GITHUB_OUTPUT`` as ``repo_setup_package_url``. A dry run or a local
    staging run uploads nothing there, so it writes no output.
    """
    package = find_package(args.dest_dir, args.pkg_type)
    root = WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform="linux")
    dest = StorageLocation(
        root.bucket,
        f"{root.prefix}/amd-repo-setup/{args.os_profile}"
        f"/amd-repo-setup-{args.tier}.{args.pkg_type}",
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    # Log where the file actually goes: with --output-dir that is a local path,
    # and an s3:// URI would describe an upload that did not happen.
    target = (
        dest.local_path(args.output_dir) if args.output_dir is not None else dest.s3_uri
    )
    action = "[DRY RUN] Would upload" if args.dry_run else "Uploading"
    print(f"{action} {package} -> {target}")
    backend.upload_file(package, dest)
    if not args.dry_run:
        print(f"Uploaded: {target}")
        if args.output_dir is None:
            gha_set_output({"repo_setup_package_url": dest.https_url})
    return dest.relative_path


def main(argv=None) -> None:
    upload(parse_args(argv))


if __name__ == "__main__":
    main()
