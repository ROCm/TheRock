#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Download a run's amd-repo-setup package for one distro.

The reverse of ``upload_repo_setup_package.py``: it finds the package that
upload put in the run's artifacts bucket and downloads it, so a test can
install it.

    {run_id}-linux/amd-repo-setup/<os_profile>/<file>

The artifacts buckets are publicly readable, so this needs no credentials: it
lists the distro's folder and downloads the one package over https. The
bucket comes from ``WorkflowOutputRoot``, the same way upload picks it.

Usage:
  python build_tools/packaging/linux/fetch_repo_setup_package.py \
      --run-id 12345678901 \
      --release-type dev \
      --os-profile rhel10 \
      --pkg-type rpm \
      --dest-dir repo-setup-package
"""

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _THIS_DIR.parent.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# Accept exactly what upload writes.
from upload_repo_setup_package import _FILENAME_RE, _OS_PROFILE_RE, _RUN_ID_RE

_TIMEOUT_SEC = 60
_KEY_RE = re.compile(r"<Key>([^<]*)</Key>")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a run's amd-repo-setup package for one distro.",
    )
    p.add_argument("--run-id", required=True, help="Workflow run ID that uploaded it")
    p.add_argument(
        "--release-type",
        required=True,
        help="Release type of that run, which selects its artifacts bucket",
    )
    p.add_argument("--os-profile", required=True, help="Target distro profile")
    p.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    p.add_argument(
        "--dest-dir", required=True, type=Path, help="Directory to download into"
    )
    args = p.parse_args(argv)
    if not _RUN_ID_RE.match(args.run_id):
        p.error(f"--run-id must be numeric: {args.run_id!r}")
    if not _OS_PROFILE_RE.match(args.os_profile):
        p.error(f"--os-profile has an unexpected format: {args.os_profile!r}")
    return args


def _bucket_url(bucket: str) -> str:
    """Return the bucket's public https endpoint."""
    return StorageLocation(bucket, "").https_url.rstrip("/")


def find_package(listing: str, folder: str, pkg_type: str) -> str:
    """Return the one package's object key in an S3 listing of ``folder``.

    The folder also holds a server-made ``index.html``, so only keys of this
    package type count. Anything other than exactly one is an error rather than
    a guess about which file to test.
    """
    keys = [
        k
        for k in _KEY_RE.findall(listing)
        if k.startswith(folder) and k.endswith(f".{pkg_type}")
    ]
    if len(keys) != 1:
        raise FileNotFoundError(
            f"expected exactly one .{pkg_type} in {folder}, found {len(keys)}: {keys}"
        )
    name = keys[0][len(folder) :]
    if not _FILENAME_RE.match(name):
        raise ValueError(f"package has an unexpected file name: {name!r}")
    return keys[0]


def fetch(args: argparse.Namespace) -> Path:
    """Download the package into ``args.dest_dir`` and return its path."""
    root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform="linux", release_type=args.release_type
    )
    folder = f"{root.repo_setup_packages().relative_path}/{args.os_profile}/"
    base = _bucket_url(root.bucket)
    query = urllib.parse.urlencode({"list-type": "2", "prefix": folder})
    print(f"Listing {base}/{folder}")
    with urllib.request.urlopen(f"{base}/?{query}", timeout=_TIMEOUT_SEC) as resp:
        listing = resp.read().decode("utf-8")
    key = find_package(listing, folder, args.pkg_type)

    url = f"{base}/{urllib.parse.quote(key)}"
    dest = args.dest_dir / key.rsplit("/", 1)[1]
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SEC) as resp:
        dest.write_bytes(resp.read())
    print(f"Downloaded: {dest}")
    return dest


def main(argv=None) -> None:
    fetch(parse_args(argv))


if __name__ == "__main__":
    main()
