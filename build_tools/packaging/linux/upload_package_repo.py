#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Upload native Linux package repositories to S3.

Expects ``--package-dir`` to already contain repo metadata from
``build_package_repo.py`` (``dists/`` or ``repodata/``). Uploads packages
with optional dedupe and always re-uploads metadata (Issue #6540).

CI flow (``multi_arch_build_native_linux_packages.yml``)::

  build_package.py → simulate test → build_package_repo.py → this script

Usage:
  python ./build_tools/packaging/linux/upload_package_repo.py \\
    --pkg-type deb \\
    --run-id 16418185899 \\
    --package-dir /path/to/packages

Bucket + prefix are resolved via ``WorkflowOutputRoot`` from
``GITHUB_REPOSITORY``, ``RELEASE_TYPE``, and fork detection:

  - CI builds    → therock-ci-artifacts / ``<run_id>-linux/packages/{deb|rpm}``
  - dev builds   → therock-dev-artifacts / ...
  - nightly      → therock-nightly-artifacts / ...
  - prerelease   → therock-prerelease-artifacts / ...
"""

import argparse
import boto3
import botocore.client
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _THIS_DIR.parent.parent
_GITHUB_ACTIONS_DIR = _BUILD_TOOLS_DIR / "github_actions"

for path in (_THIS_DIR, _BUILD_TOOLS_DIR, _GITHUB_ACTIONS_DIR):
    path_str = os.fspath(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions_api import gha_append_step_summary


def s3_object_exists(s3: botocore.client.BaseClient, bucket: str, key: str) -> bool:
    """Return whether an S3 object exists at ``bucket``/``key``.

    Used by package dedupe in :func:`upload_to_s3`. A 404 response means the
    object is absent; other ``ClientError`` codes propagate to the caller.

    Args:
        s3: boto3 S3 client (or compatible mock).
        bucket: S3 bucket name.
        key: Object key within the bucket.

    Returns:
        ``True`` if ``head_object`` succeeds, ``False`` on 404.
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_to_s3(
    source_dir: Path | str,
    bucket: str,
    prefix: str,
    dedupe: bool = False,
) -> botocore.client.BaseClient:
    """Upload packages and repository metadata under ``source_dir`` to S3.

    Walks the full tree produced by ``build_package_repo.py``. Repository metadata
    (``repodata/`` for RPM, ``dists/`` for DEB) is uploaded like any other file —
    it is **not** skipped and **not** deduped.

    Package files (``.deb`` / ``.rpm``) may be skipped when ``dedupe`` is True and
    the object already exists on S3. Metadata is always re-uploaded so a CI re-run
    with all packages deduped still refreshes ``repodata`` / ``Packages`` from the
    local tree (Issue #6540).

    Args:
        source_dir: Local package tree (``PACKAGE_DIST_DIR`` in CI).
        bucket: S3 bucket name.
        prefix: S3 key prefix for this repo (no trailing slash).
        dedupe: When True, skip uploading package files that already exist on S3.

    Returns:
        boto3 S3 client used for the upload walk.
    """
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0

    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname == "index.html":
                continue

            if fname.lower().endswith(".txt"):
                print(f"Skipping build manifest file (local only): {fname}")
                continue

            local = Path(root) / fname
            rel = local.relative_to(source_dir)
            key = f"{prefix}/{rel.as_posix()}"

            # Issue #6540: dedupe packages only — repodata/ and dists/ always upload.
            if dedupe and (fname.endswith(".deb") or fname.endswith(".rpm")):
                if s3_object_exists(s3, bucket, key):
                    print(f"Skipping existing package: {fname}")
                    skipped += 1
                    continue

            extra = {"ContentType": "text/html"} if fname.endswith(".html") else None

            print(f"Uploading: {key}")
            s3.upload_file(str(local), bucket, key, ExtraArgs=extra)
            uploaded += 1

    print(f"Uploaded: {uploaded}, Skipped: {skipped}")

    return s3


def upload_packaging_logs(
    log_dir: Path,
    pkg_type: str,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
) -> str | None:
    """Upload native packaging logs to S3.

    Uploads all log files from the packaging logs directory to the S3 location
    specified by WorkflowOutputRoot.native_linux_packages_log_dir().

    Args:
        log_dir: Directory containing packaging log files. Kept outside of
        package_dir so upload_to_s3() never walks/uploads these files
        to the packages/<pkg_type> S3 prefix.
        pkg_type: Package type ('deb' or 'rpm')
        output_root: WorkflowOutputRoot for computing S3 paths
        backend: Storage backend for uploads

    Returns:
        URL to the log index page, or None if no logs were uploaded
    """
    if not log_dir.is_dir():
        print(f"[INFO] Log directory {log_dir} not found. Skipping log upload.")
        return None

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print(f"[INFO] No log files found in {log_dir}. Skipping log upload.")
        return None

    print(f"Uploading {len(log_files)} packaging log files...")
    dest_location = output_root.native_linux_packages_log_dir(pkg_type)
    backend.upload_directory(log_dir, dest_location)

    log_index_url = output_root.native_linux_packages_log_index(pkg_type).https_url
    print(f"Packaging logs uploaded to: {log_index_url}")
    return log_index_url


def _emit_github_output(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_OUTPUT`` when running in GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def _package_install_url(bucket: str, prefix: str, pkg_type: str) -> str:
    """Compute the package-manager install URL for a repository location.

    RPM ``baseurl`` must point at ``x86_64/`` (where ``repodata/`` lives).
    DEB clients use the repo root; apt resolves ``dists/`` itself.

    Args:
        bucket: S3 artifact bucket name.
        prefix: S3 key prefix for this native package repo.
        pkg_type: ``deb`` or ``rpm``.

    Returns:
        HTTPS URL suitable for ``sources.list`` or ``.repo`` ``baseurl``.
    """
    base = StorageLocation(bucket, prefix).https_url
    if pkg_type == "rpm":
        return f"{base}/x86_64"
    return base


def _resolve_upload_target(
    args: argparse.Namespace,
    pkg_type: str,
) -> tuple[str, str, str, bool]:
    """Resolve S3 destination and install URL for a native package upload.

    Dedupe is always enabled for CI uploads: re-runs share the same S3 prefix
    under ``WorkflowOutputRoot``. Only ``.deb`` / ``.rpm`` objects are skipped
    when already present; metadata is rebuilt locally and re-uploaded each run.

    Args:
        args: Parsed CLI namespace (must include ``run_id``).
        pkg_type: ``deb`` or ``rpm``.

    Returns:
        ``(bucket, prefix, install_url, dedupe)`` where ``dedupe`` is always
        ``True`` for workflow-driven uploads.
    """
    root = WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform="linux")
    loc = root.native_linux_packages(pkg_type)
    install_url = _package_install_url(loc.bucket, loc.relative_path, pkg_type)
    return loc.bucket, loc.relative_path, install_url, True


def main() -> None:
    """Upload repo tree to S3, emit workflow outputs, and write the step summary."""
    parser = argparse.ArgumentParser(
        description="Upload native Linux package repository artifacts to S3.",
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package format (deb or rpm).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions workflow run ID (WorkflowOutputRoot path resolution).",
    )
    parser.add_argument(
        "--package-dir",
        required=True,
        help="Path to the directory containing packages and repo metadata.",
    )

    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()

    bucket, prefix, install_url, dedupe = _resolve_upload_target(args, args.pkg_type)

    upload_to_s3(package_dir, bucket, prefix, dedupe=dedupe)

    print(f"Package repository URL: {install_url}")
    _emit_github_output("package_repository_url", install_url)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform="linux"
    )
    backend = create_storage_backend()
    logs_dir = package_dir.parent / "logs"
    log_index_url = upload_packaging_logs(logs_dir, args.pkg_type, output_root, backend)
    if log_index_url:
        _emit_github_output("packaging_logs_url", log_index_url)

    pkg_type_upper = args.pkg_type.upper()
    if args.pkg_type == "deb":
        install_instructions = f"""### {pkg_type_upper} Package Build Results

[Package Repository]({install_url})

```bash
# Add the repository
echo "deb [trusted=yes] {install_url} stable main" | \\
    sudo tee /etc/apt/sources.list.d/therock.list
sudo apt update

# Install packages
sudo apt install <package-name>
```
"""
    else:
        install_instructions = f"""### {pkg_type_upper} Package Build Results

[Package Repository]({install_url})

```bash
# Add the repository
sudo tee /etc/yum.repos.d/therock.repo << 'EOF'
[therock]
name=TheRock
baseurl={install_url}
enabled=1
gpgcheck=0
EOF

# Install packages
sudo dnf install <package-name>
```
"""
    if log_index_url:
        install_instructions += f"\n[Packaging Logs]({log_index_url})\n"
    gha_append_step_summary(install_instructions)


if __name__ == "__main__":
    main()
