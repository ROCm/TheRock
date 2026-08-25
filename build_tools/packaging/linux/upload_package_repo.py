#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Packaging + repository upload tool.

Builds local repository metadata (createrepo_c or dpkg-scanpackages), uploads
packages to S3 with optional deduplication, then uploads the local metadata as-is.

Usage:
  python ./build_tools/packaging/linux/upload_package_repo.py \
    --pkg-type deb \
    --run-id 16418185899 \
    --package-dir /path/to/packages

Bucket + prefix are resolved automatically via WorkflowOutputRoot using
the GITHUB_REPOSITORY and RELEASE_TYPE environment variables:
  - CI builds    → therock-ci-artifacts / <run_id>-linux/packages/deb
  - dev builds   → therock-dev-artifacts / <run_id>-linux/packages/deb
  - nightly      → therock-nightly-artifacts / <run_id>-linux/packages/deb
  - prerelease   → therock-prerelease-artifacts / <run_id>-linux/packages/deb
"""

import argparse
import boto3
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _THIS_DIR.parent.parent
_GITHUB_ACTIONS_DIR = _BUILD_TOOLS_DIR / "github_actions"

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))
if str(_GITHUB_ACTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_GITHUB_ACTIONS_DIR))

from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions_api import gha_append_step_summary


def generate_release_file_with_checksums(release_file, job_type, dists_dir):
    """Generate a Debian Release file with MD5Sum, SHA1, and SHA256 checksums.

    Args:
        release_file: Path to the Release file to create
        job_type: Job type for metadata (nightly/dev/release)
        dists_dir: Directory containing Packages files (main/binary-amd64/)
    """
    import hashlib

    # Files to hash (relative paths from dists/stable/)
    files_to_hash = [
        (dists_dir / "Packages", "main/binary-amd64/Packages"),
        (dists_dir / "Packages.gz", "main/binary-amd64/Packages.gz"),
    ]

    # Calculate all hashes
    md5_entries = []
    sha1_entries = []
    sha256_entries = []

    for file_path, rel_path in files_to_hash:
        if not file_path.exists():
            continue

        # Get file size
        file_size = file_path.stat().st_size

        # Calculate hashes
        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                data = f.read(65536)  # Read in 64KB chunks
                if not data:
                    break
                md5_hash.update(data)
                sha1_hash.update(data)
                sha256_hash.update(data)

        # Store entries (space-aligned format)
        md5_entries.append(f" {md5_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha1_entries.append(f" {sha1_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha256_entries.append(f" {sha256_hash.hexdigest()} {file_size:16d} {rel_path}")

    # Write Release file
    with open(release_file, "w", encoding="utf-8") as f:
        # Header fields
        f.write(
            f"""Origin: AMD ROCm
Label: ROCm {job_type} Packages
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Description: ROCm APT Repository
Date: {datetime.datetime.utcnow():%a, %d %b %Y %H:%M:%S UTC}
"""
        )

        # MD5Sum section
        if md5_entries:
            f.write("MD5Sum:\n")
            f.write("\n".join(md5_entries))
            f.write("\n")

        # SHA1 section
        if sha1_entries:
            f.write("SHA1:\n")
            f.write("\n".join(sha1_entries))
            f.write("\n")

        # SHA256 section
        if sha256_entries:
            f.write("SHA256:\n")
            f.write("\n".join(sha256_entries))
            f.write("\n")

    print(f"✅ Release file generated with checksums: MD5, SHA1, SHA256")


def run_command(cmd: list[str], cwd=None, stdout=None):
    """Run a command safely without shell interpolation.

    Args:
        cmd: Command and arguments as a list of strings
        cwd: Working directory for the command
        stdout: Optional path to redirect stdout to a file

    Raises:
        subprocess.CalledProcessError: If the command exits with non-zero status
    """
    print(f"Running: {' '.join(cmd)}")
    if stdout is not None:
        with open(stdout, "wb") as f:
            subprocess.run(cmd, check=True, cwd=cwd, stdout=f)
    else:
        subprocess.run(cmd, check=True, cwd=cwd)


def find_package_dir():
    base = Path.cwd() / "output" / "packages"
    if not base.exists():
        raise RuntimeError(f"Package directory not found: {base}")
    return base


def s3_object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def create_deb_repo(package_dir, job_type):
    print("Creating APT repository...")

    package_path = Path(package_dir)
    dists = package_path / "dists" / "stable" / "main" / "binary-amd64"
    pool = package_path / "pool" / "main"

    dists.mkdir(parents=True, exist_ok=True)
    pool.mkdir(parents=True, exist_ok=True)

    for f in package_path.iterdir():
        if f.suffix == ".deb":
            shutil.move(f, pool / f.name)

    run_command(
        ["dpkg-scanpackages", "-m", "pool/main", "/dev/null"],
        cwd=str(package_path),
        stdout=dists / "Packages",
    )
    run_command(
        ["gzip", "-9c", "Packages"],
        cwd=str(dists),
        stdout=dists / "Packages.gz",
    )

    release = package_path / "dists" / "stable" / "Release"
    generate_release_file_with_checksums(release, job_type, dists)


def create_rpm_repo(package_dir):
    """Create RPM repository metadata from the complete local build tree."""
    print("Creating RPM repository...")

    package_path = Path(package_dir)
    arch_dir = package_path / "x86_64"
    arch_dir.mkdir(parents=True, exist_ok=True)

    for f in package_path.iterdir():
        if f.suffix == ".rpm":
            shutil.move(f, arch_dir / f.name)

    run_command(
        ["createrepo_c", "--no-database", "--simple-md-filenames", "."],
        cwd=str(arch_dir),
    )


def upload_to_s3(source_dir, bucket, prefix, dedupe=False):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0

    for root, _, files in os.walk(source_dir):
        for fname in files:
            # Always skip local index.html files, those are generated server-side.
            if fname == "index.html":
                continue

            # Skip build manifest files - these are for local tracking only
            if fname.lower().endswith(".txt"):
                print(f"Skipping build manifest file (local only): {fname}")
                continue

            local = Path(root) / fname
            rel = local.relative_to(source_dir)
            key = f"{prefix}/{rel.as_posix()}"

            # Dedupe applies to package files only; metadata is always re-uploaded.
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
    package_dir: Path,
    pkg_type: str,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
) -> str | None:
    """Upload native packaging logs to S3.

    Uploads all log files from the packaging logs directory to the S3 location
    specified by WorkflowOutputRoot.native_linux_packages_log_dir().

    Args:
        package_dir: Directory containing built packages (logs are in logs/ subdir)
        pkg_type: Package type ('deb' or 'rpm')
        output_root: WorkflowOutputRoot for computing S3 paths
        backend: Storage backend for uploads

    Returns:
        URL to the log index page, or None if no logs were uploaded
    """
    log_dir = package_dir / "logs"
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
    """Write a key=value pair to $GITHUB_OUTPUT if running in GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def _package_install_url(bucket: str, prefix: str, pkg_type: str) -> str:
    """Compute the package manager install URL for a given repo location.

    For RPM repos, dnf/yum baseurl must point to the x86_64/ subdirectory
    (the directory containing repodata/). For DEB repos, apt points to the
    repo root (it resolves dists/ itself).
    """
    base = StorageLocation(bucket, prefix).https_url
    if pkg_type == "rpm":
        return f"{base}/x86_64"
    return base


def _resolve_upload_target(
    args: argparse.Namespace,
    pkg_type: str,
) -> tuple[str, str, str, bool, str]:
    """Resolve S3 bucket, prefix, install URL, dedupe flag, and job type.

    Returns:
        Tuple of (bucket, prefix, install_url, dedupe, job_type)
    """
    # Derive bucket + prefix from WorkflowOutputRoot.
    # This is the single source of truth for CI path layout.
    root = WorkflowOutputRoot.from_workflow_run(run_id=args.run_id, platform="linux")
    loc = root.native_linux_packages(pkg_type)
    job_type = os.environ.get("RELEASE_TYPE", "ci")
    install_url = _package_install_url(loc.bucket, loc.relative_path, pkg_type)
    return loc.bucket, loc.relative_path, install_url, True, job_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])

    # Use WorkflowOutputRoot for bucket/prefix resolution.
    # Bucket and prefix are derived automatically from CI context
    # (GITHUB_REPOSITORY, RELEASE_TYPE, fork detection).
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions workflow run ID (required for WorkflowOutputRoot path resolution)",
    )

    parser.add_argument(
        "--package-dir",
        required=True,
        help="Path to the directory containing built packages.",
    )

    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()

    bucket, prefix, install_url, dedupe, job_type = _resolve_upload_target(
        args, args.pkg_type
    )

    # Build local repo metadata, then upload packages (dedupe OK) and metadata.
    if args.pkg_type == "deb":
        create_deb_repo(package_dir, job_type)
    else:
        create_rpm_repo(package_dir)

    upload_to_s3(package_dir, bucket, prefix, dedupe=dedupe)

    print(f"Package repository URL: {install_url}")
    _emit_github_output("package_repository_url", install_url)

    # Upload packaging logs and write step summary
    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform="linux"
    )
    backend = create_storage_backend()
    log_index_url = upload_packaging_logs(
        package_dir, args.pkg_type, output_root, backend
    )
    if log_index_url:
        _emit_github_output("packaging_logs_url", log_index_url)

    # Write GitHub Actions step summary
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
