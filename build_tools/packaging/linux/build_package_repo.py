#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Build Debian or RPM repository metadata from a local native packaging tree.

Called from ``multi_arch_build_native_linux_packages.yml`` **after** the simulate
install test and **before** ``upload_package_repo.py``. Simulate must run first
because this script moves loose ``.deb`` / ``.rpm`` files into standard repo
layout (``pool/main/`` or ``x86_64/``) before indexing.

Output is consumed unchanged by ``upload_package_repo.py`` (Issue #6540: local
metadata is authoritative; no S3 merge/regen after upload).

Usage:
  python ./build_tools/packaging/linux/build_package_repo.py \\
    --pkg-type deb \\
    --package-dir /path/to/packages

``RELEASE_TYPE`` (default ``ci``) labels DEB ``Release`` metadata for nightly,
dev, or release builds.
"""

import argparse
import datetime
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Bytes read per iteration when hashing Packages files for Release checksums.
_HASH_READ_CHUNK_BYTES = 65536

_THIS_DIR = Path(__file__).resolve().parent

if os.fspath(_THIS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(_THIS_DIR))


def generate_release_file_with_checksums(
    release_file: Path | str,
    job_type: str,
    dists_dir: Path,
) -> None:
    """Generate a Debian ``Release`` file with MD5Sum, SHA1, and SHA256 sections.

    Reads ``Packages`` and ``Packages.gz`` from ``dists_dir`` and writes a
    upload-ready ``Release`` file. ``apt update`` requires checksum sections; a
    header-only ``Release`` is not sufficient for clients.

    Args:
        release_file: Path to the ``Release`` file to create (typically
            ``dists/stable/Release`` under the package tree).
        job_type: Release label suffix (from ``RELEASE_TYPE``, e.g. ``nightly``).
        dists_dir: Directory containing ``Packages`` / ``Packages.gz``
            (``dists/stable/main/binary-amd64/``).
    """
    files_to_hash = [
        (dists_dir / "Packages", "main/binary-amd64/Packages"),
        (dists_dir / "Packages.gz", "main/binary-amd64/Packages.gz"),
    ]

    md5_entries: list[str] = []
    sha1_entries: list[str] = []
    sha256_entries: list[str] = []

    for file_path, rel_path in files_to_hash:
        if not file_path.exists():
            continue

        file_size = file_path.stat().st_size

        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                data = f.read(_HASH_READ_CHUNK_BYTES)
                if not data:
                    break
                md5_hash.update(data)
                sha1_hash.update(data)
                sha256_hash.update(data)

        md5_entries.append(f" {md5_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha1_entries.append(f" {sha1_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha256_entries.append(f" {sha256_hash.hexdigest()} {file_size:16d} {rel_path}")

    with open(release_file, "w", encoding="utf-8") as f:
        f.write(
            f"""Origin: AMD ROCm
Label: ROCm {job_type} Packages
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Description: ROCm APT Repository
Date: {datetime.datetime.now(datetime.timezone.utc):%a, %d %b %Y %H:%M:%S UTC}
"""
        )

        if md5_entries:
            f.write("MD5Sum:\n")
            f.write("\n".join(md5_entries))
            f.write("\n")

        if sha1_entries:
            f.write("SHA1:\n")
            f.write("\n".join(sha1_entries))
            f.write("\n")

        if sha256_entries:
            f.write("SHA256:\n")
            f.write("\n".join(sha256_entries))
            f.write("\n")

    print("✅ Release file generated with checksums: MD5, SHA1, SHA256")


def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    stdout: Path | str | None = None,
) -> None:
    """Run a subprocess without shell interpolation.

    Args:
        cmd: Executable and arguments as a list of strings.
        cwd: Working directory for the child process.
        stdout: When set, redirect child stdout to this file path.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
        OSError: If ``stdout`` cannot be opened for writing.
    """
    print(f"Running: {' '.join(cmd)}")
    if stdout is not None:
        with open(stdout, "wb") as f:
            subprocess.run(cmd, check=True, cwd=cwd, stdout=f)
    else:
        subprocess.run(cmd, check=True, cwd=cwd)


def create_deb_repo(package_dir: Path | str, job_type: str) -> None:
    """Build Debian repository metadata from the complete local ``.deb`` tree.

    Moves top-level ``.deb`` files into ``pool/main/``, runs ``dpkg-scanpackages``
    and ``gzip`` to produce ``Packages`` / ``Packages.gz``, then writes
    ``dists/stable/Release`` with checksums.

    Args:
        package_dir: Root of the native packaging output tree (``PACKAGE_DIST_DIR``
            in CI). Modified in place.
        job_type: ``RELEASE_TYPE`` value for ``Release`` file labeling.

    Raises:
        subprocess.CalledProcessError: If ``dpkg-scanpackages`` or ``gzip`` fails.
    """
    print("Creating APT repository...")

    package_path = Path(package_dir)
    dists = package_path / "dists" / "stable" / "main" / "binary-amd64"
    pool = package_path / "pool" / "main"

    dists.mkdir(parents=True, exist_ok=True)
    pool.mkdir(parents=True, exist_ok=True)

    # Flat .deb files from build_package.py → pool layout before scan.
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


def create_rpm_repo(package_dir: Path | str) -> None:
    """Create RPM ``repodata/`` from the complete local build tree.

    Moves top-level ``.rpm`` files into ``x86_64/`` and runs ``createrepo_c`` so
    ``repodata/`` indexes every package present locally (Fixes #6540 on CI re-runs
    when ``upload_package_repo.py`` dedupes package uploads).

    Args:
        package_dir: Root of the native packaging output tree. Modified in place.

    Raises:
        subprocess.CalledProcessError: If ``createrepo_c`` fails.
    """
    print("Creating RPM repository...")

    package_path = Path(package_dir)
    arch_dir = package_path / "x86_64"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Flat .rpm files from build_package.py → x86_64/ before createrepo_c.
    for f in package_path.iterdir():
        if f.suffix == ".rpm":
            shutil.move(f, arch_dir / f.name)

    run_command(
        ["createrepo_c", "--no-database", "--simple-md-filenames", "."],
        cwd=str(arch_dir),
    )


def main() -> None:
    """Parse CLI args and build repository metadata for ``deb`` or ``rpm``."""
    parser = argparse.ArgumentParser(
        description="Build Debian or RPM repository metadata from a local package tree.",
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package format to index (deb or rpm).",
    )
    parser.add_argument(
        "--package-dir",
        required=True,
        help="Path to the directory containing built packages (modified in place).",
    )
    args = parser.parse_args()

    package_dir = Path(args.package_dir).resolve()
    job_type = os.environ.get("RELEASE_TYPE", "ci")

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, job_type)
    else:
        create_rpm_repo(package_dir)


if __name__ == "__main__":
    main()
