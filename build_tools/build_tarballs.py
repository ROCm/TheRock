#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fetch multi-arch build artifacts and package them into per-family tarballs.

For each GPU family in --dist-amdgpu-families, this script:
1. Fetches artifacts (generic + family-specific) using artifact_manager.py
2. Flattens them into a single install-prefix-like layout
3. Compresses the result into a tarball

When KPACK_SPLIT_ARTIFACTS is enabled in the build manifest, device-specific
files are split by individual GPU target and don't conflict across families.
In that case, this script also produces a combined multi-arch tarball
containing all targets in a single install prefix.

A shared download cache avoids re-downloading generic (host) artifacts
when processing multiple families.

Tarball naming follows the existing release convention:
    therock-dist-{platform}-{family}-{version}.tar.gz
    therock-dist-{platform}-multiarch-{version}.tar.gz  (KPACK split only)

Example
-------
    python build_tools/build_tarballs.py \\
        --run-id=24104028483 \\
        --dist-amdgpu-families="gfx94X-dcgpu;gfx110X-all" \\
        --platform=linux \\
        --package-version="7.13.0.dev0+abc123" \\
        --output-dir=/tmp/tarballs

Manual testing
--------------
Find a recent multi-arch CI run at
https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml
and use its run ID. Use ``--platform`` to select which platform's
artifacts to fetch (defaults to the current system).

Expected output: one .tar.gz per family in ``--output-dir``, named
``therock-dist-{platform}-{family}-{version}.tar.gz``. If
KPACK_SPLIT_ARTIFACTS is enabled in the build, also a
``therock-dist-{platform}-multiarch-{version}.tar.gz``.

Each tarball should contain a standard install prefix layout
(``bin/``, ``lib/``, ``include/``, ``share/``, etc.) with GPU-specific
files (e.g. ``lib/hipblaslt/library/*.co``) only for the target family.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shlex
import subprocess
import sys
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


def run_command(args: list[str | Path], cwd: Path | None = None):
    args = [str(arg) for arg in args]
    log(f"++ Exec{f' [{cwd}]' if cwd else ''}$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL)


def fetch_and_flatten(
    *,
    run_id: str,
    amdgpu_families: list[str],
    platform: str,
    output_dir: Path,
    download_cache_dir: Path,
    run_github_repo: str | None = None,
):
    """Fetch artifacts for one or more families and flatten into output_dir."""
    families_str = ";".join(amdgpu_families)
    log(f"\n{'='*60}")
    log(f"Fetching artifacts for {families_str}")
    log(f"{'='*60}")

    cmd = [
        sys.executable,
        "build_tools/artifact_manager.py",
        "fetch",
        f"--run-id={run_id}",
        "--stage=all",
        f"--amdgpu-families={families_str}",
        "--expand-family-to-targets",
        f"--platform={platform}",
        f"--output-dir={output_dir}",
        "--flatten",
        f"--download-cache-dir={download_cache_dir}",
    ]
    if run_github_repo:
        cmd.append(f"--run-github-repo={run_github_repo}")
    run_command(cmd)


def is_kpack_split(flatten_dir: Path) -> bool:
    """Check if KPACK_SPLIT_ARTIFACTS is enabled from the build manifest."""
    manifest_path = flatten_dir / "share" / "therock" / "therock_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("flags", {}).get("KPACK_SPLIT_ARTIFACTS", False)


def compress_tarball(*, source_dir: Path, tarball_path: Path):
    """Compress a directory into a .tar.gz tarball.

    Uses subprocess ``tar cfz`` rather than Python's ``tarfile`` module
    (tarfile was significantly slower and produced larger output with default
    settings — its ``compresslevel`` parameter may help but was not tuned).

    Uses gzip to match the existing release tarball format. Switching to
    zstd (``tar cf - . | zstd``) would be faster with better compression,
    but requires downstream consumers to support ``.tar.zst``.
    """
    log(f"\nCompressing {source_dir} -> {tarball_path}")
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(["tar", "cfz", str(tarball_path), "."], cwd=source_dir)
    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    log(f"  Created {tarball_path.name} ({size_mb:.1f} MB)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch multi-arch artifacts and package into per-family tarballs"
    )
    parser.add_argument("--run-id", required=True, help="Workflow run ID to fetch from")
    parser.add_argument(
        "--run-github-repo",
        type=str,
        default=None,
        help="GitHub repository for --run-id in 'owner/repo' format. "
        "Defaults to GITHUB_REPOSITORY env var or 'ROCm/TheRock'",
    )
    parser.add_argument(
        "--dist-amdgpu-families",
        required=True,
        help="Semicolon-separated GPU families (e.g. 'gfx94X-dcgpu;gfx110X-all')",
    )
    parser.add_argument(
        "--platform",
        default="linux",
        choices=["linux", "windows"],
        help="Platform to fetch artifacts for",
    )
    parser.add_argument(
        "--package-version",
        required=True,
        help="ROCm package version string for tarball naming",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for tarballs",
    )
    args = parser.parse_args(argv)
    # Normalize empty string to None (workflow inputs default to "")
    args.run_github_repo = args.run_github_repo or None

    families = [f.strip() for f in args.dist_amdgpu_families.split(";") if f.strip()]
    if not families:
        raise ValueError("No GPU families specified")

    work_dir = args.output_dir / ".work"
    download_cache_dir = work_dir / "download-cache"
    download_cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Building tarballs for {len(families)} families: {', '.join(families)}")
    log(f"  Platform: {args.platform}")
    log(f"  Version: {args.package_version}")
    log(f"  Output: {args.output_dir}")

    # Phase 1: Fetch and flatten sequentially.
    # Sequential so the shared download cache avoids re-downloading generic
    # (host) artifacts for each family.
    family_dirs = []
    compress_tasks = []
    for family in families:
        flatten_dir = work_dir / family
        fetch_and_flatten(
            run_id=args.run_id,
            amdgpu_families=[family],
            platform=args.platform,
            output_dir=flatten_dir,
            download_cache_dir=download_cache_dir,
            run_github_repo=args.run_github_repo,
        )
        family_dirs.append(flatten_dir)
        tarball_name = (
            f"therock-dist-{args.platform}-{family}-{args.package_version}.tar.gz"
        )
        compress_tasks.append((flatten_dir, args.output_dir / tarball_name))

    # Phase 1.5: If KPACK_SPLIT_ARTIFACTS is enabled, fetch all families
    # into a single combined directory. With KPACK split, device-specific
    # files are per individual GPU target and don't conflict, so all
    # families can coexist in a single install prefix.
    kpack_split = is_kpack_split(family_dirs[0])
    if kpack_split and len(families) > 1:
        log("::: KPACK_SPLIT_ARTIFACTS detected — building multi-arch tarball")
        multiarch_dir = work_dir / "multiarch"
        fetch_and_flatten(
            run_id=args.run_id,
            amdgpu_families=families,
            platform=args.platform,
            output_dir=multiarch_dir,
            download_cache_dir=download_cache_dir,
            run_github_repo=args.run_github_repo,
        )
        tarball_name = (
            f"therock-dist-{args.platform}-multiarch-{args.package_version}.tar.gz"
        )
        compress_tasks.append((multiarch_dir, args.output_dir / tarball_name))

    # Phase 2: Compress all tarballs in parallel.
    # Each tar cfz is single-threaded, so running N families concurrently
    # on a multi-core runner scales well with minimal per-job slowdown.
    # TODO: Add --compress-workers flag to cap concurrency on smaller runners.
    log(f"\nCompressing {len(compress_tasks)} tarballs in parallel...")
    with ProcessPoolExecutor(max_workers=len(compress_tasks)) as executor:
        futures = {
            executor.submit(compress_tarball, source_dir=src, tarball_path=dst): dst
            for src, dst in compress_tasks
        }
        for future in as_completed(futures):
            future.result()  # Raises on failure

    log(f"\nDone. Tarballs in {args.output_dir}:")
    for tb in sorted(args.output_dir.glob("*.tar.gz")):
        size_mb = tb.stat().st_size / (1024 * 1024)
        log(f"  {tb.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
