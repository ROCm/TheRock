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

By default, generated tarballs exclude test artifacts and fftw3. Pass
``--include-test-tarballs`` to also generate full tarballs, named with a
``-tests`` suffix, that include test artifacts.

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
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from zlib_ng import gzip_ng_threaded

DEFAULT_EXCLUDED_ARTIFACTS: list[str] = ["fftw3"]
DEFAULT_EXCLUDED_COMPONENTS: list[str] = ["test"]
DEFAULT_COMPRESSION_BACKEND = "zlib-ng"
DEFAULT_COMPRESSION_LEVEL = 6
DEFAULT_COMPRESSION_THREADS = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def log_command_duration(*, start_time: float, start_cpu: os.times_result) -> None:
    elapsed = time.monotonic() - start_time
    end_cpu = os.times()
    cpu_user = (end_cpu.user - start_cpu.user) + (
        end_cpu.children_user - start_cpu.children_user
    )
    cpu_system = (end_cpu.system - start_cpu.system) + (
        end_cpu.children_system - start_cpu.children_system
    )
    log(
        f"++ Completed in {elapsed:.1f}s "
        f"(CPU: {cpu_user:.1f}s user, {cpu_system:.1f}s system)"
    )


def run_command(args: list[str | Path], cwd: Path | None = None) -> None:
    args = [str(arg) for arg in args]
    log(f"++ Exec{f' [{cwd}]' if cwd else ''}$ {shlex.join(args)}")
    start_time = time.monotonic()
    start_cpu = os.times()
    try:
        subprocess.check_call(
            args, cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL
        )
    finally:
        log_command_duration(start_time=start_time, start_cpu=start_cpu)


def fetch_and_flatten(
    *,
    run_id: str,
    amdgpu_families: list[str],
    platform: str,
    output_dir: Path,
    download_cache_dir: Path,
    extraction_cache_dir: Path,
    run_github_repo: str | None = None,
    exclude_components: list[str] | None = None,
    exclude_artifacts: list[str] | None = None,
) -> None:
    """Fetch artifacts for one or more families and flatten into output_dir."""
    families_str = ";".join(amdgpu_families)
    log(f"\n{'='*60}")
    log(f"Fetching artifacts for {families_str}")
    if exclude_components:
        log(f"Excluding components: {', '.join(exclude_components)}")
    if exclude_artifacts:
        log(f"Excluding artifacts: {', '.join(exclude_artifacts)}")
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
        f"--extraction-cache-dir={extraction_cache_dir}",
    ]
    if exclude_components:
        cmd.append(f"--exclude-components={','.join(exclude_components)}")
    if exclude_artifacts:
        cmd.append(f"--exclude-artifacts={','.join(exclude_artifacts)}")
    if run_github_repo:
        cmd.append(f"--run-github-repo={run_github_repo}")
    run_command(cmd)
    disk_usage = shutil.disk_usage(output_dir)
    log(
        "  Disk after staging: "
        f"{disk_usage.used / (1024**3):.1f} GiB used, "
        f"{disk_usage.free / (1024**3):.1f} GiB free"
    )


def is_kpack_split(flatten_dir: Path) -> bool:
    """Check if KPACK_SPLIT_ARTIFACTS is enabled from the build manifest."""
    manifest_path = flatten_dir / "share" / "therock" / "therock_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("flags", {}).get("KPACK_SPLIT_ARTIFACTS", False)


def compress_with_zlib_ng(
    *,
    source_dir: Path,
    tarball_path: Path,
    compression_level: int,
    compression_threads: int,
) -> None:
    """Stream a tar archive through zlib-ng's threaded gzip writer."""
    command = ["tar", "cf", "-", "."]
    log(
        f"++ Exec [{source_dir}]$ {shlex.join(command)} | "
        "zlib_ng.gzip_ng_threaded.open"
        f"(level={compression_level}, threads={compression_threads}) > {tarball_path}"
    )
    start_time = time.monotonic()
    start_cpu = os.times()
    tar_process = subprocess.Popen(
        command,
        cwd=source_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    assert tar_process.stdout is not None
    try:
        with tar_process.stdout:
            with gzip_ng_threaded.open(
                tarball_path,
                "wb",
                compresslevel=compression_level,
                threads=compression_threads,
            ) as gzip_output:
                shutil.copyfileobj(
                    tar_process.stdout,
                    gzip_output,
                    length=1024 * 1024,
                )
        return_code = tar_process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if tar_process.poll() is None:
            tar_process.terminate()
        tar_process.wait()
        raise
    finally:
        log_command_duration(start_time=start_time, start_cpu=start_cpu)


def compress_tarball(
    *,
    source_dir: Path,
    tarball_path: Path,
    compression_backend: str = DEFAULT_COMPRESSION_BACKEND,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    compression_threads: int = DEFAULT_COMPRESSION_THREADS,
) -> None:
    """Compress a directory into a .tar.gz tarball.

    The system ``tar`` builds the archive. By default its output is streamed
    through zlib-ng so gzip compression can use multiple threads. The system
    gzip backend is retained for direct performance comparisons.
    """
    log(f"\nCompressing {source_dir} -> {tarball_path}")
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if compression_backend == "zlib-ng":
            compress_with_zlib_ng(
                source_dir=source_dir,
                tarball_path=tarball_path,
                compression_level=compression_level,
                compression_threads=compression_threads,
            )
        elif compression_backend == "system-gzip":
            run_command(["tar", "cfz", str(tarball_path), "."], cwd=source_dir)
        else:
            raise ValueError(f"Unknown compression backend: {compression_backend}")
    except BaseException:
        tarball_path.unlink(missing_ok=True)
        raise
    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    log(f"  Created {tarball_path.name} ({size_mb:.1f} MB)")


def available_cpu_count() -> int:
    """Return the CPUs available to this process, respecting Linux affinity."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def determine_compress_workers(
    *,
    task_count: int,
    requested_workers: int | None,
    compression_backend: str,
    compression_threads: int,
) -> int:
    """Select archive concurrency without oversubscribing compression CPUs."""
    if requested_workers is not None:
        return min(task_count, requested_workers)
    cpus_per_archive = (
        compression_threads + 1 if compression_backend == "zlib-ng" else 1
    )
    return min(task_count, max(1, available_cpu_count() // cpus_per_archive))


def main(argv: list[str] | None = None) -> None:
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
    parser.add_argument(
        "--include-test-tarballs",
        action="store_true",
        help="Also produce -tests tarballs that include test artifacts",
    )
    parser.add_argument(
        "--compression-backend",
        choices=["zlib-ng", "system-gzip"],
        default=DEFAULT_COMPRESSION_BACKEND,
        help="Gzip implementation to use (default: zlib-ng)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(10),
        default=DEFAULT_COMPRESSION_LEVEL,
        help="Gzip compression level for zlib-ng (default: 6)",
    )
    parser.add_argument(
        "--compression-threads",
        type=int,
        default=DEFAULT_COMPRESSION_THREADS,
        help="Compression threads per zlib-ng tarball (default: 8)",
    )
    parser.add_argument(
        "--compress-workers",
        type=int,
        default=None,
        help="Concurrent tarballs to compress (default: auto based on CPU count)",
    )
    args = parser.parse_args(argv)
    if args.compression_threads < 1:
        parser.error("--compression-threads must be at least 1")
    if args.compress_workers is not None and args.compress_workers < 1:
        parser.error("--compress-workers must be at least 1")
    # Normalize empty string to None (workflow inputs default to "")
    args.run_github_repo = args.run_github_repo or None

    families = [f.strip() for f in args.dist_amdgpu_families.split(";") if f.strip()]
    if not families:
        raise ValueError("No GPU families specified")

    work_dir = args.output_dir / ".work"
    download_cache_dir = work_dir / "download-cache"
    extraction_cache_dir = work_dir / "extraction-cache"
    download_cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Building tarballs for {len(families)} families: {', '.join(families)}")
    log(f"  Platform: {args.platform}")
    log(f"  Version: {args.package_version}")
    log(f"  Output: {args.output_dir}")
    log(f"  Include test tarballs: {args.include_test_tarballs}")
    log(f"  Compression backend: {args.compression_backend}")
    log(f"  Compression level: {args.compression_level}")
    log(f"  Compression threads per tarball: {args.compression_threads}")

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
            extraction_cache_dir=extraction_cache_dir,
            run_github_repo=args.run_github_repo,
            exclude_components=DEFAULT_EXCLUDED_COMPONENTS,
            exclude_artifacts=DEFAULT_EXCLUDED_ARTIFACTS,
        )
        family_dirs.append(flatten_dir)
        tarball_name = (
            f"therock-dist-{args.platform}-{family}-{args.package_version}.tar.gz"
        )
        compress_tasks.append((flatten_dir, args.output_dir / tarball_name))
        if args.include_test_tarballs:
            tests_dir = work_dir / "tests" / family
            fetch_and_flatten(
                run_id=args.run_id,
                amdgpu_families=[family],
                platform=args.platform,
                output_dir=tests_dir,
                download_cache_dir=download_cache_dir,
                extraction_cache_dir=extraction_cache_dir,
                run_github_repo=args.run_github_repo,
            )
            tests_tarball_name = (
                f"therock-dist-{args.platform}-{family}-tests-"
                f"{args.package_version}.tar.gz"
            )
            compress_tasks.append((tests_dir, args.output_dir / tests_tarball_name))

    # Phase 1.5: If KPACK_SPLIT_ARTIFACTS is enabled, fetch all families
    # into a single combined directory. With KPACK split, device-specific
    # files are per individual GPU target and don't conflict, so all
    # families can coexist in a single install prefix.
    kpack_split = is_kpack_split(family_dirs[0])
    if kpack_split:
        log("::: KPACK_SPLIT_ARTIFACTS detected — building multi-arch tarball")
        multiarch_dir = work_dir / "multiarch"
        fetch_and_flatten(
            run_id=args.run_id,
            amdgpu_families=families,
            platform=args.platform,
            output_dir=multiarch_dir,
            download_cache_dir=download_cache_dir,
            extraction_cache_dir=extraction_cache_dir,
            run_github_repo=args.run_github_repo,
            exclude_components=DEFAULT_EXCLUDED_COMPONENTS,
            exclude_artifacts=DEFAULT_EXCLUDED_ARTIFACTS,
        )
        tarball_name = (
            f"therock-dist-{args.platform}-multiarch-{args.package_version}.tar.gz"
        )
        compress_tasks.append((multiarch_dir, args.output_dir / tarball_name))
        if args.include_test_tarballs:
            tests_multiarch_dir = work_dir / "tests" / "multiarch"
            fetch_and_flatten(
                run_id=args.run_id,
                amdgpu_families=families,
                platform=args.platform,
                output_dir=tests_multiarch_dir,
                download_cache_dir=download_cache_dir,
                extraction_cache_dir=extraction_cache_dir,
                run_github_repo=args.run_github_repo,
            )
            tests_tarball_name = (
                f"therock-dist-{args.platform}-multiarch-tests-"
                f"{args.package_version}.tar.gz"
            )
            compress_tasks.append(
                (tests_multiarch_dir, args.output_dir / tests_tarball_name)
            )

    # Phase 2: Compress tarballs in parallel, with optional intra-archive
    # parallelism from zlib-ng.
    compress_workers = determine_compress_workers(
        task_count=len(compress_tasks),
        requested_workers=args.compress_workers,
        compression_backend=args.compression_backend,
        compression_threads=args.compression_threads,
    )
    if args.compression_backend == "zlib-ng":
        compression_description = (
            f"zlib-ng level {args.compression_level}, "
            f"{args.compression_threads} threads per tarball"
        )
    else:
        compression_description = "system gzip"
    log(
        f"\nCompressing {len(compress_tasks)} tarballs with {compress_workers} "
        f"workers using {compression_description}..."
    )
    with ProcessPoolExecutor(max_workers=compress_workers) as executor:
        futures = {
            executor.submit(
                compress_tarball,
                source_dir=src,
                tarball_path=dst,
                compression_backend=args.compression_backend,
                compression_level=args.compression_level,
                compression_threads=args.compression_threads,
            ): dst
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
