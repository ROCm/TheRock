#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Builds UCX with ROCm support.

UCX (Unified Communication X) is an external project that builds on top of
ROCm. This script handles the autotools-based build: autogen, configure, make,
and install.

## Building interactively

A full build is a two-step process:

1. Checkout UCX sources::

    python ucx_repo.py checkout

2. Build UCX with ROCm::

    python build_ucx.py \
        --rocm-path /path/to/rocm \
        --output-dir ./output

The build produces the UCX installation (including the gtest binary used for
ROCm integration testing) in ``--output-dir``.
"""

import argparse
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

script_dir = Path(__file__).resolve().parent

is_windows = platform.system() == "Windows"


def run_command(
    args: list[str | Path],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


def do_build(args: argparse.Namespace) -> None:
    if is_windows:
        print("ERROR: UCX does not support Windows.", file=sys.stderr)
        sys.exit(1)

    ucx_dir: Path | None = args.ucx_dir
    rocm_path: Path = args.rocm_path
    output_dir: Path = args.output_dir

    if ucx_dir is None or not ucx_dir.exists():
        print(
            "ERROR: UCX source directory not found. "
            "Run 'python ucx_repo.py checkout' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not rocm_path.exists():
        print(
            f"ERROR: ROCm path does not exist: {rocm_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve all paths so they work correctly when cwd changes to build_dir.
    ucx_dir = ucx_dir.resolve()
    rocm_path = rocm_path.resolve()
    build_dir = ucx_dir / "build"
    install_prefix = output_dir.resolve()
    install_prefix.mkdir(parents=True, exist_ok=True)

    # Log the UCX commit being built for CI traceability.
    try:
        ucx_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ucx_dir), stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        ucx_commit = "unknown"

    print(f"Building UCX with ROCm support")
    print(f"  UCX source:     {ucx_dir}")
    print(f"  UCX commit:     {ucx_commit}")
    print(f"  ROCm path:      {rocm_path}")
    print(f"  Build dir:      {build_dir}")
    print(f"  Install prefix: {install_prefix}")

    num_jobs = str(args.jobs or os.cpu_count() or 4)

    # Step 1: autogen
    print("\n=== Step 1/4: autogen ===")
    run_command(["./autogen.sh"], cwd=ucx_dir)

    # Step 2: configure
    print("\n=== Step 2/4: configure ===")
    build_dir.mkdir(parents=True, exist_ok=True)
    configure_args = [
        "../contrib/configure-release",
        "--disable-logging",
        "--disable-debug",
        "--disable-assertions",
        "--enable-params-check",
        f"--prefix={install_prefix}",
        "--without-knem",
        "--without-cuda",
        f"--with-rocm={rocm_path}",
        "--enable-gtest",
        "--without-gdrcopy",
        "--without-java",
    ]
    run_command(configure_args, cwd=build_dir)

    # Step 3: make
    print("\n=== Step 3/4: make ===")
    run_command(["make", f"-j{num_jobs}"], cwd=build_dir)

    # Step 4: make install
    print("\n=== Step 4/4: make install ===")
    run_command(["make", f"-j{num_jobs}", "install"], cwd=build_dir)

    # Verify gtest binary was built
    gtest_path = build_dir / "test" / "gtest" / "gtest"
    if gtest_path.exists():
        print(f"\nUCX build successful. Gtest binary: {gtest_path}")
    else:
        print(
            f"\nWARNING: UCX gtest binary not found at {gtest_path}",
            file=sys.stderr,
        )

    print(f"UCX installed to: {install_prefix}")


def directory_if_exists(dir: Path) -> Path | None:
    if dir.exists():
        return dir
    return None


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="build_ucx.py",
        description="Build UCX with ROCm support",
    )
    p.add_argument(
        "--ucx-dir",
        type=Path,
        default=directory_if_exists(script_dir / "ucx"),
        help="UCX source directory (default: ./ucx)",
    )
    p.add_argument(
        "--rocm-path",
        type=Path,
        required=True,
        help="Path to ROCm installation",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to install UCX into",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel build jobs (default: cpu count)",
    )

    args = p.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
