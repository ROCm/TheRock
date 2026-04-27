#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Editable-install PyTorch with ROCm support for local development.

Sets up the ROCm build environment and runs `pip install -e` on a PyTorch
source tree. Intended for developers iterating on PyTorch + ROCm — for CI
and production wheel builds, use build_prod_wheels.py instead.

## Quick start

    # From a venv with rocm[libraries,devel] installed:
    python build_dev.py --pytorch-dir /path/to/pytorch

    # Or let it install rocm first:
    python build_dev.py --pytorch-dir /path/to/pytorch --install-rocm

    # With ccache for faster rebuilds:
    python build_dev.py --pytorch-dir /path/to/pytorch --use-ccache
"""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from pytorch_build_env import (
    capture,
    do_install_rocm,
    get_rocm_env,
    get_version_suffix_for_installed_rocm_package,
    prepare_pytorch_build,
    remove_dir_if_exists,
    run_command,
    sanity_check_pytorch,
)


def do_build(args: argparse.Namespace):
    if args.install_rocm:
        do_install_rocm(args)

    # Resolve to absolute so subprocess `cwd=` calls work with relative input.
    pytorch_dir: Path = args.pytorch_dir.resolve()
    version_suffix = get_version_suffix_for_installed_rocm_package()
    rocm_dir, env = get_rocm_env(pytorch_rocm_arch=args.pytorch_rocm_arch)

    if args.use_ccache:
        if not shutil.which("ccache"):
            raise RuntimeError(
                "ccache not found but --use-ccache was specified. "
                "Please install ccache before building."
            )
        print("Building with ccache, clearing stats first")
        env["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        run_command(["ccache", "--zero-stats"], cwd=tempfile.gettempdir())

    try:
        _do_build_pytorch(args, pytorch_dir, env, version_suffix)
    finally:
        if args.use_ccache:
            ccache_stats_output = capture(
                ["ccache", "--show-stats"], cwd=tempfile.gettempdir()
            )
            print(f"ccache --show-stats output:\n{ccache_stats_output}")


def _do_build_pytorch(
    args: argparse.Namespace,
    pytorch_dir: Path,
    env: dict[str, str],
    version_suffix: str,
):
    prepare_pytorch_build(
        env,
        pytorch_dir,
        version_suffix=version_suffix,
        pip_cache_dir=args.pip_cache_dir,
    )

    if args.clean:
        remove_dir_if_exists(pytorch_dir / "build")

    print("+++ Editable-installing pytorch:")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "-v",
            "--no-build-isolation",
        ],
        cwd=pytorch_dir,
        env=env,
    )

    sanity_check_pytorch()


def main(argv: list[str]):
    p = argparse.ArgumentParser(
        prog="build_dev.py",
        description="Editable-install PyTorch with ROCm support for local development.",
    )
    p.add_argument(
        "--pytorch-dir",
        type=Path,
        required=True,
        help="PyTorch source directory",
    )
    p.add_argument(
        "--pytorch-rocm-arch",
        help="gfx arch to build pytorch with (defaults to rocm-sdk targets)",
    )
    p.add_argument(
        "--use-ccache",
        action="store_true",
        default=False,
        help="Use ccache as the compiler launcher",
    )
    p.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        help="Clean build directory before building",
    )
    p.add_argument(
        "--install-rocm",
        action=argparse.BooleanOptionalAction,
        help="Install rocm-sdk before building",
    )
    p.add_argument(
        "--rocm-extras",
        default="",
        help=(
            "Comma-separated additional extras for rocm package install "
            "(e.g. 'device-gfx942,device-gfx943'). "
            "Added alongside the base 'libraries,devel' extras."
        ),
    )
    p.add_argument("--index-url", help="Base URL of the Python Package Index.")
    p.add_argument(
        "--find-links",
        help="URL or path for pip --find-links (flat package index).",
    )
    p.add_argument("--pip-cache-dir", type=Path, help="Pip cache dir")
    p.add_argument(
        "--rocm-sdk-version",
        default=">1.0",
        help="rocm-sdk version to match (with comparison prefix)",
    )
    p.add_argument(
        "--pre",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include pre-release packages (default True)",
    )

    args = p.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
