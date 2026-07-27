#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Builds a portable LMCache wheel with TheRock ROCm Python packages.

The build runs in TheRock's manylinux container and embeds the supported GPU
architectures in one wheel. For example:

```
python build_prod_wheels.py \
    --output-dir outputs
```
"""

import argparse
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_IMAGE = (
    "ghcr.io/rocm/therock_build_manylinux_x86_64@"
    "sha256:a382085df3ba2419b58aa9051350883a0d0b732a4bc0a4ef60458f8161bb08c6"
)
DEFAULT_ROCM_INDEX_URL = "https://rocm.prereleases.amd.com/whl-multi-arch/"
DEFAULT_ROCM_VERSION = "7.14.0rc3"
DEFAULT_TORCH_VERSION = "2.11.0+rocm7.14.0rc3"
DEFAULT_ROCM_ARCHES = (
    "gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1250"
)


def run_command(args: list[str | Path], cwd: Path) -> None:
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd))


def find_built_wheel(dist_dir: Path) -> Path:
    all_wheels = list(dist_dir.glob("lmcache-*.whl"))
    if not all_wheels:
        raise RuntimeError(f"No LMCache wheels found in {dist_dir}")
    if len(all_wheels) != 1:
        names = ", ".join(sorted(wheel.name for wheel in all_wheels))
        raise RuntimeError(f"Expected one LMCache wheel in {dist_dir}; found: {names}")
    return all_wheels[0]


def parse_rocm_arch(value: str) -> str:
    if not re.fullmatch(r"gfx[0-9a-f]+", value):
        raise argparse.ArgumentTypeError(
            "expected one concrete GPU architecture such as gfx942 or gfx1201"
        )
    return value


def parse_rocm_arches(value: str) -> str:
    arches = value.split(";")
    if not arches or any(not arch for arch in arches):
        raise argparse.ArgumentTypeError(
            "expected semicolon-separated concrete GPU architectures"
        )
    for arch in arches:
        parse_rocm_arch(arch)
    if len(set(arches)) != len(arches):
        raise argparse.ArgumentTypeError("GPU architecture list contains duplicates")
    return ";".join(arches)


def python_version_to_abi(value: str) -> str:
    match = re.fullmatch(r"3\.(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("expected a CPython 3.x version such as 3.12")
    python_tag = f"cp3{match.group(1)}"
    return f"{python_tag}-{python_tag}"


def parse_python_version(value: str) -> str:
    python_version_to_abi(value)
    return value


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return parsed


def do_build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lmcache_dir = args.lmcache_dir.expanduser().resolve()
    if not (lmcache_dir / "pyproject.toml").is_file():
        raise RuntimeError(
            f"{lmcache_dir} is not an LMCache checkout. "
            "Run 'python external-builds/lmcache/lmcache_repo.py checkout' first."
        )

    print(f"++ ROCm index: {args.rocm_index_url}")
    print(f"++ ROCm version: {args.rocm_version}")
    print(f"++ Build device architecture: {args.build_device_arch}")
    print(f"++ Wheel GPU architectures: {args.rocm_arches}")
    print(f"++ PyTorch version: {args.torch_version}")
    run_command(["git", "rev-parse", "HEAD"], cwd=lmcache_dir)

    build_cmd = [
        "docker",
        "build",
        "--target",
        "export",
        "--progress",
        "plain",
        "--output",
        f"type=local,dest={output_dir}",
        "--build-arg",
        f"BASE_IMAGE={args.image}",
        "--build-arg",
        f"PYTHON_ABI={python_version_to_abi(args.python_version)}",
        "--build-arg",
        f"ROCM_INDEX_URL={args.rocm_index_url}",
        "--build-arg",
        f"ROCM_VERSION={args.rocm_version}",
        "--build-arg",
        f"BUILD_DEVICE_ARCH={args.build_device_arch}",
        "--build-arg",
        f"ROCM_ARCHES={args.rocm_arches}",
        "--build-arg",
        f"TORCH_VERSION={args.torch_version}",
        "--build-arg",
        f"MAX_JOBS={args.max_jobs}",
        "--file",
        SCRIPT_DIR / "Dockerfile",
        lmcache_dir,
    ]

    if args.no_cache:
        build_cmd.insert(2, "--no-cache")

    run_command(build_cmd, cwd=SCRIPT_DIR)

    built_wheel = find_built_wheel(output_dir)
    wheel_size_mb = built_wheel.stat().st_size / 1024 / 1024
    print(f"++ Built wheel: {built_wheel} ({wheel_size_mb:.1f} MiB)")
    print(f"++ Install with: python -m pip install {shlex.quote(str(built_wheel))}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="build_prod_wheels.py",
        description="Build an LMCache wheel with ROCm support",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to write the wheel",
    )
    parser.add_argument(
        "--lmcache-dir",
        type=Path,
        default=SCRIPT_DIR / "lmcache",
        help="LMCache source checkout (default: external-builds/lmcache/lmcache)",
    )
    parser.add_argument(
        "--build-device-arch",
        type=parse_rocm_arch,
        default="gfx942",
        help="Device package used in the build environment (default: gfx942)",
    )
    parser.add_argument(
        "--rocm-arches",
        type=parse_rocm_arches,
        default=DEFAULT_ROCM_ARCHES,
        help="Semicolon-separated GPU architectures embedded in the wheel",
    )
    parser.add_argument(
        "--rocm-index-url",
        default=DEFAULT_ROCM_INDEX_URL,
        help="TheRock multi-arch Python package index",
    )
    parser.add_argument(
        "--rocm-version",
        default=DEFAULT_ROCM_VERSION,
        help="ROCm package version",
    )
    parser.add_argument(
        "--torch-version",
        default=DEFAULT_TORCH_VERSION,
        help="PyTorch package version compatible with the LMCache ref",
    )
    parser.add_argument(
        "--python-version",
        type=parse_python_version,
        default=".".join(platform.python_version_tuple()[:2]),
        help="CPython version provided by the build image (default: current version)",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Pinned TheRock manylinux build image",
    )
    parser.add_argument(
        "--max-jobs",
        type=positive_integer,
        default=8,
        help="Maximum parallel compiler jobs (default: 8)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without the Docker layer cache",
    )

    do_build(parser.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
