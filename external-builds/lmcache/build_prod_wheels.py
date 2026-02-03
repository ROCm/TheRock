#!/usr/bin/env python
r"""Builds production LMCache wheel with ROCm support.

LMCache requires PyTorch with ROCm as a build dependency. The build process
uses TheRock's manylinux build container with Python-packaged ROCm.

## Building

Typical usage to build:

```
python build_prod_wheels.py \
    --output-dir outputs \
    --index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu
```

The index URL should point to Python-packaged ROCm for your GPU's gfx architecture.

## Building Linux portable wheels

The wheel is built in a manylinux container and uses auditwheel to ensure
portability.
"""

import argparse
from pathlib import Path
import platform
import shlex
import subprocess
import sys

script_dir = Path(__file__).resolve().parent


def run_command(args: list[str | Path], cwd: Path = None):
    """Run a command and print it."""
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd or Path.cwd()}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)


def find_built_wheel(dist_dir: Path) -> Path:
    """Find the built LMCache wheel."""
    all_wheels = list(dist_dir.glob("lmcache-*.whl"))
    if not all_wheels:
        raise RuntimeError(f"No LMCache wheels found in {dist_dir}")
    if len(all_wheels) != 1:
        raise RuntimeError(f"Found multiple wheels in {dist_dir}: {all_wheels}")
    return all_wheels[0]


def do_build(args: argparse.Namespace):
    """Build LMCache wheel using Docker."""
    
    # Build using Docker
    print("++ Building LMCache wheel in Docker container...")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    build_cmd = [
        "docker", "build",
        "--progress=plain",  # Show full build output including RUN command stdout
        "--target", "export",
        "--output", f"type=local,dest={args.output_dir}",
        "--build-arg", f"PYTHON_VERSION={args.python_version}",
        "--build-arg", f"INDEX_URL={args.index_url}",
        "--build-arg", f"PYTORCH_ROCM_ARCH={args.rocm_arch}",
        "--build-arg", f"LMCACHE_BRANCH={args.lmcache_branch}",
        "-f", "Dockerfile",
        ".",
    ]
    
    if args.no_cache:
        build_cmd.insert(2, "--no-cache")
    
    run_command(build_cmd, cwd=script_dir)
    
    # Find and report the built wheel
    built_wheel = find_built_wheel(args.output_dir)
    wheel_size_mb = built_wheel.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*80}")
    print(f"✅ Build complete!")
    print(f"📦 Wheel: {built_wheel}")
    print(f"   Size: {wheel_size_mb:.1f} MB")
    print(f"{'='*80}\n")
    
    print("To install:")
    print(f"  pip install --extra-index-url {args.index_url} {built_wheel.name}")


def main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog="build_prod_wheels.py",
        description="Build LMCache wheels with ROCm support"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to copy built wheels to",
    )
    parser.add_argument(
        "--python-version",
        default=".".join(platform.python_version_tuple()[:2]),
        type=str,
        help="Python version to use for the build (e.g., 3.10, 3.11, 3.12)",
    )
    parser.add_argument(
        "--index-url",
        required=True,
        help="Index URL for Python-packaged ROCm matching your GPU's gfx arch (e.g., https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu)",
    )
    parser.add_argument(
        "--rocm-arch",
        default="gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201",
        help="ROCm GPU architectures (semicolon-separated)",
    )
    parser.add_argument(
        "--lmcache-branch",
        default="dev",
        help="LMCache branch to checkout (default: dev)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without using Docker cache",
    )
    
    args = parser.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
