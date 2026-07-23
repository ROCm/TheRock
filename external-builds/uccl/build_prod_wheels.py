#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Builds production UCCL wheel based on the rocm wheels.

The UCCL project already has a TheRock build target that is currently
based on the prerelease (or nightly) wheels.

## Building interactively

A build consists of the two steps below:

1. Checkout repository:

The following command checks out the upstream repo into this directory,
which the script will use by default if they exist. Otherwise, checkout your
own and specify with `--uccl-dir` during the build step.

```
# On Linux, using default paths (nested under this folder):
python uccl_repo.py checkout
```

2. Build UCCL for a single gfx architecture.

Typical usage to build:

```
# On Linux, using the default path for the repository:
python build_prod_wheels.py \
    --output-dir $HOME/tmp/pyout \
    --index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu
```

## Building Linux portable wheels

UCCL's build process already produces a manylinux portable wheel. No
additional processing is required.
"""

import argparse
from datetime import date
import json
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import textwrap

script_dir = Path(__file__).resolve().parent

is_windows = platform.system() == "Windows"

# Map rocm-sdk wheel family names (the trailing component of the package index
# URL, e.g. "gfx94X-dcgpu") to the concrete --offload-arch list that hipcc /
# clang++ actually accepts. UCCL's ep/setup.py reads PYTORCH_ROCM_ARCH and
# passes each entry to --offload-arch=..., so the values here must be real
# GPU arch IDs (gfx942, gfx950, ...), not family names.
FAMILY_TO_DEFAULT_ARCH: dict[str, str] = {
    "gfx94X-dcgpu": "gfx942",
}


def parse_family_from_index_url(index_url: str) -> str | None:
    # Index URLs look like https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu
    tail = index_url.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.startswith("gfx") else None


def resolve_rocm_arch(args: argparse.Namespace) -> str | None:
    # Priority: explicit env override > --amdgpu-family flag > parsed from --index-url.
    env_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    if env_arch:
        return env_arch
    family = args.amdgpu_family or parse_family_from_index_url(args.index_url)
    if family is None:
        return None
    arch = FAMILY_TO_DEFAULT_ARCH.get(family)
    if arch is None:
        print(
            f"WARNING: no PYTORCH_ROCM_ARCH mapping for family '{family}'. "
            f"Falling back to ep/setup.py defaults; build may fail with "
            f"'unsupported HIP gpu architecture'. Override with "
            f"PYTORCH_ROCM_ARCH=<gfxNNN> or --amdgpu-family.",
            file=sys.stderr,
        )
    return arch


def run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(f":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


def copy_to_output(args: argparse.Namespace, src_file: Path):
    output_dir: Path = args.output_dir
    print(f"++ Copy {src_file} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, output_dir)


def directory_if_exists(dir: Path) -> Path | None:
    if dir.exists():
        return dir
    else:
        return None


def find_built_wheel(dist_dir: Path, dist_package: str) -> Path:
    dist_package = dist_package.replace("-", "_")
    glob = f"{dist_package}-*.whl"
    all_wheels = list(dist_dir.glob(glob))
    if not all_wheels:
        raise RuntimeError(f"No wheels matching '{glob}' found in {dist_dir}")
    if len(all_wheels) != 1:
        raise RuntimeError(f"Found multiple wheels matching '{glob}' in {dist_dir}")
    return all_wheels[0]


def do_build(args: argparse.Namespace):
    uccl_dir: Path | None = args.uccl_dir

    if is_windows:
        print("WARNING: UCCL does not builds on Windows.", file=sys.stderr)

    # Build UCCL
    if uccl_dir:
        build_env: dict[str, str] = {}
        rocm_arch = resolve_rocm_arch(args)
        if rocm_arch:
            # build.sh forwards PYTORCH_ROCM_ARCH into the build container, and
            # uccl's ep/setup.py reads it to drive --offload-arch.
            build_env["PYTORCH_ROCM_ARCH"] = rocm_arch
        run_command(
            [
                "./build.sh",
                "therock",
                "all",
                args.python_version,
                args.index_url,
                args.image,
            ],
            cwd=uccl_dir,
            env=build_env or None,
        )

        built_wheel = find_built_wheel(uccl_dir / "wheelhouse-therock", "uccl")
        print(f"Found built wheel: {built_wheel}")
        copy_to_output(args, built_wheel)
    else:
        print("--- Not building UCCL (no --uccl-dir)")


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="build_prod_wheels.py")

    p.add_argument(
        "--image",
        default="ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:a382085df3ba2419b58aa9051350883a0d0b732a4bc0a4ef60458f8161bb08c6",
        help="Base docker image for UCCL's build",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to copy built wheels to",
    )
    p.add_argument(
        "--uccl-dir",
        default=directory_if_exists(script_dir / "uccl"),
        type=Path,
        help="UCCL source directory",
    )
    p.add_argument(
        "--python-version",
        default=".".join(platform.python_version_tuple()[:2]),
        type=str,
        help="Python version to use for the build",
    )
    p.add_argument(
        "--index-url", required=True, help="Base URL of the Python Package Index."
    )
    p.add_argument(
        "--amdgpu-family",
        default=None,
        help=(
            "AMD GPU family name as used by the rocm-sdk wheels (e.g. "
            "'gfx94X-dcgpu'). Defaults to the trailing path component of "
            "--index-url. Maps to a concrete PYTORCH_ROCM_ARCH via "
            "FAMILY_TO_DEFAULT_ARCH. Override directly via the "
            "PYTORCH_ROCM_ARCH environment variable."
        ),
    )

    args = p.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
