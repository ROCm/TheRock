#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Build a production, multi-architecture UCCL wheel from ROCm wheels.

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

2. Build UCCL for the supported CDNA architectures.

Typical usage to build:

```
# On Linux, using the default path for the repository:
python build_prod_wheels.py \
    --output-dir /tmp/pyout
```

## Building Linux portable wheels

UCCL's build process already produces a manylinux portable wheel. No
additional processing is required.
"""

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys

script_dir = Path(__file__).resolve().parent

is_windows = platform.system() == "Windows"

DEFAULT_ROCM_ARCHES = "gfx90a,gfx942,gfx950,gfx1250"
DEFAULT_ROCM_INDEX_URL = "https://rocm.prereleases.amd.com/whl-multi-arch/"
ROCM_ARCH_RE = re.compile(r"gfx[0-9a-f]+")


def parse_rocm_arches(value: str) -> str:
    """Validate and normalize UCCL's comma-separated ROCm target list."""
    arches = [arch.strip().lower() for arch in value.split(",")]
    if not arches or any(not arch for arch in arches):
        raise argparse.ArgumentTypeError(
            "ROCm architectures must be a comma-separated list"
        )
    invalid = [arch for arch in arches if ROCM_ARCH_RE.fullmatch(arch) is None]
    if invalid:
        raise argparse.ArgumentTypeError(
            "Invalid ROCm architecture(s): " + ", ".join(invalid)
        )
    if len(arches) != len(set(arches)):
        raise argparse.ArgumentTypeError("ROCm architectures must not be duplicated")
    return ",".join(arches)


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
        raise RuntimeError("UCCL does not build on Windows")

    if uccl_dir:
        build_script = uccl_dir / "build.sh"
        if not build_script.is_file():
            raise FileNotFoundError(f"UCCL build script not found: {build_script}")

        # build.sh forwards PYTORCH_ROCM_ARCH into the build container. UCCL's
        # ep/setup.py turns each comma-separated entry into --offload-arch.
        build_env = {"PYTORCH_ROCM_ARCH": args.rocm_arches}
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
            env=build_env,
        )

        built_wheel = find_built_wheel(uccl_dir / "wheelhouse-therock", "uccl")
        print(f"Found built wheel: {built_wheel}")
        copy_to_output(args, built_wheel)
    else:
        raise FileNotFoundError(
            "UCCL source directory was not found; run "
            "'python uccl_repo.py checkout' or pass --uccl-dir"
        )


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
        "--index-url",
        default=DEFAULT_ROCM_INDEX_URL,
        help=f"Python package index URL (default: {DEFAULT_ROCM_INDEX_URL})",
    )
    p.add_argument(
        "--rocm-arches",
        default=os.environ.get("PYTORCH_ROCM_ARCH", DEFAULT_ROCM_ARCHES),
        type=parse_rocm_arches,
        help=(
            "Comma-separated ROCm targets compiled into the wheel "
            f"(default: {DEFAULT_ROCM_ARCHES})"
        ),
    )

    args = p.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
