#!/usr/bin/env python
r"""Builds production UCCL wheel based on the rocm wheels.

The UCCL project already has a TheRock build target that is currently
based on the nightly wheels.
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


def exec(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
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
    uccl_dir: Path = args.uccl_dir

    if is_windows:
        print("WARNING: Unsure if UCCL builds on Windows.", file=sys.stderr)

    if not uccl_dir.exists():
        exec(
            [
                "git",
                "clone",
                "--recursive",
                "https://github.com/uccl-project/uccl.git",
                str(uccl_dir),
            ],
            cwd=script_dir,
        )

    exec(
        ["./build.sh", "therock", "all", args.python_version, args.index_url],
        cwd=uccl_dir,
    )

    built_wheel = find_built_wheel(uccl_dir / "wheelhouse-therock", "uccl")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="build_prod_wheels.py")

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to copy built wheels to",
    )
    p.add_argument(
        "--uccl-dir",
        default=Path(script_dir / "uccl"),
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

    args = p.parse_args(argv)
    do_build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
