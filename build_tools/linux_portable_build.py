#!/usr/bin/env python
"""Builds TheRock in a manylinux based container.

Example usage:

    # Build for a specific family. Note that all options after the "--" are
    # passed verbatim to CMake.
    python linux_build_portable.py -- -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu

    # Build with podman vs docker.
    python linux_build_portable.py --docker=podman

    # Enter an interactive shell set up like the build.
    python linux_build_portable.py --interactive

Other options of note:

* `--image`: Change the default build image
* `--output-dir`: Change the output directory, which contains caches and build
     or Python packages
* `--artifact-dir`: Change the source artifacts/ directory to build Python
     packages from
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent


def exec(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd))


def do_build(args: argparse.Namespace, *, rest_args: list[str]):
    if args.pull:
        exec([args.docker, "pull", args.image], cwd=THIS_DIR)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cl = [
        args.docker,
        "run",
        "--rm",
    ]
    if sys.stdin.isatty():
        cl.extend(["-i", "-t"])

    # Mount options that must go before execution options.
    cl.extend(
        [
            "--mount",
            f"type=bind,src={output_dir},dst=/therock/output",
            "--mount",
            f"type=bind,src={args.repo_dir},dst=/therock/src",
        ]
    )
    if args.build_python_only:
        cl.extend(
            [
                "--mount",
                f"type=bind,src={args.artifact_dir},dst=/therock/artifacts",
            ]
        )

    # Type of execution requested.
    if args.exec:
        cl.extend(
            [
                args.image,
            ]
        )
        cl += rest_args
    elif args.interactive:
        cl.extend(
            [
                "-it",
                args.image,
                "/bin/bash",
            ]
        )
    elif args.build_python_only:
        cl.extend(
            [
                args.image,
                "/bin/bash",
                "/therock/src/build_tools/detail/linux_python_package_in_container.sh",
            ]
        )
        cl += rest_args
    else:
        cl.extend(
            [
                args.image,
                "/bin/bash",
                "/therock/src/build_tools/detail/linux_portable_build_in_container.sh",
            ]
        )
        cl += rest_args

    cl = [str(arg) for arg in cl]
    print(f"++ Exec [{THIS_DIR}]$ {shlex.join(cl)}")
    try:
        p = subprocess.Popen(cl, cwd=str(THIS_DIR))
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
    p.wait()
    sys.exit(p.returncode)


def main(argv: list[str]):
    try:
        rest_pos = argv.index("--")
    except ValueError:
        rest_args = []
    else:
        rest_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    p = argparse.ArgumentParser(prog="linux_build_portable.py")
    p.add_argument("--docker", default="docker", help="Docker or podman binary")
    p.add_argument(
        "--image",
        default="ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:3ac188c17f88f08ce522297b616d8308361b8e9a9b31bcc3c3bbb9429b1efa6c",
        help="Build docker image",
    )
    p.add_argument(
        "--pull",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull docker image",
    )
    p.add_argument(
        "--output-dir",
        default=Path(REPO_DIR / "output-linux-portable"),
        type=Path,
        help="Output directory",
    )
    p.add_argument(
        "--repo-dir", default=REPO_DIR, help="Root directory of this repository"
    )
    p.add_argument(
        "--exec",
        action=argparse.BooleanOptionalAction,
        help="Execute arguments verbatim vs running a specific script",
    )
    p.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        help="Enter interactive shell vs invoking the build",
    )
    p.add_argument(
        "--build-python-only",
        action="store_true",
        default=False,
        help="Build only Python packages",
    )
    p.add_argument(
        "--artifact-dir",
        default=Path(REPO_DIR / "output-linux-portable" / "build" / "artifacts"),
        type=Path,
        help="Source artifacts/ dir from a build",
    )

    args = p.parse_args(argv)
    do_build(args, rest_args=rest_args)


if __name__ == "__main__":
    main(sys.argv[1:])
