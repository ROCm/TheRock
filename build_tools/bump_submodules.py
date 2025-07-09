#!/usr/bin/env python

import argparse
from pathlib import Path
from datetime import datetime
import shlex
import subprocess
import sys

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def exec(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def pin_tensile():
    tag_file_path = THEROCK_DIR / "math-libs" / "BLAS" / "rocBLAS" / "tensile_tag.txt"
    with open(tag_file_path) as tag_file:
        tensile_tag = tag_file.read().splitlines()

    exec(
        ["git", "checkout", tensile_tag[0]],
        cwd=THEROCK_DIR / "math-libs" / "BLAS" / "Tensile",
    )


def pin_ck():
    requirements_file_path = THEROCK_DIR / "ml-libs" / "MIOpen" / "requirements.txt"
    with open(requirements_file_path) as requirements_file:
        requirements = requirements_file.read().splitlines()

    # The requirements file pins several dependencies. And entry for CK looks like:
    # 'ROCm/composable_kernel@778ac24376813d18e63c9f77a2dd51cf87eb4a80 -DCMAKE_BUILD_TYPE=Release'
    # After filtering, the string is split to isolate the CK commit.
    ck_requirement = list(
        filter(lambda x: "rocm/composable_kernel" in x.lower(), requirements)
    )[0]
    ck_commit = ck_requirement.split("@")[-1].split()[0]

    exec(
        ["git", "checkout", ck_commit],
        cwd=THEROCK_DIR / "ml-libs" / "composable_kernel",
    )


def run(args):
    date = datetime.today().strftime("%Y%m%d")

    if args.create_branch or args.push_branch:
        exec(
            ["git", "checkout", "-b", args.branch_name],
            cwd=THEROCK_DIR,
        )

    exec(
        [
            sys.executable,
            "./build_tools/fetch_sources.py",
            "--remote",
            "--no-apply-patches",
        ],
        cwd=THEROCK_DIR,
    )

    if args.pin_tensile:
        pin_tensile()

    if args.pin_ck:
        pin_ck()

    exec(
        ["git", "commit", "-a", "-m", "Bump submodules " + date],
        cwd=THEROCK_DIR,
    )

    try:
        exec(
            [sys.executable, "./build_tools/fetch_sources.py"],
            cwd=THEROCK_DIR,
        )
    except subprocess.CalledProcessError as patching_error:
        log("Failed to apply patches")
        sys.exit(1)

    if args.push_branch:
        exec(
            ["git", "push", "-u", "origin", args.branch_name],
            cwd=THEROCK_DIR,
        )


def main(argv):
    parser = argparse.ArgumentParser(prog="bump_submodules")
    parser.add_argument(
        "--create-branch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Create a branch without pushing",
    )
    parser.add_argument(
        "--branch-name",
        type=str,
        default="integrate",
        help="Name of the branch to create",
    )
    parser.add_argument(
        "--push-branch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Create and push a branch",
    )
    parser.add_argument(
        "--pin-tensile",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Pin Tensile to version tagged in rocBLAS",
    )
    parser.add_argument(
        "--pin-ck",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Pin composable_kernel to version tagged in MIOpen",
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
