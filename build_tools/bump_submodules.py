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
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
