#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks out UCX (Unified Communication X).

Helper script to checkout upstream UCX sources.

Primary usage:

    ./ucx_repo.py checkout

The checkout process:

* Clones the UCX repository with a requested ``--repo-hashtag``
  (defaults to ``master``).
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_NAME = "ucx"
DEFAULT_ORIGIN = "https://github.com/openucx/ucx.git"
DEFAULT_HASHTAG = "master"


def run_command(
    args: list[str | Path], cwd: Path, *, stdout_devnull: bool = False
) -> None:
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(
        args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL if stdout_devnull else None,
    )


def do_checkout(args: argparse.Namespace) -> None:
    repo_dir: Path = args.repo
    check_git_dir = repo_dir / ".git"

    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
        run_command(
            ["git", "remote", "set-url", "origin", args.gitrepo_origin],
            cwd=repo_dir,
        )
    else:
        print(f"Cloning UCX repository at {args.repo_hashtag}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        run_command(["git", "init", "--initial-branch=main"], cwd=repo_dir)
        run_command(
            ["git", "config", "advice.detachedHead", "false"], cwd=repo_dir
        )
        run_command(
            ["git", "remote", "add", "origin", args.gitrepo_origin],
            cwd=repo_dir,
        )

    # Fetch and checkout.
    fetch_args = []
    if args.depth is not None:
        fetch_args.extend(["--depth", str(args.depth)])
    if args.jobs:
        fetch_args.extend(["-j", str(args.jobs)])
    run_command(
        ["git", "fetch"] + fetch_args + ["origin", args.repo_hashtag],
        cwd=repo_dir,
    )
    run_command(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

    print(f"UCX checkout complete: {repo_dir}")


def main(cl_args: list[str]) -> None:
    p = argparse.ArgumentParser(prog="ucx_repo.py")
    sub_p = p.add_subparsers(required=True)

    checkout_p = sub_p.add_parser(
        "checkout", help="Clone UCX locally and checkout"
    )
    checkout_p.add_argument(
        "--repo",
        type=Path,
        default=THIS_DIR / DEFAULT_REPO_NAME,
        help="Git repository path",
    )
    checkout_p.add_argument(
        "--gitrepo-origin",
        default=DEFAULT_ORIGIN,
        help="Git repository URL",
    )
    checkout_p.add_argument(
        "--repo-hashtag",
        default=DEFAULT_HASHTAG,
        help="Git repository ref/tag to checkout",
    )
    checkout_p.add_argument(
        "--depth", type=int, help="Fetch depth"
    )
    checkout_p.add_argument(
        "--jobs", default=10, type=int, help="Number of fetch jobs"
    )
    checkout_p.set_defaults(func=do_checkout)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
