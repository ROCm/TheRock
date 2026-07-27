#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks out the LMCache source used by TheRock builds.

Primary usage:

    python lmcache_repo.py checkout
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKOUT_DIR = SCRIPT_DIR / "lmcache"
DEFAULT_ORIGIN = "https://github.com/LMCache/LMCache.git"
DEFAULT_REF = "fa8d5e7f4afdd9450d814ea697c73a576d85d4bb"


def run_command(args: list[str | Path], cwd: Path) -> None:
    command = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(command)}", flush=True)
    subprocess.check_call(command, cwd=str(cwd), stdin=subprocess.DEVNULL)


def checkout(args: argparse.Namespace) -> None:
    checkout_dir = args.checkout_dir.expanduser().resolve()
    git_dir = checkout_dir / ".git"

    if git_dir.exists():
        print(f"++ Reusing LMCache checkout: {checkout_dir}")
        run_command(
            ["git", "remote", "set-url", "origin", args.gitrepo_origin],
            cwd=checkout_dir,
        )
    else:
        if checkout_dir.exists() and any(checkout_dir.iterdir()):
            raise RuntimeError(
                f"Checkout directory exists and is not empty: {checkout_dir}"
            )
        checkout_dir.mkdir(parents=True, exist_ok=True)
        run_command(["git", "init", "--initial-branch=main"], cwd=checkout_dir)
        run_command(["git", "config", "advice.detachedHead", "false"], cwd=checkout_dir)
        run_command(
            ["git", "remote", "add", "origin", args.gitrepo_origin],
            cwd=checkout_dir,
        )

    fetch_command = ["git", "fetch", "--tags"]
    if args.depth is not None:
        fetch_command.extend(["--depth", str(args.depth)])
    fetch_command.extend(["origin", args.repo_hashtag])
    run_command(fetch_command, cwd=checkout_dir)
    run_command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout_dir)
    run_command(["git", "rev-parse", "HEAD"], cwd=checkout_dir)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="lmcache_repo.py")
    subparsers = parser.add_subparsers(required=True)

    checkout_parser = subparsers.add_parser(
        "checkout", help="Clone LMCache locally and check out a revision"
    )
    checkout_parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=DEFAULT_CHECKOUT_DIR,
        help=f"Checkout destination (default: {DEFAULT_CHECKOUT_DIR})",
    )
    checkout_parser.add_argument(
        "--gitrepo-origin",
        default=DEFAULT_ORIGIN,
        help="LMCache Git repository URL",
    )
    checkout_parser.add_argument(
        "--repo-hashtag",
        default=DEFAULT_REF,
        help="LMCache branch, tag, or commit to check out",
    )
    checkout_parser.add_argument(
        "--depth",
        type=int,
        help="Limit fetch history (may reduce source-derived version detail)",
    )
    checkout_parser.set_defaults(func=checkout)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
