#!/usr/bin/env python
"""Fetches the specific pull request, commit or branch from a repo.
This script is available for users, but it is primarily the mechanism
to fetch a monorepo in the CI.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def fetch_branch(args):
    run_command(
        ["git", "checkout", args.remote_branch],
        cwd=args.directory,
    )


def fetch_commit(args):
    additional_args = []
    if args.local_branch:
        additional_args += ["-b", args.local_branch]
    run_command(
        ["git", "checkout"] + additional_args + [args.commit],
        cwd=args.directory,
    )


def fetch_pr(args):
    if args.local_branch:
        local_branch = args.local_branch
    else:
        local_branch = args.pr_number

    run_command(
        ["git", "fetch", "origin", f"pull/{args.pr_number}/head:{local_branch}"],
        cwd=args.directory,
    )
    run_command(
        ["git", "switch", local_branch],
        cwd=args.directory,
    )


def run(args):
    additional_args = []
    if args.depth:
        additional_args += ["--depth", str(args.depth)]
    if args.jobs:
        additional_args += ["--jobs", str(args.jobs)]

    run_command(
        ["git", "clone"] + additional_args + [args.repo, args.directory],
        cwd=THIS_SCRIPT_DIR,
    )

    if args.pr_number:
        fetch_pr(args)

    if args.commit:
        fetch_commit(args)

    if args.remote_branch:
        fetch_branch(args)


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_repo")

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Monorepo to fetch",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Name of a new directory to clone into.",
    )
    parser.add_argument(
        "--local-branch",
        type=str,
        required=False,
        help="(Optional) Local branch to create if checking out a PR or commit",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Git depth when fetch the repository",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Number of jobs to use for fetching the repo",
        default=None,
    )

    checkout = parser.add_mutually_exclusive_group()
    checkout.add_argument(
        "--pr-number",
        type=int,
        help="Pull request (PR) number to checkout",
    )
    checkout.add_argument(
        "--commit",
        type=str,
        help="Commit hash to checkout",
    )
    checkout.add_argument(
        "--remote-branch",
        type=str,
        help="Remote branch to checkout",
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
