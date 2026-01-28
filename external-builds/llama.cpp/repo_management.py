import argparse
from pathlib import Path
import shlex
import subprocess

TAG_UPSTREAM_DIFFBASE = "THEROCK_UPSTREAM_DIFFBASE"


def exec(args: list[str | Path], cwd: Path, *, stdout_devnull: bool = False):
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(
        args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL if stdout_devnull else None,
    )


def do_checkout(args: argparse.Namespace):
    repo_dir: Path = args.repo
    check_git_dir = repo_dir / ".git"

    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
        exec(["git", "remote", "set-url", "origin", args.gitrepo_origin], cwd=repo_dir)
    else:
        print(f"Cloning repository at {args.repo_hashtag}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        exec(["git", "init", "--initial-branch=main"], cwd=repo_dir)
        exec(["git", "config", "advice.detachedHead", "false"], cwd=repo_dir)
        exec(["git", "remote", "add", "origin", args.gitrepo_origin], cwd=repo_dir)

    fetch_args = []
    if args.depth is not None:
        fetch_args.extend(["--depth", str(args.depth)])
    if args.jobs:
        fetch_args.extend(["-j", str(args.jobs)])

    exec(["git", "fetch"] + fetch_args + ["origin", args.repo_hashtag], cwd=repo_dir)
    exec(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
    exec(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE, "--no-sign"], cwd=repo_dir)
