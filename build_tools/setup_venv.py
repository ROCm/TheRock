#!/usr/bin/env python

"""Sets up a Python venv and optionally installs rocm packages into it.

* https://docs.python.org/3/library/venv.html
* https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

There are a few modes this can be used in:

* Minimally, this is equivalent to `python -m venv .venv`:

    ```
    python setup_venv.py .venv
    ```

* To install the latest nightly rocm packages for gfx110X-dgpu into the venv:

    ```
    python setup_venv.py .venv --packages rocm[libraries,devel] \
        --index-name nightly --index-subdir gfx110X-dgpu
    ```

    This is roughly equivalent to:

    ```
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install rocm[libraries,devel] --index-url=https://.../gfx110X-dgpu
    deactivate
    ```
"""

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

from github_actions.github_actions_utils import *

is_windows = platform.system() == "Windows"

INDEX_URLS_MAP = {
    "nightly": "https://d2awnip2yjpvqn.cloudfront.net/v2",
    "dev": "https://d25kgig7rdsyks.cloudfront.net/v2",
}


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def exec(args: list[str | Path], cwd: Path = Path.cwd()):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def find_venv_python(venv_path: Path) -> Path | None:
    paths = [venv_path / "bin" / "python", venv_path / "Scripts" / "python.exe"]
    for p in paths:
        if p.exists():
            return p
    return None


def create_venv(venv_dir: Path):
    cwd = Path.cwd()

    log(f"Creating venv at '{venv_dir}'")

    # Log some other variations of the path too.
    try:
        venv_dir_relative = venv_dir.relative_to(cwd)
    except ValueError:
        venv_dir_relative = venv_dir
    venv_dir_resolved = venv_dir.resolve()
    log(f"  Relative dir: '{venv_dir_relative}'")
    log(f"  Resolved dir: '{venv_dir_resolved}'")
    log("")

    # Create with 'python -m venv' as needed.
    python_exe = find_venv_python(venv_dir)
    if python_exe:
        log(f"  Found existing python executable at '{python_exe}', skipping creation")
        log("  Run again with --clean to clear the existing directory instead")
    else:
        exec([sys.executable, "-m", "venv", str(venv_dir)])


def upgrade_pip(python_exe: Path):
    log("")
    exec([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])


def install_packages(args: argparse.Namespace):
    log("")

    python_exe = find_venv_python(args.venv_dir)

    if args.index_name:
        index_url = INDEX_URLS_MAP[args.index_name]
    else:
        index_url = args.index_url
    index_url = index_url.rstrip("/") + "/" + args.index_subdir.strip("/")

    command = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        f"--index-url={index_url}",
        args.packages,
    ]
    if args.disable_cache:
        command.append("--no-cache-dir")
    exec(command)


def activate_venv_in_gha(venv_dir: Path):
    log("")
    log(f"Activating venv for future GitHub Actions workflow steps")
    gha_warn_if_not_running_on_ci()

    # See https://docs.python.org/3/library/venv.html#how-venvs-work.
    #
    # The usual way to activate a venv is to run the platform-specific command:
    #   POSIX bash         : `source <venv>/bin/activate`
    #   Windows cmd.exe    : `<venv>\Scripts\activate.bat`
    #   Windows powershell : `<venv>\Scripts\Activate.ps1`
    #   etc.
    #
    # What these scripts actually do is a combination of setting environment
    # variables, which we can't normally do (persistently) from a Python script.
    # However, in the context of a GitHub Actions workflow, we *can* set
    # environment variables (and job outputs, and step summaries, etc.) using
    # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions.

    if is_windows:
        gha_add_to_path(venv_dir / "Scripts")
    else:
        gha_add_to_path(venv_dir / "bin")
    gha_set_env({"VIRTUAL_ENV": venv_dir})


def log_activate_instructions(venv_dir: Path):
    log("")
    log(f"Setup complete at '{venv_dir}'! Activate the venv with:")
    if is_windows:
        log(f"  {venv_dir}\\Scripts\\activate.bat")
    else:
        log(f"  source {venv_dir}/bin/activate")


def run(args: argparse.Namespace):
    venv_dir = args.venv_dir

    if args.clean and venv_dir.exists():
        log(f"Clearing existing venv_dir '{venv_dir}'")
        shutil.rmtree(venv_dir)

    create_venv(venv_dir)
    python_exe = find_venv_python(venv_dir)

    upgrade_pip(python_exe)
    if args.packages:
        install_packages(args)

    if args.activate_in_future_github_actions_steps:
        activate_venv_in_gha(venv_dir)
    else:
        log_activate_instructions(venv_dir)


def main(argv: list[str]):
    p = argparse.ArgumentParser("setup_venv.py")
    p.add_argument(
        "venv_dir",
        type=Path,
        help="Directory in which to create the venv, such as '.venv'",
    )

    general_options = p.add_argument_group("General options")
    general_options.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        help="If the venv directory already exists, clear it and start fresh",
    )
    general_options.add_argument(
        "--disable-cache",
        action=argparse.BooleanOptionalAction,
        help="Disables the pip cache through the --no-cache-dir option",
    )
    general_options.add_argument(
        "--activate-in-future-github-actions-steps",
        action=argparse.BooleanOptionalAction,
        help="Attempts to activate the venv persistently when running in a GitHub Action. This is less reliable than running the official activate command",
    )

    install_options = p.add_argument_group("Install options")

    # TODO(#1036): Other flags or helper scripts to help map between versions,
    #              git commits/refs, workflow runs, etc.
    #              I'd like a shorthand for "install packages from commit abcde"
    install_options.add_argument(
        "--packages",
        type=str,
        help="List of packages to install, including any extras or explicit versions",
    )

    index_group = install_options.add_mutually_exclusive_group()
    # TODO(#1036): add "auto" mode here that infers the index from the version?
    # TODO(#1036): Default to nightly?
    index_group.add_argument(
        "--index-name",
        type=str,
        choices=["nightly", "dev"],
        help="Shorthand name for an index to use with 'pip install --index-url='",
    )
    index_group.add_argument(
        "--index-url",
        type=str,
        help="Full URL for a release index to use with 'pip install --index-url='",
    )

    # TODO(#1036): Enumerate possible options here (hardcode or scrape) and
    #              help the user make a choice
    install_options.add_argument(
        "--index-subdir",
        "--index-subdirectory",
        type=str,
        help="Index subdirectory, such as 'gfx110X-dgpu'",
    )

    args = p.parse_args(argv)

    # Validate arguments.
    if args.venv_dir.exists() and not args.venv_dir.is_dir():
        p.error(f"venv_dir '{args.venv_dir}' exists and is not a directory")
    if args.packages and not (args.index_name or args.index_url):
        p.error("If --packages is set, one of --index-name or --index-url must be set")
    if args.packages and not args.index_subdir:
        p.error("If --packages is set, --index-subdir must be set")

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
