#!/usr/bin/env python
"""Helper script to bump TheRock's submodules, doing the following:
 * (Optional) Creates a new branch
 * Updates submodules from remote using `fetch_sources.py`
 * Creares a commit and tries to apply local patches
 * (Optional) Pushed the new branch to origin

The submodules to bump can be specified via `--components`.

Examples:
Bump submpdules in base, core and profiler
```
./build_tools/bump_submodules.py \
    --components base core profiler
```

Bump rocm-systems submodule and create a branch
```
./build_tools/bump_submodules.py \
    --create-branch --branch-name shared/bump-rocm-systems --components rocm-systems
```
"""

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


def run_command(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def parse_components(components: list[str]) -> list[list]:
    arguments = []
    system_projects = []

    # If `default` is passed, use the defaults set in `fetch_sources.py` by not passing additonal arguments.
    if "default" in components:
        return [], []

    if any(comp in components for comp in ["base", "comm-libs", "core", "profiler"]):
        arguments.append("--include-system-projects")
    else:
        arguments.append("--no-include-system-projects")

    if "base" in components:
        system_projects += [
            "half",
            "rocm-cmake",
        ]

    if "profiler" in components:
        system_projects += [
            "rocprof-trace-decoder",
        ]

    if "rocm-libraries" in components:
        arguments.append("--include-rocm-libraries")
        arguments.append("--include-ml-frameworks")
    else:
        arguments.append("--no-include-rocm-libraries")

        if "ml-libs" in components:
            arguments.append("--include-ml-frameworks")
        else:
            arguments.append("--no-include-ml-frameworks")

    if "rocm-systems" in components:
        arguments.append("--include-rocm-systems")
    else:
        arguments.append("--no-include-rocm-systems")

    if "compiler" in components:
        arguments.append("--include-compilers")
    else:
        arguments.append("--no-include-compilers")

    if "iree-libs" in components:
        arguments.append("--include-iree-libs")
    else:
        arguments.append("--no-include-iree-libs")

    if "debug-tools" in components:
        arguments.append("--include-debug-tools")
    else:
        arguments.append("--no-include-debug-tools")

    if "rocm-media" in components:
        arguments.append("--include-rocm-media")
    else:
        arguments.append("--no-include-rocm-media")

    if "math-libraries" in components:
        arguments.append("--include-math-libraries")
    else:
        arguments.append("--no-include-math-libraries")

    log(f"++ Arguments: {shlex.join(arguments)}")
    if system_projects:
        log(f"++ System projects: {shlex.join(system_projects)}")

    return [arguments, system_projects]


def run(args: argparse.Namespace, fetch_args: list[str], system_projects: list[str]):
    date = datetime.today().strftime("%Y%m%d")

    if args.create_branch or args.push_branch:
        run_command(
            ["git", "checkout", "-b", args.branch_name],
            cwd=THEROCK_DIR,
        )

    if system_projects:
        projects_args = ["--system-projects"] + system_projects
    else:
        projects_args = []

    run_command(
        [
            sys.executable,
            "./build_tools/fetch_sources.py",
            "--remote",
            "--no-apply-patches",
        ]
        + fetch_args
        + projects_args,
        cwd=THEROCK_DIR,
    )

    run_command(
        ["git", "commit", "-a", "-m", "Bump submodules " + date],
        cwd=THEROCK_DIR,
    )

    try:
        run_command(
            [sys.executable, "./build_tools/fetch_sources.py"],
            cwd=THEROCK_DIR,
        )
    except subprocess.CalledProcessError as patching_error:
        log("Failed to apply patches")
        sys.exit(1)

    if args.push_branch:
        run_command(
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
        "--components",
        type=str,
        nargs="+",
        default="default",
        help="""List of components (subdirectories) to bump. Choices:
                  default,
                  base,
                  compiler,
                  ml-libs,
                  rocm-libraries,
                  rocm-systems,
                  profiler,
                  iree-libs,
                  debug-tools,
                  rocm-media,
                  math-libraries
             """,
    )
    args = parser.parse_args(argv)
    fetch_args, system_projects = parse_components(args.components)
    run(args, fetch_args, system_projects)


if __name__ == "__main__":
    main(sys.argv[1:])
