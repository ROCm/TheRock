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

Bump comm-lib submodules and create a branch
```
./build_tools/bump_submodules.py \
    --create-branch --branch-name shared/bump-comm-libs --components comm-libs
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
            "rocm-core",
            "rocm_smi_lib",
            "rocprofiler-register",
        ]

    if "comm-libs" in components:
        system_projects += [
            "rccl",
            "rccl-tests",
        ]

    if "core" in components:
        # amdgpu-windows-interop is Windows only and is updated manually.
        system_projects += [
            "HIP",
            "ROCR-Runtime",
            "clr",
            "rocminfo",
        ]

    if "profiler" in components:
        system_projects += [
            "aqlprofile",
            "rocprof-trace-decoder",
            "rocprofiler-sdk",
            "roctracer",
        ]

    if "math-libs" in components:
        arguments.append("--include-math-libs")
    else:
        arguments.append("--no-include-math-libs")

    if "ml-libs" in components:
        arguments.append("--include-ml-frameworks")
    else:
        arguments.append("--no-include-ml-frameworks")

    if "compiler" in components:
        arguments.append("--include-compilers")
    else:
        arguments.append("--no-include-compilers")

    log(f"++ Arguments: {shlex.join(arguments)}")
    if system_projects:
        log(f"++ System projects: {shlex.join(system_projects)}")

    return [arguments, system_projects]


def run(args: argparse.Namespace, fetch_args: list[str], system_projects: list[str]):
    date = datetime.today().strftime("%Y%m%d")

    if args.create_branch or args.push_branch:
        exec(
            ["git", "checkout", "-b", args.branch_name],
            cwd=THEROCK_DIR,
        )

    if system_projects:
        projects_args = ["--system-projects"] + system_projects
    else:
        projects_args = []

    exec(
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
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default="default",
        help="""List of components (subdirectories) to bump. Choices:
                  default,
                  base,
                  comm-libs,
                  compiler,
                  core,
                  math-libs,
                  ml-libs,
                  profiler
             """,
    )
    args = parser.parse_args(argv)
    fetch_args, system_projects = parse_components(args.components)
    run(args, fetch_args, system_projects)


if __name__ == "__main__":
    main(sys.argv[1:])
