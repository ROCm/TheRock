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


def pin_ck():
    requirements_file_path = (
        THEROCK_DIR / "rocm-libraries" / "projects" / "miopen" / "requirements.txt"
    )
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

def get_component_submodule_hashes(component: str, cwd: Path):
    """
    Returns a dict {submodule_path: commit_hash} for submodules located
    inside the given component directory.
    """
    output = subprocess.check_output(
        ["git", "submodule", "status"],
        cwd=str(cwd),
        text=True
    )

    results = {}
    prefix = f"{component}/"

    for line in output.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        commit = parts[0].lstrip("+ -")
        path = parts[1]

        if path.startswith(prefix):
            results[path] = commit

    return results


def diff_component_hashes(old: dict, new: dict):
    """
    Compute old → new hash changes for submodules in the component.
    Returns (old_hash, new_hash).
    If multiple submodules exist, returns the *first changed pair*.
    """
    for path in new:
        old_h = old.get(path)
        new_h = new[path]
        if old_h != new_h:
            return old_h, new_h
    return None, None


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

    if "comm-libs" in components:
        system_projects += [
            "rccl",
            "rccl-tests",
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
    
    old_hashes = {}
    for comp in args.components:
        old_hashes[comp] = get_component_submodule_hashes(comp, THEROCK_DIR)

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
        # Capture new hashes for each component
        new_hashes = {}
        for comp in args.components:
            new_hashes[comp] = get_component_submodule_hashes(comp, THEROCK_DIR)

        exec(
            ["git", "push", "-u", "origin", args.branch_name],
            cwd=THEROCK_DIR,
        )

        # Determine hash changes for the *single component or combined name*
        component_name = "-".join(args.components)

        old_hash = None
        new_hash = None

        # Get the first changed pair across requested components
        for comp in args.components:
            o, n = diff_component_hashes(old_hashes[comp], new_hashes[comp])
            if o and n:
                old_hash, new_hash = o[:7], n[:7]
                break

        if not old_hash or not new_hash:
            log("WARNING: Could not detect hash changes for the requested components.")
            old_hash = "unknown"
            new_hash = "unknown"

        # PR Title and Body
        pr_title = f"Bump {component_name} from {old_hash} to {new_hash}"
        pr_date = datetime.today().strftime("%m%d%Y")
        pr_body = f"Bump happened on {pr_date}"

        # Create PR with gh
        exec(
            [
                "gh", "pr", "create",
                "--title", pr_title,
                "--body", pr_body,
                "--head", args.branch_name,
                "--base", "main",        
                "--reviewer", "ScottTodd",
                "--reviewer", "marbre",
                "--reviewer", "geomin12",
                "--reviewer", "jayhawk-commits",
            ],
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
                  ml-libs,
                  rocm-libraries,
                  rocm-systems,
                  profiler
             """,
    )
    args = parser.parse_args(argv)
    fetch_args, system_projects = parse_components(args.components)
    run(args, fetch_args, system_projects)


if __name__ == "__main__":
    main(sys.argv[1:])
