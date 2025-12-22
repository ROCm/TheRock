#!/usr/bin/env python
"""Helper script to bump TheRock's submodules, doing the following:
 * (Optional) Creates a new branch
 * Updates submodules from remote using `fetch_sources.py`
 * Creates a commit and tries to apply local patches
 * (Optional) Pushed the new branch to origin
 * (Optional) Create pull request from the new branch to origin (requires gh cli installed)
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
import shutil

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

def get_component_submodules(component: str) -> list[str]:
    """
    Returns the list of submodule paths belonging to the component.
    Reads .gitmodules to ensure accuracy.
    """

    gitmodules_path = THEROCK_DIR / ".gitmodules"
    content = gitmodules_path.read_text()

    paths = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("path = "):
            sub_path = line.split("=", 1)[1].strip()
            if sub_path == component or sub_path.startswith(component + "/"):
                paths.append(sub_path)
    return paths


def get_submodule_hash(path: str) -> str:
    """
    Returns the commit hash for a submodule path.
    """
    output = subprocess.check_output(
        ["git", "submodule", "status", path],
        cwd=str(THEROCK_DIR),
        text=True
    ).strip()

    commit = output.split()[0].lstrip("+ -")
    return commit


def get_hashes_for_component(component: str) -> dict:
    """
    Returns {submodule_path: hash} for the component.
    """
    submodules = get_component_submodules(component)

    hashes = {}
    for sm in submodules:
        try:
            h = get_submodule_hash(sm)
            hashes[sm] = h
        except Exception as e:
            log(f"Failed to read hash for {sm}: {e}")
    return hashes


def detect_hash_change(old: dict, new: dict):
    """
    Return (old_hash, new_hash, path) for the first changed submodule.
    """
    for path in new:
        old_h = old.get(path)
        new_h = new[path]
        if old_h != new_h:
            log(f"CHANGE DETECTED in {path}: {old_h} -> {new_h}")
            return old_h, new_h, path

    log("No changes detected for component")
    return None, None, None

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
    component = args.components[0]
    old_hashes = get_hashes_for_component(component)

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

    # Detect new hashes BEFORE commit/branch creation
    new_hashes = get_hashes_for_component(component)
    old_h, new_h, changed_path = detect_hash_change(old_hashes, new_hashes)
    if not old_h or not new_h:
        log("No submodule changes detected – aborting without commit, branch, or PR.")
        return
    
    if args.create_branch or args.push_branch or args.create_pr:
        exec(["git", "checkout", "-b", args.branch_name], cwd=THEROCK_DIR)
    
    # Prepare commit/PR title/body
    pr_title = f"Bump {component} from {old_h[:7]} to {new_h[:7]}"
    pr_body = (
        f"Bump happened on {datetime.today().strftime('%m%d%Y')}\n\n"
        "### Submodule changes:\n"
        + "".join(
            f"- `{sm}`: {old_hashes.get(sm)} → {new_hashes.get(sm)}\n"
            for sm in new_hashes
        )
    )
    commit_msg = pr_title + "\n\n" + pr_body
    exec(["git", "commit", "-a", "-m", commit_msg], cwd=THEROCK_DIR)

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
        # Create PR
        if args.create_pr:
            if not shutil.which("gh"):
                raise SystemExit("ERROR: --create-pr requires the `gh` CLI.\n")      
            exec(
                [
                    "gh", "pr", "create",
                    "--title", pr_title,
                    "--body", pr_body,
                    "--head", args.branch_name,
                    "--base", "main",
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
    parser.add_argument(
        "--create-pr",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Create a GitHub PR (requires `gh` CLI)",
    )
    args = parser.parse_args(argv)
    fetch_args, system_projects = parse_components(args.components)
    run(args, fetch_args, system_projects)


if __name__ == "__main__":
    main(sys.argv[1:])